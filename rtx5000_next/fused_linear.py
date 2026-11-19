"""FusedNVFP4Linear: an NVFP4 linear layer that torch.compile can see.

Why this exists
    The old fusion patched torchao's NVFP4Tensor.to_nvfp4. Inside torch.compile'd blocks that patch was
    not taken (nsys: our kernel ran 6 times per step instead of 286), so "fused + compile" measured the
    compiled stock graph. This module does not use torchao's tensor-subclass dispatch at all:

        x --(custom op: quantize, writes FP4 codes + swizzled block scales)--> --(torch._scaled_mm)--> y
        y = y * (act_pts * weight_pts) [+ bias]        (same arithmetic as torchao's _addmm_nvfp4_dispatch)

    Both pieces are plain ops in the traced graph, so inductor fuses the output scale, bias, gate and
    residual into one elementwise pass. The custom ops are opaque, so a counter inside them proves that
    every site really ran our kernel (see COUNTS).

Options (all training-free)
    static activation scale   act_amax given: no per-call amax pass over x (one less full read of x)
    shared quantize           several linears that read the same input (Q, K, V) quantize it once
    concat                    shared quantize plus one GEMM with the weights concatenated along N
                              (one per-tensor weight scale for the group; slightly different from separate)
    selective                 convert() can skip small / sensitive layers (they stay bf16)

CPU use: with fast=False the GEMM is a reference op that dequantizes both operands (torchao's own
dequantize) and calls F.linear. That makes every CPU test possible. fast=True (default on CUDA) uses
torch._scaled_mm and needs an sm100+/sm120 GPU.
"""
import collections

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchao.prototype.mx_formats.nvfp4_tensor import (
    NVFP4Tensor,
    hp_data_dims_to_swizzled_scale_dims_nvfp4,
    per_tensor_amax_to_scale,
)
from torchao.prototype.mx_formats.utils import to_blocked

import quant_backends as qb

BF16 = torch.bfloat16
BLOCK = 16
COUNTS = collections.Counter()  # incremented inside the custom ops, so it also counts under torch.compile


def reset_counts():
    COUNTS.clear()


def _ref_quant_impl(x, pts):
    """Pure-torch quantize (torchao eager path). Returns FP4 codes and flat swizzled scales."""
    q = NVFP4Tensor.to_nvfp4(x, block_size=BLOCK, per_tensor_scale=pts.reshape(()), is_swizzled_scales=True)
    return q.qdata, q.scale.reshape(-1)


qb.IMPLS.setdefault("torchref", (_ref_quant_impl, True))


def _counted(name, impl):
    def counted(x, pts):
        COUNTS[f"quant:{name}"] += 1
        return impl(x, pts)

    return counted


def quant_op(backend):
    """The custom op nvfp4fusion::<backend> with a counter. Created once per backend."""
    impl, native_swizzled = qb.IMPLS[backend]
    return qb._as_op(backend, _counted(backend, impl), native_swizzled), native_swizzled


_REF_MM = {}


def ref_mm_op():
    """Reference GEMM as an opaque op (dequantize both sides, F.linear). CPU-safe, used when fast=False."""
    if "op" in _REF_MM:
        return _REF_MM["op"]

    @torch.library.custom_op("nvfp4fusion::ref_mm", mutates_args=())
    def op(a_q: torch.Tensor, a_s: torch.Tensor, a_pts: torch.Tensor,
           w_q: torch.Tensor, w_s: torch.Tensor, w_pts: torch.Tensor) -> torch.Tensor:
        COUNTS["mm:ref"] += 1
        a = NVFP4Tensor(a_q, a_s, BLOCK, BF16, a_pts, None, True, False, None).dequantize(BF16)
        w = NVFP4Tensor(w_q, w_s, BLOCK, BF16, w_pts, None, True, False, None).dequantize(BF16)
        return F.linear(a, w)

    @op.register_fake
    def _(a_q, a_s, a_pts, w_q, w_s, w_pts):
        return a_q.new_empty((a_q.shape[0], w_q.shape[0]), dtype=BF16)

    _REF_MM["op"] = op
    return op


def quantize_weight(w):
    """bf16 weight (N, K) -> FP4 codes (N, K/2), swizzled block scales, per-tensor scale (fp32, 0-dim)."""
    pts = per_tensor_amax_to_scale(w.detach().abs().max())
    q = NVFP4Tensor.to_nvfp4(w.detach().contiguous(), block_size=BLOCK, per_tensor_scale=pts, is_swizzled_scales=True)
    return q.qdata.contiguous(), q.scale.contiguous(), pts.float()


class FusedNVFP4Linear(nn.Module):
    """One or more linears that read the same input. Returns a tensor (one linear) or a tuple."""

    def __init__(self, linears, backend="triton", act_amax=None, concat=False, fast=None):
        super().__init__()
        ws = [l.weight.detach() for l in linears]
        bs = [None if l.bias is None else l.bias.detach() for l in linears]
        self.sizes = [w.shape[0] for w in ws]
        self.K = ws[0].shape[1]
        assert all(w.shape[1] == self.K for w in ws) and self.K % BLOCK == 0
        self.concat = concat and len(ws) > 1
        if self.concat:
            ws = [torch.cat(ws, 0)]
            bs = [None if any(b is None for b in bs) else torch.cat(bs, 0)]
        self.n_sets = len(ws)
        for i, (w, b) in enumerate(zip(ws, bs)):
            q, s, p = quantize_weight(w)
            self.register_buffer(f"wq{i}", q)
            self.register_buffer(f"ws{i}", s)
            self.register_buffer(f"wp{i}", p)
            self.register_buffer(f"b{i}", None if b is None else b.to(BF16).contiguous())
        if act_amax is None:
            self.register_buffer("act_pts", None)
        else:
            self.register_buffer("act_pts", per_tensor_amax_to_scale(torch.tensor(float(act_amax), device=ws[0].device)))
        self.backend, self.fast = backend, fast

    def extra_repr(self):
        return f"K={self.K}, sizes={self.sizes}, concat={self.concat}, static={self.act_pts is not None}, backend={self.backend}"

    def _quantize(self, x2):
        M, K = x2.shape
        pts = self.act_pts if self.act_pts is not None else per_tensor_amax_to_scale(x2.detach().abs().max())
        op, native = quant_op(self.backend)
        packed, scales = op(x2, pts)
        if not native:  # logical (M, K/16) scales -> the 128x4 swizzled layout _scaled_mm expects
            scales = to_blocked(scales.view(M, K // BLOCK)).flatten()
        sm, sk = hp_data_dims_to_swizzled_scale_dims_nvfp4(M, K)
        return packed, scales.view(sm, sk), pts

    def _mm(self, a_q, a_s, a_pts, i):
        wq, ws, wp, b = getattr(self, f"wq{i}"), getattr(self, f"ws{i}"), getattr(self, f"wp{i}"), getattr(self, f"b{i}")
        fast = a_q.is_cuda if self.fast is None else self.fast
        if fast:
            y = torch._scaled_mm(a_q.view(torch.float4_e2m1fn_x2), wq.t().view(torch.float4_e2m1fn_x2),
                                 a_s.view(torch.float8_e4m3fn), ws.view(torch.float8_e4m3fn), out_dtype=BF16)
        else:
            y = ref_mm_op()(a_q, a_s, a_pts, wq, ws, wp)
            return y + b if b is not None else y  # reference op already applied both per-tensor scales
        y = y * (a_pts * wp).to(BF16)  # per-tensor scales after the GEMM, then bias: torchao's order
        return y + b if b is not None else y

    def forward(self, x):
        lead = x.shape[:-1]
        x2 = x.reshape(-1, self.K)
        if x2.dtype != BF16:
            x2 = x2.to(BF16)
        a_q, a_s, a_pts = self._quantize(x2.contiguous())
        outs = [self._mm(a_q, a_s, a_pts, i) for i in range(self.n_sets)]
        if self.concat:
            outs = list(outs[0].split(self.sizes, -1))
        outs = [o.reshape(*lead, o.shape[-1]) for o in outs]
        return outs[0] if len(outs) == 1 else tuple(outs)


def convert_pixart(transformer, backend="triton", static_amax=None, shared_qkv=True, concat=False, selective=False,
                   outside_blocks=True, fast=None):
    """Replace the NVFP4 sites of a bf16 PixArt transformer in place. Returns a report dict.

    static_amax  dict {original linear module name: amax} from calibrate_static_scales.py, or None (dynamic)
    shared_qkv   quantize the shared input once for Q/K/V (self) and K/V (cross); concat adds one GEMM
    selective    keep small or sensitive layers in bf16: cross-attention K/V (only 300 text rows, highest
                 error in the Q01 ablation), caption projection, timestep embedder, adaln, final proj
    outside_blocks  also convert caption_projection / adaln / proj_out (stock NVFP4 quantizes them too)
    """
    static_amax = static_amax or {}
    rep = collections.Counter()

    def amax_for(name):
        return static_amax.get(name)

    def make(linears, name, group_concat=False):
        rep["modules"] += 1
        rep["sites"] += 1
        return FusedNVFP4Linear(linears, backend, amax_for(name), concat=group_concat, fast=fast)

    for bi, block in enumerate(transformer.transformer_blocks):
        pre = f"transformer_blocks.{bi}"
        a1, a2 = block.attn1, block.attn2
        if shared_qkv:
            a1.fused_qkv = make([a1.to_q, a1.to_k, a1.to_v], f"{pre}.attn1.to_q", concat)
            a1.to_q = a1.to_k = a1.to_v = None
        else:
            for n in ("to_q", "to_k", "to_v"):
                setattr(a1, n, make([getattr(a1, n)], f"{pre}.attn1.{n}"))
        a1.to_out[0] = make([a1.to_out[0]], f"{pre}.attn1.to_out.0")
        a2.to_q = make([a2.to_q], f"{pre}.attn2.to_q")
        if selective:
            rep["skipped"] += 2
        elif shared_qkv:
            a2.fused_kv = make([a2.to_k, a2.to_v], f"{pre}.attn2.to_k", concat)
            a2.to_k = a2.to_v = None
        else:
            a2.to_k = make([a2.to_k], f"{pre}.attn2.to_k")
            a2.to_v = make([a2.to_v], f"{pre}.attn2.to_v")
        a2.to_out[0] = make([a2.to_out[0]], f"{pre}.attn2.to_out.0")
        block.ff.net[0].proj = make([block.ff.net[0].proj], f"{pre}.ff.net.0.proj")
        block.ff.net[2] = make([block.ff.net[2]], f"{pre}.ff.net.2")

    if outside_blocks and not selective:
        tr = transformer
        for holder, attr, name in (
            (tr.caption_projection, "linear_1", "caption_projection.linear_1"),
            (tr.caption_projection, "linear_2", "caption_projection.linear_2"),
            (tr.adaln_single.emb.timestep_embedder, "linear_1", "adaln_single.emb.timestep_embedder.linear_1"),
            (tr.adaln_single.emb.timestep_embedder, "linear_2", "adaln_single.emb.timestep_embedder.linear_2"),
            (tr.adaln_single, "linear", "adaln_single.linear"),
            (tr, "proj_out", "proj_out"),
        ):
            lin = getattr(holder, attr)
            if lin.in_features % BLOCK == 0:
                setattr(holder, attr, make([lin], name))
    return dict(rep)
