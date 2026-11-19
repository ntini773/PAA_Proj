"""CPU tests for fused_linear.py: quantize op, reference GEMM, shared QKV, static scales, fake shapes, compile engagement.

The compile-engagement test is the detector for the bug seen in the nsys data: the old monkeypatch was not taken
inside torch.compile'd blocks. Here the counters live inside the custom ops, so a compiled run must call them once per site.
"""
import _boot
from _boot import check, torch

import fused_linear as fl
from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor, per_tensor_amax_to_scale

F = torch.nn.functional
BF = torch.bfloat16


def rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm())


def lin(k, n, bias=True, seed=0):
    torch.manual_seed(seed)
    m = torch.nn.Linear(k, n, bias=bias).to(BF)
    return m


def test_quantize_op_and_reference_gemm():
    x = (torch.randn(48, 64) * 2).to(BF)
    L = lin(64, 32)
    m = fl.FusedNVFP4Linear([L], backend="torchref", fast=False)
    y = m(x)
    # reference: dequantize both operands with torchao and run F.linear, then bias (same arithmetic as the module)
    pts = per_tensor_amax_to_scale(x.abs().max())
    xq = NVFP4Tensor.to_nvfp4(x, per_tensor_scale=pts, is_swizzled_scales=True).dequantize(BF)
    wq = NVFP4Tensor.to_nvfp4(L.weight.detach(), per_tensor_scale=per_tensor_amax_to_scale(L.weight.abs().max()),
                              is_swizzled_scales=True).dequantize(BF)
    ref = F.linear(xq, wq) + L.bias
    check("fused linear == dequantized reference", torch.equal(y, ref.to(BF)) or rel(y, ref) < 1e-2, f"rel {rel(y, ref):.4f}")
    check("fused linear close to bf16 linear (fp4 error)", rel(y, L(x)) < 0.25, f"rel {rel(y, L(x)):.3f}")


def test_shared_qkv_equals_separate():
    x = torch.randn(2, 8, 64).to(BF)
    qkv = [lin(64, 64, seed=s) for s in (1, 2, 3)]
    shared = fl.FusedNVFP4Linear(qkv, backend="torchref", fast=False)
    outs = shared(x)
    sep = [fl.FusedNVFP4Linear([l], backend="torchref", fast=False)(x) for l in qkv]
    check("shared quantize: 3 outputs", len(outs) == 3)
    check("shared quantize equals three separate layers", all(torch.equal(a, b) for a, b in zip(outs, sep)))
    fl.reset_counts()
    shared(x)
    check("shared quantize runs the quantize op once", fl.COUNTS["quant:torchref"] == 1, str(dict(fl.COUNTS)))


def test_concat_close_to_separate():
    x = torch.randn(32, 64).to(BF)
    qkv = [lin(64, 64, seed=s) for s in (1, 2, 3)]
    cat = fl.FusedNVFP4Linear(qkv, backend="torchref", concat=True, fast=False)(x)
    sep = fl.FusedNVFP4Linear(qkv, backend="torchref", fast=False)(x)
    check("concat output shapes", [tuple(o.shape) for o in cat] == [(32, 64)] * 3)
    check("concat close to separate (one shared weight scale)", all(rel(a, b) < 0.2 for a, b in zip(cat, sep)),
          str([round(rel(a, b), 3) for a, b in zip(cat, sep)]))


def test_static_scale_used():
    x = torch.randn(32, 64).to(BF)
    L = lin(64, 32)
    amax = float(x.abs().max())
    dyn = fl.FusedNVFP4Linear([L], backend="torchref", fast=False)
    sta = fl.FusedNVFP4Linear([L], backend="torchref", act_amax=amax, fast=False)
    check("static scale equals dynamic when amax matches", torch.equal(dyn(x), sta(x)))
    # a power-of-two factor would rescale the e4m3 block scales exactly and change nothing, so use 3 and 0.5
    # (0.5 also saturates the largest blocks at 448, which is what a too-small static scale does)
    odd = fl.FusedNVFP4Linear([L], backend="torchref", act_amax=amax * 3, fast=False)
    small = fl.FusedNVFP4Linear([L], backend="torchref", act_amax=amax * 0.5, fast=False)
    check("static scale is really used (other amax changes the result)", not torch.equal(dyn(x), odd(x)) or not torch.equal(dyn(x), small(x)))
    check("too-small static scale saturates (error grows)", rel(small(x), L(x)) > rel(dyn(x), L(x)))


def test_fake_matches_real():
    from torch._subclasses.fake_tensor import FakeTensorMode
    op, native = fl.quant_op("torchref")
    x = torch.randn(40, 64).to(BF)
    pts = per_tensor_amax_to_scale(x.abs().max())
    real = op(x, pts)
    with FakeTensorMode():
        fx = torch.empty(40, 64, dtype=BF)
        fake = op(fx, torch.empty((), dtype=torch.float32))
    check("fake quantize shapes/dtypes equal real", [(t.shape, t.dtype) for t in real] == [(t.shape, t.dtype) for t in fake],
          f"{[(t.shape, t.dtype) for t in real]} vs {[(t.shape, t.dtype) for t in fake]}")


def test_compile_engagement():
    """Every NVFP4 site must run our op under torch.compile, exactly as often as in eager."""
    import attention_patch as ap
    pipe = _boot.tiny_pipe()
    tr = pipe.transformer.to(BF)
    rep = fl.convert_pixart(tr, backend="torchref", fast=False)
    ap.install_attention(tr, ap.AttnConfig())
    B = 2
    x = torch.randn(B, 4, 8, 8).to(BF)
    enc = torch.randn(B, _boot.TEXT, 32).to(BF)
    mask = torch.ones(B, _boot.TEXT, dtype=torch.long)
    t = torch.tensor([500, 500])
    kw = dict(encoder_hidden_states=enc, encoder_attention_mask=mask, timestep=t,
              added_cond_kwargs={"resolution": None, "aspect_ratio": None}, return_dict=False)
    with torch.no_grad():
        fl.reset_counts()
        eager = tr(x, **kw)[0]
        n_eager = fl.COUNTS["quant:torchref"]
        fl.reset_counts()
        comp = torch.compile(tr, backend="aot_eager")
        out = comp(x, **kw)[0]
        n_comp = fl.COUNTS["quant:torchref"]
    layers = len(tr.transformer_blocks)
    expected = layers * 7 + 6  # per block: qkv, attn1 out, attn2 q, kv, attn2 out, fc1, fc2; outside: 6
    check("eager: one quantize op per site", n_eager == expected, f"{n_eager} vs {expected}, report {rep}")
    check("compiled: one quantize op per site (patch bypass bug would give ~0)", n_comp == expected, f"{n_comp} vs {expected}")
    check("compiled output equals eager", rel(out, eager) < 1e-3, f"rel {rel(out, eager):.2e}")


if __name__ == "__main__":
    _boot.run(test_quantize_op_and_reference_gemm, test_shared_qkv_equals_separate, test_concat_close_to_separate,
              test_static_scale_used, test_fake_matches_real, test_compile_engagement)
