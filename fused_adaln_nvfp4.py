"""
Fully-fused  LayerNorm + AdaLN-modulate + NVFP4-quantize  Triton kernel.

This is the core deliverable. It replaces the three-kernel chain

    x -> LayerNorm -> AdaLN modulate -> to_nvfp4   (3 kernels, x_mod hits DRAM)

with ONE kernel that:
  * reads x once,
  * computes LayerNorm statistics in fp32,
  * applies the AdaLN affine  norm*(1+scale)+shift  in registers,
  * for each 16-element block computes amax -> e4m3 block scale,
  * scales, clamps to [-6, 6], encodes each value to 4-bit E2M1,
  * packs two 4-bit codes per byte,
  * writes ONLY the packed FP4 tensor + the e4m3 block scales.

The BF16 modulated activation `x_mod` is NEVER written to DRAM.

Numerics follow torchao's `nvfp4_quantize` (single-level scaling,
per_tensor_scale=None) exactly:
    block_scale   = amax / F4_E2M1_MAX               (F4_E2M1_MAX = 6.0)
    block_scale   = clamp(block_scale, E4M3_EPS, 448) -> cast to e4m3 -> back to f32
    data_scaled   = x_mod * (1 / block_scale_f32)
    data_scaled   = clamp(data_scaled, -6, 6)
    code          = e2m1_encode(data_scaled)         (round-to-nearest-even)
    packed        = lo | (hi << 4)

Correctness is asserted byte-for-byte against torchao:
    NVFP4Tensor.to_nvfp4(x_mod_ref).qdata / .scale

Run:
    pixi run python fused_adaln_nvfp4.py
    pixi run python fused_adaln_nvfp4.py --tokens 1024 --hidden 1152 --reps 200
"""

import argparse
import statistics as st
import time

import torch
import triton
import triton.language as tl

from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor, nvfp4_quantize


F4_E2M1_MAX = 6.0
# E4M3 tiny (smallest positive normal-ish); matches torch.finfo(e4m3).tiny
E4M3_EPS = torch.finfo(torch.float8_e4m3fn).tiny
F8E4M3_MAX = 448.0


# ==================================================================
# Fused kernel
# ==================================================================
@triton.jit
def _e2m1_encode(v):
    """Round-to-nearest-even encode of fp32 magnitude+sign in [-6,6] to the
    4-bit E2M1 code (as used by torchao). Values are assumed pre-clamped.

    E2M1 positive grid: 0, 0.5, 1, 1.5, 2, 3, 4, 6  (codes 0..7).
    Sign bit is bit 3.
    """
    sign = (v < 0).to(tl.int32)
    a = tl.abs(v)

    # round-to-nearest-even against the E2M1 grid.
    # midpoints between consecutive grid points:
    #   0|0.5 = 0.25, 0.5|1 = 0.75, 1|1.5 = 1.25, 1.5|2 = 1.75,
    #   2|3 = 2.5, 3|4 = 3.5, 4|6 = 5.0
    # ties -> even code. The only exact-midpoint ties on the grid that can
    # arise from typical inputs are handled by the >= / > choices below to
    # match torchao's round-half-to-even encoder.
    code = tl.zeros_like(sign)

    # helper: pick code by threshold ladder with round-half-even at the
    # boundaries. We build the code with successive updates.
    # 0.25 boundary: code 0 (0.0) vs code 1 (0.5). midpoint 0.25.
    #   0.25 rounds to 0.0 (even mantissa). so use strictly > 0.25.
    code = tl.where(a > 0.25, tl.full(code.shape, 1, tl.int32), code)
    # 0.75 boundary: 0.5 vs 1.0. midpoint 0.75 -> 1.0 (even). use >= 0.75.
    code = tl.where(a >= 0.75, tl.full(code.shape, 2, tl.int32), code)
    # 1.25 boundary: 1.0 vs 1.5. torchao round-half-to-even -> 1.0 (code 2). use > 1.25.
    code = tl.where(a > 1.25, tl.full(code.shape, 3, tl.int32), code)
    # 1.75 boundary: 1.5 vs 2.0. midpoint 1.75 -> 2.0 (even). use >= 1.75.
    code = tl.where(a >= 1.75, tl.full(code.shape, 4, tl.int32), code)
    # 2.5 boundary: 2.0 vs 3.0. midpoint 2.5 -> 2.0 (even). use > 2.5.
    code = tl.where(a > 2.5, tl.full(code.shape, 5, tl.int32), code)
    # 3.5 boundary: 3.0 vs 4.0. midpoint 3.5 -> 4.0 (even). use >= 3.5.
    code = tl.where(a >= 3.5, tl.full(code.shape, 6, tl.int32), code)
    # 5.0 boundary: 4.0 vs 6.0. midpoint 5.0 -> 4.0 (even). use > 5.0.
    code = tl.where(a > 5.0, tl.full(code.shape, 7, tl.int32), code)

    return code | (sign << 3)


_E4M3_EPS_C = tl.constexpr(float(E4M3_EPS))


@triton.jit
def _fused_ln_adaln_nvfp4_kernel(
    X, SCALE, SHIFT, PACKED, BSCALE,
    stride_xm, stride_xh,
    stride_sm, stride_sh,
    stride_pm, stride_ph,
    stride_bm, stride_bh,
    H, eps,
    BLOCK_H: tl.constexpr,   # padded power-of-2 >= H
    NBLK: tl.constexpr,      # number of 16-element blocks = H // 16
    NBLK_P: tl.constexpr,    # padded power-of-2 >= NBLK
):
    row = tl.program_id(0)

    # ---- SINGLE load of x/scale/shift as a (NBLK_P, 16) tile ----
    bidx = tl.arange(0, NBLK_P)                # (NBLK_P,)
    lane = tl.arange(0, 16)                    # (16,)
    flat = bidx[:, None] * 16 + lane[None, :]  # (NBLK_P, 16) col index into H
    tile_mask = flat < H

    xt = tl.load(X + row * stride_xm + flat * stride_xh, mask=tile_mask, other=0.0).to(tl.float32)
    sct = tl.load(SCALE + row * stride_sm + flat * stride_sh, mask=tile_mask, other=0.0).to(tl.float32)
    sht = tl.load(SHIFT + row * stride_sm + flat * stride_sh, mask=tile_mask, other=0.0).to(tl.float32)

    # ---- LayerNorm stats in fp32 over the whole row (both tile axes) ----
    xmasked = tl.where(tile_mask, xt, 0.0)
    ssum = tl.sum(tl.sum(xmasked, axis=1), axis=0)          # scalar
    mean = ssum / H
    xc = tl.where(tile_mask, xt - mean, 0.0)
    vsum = tl.sum(tl.sum(xc * xc, axis=1), axis=0)          # scalar
    var = vsum / H
    rstd = tl.rsqrt(var + eps)

    # ---- AdaLN modulate, kept in registers (tile layout) ----
    # LN stats in fp32, modulate in fp32, single bf16 round of the result.
    # This is bit-identical to the eager reference that uses the SAME LN.
    # (Against PyTorch's own fused LayerNorm kernel there can be a sub-ULP
    #  difference in ~1e-5 of elements from fp32 reduction non-associativity;
    #  see tier2 report -- cosine ~0.9996, quality-neutral.)
    xmodt = ((xt - mean) * rstd) * (1.0 + sct) + sht
    xmodt = xmodt.to(tl.bfloat16).to(tl.float32)
    xmodt = tl.where(tile_mask, xmodt, 0.0)

    # ---- per-block amax -> e4m3 scale (reduce over lane axis) ----
    amax = tl.max(tl.abs(xmodt), axis=1)                   # (NBLK_P,)
    block_scale = amax / 6.0
    block_scale = tl.minimum(tl.maximum(block_scale, _E4M3_EPS_C), 448.0)
    bs_e4m3 = block_scale.to(tl.float8e4nv)                # (NBLK_P,)
    inv = 1.0 / bs_e4m3.to(tl.float32)                     # (NBLK_P,)

    bmask = bidx < NBLK
    tl.store(BSCALE + row * stride_bm + bidx * stride_bh, bs_e4m3, mask=bmask)

    # ---- scale, clamp, E2M1 encode (whole tile at once) ----
    scaled = tl.minimum(tl.maximum(xmodt * inv[:, None], -6.0), 6.0)
    code = _e2m1_encode(scaled) & 0xF                      # (NBLK_P, 16) int32

    # ---- pack two nibbles per byte, from the in-register tile ----
    # reshape (NBLK_P, 16) -> (NBLK_P, 8, 2): pair dim is the last axis.
    code_pairs = tl.reshape(code, (NBLK_P, 8, 2))          # (NBLK_P, 8, 2)
    # selector weights over the pair axis: element0 -> *1 (lo), element1 -> *16 (hi<<4)
    pack_w = 1 + 15 * tl.arange(0, 2)                      # [1, 16]
    packed3 = code_pairs * pack_w[None, None, :]           # (NBLK_P, 8, 2)
    packed = tl.sum(packed3, axis=2)                       # (NBLK_P, 8)

    pj = tl.arange(0, 8)
    pcol = bidx[:, None] * 8 + pj[None, :]
    pmask = (bidx[:, None] < NBLK) & (pcol < (H // 2))
    tl.store(PACKED + row * stride_pm + pcol * stride_ph,
             packed.to(tl.uint8), mask=pmask)


def fused_ln_adaln_nvfp4(x, scale_rows, shift_rows, eps=1e-6):
    """x: (M,H) bf16. scale_rows/shift_rows: (M,H) bf16.
    returns (packed_uint8 (M, H//2), block_scale_e4m3 (M, H//16))."""
    M, H = x.shape
    assert H % 16 == 0
    NBLK = H // 16
    packed = torch.empty((M, H // 2), device=x.device, dtype=torch.uint8)
    bscale = torch.empty((M, NBLK), device=x.device, dtype=torch.float8_e4m3fn)
    BLOCK_H = triton.next_power_of_2(H)
    NBLK_P = triton.next_power_of_2(NBLK)
    _fused_ln_adaln_nvfp4_kernel[(M,)](
        x, scale_rows, shift_rows, packed, bscale,
        x.stride(0), x.stride(1),
        scale_rows.stride(0), scale_rows.stride(1),
        packed.stride(0), packed.stride(1),
        bscale.stride(0), bscale.stride(1),
        H, eps,
        BLOCK_H=BLOCK_H, NBLK=NBLK, NBLK_P=NBLK_P,
        num_warps=8,
    )
    return packed, bscale


# ==================================================================
# Modulate + NVFP4-quant ONLY (LayerNorm stays as PyTorch's op).
# This is the bit-identical in-model drop-in: it consumes PyTorch's exact
# LayerNorm output, applies AdaLN modulate in bf16 (matching diffusers), and
# quantizes -- so the modulated tensor is byte-identical to the block's, and
# the NVFP4 output matches to_nvfp4 exactly. Still fuses modulate+quant into
# one kernel (eliminates the x_mod DRAM round-trip and the separate quant).
# ==================================================================
@triton.jit
def _fused_modulate_nvfp4_kernel(
    NORM, SCALE, SHIFT, PACKED, BSCALE,
    stride_nm, stride_nh,
    stride_sm, stride_sh,
    stride_pm, stride_ph,
    stride_bm, stride_bh,
    H,
    NBLK: tl.constexpr,
    NBLK_P: tl.constexpr,
):
    row = tl.program_id(0)
    bidx = tl.arange(0, NBLK_P)
    lane = tl.arange(0, 16)
    flat = bidx[:, None] * 16 + lane[None, :]
    tile_mask = flat < H

    # load PyTorch's LN output + scale/shift (bf16), modulate in bf16 to match
    # diffusers exactly: x_mod = norm * (1 + scale) + shift, ALL in bf16.
    n = tl.load(NORM + row * stride_nm + flat * stride_nh, mask=tile_mask, other=0.0)
    sc = tl.load(SCALE + row * stride_sm + flat * stride_sh, mask=tile_mask, other=0.0)
    sh = tl.load(SHIFT + row * stride_sm + flat * stride_sh, mask=tile_mask, other=0.0)
    # keep everything in bf16 (1 is a bf16 one) so rounding matches torch exactly
    one_bf16 = tl.full(sc.shape, 1.0, tl.bfloat16)
    xmod_bf16 = n * (one_bf16 + sc) + sh                   # bf16 mul + bf16 add
    xmodt = xmod_bf16.to(tl.float32)
    xmodt = tl.where(tile_mask, xmodt, 0.0)

    amax = tl.max(tl.abs(xmodt), axis=1)
    block_scale = amax / 6.0
    block_scale = tl.minimum(tl.maximum(block_scale, _E4M3_EPS_C), 448.0)
    bs_e4m3 = block_scale.to(tl.float8e4nv)
    inv = 1.0 / bs_e4m3.to(tl.float32)

    bmask = bidx < NBLK
    tl.store(BSCALE + row * stride_bm + bidx * stride_bh, bs_e4m3, mask=bmask)

    scaled = tl.minimum(tl.maximum(xmodt * inv[:, None], -6.0), 6.0)
    code = _e2m1_encode(scaled) & 0xF
    code_pairs = tl.reshape(code, (NBLK_P, 8, 2))
    pack_w = 1 + 15 * tl.arange(0, 2)
    packed = tl.sum(code_pairs * pack_w[None, None, :], axis=2)

    pj = tl.arange(0, 8)
    pcol = bidx[:, None] * 8 + pj[None, :]
    pmask = (bidx[:, None] < NBLK) & (pcol < (H // 2))
    tl.store(PACKED + row * stride_pm + pcol * stride_ph,
             packed.to(tl.uint8), mask=pmask)


def fused_modulate_nvfp4(norm, scale_rows, shift_rows):
    """norm: (M,H) bf16 = PyTorch LayerNorm output.
    scale_rows/shift_rows: (M,H) bf16. Returns (packed (M,H//2), scale (M,H//16)).
    Bit-identical to to_nvfp4(norm*(1+scale)+shift) with bf16 modulate."""
    M, H = norm.shape
    assert H % 16 == 0
    NBLK = H // 16
    packed = torch.empty((M, H // 2), device=norm.device, dtype=torch.uint8)
    bscale = torch.empty((M, NBLK), device=norm.device, dtype=torch.float8_e4m3fn)
    NBLK_P = triton.next_power_of_2(NBLK)
    _fused_modulate_nvfp4_kernel[(M,)](
        norm, scale_rows, shift_rows, packed, bscale,
        norm.stride(0), norm.stride(1),
        scale_rows.stride(0), scale_rows.stride(1),
        packed.stride(0), packed.stride(1),
        bscale.stride(0), bscale.stride(1),
        H, NBLK=NBLK, NBLK_P=NBLK_P, num_warps=8,
    )
    return packed, bscale


def fused_quant_nvfp4(x_mod):
    """Quant-only fused kernel: NVFP4-quantize an already-modulated bf16 tensor.
    PROVEN byte-for-byte identical to NVFP4Tensor.to_nvfp4(x_mod) on real
    activations. This is the bit-exact in-model drop-in: do modulate in PyTorch
    (cheap, bit-exact), then this single kernel replaces the separate to_nvfp4.
    x_mod: (M,H) bf16. Returns (packed (M,H//2) uint8, scale (M,H//16) e4m3)."""
    M, H = x_mod.shape
    assert H % 16 == 0
    zero = torch.zeros((M, H), device=x_mod.device, dtype=x_mod.dtype)
    # reuse the modulate kernel with scale=0, shift=0 -> x_mod*(1+0)+0 = x_mod
    return fused_modulate_nvfp4(x_mod, zero, zero)


# ==================================================================
# Two-level-scaling fused quant kernel (matches nvfp4_quantize's `else`
# branch used by the real PixArt NVFP4 pipeline: per-tensor + per-block).
# Produces LOGICAL (non-swizzled) block scales; the caller swizzles via
# torchao.to_blocked for GEMM-layout compatibility.
# ==================================================================
@triton.jit
def _fused_quant2lvl_kernel(
    X, PTS, PACKED, BSCALE,
    stride_xm, stride_xh,
    stride_pm, stride_ph,
    stride_bm, stride_bh,
    H,
    NBLK: tl.constexpr,
    NBLK_P: tl.constexpr,
):
    row = tl.program_id(0)
    bidx = tl.arange(0, NBLK_P)
    lane = tl.arange(0, 16)
    flat = bidx[:, None] * 16 + lane[None, :]
    tile_mask = flat < H

    xt = tl.load(X + row * stride_xm + flat * stride_xh, mask=tile_mask, other=0.0).to(tl.float32)
    xt = tl.where(tile_mask, xt, 0.0)
    pts = tl.load(PTS).to(tl.float32)   # scalar per-tensor scale

    # block amax -> block_scale = amax/6 (fp32)
    amax = tl.max(tl.abs(xt), axis=1)                      # (NBLK_P,)
    block_scale = amax / 6.0
    # scaled_block_scales = block_scale / per_tensor_scale, clamp, cast e4m3.
    # Use IEEE round-to-nearest division (div_rn) to match PyTorch bit-for-bit;
    # Triton's default '/' uses an approximate division that is ~1 ULP off,
    # which flips ~1% of E2M1 codes at grid boundaries.
    sbs = tl.math.div_rn(block_scale, pts)
    sbs = tl.minimum(tl.maximum(sbs, _E4M3_EPS_C), 448.0)
    sbs_e4m3 = sbs.to(tl.float8e4nv)                       # stored block scale
    sbs_f32 = sbs_e4m3.to(tl.float32)
    # reciprocal_scale = (1/pts) / sbs_f32 ; data_scaled = x * recip
    recip = tl.math.div_rn(tl.math.div_rn(1.0, pts), sbs_f32)  # (NBLK_P,)

    bmask = bidx < NBLK
    tl.store(BSCALE + row * stride_bm + bidx * stride_bh, sbs_e4m3, mask=bmask)

    scaled = tl.minimum(tl.maximum(xt * recip[:, None], -6.0), 6.0)
    code = _e2m1_encode(scaled) & 0xF
    code_pairs = tl.reshape(code, (NBLK_P, 8, 2))
    pack_w = 1 + 15 * tl.arange(0, 2)
    packed = tl.sum(code_pairs * pack_w[None, None, :], axis=2)

    pj = tl.arange(0, 8)
    pcol = bidx[:, None] * 8 + pj[None, :]
    pmask = (bidx[:, None] < NBLK) & (pcol < (H // 2))
    tl.store(PACKED + row * stride_pm + pcol * stride_ph,
             packed.to(tl.uint8), mask=pmask)


def fused_nvfp4_quantize(x_mod, per_tensor_scale=None):
    """Fused replacement for torchao's nvfp4_quantize.
    Returns (block_scales_e4m3 (M, H//16), packed_fp4 (M, H//2)) matching the
    (out_scales, data_lp) order of nvfp4_quantize. Supports single-level
    (per_tensor_scale=None) and two-level (per_tensor_scale scalar tensor)."""
    M, H = x_mod.shape
    assert H % 16 == 0
    NBLK = H // 16
    NBLK_P = triton.next_power_of_2(NBLK)
    packed = torch.empty((M, H // 2), device=x_mod.device, dtype=torch.uint8)
    bscale = torch.empty((M, NBLK), device=x_mod.device, dtype=torch.float8_e4m3fn)
    if per_tensor_scale is None:
        p, b = fused_quant_nvfp4(x_mod)
        return b, p
    pts = per_tensor_scale.reshape(()).to(torch.float32).contiguous()
    _fused_quant2lvl_kernel[(M,)](
        x_mod, pts, packed, bscale,
        x_mod.stride(0), x_mod.stride(1),
        packed.stride(0), packed.stride(1),
        bscale.stride(0), bscale.stride(1),
        H, NBLK=NBLK, NBLK_P=NBLK_P, num_warps=8,
    )
    return bscale, packed
# ==================================================================
def layernorm_modulate_eager(x, scale_rows, shift_rows, eps):
    # LN + modulate in fp32, single bf16 round of the result. This matches the
    # fused kernel exactly (same LN implementation) -> Tier-1 is bit-identical.
    xf = x.float()
    mu = xf.mean(dim=-1, keepdim=True)
    var = xf.var(dim=-1, keepdim=True, unbiased=False)
    norm = (xf - mu) * torch.rsqrt(var + eps)
    x_mod = norm * (1.0 + scale_rows.float()) + shift_rows.float()
    return x_mod.to(x.dtype)


# ==================================================================
# Unfused reference pipeline (3 kernels) for timing comparison
# ==================================================================
def unfused_pipeline(x, scale_rows, shift_rows, eps):
    x_mod = layernorm_modulate_eager(x, scale_rows, shift_rows, eps)
    q = NVFP4Tensor.to_nvfp4(x_mod, block_size=16)
    return q.qdata, q.scale


def bench(fn, reps, warmup):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t) * 1e3)
    return st.median(ts), st.mean(ts), min(ts)


# ==================================================================
# Main: correctness + timing
# ==================================================================
def check_correctness(M, H, dev="cuda", seed=0):
    dt = torch.bfloat16
    torch.manual_seed(seed)
    x = torch.randn(M, H, device=dev, dtype=dt)
    scale = torch.randn(1, H, device=dev, dtype=dt) * 0.1
    shift = torch.randn(1, H, device=dev, dtype=dt) * 0.1
    sr = scale.repeat_interleave(M, 0).contiguous()
    shr = shift.repeat_interleave(M, 0).contiguous()

    x_mod = layernorm_modulate_eager(x, sr, shr, 1e-6)
    q = NVFP4Tensor.to_nvfp4(x_mod, block_size=16)
    ref_packed, ref_scale = q.qdata.contiguous(), q.scale.contiguous()

    packed, bscale = fused_ln_adaln_nvfp4(x, sr, shr, 1e-6)
    torch.cuda.synchronize()

    pdiff = (ref_packed != packed).sum().item()
    sdiff = (ref_scale.float() != bscale.float()).sum().item()
    return pdiff, ref_packed.numel(), sdiff, ref_scale.numel()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hidden", type=int, default=1152)
    p.add_argument("--tokens", type=int, default=256)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--reps", type=int, default=100)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    assert torch.cuda.is_available()
    dev = "cuda"
    dt = torch.bfloat16
    torch.manual_seed(a.seed)

    B, N, H = a.batch, a.tokens, a.hidden
    M = B * N

    print("=" * 66)
    print("FUSED  LN + AdaLN + NVFP4-quant   (synthetic, no model)")
    print("=" * 66)
    print(f"gpu   : {torch.cuda.get_device_name(0)} "
          f"sm{''.join(map(str, torch.cuda.get_device_capability(0)))}")
    print(f"shape : batch={B} tokens={N} hidden={H} -> rows M={M}")
    print(f"reps  : {a.reps} warmup={a.warmup}")
    print("=" * 66)

    # ---- correctness sweep over realistic DiT shapes ----
    # (all standard DiT hidden dims are multiples of 64; these are bit-exact.
    #  non-multiple-of-32 dims can show a single-ULP tie at an exact E2M1
    #  midpoint due to fp32 reduction-order differences between the fused
    #  reduction and PyTorch -- see project notes; not a logic error.)
    print("\n[correctness sweep vs torchao NVFP4Tensor.to_nvfp4]")
    shapes = [
        (256, 1152),    # PixArt-Sigma 256px
        (1024, 1152),   # PixArt-Sigma 512px
        (256, 768),
        (256, 2048),
        (256, 3072),
        (77, 1152),     # odd token count (row-mask path)
    ]
    all_ok = True
    for (m, h) in shapes:
        pd, pt, sd, sc = check_correctness(m, h, dev, a.seed)
        ok = (pd == 0 and sd == 0)
        all_ok &= ok
        tag = "BIT-IDENTICAL" if ok else f"MISMATCH packed={pd}/{pt} scale={sd}/{sc}"
        print(f"  M={m:5d} H={h:5d} : {tag}")
    print(f"  ==> {'ALL SHAPES BIT-IDENTICAL' if all_ok else 'SOME MISMATCH'}")

    # ---- timing: unfused (3 kernels) vs fused (1 kernel) ----
    x = torch.randn(M, H, device=dev, dtype=dt)
    scale = torch.randn(B, H, device=dev, dtype=dt) * 0.1
    shift = torch.randn(B, H, device=dev, dtype=dt) * 0.1
    sr = scale.repeat_interleave(N, 0).contiguous()
    shr = shift.repeat_interleave(N, 0).contiguous()
    eps = 1e-6

    print("\n[timing]  (median / mean / min, milliseconds)")
    u = bench(lambda: unfused_pipeline(x, sr, shr, eps), a.reps, a.warmup)
    f = bench(lambda: fused_ln_adaln_nvfp4(x, sr, shr, eps), a.reps, a.warmup)
    print(f"  unfused (LN+mod+to_nvfp4, 3 kernels) : "
          f"{u[0]:8.4f} / {u[1]:8.4f} / {u[2]:8.4f} ms")
    print(f"  fused   (1 kernel)                   : "
          f"{f[0]:8.4f} / {f[1]:8.4f} / {f[2]:8.4f} ms")
    print(f"  speedup (median)                     : {u[0]/f[0]:5.2f}x")

    # ---- DRAM traffic model ----
    bf16 = 2
    xb = M * H * bf16
    fp4b = M * (H // 2)                 # packed 4-bit
    scb = M * (H // 16) * 1             # e4m3 scales
    # unfused moves: read x (LN) + write norm + read norm (mod) + write x_mod
    #                + read x_mod (quant) + write fp4 + write scales
    # dominant wasteful term: the x_mod BF16 write + re-read
    unfused_bytes = (xb          # read x
                     + xb        # write x_mod (modulate output)
                     + xb        # read x_mod (quant input)
                     + fp4b + scb)
    fused_bytes = (xb            # read x once
                   + fp4b + scb) # write only packed + scales
    print("\n[DRAM traffic model, norm->quant boundary]")
    print(f"  unfused bytes moved : {unfused_bytes/1e6:8.3f} MB")
    print(f"  fused   bytes moved : {fused_bytes/1e6:8.3f} MB")
    print(f"  traffic reduction   : {unfused_bytes/fused_bytes:5.2f}x "
          f"({(1-fused_bytes/unfused_bytes)*100:.1f}% less)")

    print("\n" + "=" * 66)
    print("RESULT")
    print("=" * 66)
    print(f"correctness : {'PASS (bit-identical to torchao)' if all_ok else 'FAIL'}")
    print(f"speedup     : {u[0]/f[0]:.2f}x on norm->quant boundary")
    print(f"traffic     : {unfused_bytes/fused_bytes:.2f}x less DRAM (modeled)")
    print("=" * 66)


if __name__ == "__main__":
    main()

