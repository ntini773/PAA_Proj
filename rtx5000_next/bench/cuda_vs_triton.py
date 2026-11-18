"""CUDA kernels vs Triton vs torchao: byte exactness and time per call.

    python bench/cuda_vs_triton.py [--only cuda_row cuda_swizzled amax_swizzled ln_mod_quant]
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import torch

import cuda_ext
import quant_backends as qb
from common import append_jsonl, cuda_ms, stats

SHAPES = [(512, 1152), (2048, 1152), (8192, 1152), (600, 1152), (512, 4608), (2048, 4608), (8192, 4608), (512, 1536), (4096, 3072)]
E4M3_X_E2M1 = 448.0 * 6.0


def make_x(M, K, seed):
    g = torch.Generator("cuda").manual_seed(seed)
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16, generator=g)
    mask = torch.rand(M, K, device="cuda", generator=g) < 0.002
    return torch.where(mask, x * 40, x)


def diff_bytes(a, b):
    return int((a.contiguous().view(torch.uint8) != b.contiguous().view(torch.uint8)).sum())


def swizzle(scales, M, K):
    from torchao.prototype.mx_formats.utils import to_blocked
    return to_blocked(scales.view(M, K // 16)).flatten()


def torchao_stock(x, pts):
    from torchao.prototype.mx_formats.nvfp4_tensor import nvfp4_quantize
    scales, packed = nvfp4_quantize(x, 16, pts)
    return packed, scales


def test_quant(name, fn, swizzled, M, K, rec):
    x = make_x(M, K, 0)
    pts = (x.abs().max().float() / E4M3_X_E2M1).reshape(())
    ref_p, ref_s = qb._triton_impl(x, pts)
    p, s = fn(x, pts)
    ref_s_cmp = swizzle(ref_s, M, K) if swizzled else ref_s
    rec.update(packed_diff_vs_triton=diff_bytes(p, ref_p), scale_diff_vs_triton=diff_bytes(s, ref_s_cmp), packed_bytes=p.numel())
    try:
        sp, ss = torchao_stock(x, pts)
        ss_cmp = swizzle(ss, M, K) if swizzled else ss
        rec.update(packed_diff_vs_torchao=diff_bytes(p, sp), scale_diff_vs_torchao=diff_bytes(s, ss_cmp))
    except Exception as e:
        rec["torchao_ref"] = f"skipped {type(e).__name__}"
    rec["time_us"] = stats([t * 1e3 for t in cuda_ms(lambda: fn(x, pts), 200, 30)])
    rec["triton_us"] = stats([t * 1e3 for t in cuda_ms(lambda: qb._triton_impl(x, pts), 200, 30)])


def test_amax(M, K, rec):
    x = make_x(M, K, 0)
    ext = cuda_ext.load_ext("quant2lvl_swizzled")
    p, s, pts = ext.amax_quant_swizzled(x)
    ref_pts = x.abs().max().float() / E4M3_X_E2M1
    rp, rs = qb._triton_impl(x, ref_pts)
    rec.update(pts_diff=float((pts.float() - ref_pts).abs()), packed_diff_vs_triton=diff_bytes(p, rp),
               scale_diff_vs_triton=diff_bytes(s, swizzle(rs, M, K)))

    def stock():
        pt = x.abs().max().float() / E4M3_X_E2M1
        sp, ss = qb._triton_impl(x, pt)
        return sp, swizzle(ss, M, K)

    rec["time_us"] = stats([t * 1e3 for t in cuda_ms(lambda: ext.amax_quant_swizzled(x), 200, 30)])
    rec["triton_plus_amax_plus_swizzle_us"] = stats([t * 1e3 for t in cuda_ms(stock, 200, 30)])


def test_ln(M, K, rec):
    from fused_adaln_nvfp4 import fused_ln_adaln_nvfp4
    x = make_x(M, K, 1)
    sc = (torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.1)
    sh = (torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.1)
    ext = cuda_ext.load_ext("ln_mod_quant_cuda")
    rp, rs = fused_ln_adaln_nvfp4(x, sc, sh, 1e-6)
    p, s = ext.ln_mod_quant(x, sc, sh, 1e-6)
    rec.update(packed_diff_vs_triton=diff_bytes(p, rp), scale_diff_vs_triton=diff_bytes(s, rs), packed_bytes=p.numel())
    rec["time_us"] = stats([t * 1e3 for t in cuda_ms(lambda: ext.ln_mod_quant(x, sc, sh, 1e-6), 200, 30)])
    rec["triton_us"] = stats([t * 1e3 for t in cuda_ms(lambda: fused_ln_adaln_nvfp4(x, sc, sh, 1e-6), 200, 30)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=["cuda_row", "cuda_swizzled", "amax_swizzled", "ln_mod_quant"])
    a = ap.parse_args()
    for name in a.only:
        for M, K in SHAPES:
            rec = dict(kernel=name, M=M, K=K)
            try:
                if name in ("cuda_row", "cuda_swizzled"):
                    impl, swz = qb.IMPLS[name]
                    test_quant(name, impl, swz, M, K, rec)
                elif name == "amax_swizzled":
                    test_amax(M, K, rec)
                else:
                    test_ln(M, K, rec)
            except Exception as e:
                rec["error"] = f"{type(e).__name__}: {str(e)[:300]}"
            t = rec.get("time_us", {}).get("median")
            print(name, M, K, {k: v for k, v in rec.items() if k.endswith("diff_vs_triton") or k in ("error",)}, f"{t:.1f} us" if t else "")
            append_jsonl("cuda_vs_triton.jsonl", rec)


if __name__ == "__main__":
    main()
