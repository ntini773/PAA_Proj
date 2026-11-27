"""Roofs from a sweep, not from one point. Needs an idle GPU, about 3-5 min.   (eval3 T-roof, fixes the "FP4 roof is too low" finding)

    python bench/peaks_sweep.py [--quick]
Writes results/eval3/peaks_sweep.json. results/peaks.json is left alone.
A roof is the BEST achieved value over the whole sweep, so no kernel can show an efficiency above 100% against it:
  fp4_tflops    NVFP4 block-scaled GEMM, activation already quantized, over PixArt / SD3.5 / FLUX shapes and square shapes
  bf16_tflops   torch.matmul bf16, same shapes
  attn_tflops   flash SDPA per head dim (64, 72, 96, 128) at the token counts of the three models
  copy_gbs      device-to-device copy and a read-only reduction (DRAM bandwidth roof)
Also kept per shape (rows): so the report can show the roof as a function of K and N (small K = low roof) and give a per-layer roof
instead of one global number.
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import torch
import torch.nn.functional as F

from common import cuda_ms, stats, write_json

BF = torch.bfloat16
# (K, N) of the linear layers: PixArt-Sigma (1152), SD3.5 Medium (1536), FLUX (3072), plus square roof shapes
KN = {
    "pixart": [(1152, 1152), (1152, 4608), (4608, 1152), (1152, 3456)],
    "sd35m": [(1536, 1536), (1536, 6144), (6144, 1536), (1536, 4608)],
    "flux": [(3072, 3072), (3072, 12288), (12288, 3072), (3072, 9216)],
    "square": [(8192, 8192), (4096, 4096)],
}
M_FULL = [1024, 2048, 4096, 8192, 16384, 32768]
M_QUICK = [4096, 16384]
# (batch, heads, tokens, head dim): self-attention shapes at 1024 px with CFG batch 2; FLUX has no CFG batch
ATTN = [(2, 16, 1024, 72), (2, 16, 4096, 72), (2, 16, 4096, 96), (2, 24, 4096, 64), (2, 24, 4429, 64), (1, 24, 4608, 128), (2, 16, 16384, 72), (2, 24, 16384, 64)]


def med(fn, iters=30):
    return stats(cuda_ms(fn, iters=iters, warmup=8))["median"]


def copy_bw():
    rows = []
    for mb in (512, 2048):
        n = mb * 2**20 // 2
        x, y = torch.empty(n, dtype=BF, device="cuda"), torch.empty(n, dtype=BF, device="cuda")
        ms = med(lambda: y.copy_(x))
        rows.append(dict(kind="copy", mb=mb, gbs=2 * n * 2 / ms / 1e6))
        ms = med(lambda: x.sum())
        rows.append(dict(kind="read", mb=mb, gbs=n * 2 / ms / 1e6))
        del x, y
    return rows


def fp4_row(m, k, n):
    from torchao.prototype.mx_formats.inference_workflow import NVFP4DynamicActivationNVFP4WeightConfig
    from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor, per_tensor_amax_to_scale
    from torchao.quantization import quantize_
    lin = torch.nn.Linear(k, n, bias=False, device="cuda", dtype=BF)
    quantize_(lin, NVFP4DynamicActivationNVFP4WeightConfig(use_dynamic_per_tensor_scale=True, use_triton_kernel=False))
    x = torch.randn(m, k, device="cuda", dtype=BF)
    kw = lin.weight.act_quant_kwargs
    xq = NVFP4Tensor.to_nvfp4(x, block_size=16, per_tensor_scale=per_tensor_amax_to_scale(x.abs().max()), is_swizzled_scales=kw.is_swizzled_scales)
    ms = med(lambda: F.linear(xq, lin.weight))
    return ms


def gemm_rows(ms_list, quick):
    rows = []
    for fam, kns in KN.items():
        for k, n in (kns[:2] if quick else kns):
            for m in ms_list:
                if m * max(k, n) * 2 > 3 * 2**30:  # keep each operand under 3 GB
                    continue
                row = dict(family=fam, m=m, k=k, n=n)
                a, b = torch.randn(m, k, device="cuda", dtype=BF), torch.randn(k, n, device="cuda", dtype=BF)
                ms = med(lambda: a @ b)
                row.update(bf16_ms=ms, bf16_tflops=2 * m * k * n / ms / 1e9)
                del a, b
                try:
                    ms = fp4_row(m, k, n)
                    row.update(fp4_ms=ms, fp4_tflops=2 * m * k * n / ms / 1e9)
                except Exception as e:  # noqa: BLE001
                    row["fp4_error"] = repr(e)[:120]
                rows.append(row)
                print(f"{fam:7s} M={m:6d} K={k:6d} N={n:6d}  bf16 {row['bf16_tflops']:6.0f}  fp4 {row.get('fp4_tflops', float('nan')):6.0f} TFLOP/s", flush=True)
    return rows


def attn_rows(quick):
    rows = []
    for b, h, t, d in (ATTN[:4] if quick else ATTN):
        try:
            q, k, v = (torch.randn(b, h, t, d, device="cuda", dtype=BF) for _ in range(3))
            with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
                ms = med(lambda: F.scaled_dot_product_attention(q, k, v))
            rows.append(dict(batch=b, heads=h, tokens=t, head_dim=d, ms=ms, tflops=4 * b * h * t * t * d / ms / 1e9))
            print(f"attn B={b} H={h} T={t} d={d}: {rows[-1]['tflops']:.0f} TFLOP/s", flush=True)
        except Exception as e:  # noqa: BLE001  (flash may refuse some head dims on some builds)
            rows.append(dict(batch=b, heads=h, tokens=t, head_dim=d, error=repr(e)[:120]))
            print(f"attn B={b} H={h} T={t} d={d}: {e!r}"[:160])
        torch.cuda.empty_cache()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="2 M values, 2 layer shapes per family (about 1 min)")
    a = ap.parse_args()
    p = torch.cuda.get_device_properties(0)
    res = dict(device=p.name, torch=torch.__version__, quick=a.quick)
    res["bandwidth"] = copy_bw()
    res["gemm"] = gemm_rows(M_QUICK if a.quick else M_FULL, a.quick)
    res["attn"] = attn_rows(a.quick)
    fp4 = [r["fp4_tflops"] for r in res["gemm"] if "fp4_tflops" in r]
    res["best"] = dict(
        copy_gbs=max(r["gbs"] for r in res["bandwidth"] if r["kind"] == "copy"),
        read_gbs=max(r["gbs"] for r in res["bandwidth"] if r["kind"] == "read"),
        bf16_tflops=max(r["bf16_tflops"] for r in res["gemm"]),
        fp4_tflops=max(fp4) if fp4 else None,
        attn_tflops_by_head_dim={d: max((r["tflops"] for r in res["attn"] if r.get("head_dim") == d and "tflops" in r), default=None) for d in (64, 72, 96, 128)},
    )
    write_json("eval3/peaks_sweep.json", res)
    print("best (the roofs):", res["best"])


if __name__ == "__main__":
    main()
