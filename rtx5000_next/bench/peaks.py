"""Measured limits of this GPU, used as the roofs in analysis/roofline_pixart.py. Needs an idle GPU, about 1-2 min.

    python bench/peaks.py
Writes results/peaks.json. These are best achieved numbers, not spec-sheet numbers:
  copy_gbs     device-to-device copy (read + write bytes)
  bf16_tflops  best torch.matmul over large and PixArt-shaped GEMMs
  fp4_tflops   NVFP4 GEMM with the activation already quantized (null if the torchao call fails)
  attn_tflops  flash SDPA at PixArt head shape (16 heads, head dim 72)
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import torch
import torch.nn.functional as F

from common import cuda_ms, stats, write_json

BF = torch.bfloat16
SHAPES = [(8192, 8192, 8192), (8192, 1152, 1152), (8192, 1152, 4608), (8192, 4608, 1152), (2048, 1152, 4608), (32768, 1152, 4608)]


def med(fn, iters=50):
    return stats(cuda_ms(fn, iters=iters, warmup=10))["median"]


def copy_bandwidth():
    rows = []
    for mb in (256, 1024, 2048):
        n = mb * 2**20 // 2
        x, y = torch.empty(n, dtype=BF, device="cuda"), torch.empty(n, dtype=BF, device="cuda")
        ms = med(lambda: y.copy_(x))
        rows.append(dict(mb=mb, ms=ms, gbs=2 * n * 2 / ms / 1e6))
        del x, y
    return rows


def bf16_gemm():
    rows = []
    for m, k, n in SHAPES:
        a, b = torch.randn(m, k, device="cuda", dtype=BF), torch.randn(k, n, device="cuda", dtype=BF)
        ms = med(lambda: a @ b)
        rows.append(dict(m=m, k=k, n=n, ms=ms, tflops=2 * m * k * n / ms / 1e9))
    return rows


def fp4_gemm():
    from torchao.prototype.mx_formats.inference_workflow import NVFP4DynamicActivationNVFP4WeightConfig
    from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor, per_tensor_amax_to_scale
    from torchao.quantization import quantize_
    rows = []
    for m, k, n in SHAPES:
        lin = torch.nn.Linear(k, n, bias=False, device="cuda", dtype=BF)
        quantize_(lin, NVFP4DynamicActivationNVFP4WeightConfig(use_dynamic_per_tensor_scale=True, use_triton_kernel=False))
        x = torch.randn(m, k, device="cuda", dtype=BF)
        kw = lin.weight.act_quant_kwargs
        xq = NVFP4Tensor.to_nvfp4(x, block_size=16, per_tensor_scale=per_tensor_amax_to_scale(x.abs().max()),
                                  is_swizzled_scales=kw.is_swizzled_scales)
        ms_gemm = med(lambda: F.linear(xq, lin.weight))
        ms_full = med(lambda: lin(x))
        rows.append(dict(m=m, k=k, n=n, ms_gemm=ms_gemm, tflops=2 * m * k * n / ms_gemm / 1e9,
                         ms_with_quantize=ms_full, tflops_with_quantize=2 * m * k * n / ms_full / 1e9))
    return rows


def attention():
    rows = []
    for batch, tokens in ((2, 1024), (2, 4096), (4, 4096)):
        h, d = 16, 72
        q, k, v = (torch.randn(batch, h, tokens, d, device="cuda", dtype=BF) for _ in range(3))
        ms = med(lambda: F.scaled_dot_product_attention(q, k, v))
        rows.append(dict(batch=batch, tokens=tokens, ms=ms, tflops=4 * batch * h * tokens * tokens * d / ms / 1e9))
    return rows


def main():
    props = torch.cuda.get_device_properties(0)
    res = dict(device=props.name, total_gb=props.total_memory / 2**30, torch=torch.__version__)
    res["copy"] = copy_bandwidth()
    res["copy_gbs"] = max(r["gbs"] for r in res["copy"])
    res["bf16"] = bf16_gemm()
    res["bf16_tflops"] = max(r["tflops"] for r in res["bf16"])
    try:
        res["fp4"] = fp4_gemm()
        res["fp4_tflops"] = max(r["tflops"] for r in res["fp4"])
    except Exception as e:
        res["fp4"], res["fp4_tflops"], res["fp4_error"] = [], None, repr(e)
        print("fp4 GEMM failed:", repr(e))
    res["attn"] = attention()
    res["attn_tflops"] = max(r["tflops"] for r in res["attn"])
    write_json("peaks.json", res)
    print(f"{props.name}: copy {res['copy_gbs']:.0f} GB/s  bf16 {res['bf16_tflops']:.0f} TFLOP/s  "
          f"fp4 {res['fp4_tflops'] and round(res['fp4_tflops'])} TFLOP/s  attention {res['attn_tflops']:.0f} TFLOP/s")


if __name__ == "__main__":
    main()
