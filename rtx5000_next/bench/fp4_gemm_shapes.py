"""NVFP4 GEMM efficiency versus shape: where does a PixArt GEMM reach the tensor-core peak? Needs an idle GPU, about 2 min.

    python bench/fp4_gemm_shapes.py

Why: PixArt linears have K=1152, so the NVFP4 GEMM reached only 274-302 TFLOP/s against 707 at 8192^3 (peaks.json). This sweep
varies M (rows = 2 x images x tokens, so it grows with batch and resolution) and the N of the real sites, including the concatenated
Q/K/V (N=3456) and K/V (N=2304), to answer two compute-side questions without touching the model:
    1. does concatenating weights make the GEMM more efficient?   2. at what batch does the GEMM saturate (throughput mode)?
Method: both operands pre-quantized once; time only torch._scaled_mm (CUDA events, median); bf16 GEMM at the same shape for the ratio.
Output: results/fp4_gemm_shapes.jsonl and .md (TFLOP/s fp4, TFLOP/s bf16, fp4 speedup).
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import torch
from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor, per_tensor_amax_to_scale

from common import RESULTS, cuda_ms, stats

BF = torch.bfloat16
KN = [("qkv_sep (1152->1152)", 1152, 1152), ("qkv_concat (1152->3456)", 1152, 3456), ("kv_concat (1152->2304)", 1152, 2304),
      ("fc1 (1152->4608)", 1152, 4608), ("fc2 (4608->1152)", 4608, 1152)]


def q(t):
    return NVFP4Tensor.to_nvfp4(t, block_size=16, per_tensor_scale=per_tensor_amax_to_scale(t.abs().max()), is_swizzled_scales=True)


def main():
    out, lines = [], ["| site | tokens/image | images | M | fp4 TFLOP/s | bf16 TFLOP/s | fp4 speedup |", "|---|---|---|---|---|---|---|"]
    for name, K, N in KN:
        w = torch.randn(N, K, device="cuda", dtype=BF)
        wq = q(w)
        for tokens in (1024, 4096):
            for images in (1, 2, 4, 8):
                M = 2 * images * tokens
                x = torch.randn(M, K, device="cuda", dtype=BF)
                xq = q(x)
                f4 = lambda: torch._scaled_mm(xq.qdata.view(torch.float4_e2m1fn_x2), wq.qdata.t().view(torch.float4_e2m1fn_x2),
                                              xq.scale.view(torch.float8_e4m3fn), wq.scale.view(torch.float8_e4m3fn), out_dtype=BF)
                wt = w.t().contiguous()
                fb = lambda: x @ wt
                t4, tb = (stats(cuda_ms(f, iters=20, warmup=5))["median"] for f in (f4, fb))
                fl = 2 * M * K * N
                rec = dict(site=name, tokens=tokens, images=images, M=M, K=K, N=N, fp4_tflops=fl / t4 / 1e9, bf16_tflops=fl / tb / 1e9, speedup=tb / t4)
                out.append(rec)
                lines.append(f"| {name} | {tokens} | {images} | {M} | {rec['fp4_tflops']:.0f} | {rec['bf16_tflops']:.0f} | {rec['speedup']:.2f} |")
                del x, xq
        del w, wq
        torch.cuda.empty_cache()
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "fp4_gemm_shapes.jsonl").write_text("\n".join(json.dumps(r) for r in out) + "\n")
    (RESULTS / "fp4_gemm_shapes.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
