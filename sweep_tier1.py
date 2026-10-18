"""Tier-1 scaling sweep on synthetic tensors: correctness + GPU time vs rows M.

M = batch * tokens. PixArt-Sigma: tokens = (size/16)^2, CFG doubles batch.
Paths timed (CUDA events, median of --iters):
  unfused3   : eager LN+mod (fp32) -> torchao to_nvfp4          (3+ kernels)
  fused_full : fused_ln_adaln_nvfp4 (LN+mod+quant, 1 kernel)
  quant_old  : torchao to_nvfp4 on a ready x_mod               (what e2e replaces)
  quant_new  : fused_nvfp4_quantize on a ready x_mod           (what e2e wires in)
  compiled   : torch.compile(unfused3)  -> checks the eager-baseline strawman
Writes results/tier1.csv. 'bitexact' = differing packed bytes + differing scale bytes
(a few at large M are the known fp32 reduction-order midpoint ties).   Run: pixi run python sweep_tier1.py
"""
import os, sys, csv, statistics
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import torch
from fused_adaln_nvfp4 import (fused_ln_adaln_nvfp4, fused_nvfp4_quantize,
                               layernorm_modulate_eager, check_correctness)
from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor

H, EPS, ITERS = 1152, 1e-6, 200


def gpu_ms(fn, iters=ITERS, warm=30):
    for _ in range(warm): fn()
    torch.cuda.synchronize()
    s = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    e = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        s[i].record(); fn(); e[i].record()
    torch.cuda.synchronize()
    return statistics.median(a.elapsed_time(b) for a, b in zip(s, e))


def main():
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    rows = []
    print(f"{'size':>5} {'B':>2} {'M':>6} | {'bitexact':>8} | {'unfused3':>9} {'fused':>8} {'x':>5} | "
          f"{'q_old':>8} {'q_new':>8} {'x':>5} | {'compiled':>9}")
    for size in (256, 512, 768, 1024, 1536, 2048):
        N = (size // 16) ** 2
        for B in (1, 2):
            M = B * N
            torch.manual_seed(0)
            x = torch.randn(M, H, device="cuda", dtype=torch.bfloat16)
            sc = (torch.randn(1, H, device="cuda", dtype=torch.bfloat16) * 0.1).repeat_interleave(M, 0).contiguous()
            sh = (torch.randn(1, H, device="cuda", dtype=torch.bfloat16) * 0.1).repeat_interleave(M, 0).contiguous()
            xm = layernorm_modulate_eager(x, sc, sh, EPS)
            pd, _, sd, _ = check_correctness(M, H)
            def unf():
                q = NVFP4Tensor.to_nvfp4(layernorm_modulate_eager(x, sc, sh, EPS), block_size=16)
                return q.qdata, q.scale
            t_unf = gpu_ms(unf)
            t_fus = gpu_ms(lambda: fused_ln_adaln_nvfp4(x, sc, sh, EPS))
            t_qo = gpu_ms(lambda: NVFP4Tensor.to_nvfp4(xm, block_size=16))
            t_qn = gpu_ms(lambda: fused_nvfp4_quantize(xm))
            try:
                cf = torch.compile(unf)
                t_cmp = gpu_ms(cf)
            except Exception as ex:
                t_cmp = float("nan"); print("   compile failed:", type(ex).__name__)
            r = dict(size=size, batch=B, M=M, bitexact=f"{pd}B+{sd}s",
                     unfused3_ms=t_unf, fused_full_ms=t_fus, quant_old_ms=t_qo,
                     quant_new_ms=t_qn, compiled_unfused_ms=t_cmp)
            rows.append(r)
            print(f"{size:>5} {B:>2} {M:>6} | {r['bitexact']:>8} | {t_unf:9.3f} {t_fus:8.3f} {t_unf/t_fus:5.1f} | "
                  f"{t_qo:8.3f} {t_qn:8.3f} {t_qo/t_qn:5.1f} | {t_cmp:9.3f}")
            del x, sc, sh, xm
            torch.cuda.empty_cache()
    with open(os.path.join(HERE, "results", "tier1.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
