"""
Minimal ncu/nsys entrypoint. Runs ONE path (fused or unfused) in a tight loop
on synthetic DiT-shaped tensors so a profiler can attribute GPU time and DRAM
throughput to the exact kernels.

Usage (under profiler):
    ncu --set full --launch-count N -o prof \
        python profile_kernel.py --path fused   --iters 50
    ncu --set full --launch-count N -o prof \
        python profile_kernel.py --path unfused --iters 50

Standalone (just runs, no profiler):
    python profile_kernel.py --path fused --iters 50
"""
import argparse
import torch

from fused_adaln_nvfp4 import (
    fused_ln_adaln_nvfp4,
    layernorm_modulate_eager,
)
from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--path", choices=["fused", "unfused"], default="fused")
    p.add_argument("--hidden", type=int, default=1152)
    p.add_argument("--tokens", type=int, default=256)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    dev = "cuda"
    dt = torch.bfloat16
    torch.manual_seed(a.seed)
    M, H = a.tokens, a.hidden

    x = torch.randn(M, H, device=dev, dtype=dt)
    scale = torch.randn(1, H, device=dev, dtype=dt) * 0.1
    shift = torch.randn(1, H, device=dev, dtype=dt) * 0.1
    sr = scale.repeat_interleave(M, 0).contiguous()
    shr = shift.repeat_interleave(M, 0).contiguous()
    eps = 1e-6

    if a.path == "fused":
        def step():
            return fused_ln_adaln_nvfp4(x, sr, shr, eps)
    else:
        def step():
            xm = layernorm_modulate_eager(x, sr, shr, eps)
            q = NVFP4Tensor.to_nvfp4(xm, block_size=16)
            return q.qdata, q.scale

    # warmup (JIT compile + autotune)
    for _ in range(a.warmup):
        step()
    torch.cuda.synchronize()

    # ---- CUDA-event GPU timing (true device time, not wall clock) ----
    ev_start = [torch.cuda.Event(enable_timing=True) for _ in range(a.iters)]
    ev_end = [torch.cuda.Event(enable_timing=True) for _ in range(a.iters)]
    for i in range(a.iters):
        ev_start[i].record()
        step()
        ev_end[i].record()
    torch.cuda.synchronize()
    gpu_ms = [ev_start[i].elapsed_time(ev_end[i]) for i in range(a.iters)]
    gpu_ms.sort()
    med = gpu_ms[len(gpu_ms) // 2]
    mn = gpu_ms[0]

    # profiled region for nsys (nvtx-wrapped)
    torch.cuda.nvtx.range_push(f"{a.path}_loop")
    for _ in range(a.iters):
        step()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()

    print(f"path={a.path} M={M} H={H} iters={a.iters}  "
          f"GPU-time median={med*1000:.2f}us  min={mn*1000:.2f}us")


if __name__ == "__main__":
    main()
