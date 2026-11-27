"""Reviewer point 1, op level: how much does our quantize kernel itself buy, with the rest of the model removed?

    python eval3/kernel_isolation.py                    # timing + bit-exactness, all shapes (about 2 min)
    python eval3/kernel_isolation.py --quick            # 2 shapes (smoke)
    python eval3/kernel_isolation.py --once             # one call per backend at M=8192, K=1152, under NVTX ranges (for ncu)
Compares, on the real activation shapes of PixArt (K 1152, 4608), SD3.5 Medium (K 1536) and FLUX (K 3072):
  torchao_eager     NVFP4Tensor.to_nvfp4 (what stock torchao runs without compile)
  torchao_compiled  the same function under torch.compile (what stock torchao runs inside a compiled block)
  ours_triton       our Triton kernel through the custom op the module stack uses, plus the to_blocked swizzle it needs
  ours_cuda         our CUDA kernel that writes the swizzled scales directly (the module stack default)
Each is timed twice: quantize only (per-tensor scale given) and amax + quantize (the dynamic path: one extra read of x).
Timing: CUDA Graph replay of 20 calls (no launch gaps), median of 5 replays; falls back to CUDA events if capture fails.
Achieved GB/s uses the ideal bytes (read bf16 x once, write FP4 codes + e4m3 scales; +1 read for the amax pass) against the copy roof
from results/eval3/peaks_sweep.json. Bit-exactness: codes and swizzled scales of every backend equal torchao_eager's, byte for byte.
Output: results/eval3/kernel_isolation.json and .md. Exit code 1 if any backend that ran is not bit-exact.
"""
import argparse
import json
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import torch

import fused_linear as fl
from common import RESULTS, cuda_ms, write_json

BF = torch.bfloat16
SHAPES = [(2048, 1152, "pixart 512px, K=1152"), (8192, 1152, "pixart 1024px, K=1152"), (8192, 4608, "pixart 1024px fc2, K=4608"),
          (9216, 1536, "sd35 1024px, K=1536"), (9216, 3072, "flux 1024px, K=3072")]
QUICK = SHAPES[:2]


def graph_ms(fn, n=20):
    """Device time per call: capture n calls in one CUDA Graph, replay 5 times, median / n. None if capture fails."""
    try:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(n):
                fn()
        g.replay()
        torch.cuda.synchronize()
        ts = []
        for _ in range(5):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            g.replay()
            b.record()
            torch.cuda.synchronize()
            ts.append(a.elapsed_time(b) / n)
        return statistics.median(ts), "graph"
    except Exception as e:  # noqa: BLE001  (some paths cannot be captured; report that, then time with events)
        torch.cuda.synchronize()
        return statistics.median(cuda_ms(fn, iters=50, warmup=10)), f"events ({type(e).__name__})"


def backends():
    """name -> fn(x, pts) returning (codes uint8 (M, K/2), swizzled scales as flat uint8)."""
    from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor
    from torchao.prototype.mx_formats.utils import to_blocked

    def torchao_q(x, pts):
        q = NVFP4Tensor.to_nvfp4(x, block_size=16, per_tensor_scale=pts, is_swizzled_scales=True)
        return q.qdata, q.scale

    out = {"torchao_eager": torchao_q}
    try:
        out["torchao_compiled"] = torch.compile(torchao_q, dynamic=False)
    except Exception as e:  # noqa: BLE001
        print("torch.compile of torchao quantize failed:", repr(e)[:150])
    for be, name in (("triton", "ours_triton"), ("cuda_swizzled", "ours_cuda")):
        try:
            op, native = fl.quant_op(be)
        except Exception as e:  # noqa: BLE001  (CUDA extension may not build on this machine)
            print(f"{name} unavailable:", repr(e)[:150])
            continue

        def f(x, pts, op=op, native=native):
            codes, scales = op(x, pts)
            if not native:
                M, K = x.shape
                scales = to_blocked(scales.view(M, K // 16)).flatten()
            return codes, scales
        out[name] = f
    return out


def as_bytes(t):
    return t.contiguous().view(torch.uint8).flatten()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--once", action="store_true", help="one NVTX-labelled call per backend for ncu (M=8192, K=1152)")
    a = ap.parse_args()
    from torchao.prototype.mx_formats.nvfp4_tensor import per_tensor_amax_to_scale
    bes = backends()
    g = torch.Generator("cuda").manual_seed(0)

    if a.once:
        x = torch.randn(8192, 1152, device="cuda", dtype=BF, generator=g)
        pts = per_tensor_amax_to_scale(x.abs().max())
        for name, fn in bes.items():
            fn(x, pts)  # compile / build outside the profiled range
        torch.cuda.synchronize()
        for name, fn in bes.items():
            torch.cuda.nvtx.range_push(f"kiso_{name}")
            fn(x, pts)
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_pop()
        print("ran one call per backend:", list(bes))
        return

    peaks_p = RESULTS / "eval3" / "peaks_sweep.json"
    roof = json.loads(peaks_p.read_text())["best"]["copy_gbs"] if peaks_p.exists() else None
    rows, not_exact = [], []
    for M, K, label in (QUICK if a.quick else SHAPES):
        x = torch.randn(M, K, device="cuda", dtype=BF, generator=g)
        pts = per_tensor_amax_to_scale(x.abs().max())
        ref_c, ref_s = bes["torchao_eager"](x, pts)
        ideal_q = M * K * 2 + M * K // 2 + M * K // 16
        for name, fn in bes.items():
            row = dict(m=M, k=K, shape=label, backend=name)
            try:
                c, s = fn(x, pts)
                row["bit_exact"] = bool(torch.equal(as_bytes(c), as_bytes(ref_c)) and torch.equal(as_bytes(s)[:as_bytes(ref_s).numel()], as_bytes(ref_s)))
                if not row["bit_exact"]:
                    not_exact.append(f"{name} {label}")
                ms, how = graph_ms(lambda: fn(x, pts))
                ms_amax, how2 = graph_ms(lambda: fn(x, per_tensor_amax_to_scale(x.abs().max())))
                row.update(quant_us=ms * 1e3, quant_gbs=ideal_q / ms / 1e6, timing=how,
                           amax_quant_us=ms_amax * 1e3, amax_quant_gbs=(ideal_q + M * K * 2) / ms_amax / 1e6, timing_amax=how2)
                if roof:
                    row["quant_frac_of_copy_roof"] = row["quant_gbs"] / roof
            except Exception as e:  # noqa: BLE001
                row["error"] = f"{type(e).__name__}: {str(e)[:200]}"
            rows.append(row)
            print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
        del x
        torch.cuda.empty_cache()
    write_json("eval3/kernel_isolation.json", dict(rows=rows, copy_roof_gbs=roof, not_bit_exact=not_exact))
    L = ["# Quantize kernel in isolation (one call, CUDA Graph timed)\n",
         f"Copy roof: {roof:.0f} GB/s.\n" if roof else "Copy roof: missing (run W104).\n",
         "| shape | backend | bit-exact | quantize us | GB/s | % of copy roof | amax+quantize us | timing |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if "error" in r:
            L.append(f"| {r['shape']} | {r['backend']} | - | error: {r['error'][:60]} | | | | |")
            continue
        pct = f"{100 * r['quant_frac_of_copy_roof']:.0f}%" if "quant_frac_of_copy_roof" in r else "-"
        L.append(f"| {r['shape']} | {r['backend']} | {'yes' if r['bit_exact'] else 'NO'} | {r['quant_us']:.1f} | {r['quant_gbs']:.0f} | {pct} | "
                 f"{r['amax_quant_us']:.1f} | {r['timing']} |")
    (RESULTS / "eval3" / "kernel_isolation.md").write_text("\n".join(L) + "\n")
    print("wrote results/eval3/kernel_isolation.md; not bit-exact:", not_exact or "none")
    sys.exit(1 if not_exact else 0)


if __name__ == "__main__":
    main()
