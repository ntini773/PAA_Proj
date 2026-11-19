"""Where does one transformer forward spend its time? Per stage: GPU span, GPU busy, gaps, kernels, CPU launch time.

    python timing/profile_stages.py --variant module+static --size 512
    python timing/profile_stages.py --variant stock --size 1024 --iters 5

Why: the roofline and nsys class tables say how long kernel classes take, not which part of the block they belong to or
whether the GPU is waiting for the CPU. This table answers both per stage (norm1, attn1, attn2, norm2, ff, quant, gemm, ...)
so the memory-movement work (quantize, glue) and the launch-bound stages can be told apart.

Method (timing/stages.py): hooks wrap the block pieces in stage ranges. Pass 1 (no profiler) gives span_ms from CUDA events
and cpu_ms from the host clock; pass 2 (torch.profiler) gives busy_ms and kernel counts per stage from kernel launches
nested inside each stage range. gap = span - busy; a stage is launch-bound when gap > busy.
Compiled variants have no inner stages (kernels are fused across module boundaries): they report the whole forward and
kernel classes only.
Output: results/stage_timing.jsonl (one record per run) and a Markdown table on stdout.
"""
import argparse
import collections
import json
import pathlib
import statistics
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import torch
from torch.profiler import ProfilerActivity, profile

import variants as vr
from analysis.kernel_classes import classify
from common import RESULTS, append_jsonl, load_prompt_embeds
from timing.stages import StageRecorder, install


def _kernels(ev):
    ks = list(getattr(ev, "kernels", []))
    for c in getattr(ev, "cpu_children", []):
        ks += _kernels(c)
    return ks


def measure(fwd, rec, iters):
    """span/cpu per stage per forward from CUDA events and the host clock."""
    rec.clear()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        with torch.no_grad():
            fwd()
    e.record()
    cpu_total = (time.perf_counter() - t0) * 1e3
    torch.cuda.synchronize()
    stages = rec.summary()
    for d in stages.values():
        d["span_ms"] /= iters
        d["cpu_ms"] /= iters
        d["calls"] //= iters
    return stages, s.elapsed_time(e) / iters, cpu_total / iters


def profile_pass(fwd, iters):
    """busy_ms and kernel count per stage and per kernel class from torch.profiler (per forward)."""
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            with torch.no_grad():
                fwd()
        torch.cuda.synchronize()
    busy, count = collections.defaultdict(float), collections.defaultdict(int)
    classes = collections.defaultdict(float)
    for ev in prof.events():
        if ev.name.startswith("stage::"):
            ks = _kernels(ev)
            name = ev.name[len("stage::"):]
            busy[name] += sum(k.duration for k in ks) / 1e3
            count[name] += len(ks)
    total_busy, total_k = 0.0, 0
    for ev in prof.events():
        for k in getattr(ev, "kernels", []):
            classes[classify(k.name)] += k.duration / 1e3
            total_busy += k.duration / 1e3
            total_k += 1
    return ({n: dict(busy_ms=busy[n] / iters, kernels=count[n] / iters) for n in busy},
            total_busy / iters, total_k / iters, {c: v / iters for c, v in classes.items()})


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--variant", default="stock")
    a.add_argument("--size", type=int, default=512)
    a.add_argument("--batch", type=int, default=1)
    a.add_argument("--iters", type=int, default=3)
    a.add_argument("--embeds", default="prompts/embeds_pixart.pt")
    a.add_argument("--out", default="stage_timing.jsonl")
    a = a.parse_args()

    items = load_prompt_embeds(a.embeds)
    b = vr.build(a.variant, a.size)
    fwd = vr.make_forward(b, items, a.size, a.batch)
    for _ in range(3 if not b.v.compile else 4):
        with torch.no_grad():
            fwd()
    torch.cuda.synchronize()

    rec = StageRecorder(profiler_labels=True)
    undo = (lambda: None) if b.v.compile else install(b.pipe.transformer, rec)
    stages, wall_ms, cpu_ms = measure(fwd, rec, a.iters)
    busy, total_busy, total_k, classes = profile_pass(fwd, a.iters)
    undo()

    rows = []
    for name in stages:
        d = stages[name]
        bz = busy.get(name, dict(busy_ms=0.0, kernels=0))
        gap = d["span_ms"] - bz["busy_ms"]
        rows.append(dict(stage=name, span_ms=d["span_ms"], busy_ms=bz["busy_ms"], gap_ms=gap, cpu_ms=d["cpu_ms"], calls=d["calls"],
                         kernels=bz["kernels"], mean_kernel_us=(bz["busy_ms"] * 1e3 / bz["kernels"]) if bz["kernels"] else 0.0,
                         launch_bound=bool(gap > bz["busy_ms"])))
    rec_out = dict(variant=a.variant, size=a.size, batch=a.batch, wall_ms=wall_ms, cpu_launch_ms=cpu_ms, gpu_busy_ms=total_busy,
                   kernels=total_k, gpu_busy_pct=100 * total_busy / wall_ms, kernel_class_ms=classes, stages=rows, report=b.report)
    append_jsonl(a.out, rec_out)
    print(f"{a.variant} {a.size}px batch {a.batch}: forward {wall_ms:.2f} ms, GPU busy {total_busy:.2f} ms ({rec_out['gpu_busy_pct']:.0f}%), "
          f"{total_k:.0f} kernels, CPU launch {cpu_ms:.2f} ms")
    print("| stage | span ms | busy ms | gap ms | cpu ms | calls | kernels | mean kernel us | launch-bound |\n|---|---|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda r: -r["span_ms"]):
        print(f"| {r['stage']} | {r['span_ms']:.2f} | {r['busy_ms']:.2f} | {r['gap_ms']:.2f} | {r['cpu_ms']:.2f} | {r['calls']} | "
              f"{r['kernels']:.0f} | {r['mean_kernel_us']:.1f} | {'yes' if r['launch_bound'] else ''} |")
    print("kernel classes ms:", {k: round(v, 2) for k, v in sorted(classes.items(), key=lambda kv: -kv[1])})


if __name__ == "__main__":
    main()
