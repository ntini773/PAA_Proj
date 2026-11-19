"""Stage timing tables and figures from results/stage_timing.jsonl (written by timing/profile_stages.py). No GPU needed.

    python analysis/analyze_stages.py

Outputs
    results/analysis/stage_timing.csv       one row per (variant, size, stage)
    results/analysis/stage_timing.md        per size: forward ms, GPU busy %, kernels, CPU launch ms, then per-stage table
    results/figures/stage_timing_<size>.png stacked bars per variant: GPU busy per stage, GPU gaps, time outside the stages
How to read it: busy is real GPU work; gap is the GPU waiting inside a stage (launch-bound when gap > busy); "outside"
is forward time not inside any stage (embedding, final layer, glue between stages).
Only eager variants have inner stages. Compiled variants show only the whole forward and the kernel-class split.
"""
import collections
import csv
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "results" / "stage_timing.jsonl"
OUT = ROOT / "results" / "analysis"
BLOCK_STAGES = ("norm1", "attn1", "attn2", "norm2", "ff")  # top-level pieces; quant/gemm are nested inside them


def main():
    if not SRC.exists():
        raise SystemExit(f"missing {SRC}: run timing/profile_stages.py first")
    recs = [json.loads(l) for l in SRC.read_text().splitlines() if l.strip()]
    latest = {}
    for r in recs:  # keep the last run of each (variant, size, batch)
        latest[r["variant"], r["size"], r["batch"]] = r
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "stage_timing.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "size", "batch", "stage", "span_ms", "busy_ms", "gap_ms", "cpu_ms", "calls", "kernels", "mean_kernel_us", "launch_bound"])
        for (v, s, b), r in sorted(latest.items(), key=lambda kv: (kv[0][1], kv[0][0])):
            for st in r["stages"]:
                w.writerow([v, s, b, st["stage"], *(round(st[k], 3) for k in ("span_ms", "busy_ms", "gap_ms", "cpu_ms")), st["calls"],
                            round(st["kernels"], 1), round(st["mean_kernel_us"], 1), st["launch_bound"]])
    md = []
    for size in sorted({k[1] for k in latest}):
        md.append(f"\n## {size} px\n\n| variant | forward ms | GPU busy ms | busy % | kernels | CPU launch ms |\n|---|---|---|---|---|---|")
        for (v, s, b), r in sorted(latest.items()):
            if s == size:
                md.append(f"| {v} (batch {b}) | {r['wall_ms']:.2f} | {r['gpu_busy_ms']:.2f} | {r['gpu_busy_pct']:.0f} | {r['kernels']:.0f} | {r['cpu_launch_ms']:.2f} |")
        for (v, s, b), r in sorted(latest.items()):
            if s == size and r["stages"]:
                md.append(f"\n### {v}\n\n| stage | span ms | busy ms | gap ms | cpu ms | calls | kernels | mean kernel us | launch-bound |\n|---|---|---|---|---|---|---|---|---|")
                for st in sorted(r["stages"], key=lambda x: -x["span_ms"]):
                    md.append(f"| {st['stage']} | {st['span_ms']:.2f} | {st['busy_ms']:.2f} | {st['gap_ms']:.2f} | {st['cpu_ms']:.2f} | {st['calls']} | "
                              f"{st['kernels']:.0f} | {st['mean_kernel_us']:.1f} | {'yes' if st['launch_bound'] else ''} |")
    (OUT / "stage_timing.md").write_text("\n".join(md) + "\n")
    print("wrote", OUT / "stage_timing.csv", "and stage_timing.md")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib missing: tables only")
        return
    fig_dir = ROOT / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for size in sorted({k[1] for k in latest}):
        runs = [(v, r) for (v, s, b), r in sorted(latest.items()) if s == size]
        fig, ax = plt.subplots(figsize=(max(7, 1.6 * len(runs)), 5))
        for i, (v, r) in enumerate(runs):
            bottom = 0.0
            top = {st["stage"]: st for st in r["stages"] if st["stage"] in BLOCK_STAGES}
            for name in BLOCK_STAGES:
                if name in top:
                    ax.bar(i, top[name]["busy_ms"], bottom=bottom, color=plt.cm.tab10(BLOCK_STAGES.index(name)), label=f"{name} busy" if i == 0 else None)
                    bottom += top[name]["busy_ms"]
            gaps = sum(st["gap_ms"] for st in top.values())
            outside = max(r["wall_ms"] - sum(st["span_ms"] for st in top.values()), 0)
            if top:
                ax.bar(i, gaps, bottom=bottom, color="#cccccc", hatch="//", label="GPU gap inside stages" if i == 0 else None)
                ax.bar(i, outside, bottom=bottom + gaps, color="#888888", label="outside stages" if i == 0 else None)
            else:
                ax.bar(i, r["gpu_busy_ms"], color="#4c78a8", label="GPU busy (no stages)" if i == 0 else None)
                ax.bar(i, max(r["wall_ms"] - r["gpu_busy_ms"], 0), bottom=r["gpu_busy_ms"], color="#cccccc", hatch="//")
        ax.set_xticks(range(len(runs)))
        ax.set_xticklabels([v for v, _ in runs], rotation=40, ha="right", fontsize=8)
        ax.set_ylabel("ms per forward")
        ax.set_title(f"PixArt-Sigma {size}px: GPU busy time per stage, gaps and time outside stages")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(fig_dir / f"stage_timing_{size}.png", dpi=150)
        plt.close(fig)
        print("wrote", fig_dir / f"stage_timing_{size}.png")


if __name__ == "__main__":
    main()
