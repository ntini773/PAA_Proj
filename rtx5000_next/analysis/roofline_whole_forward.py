"""Whole-forward roofline for PixArt-Sigma: every config and every size from the step-time runs. No GPU needed.

    python analysis/roofline_whole_forward.py [--batch 2]
Inputs : results/step_target.jsonl (S01, any size and config), results/peaks.json (R01)
Outputs: results/analysis/roofline_whole_forward.csv, .md, results/figures/roofline_whole_forward.png

For each (config, size) the whole forward is one point:
  FLOPs      useful work only: linear layers + attention (quantize FLOPs are overhead and not counted)
  bytes      ideal DRAM traffic of the modelled ops: linears, attention, quantize (none for bf16 and free_bound)
  speed of light (SoL) time  sum over classes of max(FLOPs / class compute roof, bytes / DRAM roof)
  efficiency = SoL time / measured time. 1.0 means every class ran at its roof. Eager and CUDA Graph times both shown.
Limits: LayerNorm, modulate, GELU and residual traffic are not in the bytes, so SoL is optimistic and efficiency is a ceiling
        on what the model can say. Roofs are the measured limits in peaks.json. fp8 is skipped (no fp8 model of bytes or roof).
        PixArt shapes only: the analytic model is in analysis/roofline_pixart.py.
"""
import argparse
import csv
import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
from roofline_pixart import model  # noqa: E402

MARK = {256: "o", 512: "s", 1024: "^"}


def latest_rows():
    """One row per (config, size). Prefer rows that carry a CUDA Graph time (S01), later rows win."""
    best = {}
    for line in (ROOT / "results" / "step_target.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if "layer" in r or r.get("mode") not in ("bf16", "stock", "fused"):
            continue
        key = (r["config"], r["size"])
        has_graph = "graph_ms" in r
        if key not in best or has_graph >= ("graph_ms" in best[key]):
            best[key] = r
    return best


def sol(mode, config, m, peaks, bw):
    bf16 = mode == "bf16"
    gpeak = peaks["bf16_tflops"] if bf16 else peaks.get("fp4_tflops")
    gbytes = m["gemm_bytes_bf16"] if bf16 else m["gemm_bytes_fp4"]
    t = max(m["gemm_flops"] / (gpeak * 1e12) if gpeak else 0, gbytes / (bw * 1e9))
    t += max(m["attn_flops"] / (peaks["attn_tflops"] * 1e12), m["attn_bytes"] / (bw * 1e9))
    qbytes = 0 if bf16 or config == "free_bound" else m["quant_bytes"]
    t += qbytes / (bw * 1e9)
    return t * 1e3, gbytes + m["attn_bytes"] + qbytes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=2, help="transformer batch (2 = CFG with batch 1)")
    a = ap.parse_args()

    peaks_p = ROOT / "results" / "peaks.json"
    steps_p = ROOT / "results" / "step_target.jsonl"
    if not peaks_p.exists():
        raise SystemExit(f"missing {peaks_p}: run bench/peaks.py (R01) first")
    if not steps_p.exists():
        raise SystemExit(f"missing {steps_p}: run profiling/run_step_matrix.sh (S01) first")
    peaks = json.loads(peaks_p.read_text())
    bw = peaks["copy_gbs"]

    rows = []
    for (config, size), r in sorted(latest_rows().items(), key=lambda kv: (kv[0][1], kv[0][0])):
        m = model(size, a.batch, 0)
        t_sol, ideal_bytes = sol(r["mode"], config, m, peaks, bw)
        flops = m["gemm_flops"] + m["attn_flops"]
        eager = r["gpu_ms"]["median"]
        graph = r["graph_ms"]["median"] if "graph_ms" in r else None
        rows.append(dict(config=config, size=size, eager_ms=round(eager, 2), graph_ms=round(graph, 2) if graph else "",
                         gflop=round(flops / 1e9, 1), ideal_gb=round(ideal_bytes / 1e9, 3), intensity=round(flops / ideal_bytes, 1),
                         sol_ms=round(t_sol, 2), eff_eager=round(t_sol / eager, 3), eff_graph=round(t_sol / graph, 3) if graph else "",
                         tflops_eager=round(flops / eager / 1e9, 2), tflops_graph=round(flops / graph / 1e9, 2) if graph else ""))
    if not rows:
        raise SystemExit("no usable rows in step_target.jsonl")

    out = ROOT / "results" / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "roofline_whole_forward.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(out / "roofline_whole_forward.md", "w") as f:
        f.write(f"Roofs: DRAM {bw:.0f} GB/s, bf16 {peaks['bf16_tflops']:.0f}, fp4 {peaks.get('fp4_tflops') or 'n/a'}, attention {peaks['attn_tflops']:.0f} TFLOP/s\n")
        for size in sorted({r["size"] for r in rows}):
            f.write(f"\n### {size} px\n| config | eager ms | graph ms | SoL ms | eff eager | eff graph | TFLOP/s eager |\n|---|---|---|---|---|---|---|\n")
            for r in (x for x in rows if x["size"] == size):
                f.write(f"| {r['config']} | {r['eager_ms']} | {r['graph_ms']} | {r['sol_ms']} | {r['eff_eager']} | {r['eff_graph']} | {r['tflops_eager']} |\n")
    print((out / "roofline_whole_forward.md").read_text())

    cfgs = sorted({r["config"] for r in rows})
    sizes = sorted({r["size"] for r in rows})
    colors = {c: plt.cm.tab20(i % 20) for i, c in enumerate(cfgs)}
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    xs = [10 ** (i / 20) for i in range(0, 81)]
    for peak, ls in ((peaks["bf16_tflops"], "-"), (peaks.get("fp4_tflops"), "--")):
        if peak:
            ax[0].plot(xs, [min(bw * x / 1e3, peak) for x in xs], ls, color="k", lw=1, alpha=0.5)
            ax[0].annotate(f"{peak:.0f} TFLOP/s", (xs[-1], peak), ha="right", va="bottom", fontsize=8)
    for r in rows:
        ax[0].scatter(r["intensity"], r["tflops_eager"], marker=MARK.get(r["size"], "o"), s=60, color=colors[r["config"]], edgecolor="k", zorder=3)
        if r["tflops_graph"] != "":
            ax[0].scatter(r["intensity"], r["tflops_graph"], marker=MARK.get(r["size"], "o"), s=60, facecolor="none", edgecolor=colors[r["config"]], linewidth=1.5, zorder=3)
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("useful FLOP / ideal DRAM byte (whole forward)")
    ax[0].set_ylabel("achieved TFLOP/s (filled: eager, open: CUDA Graph)")
    ax[0].set_title("PixArt-Sigma whole forward (o 256, s 512, ^ 1024 px; solid roof bf16, dashed fp4)")
    ax[0].legend(handles=[plt.Line2D([], [], marker="o", ls="", color=colors[c], label=c) for c in cfgs], fontsize=7, loc="lower right")
    width = 0.8 / len(sizes)
    for j, size in enumerate(sizes):
        vals = []
        for c in cfgs:
            r = next((x for x in rows if x["config"] == c and x["size"] == size), None)
            vals.append((r["eff_graph"] if r and r["eff_graph"] != "" else r["eff_eager"]) if r else 0)
        ax[1].bar([i + j * width for i in range(len(cfgs))], vals, width, label=f"{size} px")
    ax[1].set_xticks([i + 0.4 - width / 2 for i in range(len(cfgs))])
    ax[1].set_xticklabels(cfgs, rotation=60, ha="right", fontsize=8)
    ax[1].axhline(1, color="k", lw=0.8)
    ax[1].set_ylabel("speed-of-light time / measured time (CUDA Graph if available)")
    ax[1].set_title("whole-forward roofline efficiency")
    ax[1].legend()
    fig.tight_layout()
    fig_dir = ROOT / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "roofline_whole_forward.png", dpi=150)
    print("wrote", fig_dir / "roofline_whole_forward.png", "and", out / "roofline_whole_forward.csv")


if __name__ == "__main__":
    main()
