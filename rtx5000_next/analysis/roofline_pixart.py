"""Whole-step roofline for PixArt-Sigma. No GPU needed.

    python analysis/roofline_pixart.py [--batch 2] [--quant-flops-per-elem 12]
Inputs : results/analysis/nsys_summary.csv (run analysis/analyze_nsys.py first), results/peaks.json (bench/peaks.py)
Outputs: results/analysis/roofline_points.csv, results/figures/roofline_pixart_<size>.png

Per config and size, three classes become one point each:
  gemm       linear layers, analytic FLOPs and ideal DRAM bytes (bf16 for the bf16 config, NVFP4 for the others)
  attention  self + cross attention, analytic FLOPs, bytes = read Q K V, write O (flash, no score matrix)
  quant      the quantize path and glue kernels, nominal FLOPs (--quant-flops-per-elem), ideal bytes = read bf16, write codes + scales
x = FLOP / ideal byte, y = achieved TFLOP/s = FLOPs / kernel time of that class from nsys.
Efficiency = achieved / min(compute roof of the class, DRAM roof at that intensity).
Limits: the quant class time includes every non-GEMM, non-attention kernel (GELU, residual, norms), so its efficiency is a lower bound.
        Ideal bytes ignore L2 reuse and the per-tensor max pass.
"""
import argparse
import csv
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent.parent
D, FF, HEADS, LAYERS, TEXT = 1152, 4608, 16, 28, 300
FP4_B = 0.5 + 1 / 16  # packed code + e4m3 scale per element
MARK = {"gemm": "o", "attention": "s", "quant": "^"}


def linears(rows, text_rows):
    """(m, k, n) of every Linear in one block."""
    return ([(rows, D, D)] * 3 + [(rows, D, D)] + [(rows, D, D)] + [(text_rows, D, D)] * 2 + [(rows, D, D)]
            + [(rows, D, FF), (rows, FF, D)])


def model(size, batch, qf):
    n = (size // 16) ** 2
    rows, trows = batch * n, batch * TEXT
    lins = linears(rows, trows)
    gemm_flops = LAYERS * sum(2 * m * k * nn for m, k, nn in lins)
    gemm_bytes_bf16 = LAYERS * sum(2 * (m * k + k * nn + m * nn) for m, k, nn in lins)
    gemm_bytes_fp4 = LAYERS * sum(FP4_B * (m * k + k * nn) + 2 * m * nn for m, k, nn in lins)
    attn_flops = LAYERS * (4 * batch * n * n * D + 4 * batch * n * TEXT * D)
    attn_bytes = LAYERS * 2 * (4 * batch * n * D + 2 * batch * n * D + 2 * batch * TEXT * D)
    quant_elems = LAYERS * sum(m * k for m, k, _ in lins)
    return dict(gemm_flops=gemm_flops, gemm_bytes_bf16=gemm_bytes_bf16, gemm_bytes_fp4=gemm_bytes_fp4,
                attn_flops=attn_flops, attn_bytes=attn_bytes,
                quant_flops=qf * quant_elems, quant_bytes=quant_elems * (2 + FP4_B))


def points(row, size, m, peaks):
    bf16 = row["config"] == "bf16"
    fp8 = row["config"] == "fp8"
    out = []
    g_ms = float(row["bf16_gemm_ms"] if bf16 else row["fp4_gemm_ms"])
    if not fp8 and g_ms > 0:
        out.append(("gemm", g_ms, m["gemm_flops"], m["gemm_bytes_bf16"] if bf16 else m["gemm_bytes_fp4"],
                    peaks["bf16_tflops"] if bf16 else peaks.get("fp4_tflops")))
    if float(row["attention_ms"]) > 0:
        out.append(("attention", float(row["attention_ms"]), m["attn_flops"], m["attn_bytes"], peaks["attn_tflops"]))
    if not bf16 and float(row["quant_path_ms"]) > 0:
        out.append(("quant", float(row["quant_path_ms"]), m["quant_flops"], m["quant_bytes"], None))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=2, help="transformer batch (2 = CFG with batch 1)")
    ap.add_argument("--quant-flops-per-elem", type=float, default=12.0)
    a = ap.parse_args()

    peaks = json.loads((ROOT / "results" / "peaks.json").read_text())
    bw = peaks["copy_gbs"]
    runs = list(csv.DictReader(open(ROOT / "results" / "analysis" / "nsys_summary.csv")))
    pts = []
    for row in runs:
        size = int(row["size"])
        m = model(size, a.batch, a.quant_flops_per_elem)
        for cls, ms, flops, byts, roof in points(row, size, m, peaks):
            ai = flops / byts
            ach = flops / ms / 1e9
            attain = min(bw * ai / 1e3, roof) if roof else bw * ai / 1e3
            pts.append(dict(config=row["config"], size=size, cls=cls, time_ms=round(ms, 3), gflop=round(flops / 1e9, 1),
                            ideal_gb=round(byts / 1e9, 3), intensity=round(ai, 2), achieved_tflops=round(ach, 2),
                            achieved_gbs=round(byts / ms / 1e6, 1), attainable_tflops=round(attain, 2),
                            efficiency=round(ach / attain, 3)))
    out = ROOT / "results" / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "roofline_points.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pts[0]))
        w.writeheader()
        w.writerows(pts)

    fig_dir = ROOT / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    cfgs = sorted({p["config"] for p in pts})
    colors = {c: plt.cm.tab20(i % 20) for i, c in enumerate(cfgs)}
    for size in sorted({p["size"] for p in pts}):
        sp = [p for p in pts if p["size"] == size]
        fig, ax = plt.subplots(1, 2, figsize=(15, 6))
        xs = [10 ** (i / 20) for i in range(-20, 81)]
        for name, peak, ls in (("bf16 GEMM", peaks["bf16_tflops"], "-"), ("NVFP4 GEMM", peaks.get("fp4_tflops"), "-"),
                               ("attention", peaks["attn_tflops"], ":")):
            if peak:
                ax[0].plot(xs, [min(bw * x / 1e3, peak) for x in xs], ls, color="k", lw=1, alpha=0.5)
                ax[0].annotate(f"{name} {peak:.0f} TFLOP/s", (xs[-1], peak), ha="right", va="bottom", fontsize=8)
        ax[0].annotate(f"DRAM {bw:.0f} GB/s (copy)", (0.3, bw * 0.3 / 1e3), fontsize=8, rotation=35)
        for p in sp:
            ax[0].scatter(p["intensity"], p["achieved_tflops"], marker=MARK[p["cls"]], s=70, color=colors[p["config"]], edgecolor="k", zorder=3)
        ax[0].set_xscale("log")
        ax[0].set_yscale("log")
        ax[0].set_xlabel("FLOP / ideal DRAM byte")
        ax[0].set_ylabel("achieved TFLOP/s")
        ax[0].set_title(f"PixArt-Sigma {size}px, batch {a.batch}: roofline (o gemm, s attention, ^ quantize path)")
        handles = [plt.Line2D([], [], marker="o", ls="", color=colors[c], label=c) for c in cfgs]
        ax[0].legend(handles=handles, fontsize=7, loc="upper left")
        for j, cls in enumerate(("gemm", "attention", "quant")):
            cp = [p for p in sp if p["cls"] == cls]
            ax[1].bar([i + j * 0.27 for i in range(len(cfgs))], [next((p["efficiency"] for p in cp if p["config"] == c), 0) for c in cfgs],
                      0.27, label=cls)
        ax[1].set_xticks([i + 0.27 for i in range(len(cfgs))])
        ax[1].set_xticklabels(cfgs, rotation=60, ha="right", fontsize=8)
        ax[1].axhline(1, color="k", lw=0.8)
        ax[1].set_ylabel("achieved / attainable (roofline efficiency)")
        ax[1].set_title("how close each class is to its roof")
        ax[1].legend()
        fig.tight_layout()
        fig.savefig(fig_dir / f"roofline_pixart_{size}.png", dpi=150)
        plt.close(fig)
        print("wrote", fig_dir / f"roofline_pixart_{size}.png")
    print("wrote", out / "roofline_points.csv", f"({len(pts)} points)")


if __name__ == "__main__":
    main()
