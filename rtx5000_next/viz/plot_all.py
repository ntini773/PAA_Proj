"""Figures for the hypotheses. Every plot is skipped when its data is missing.

    python viz/plot_all.py   ->  results/figures/*.png
"""
import csv
import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
from kernel_classes import CATEGORIES

R = ROOT / "results"
FIG = R / "figures"
FIG.mkdir(parents=True, exist_ok=True)
ORDER = ["bf16", "fp8", "stock", "stock_compile", "fused_triton_narrow", "fused_triton", "fused_triton_op_compile",
         "fused_cuda_row", "fused_cuda_swizzled", "fused_cuda_swizzled_compile", "free_bound"]
COLORS = {"fp4_gemm": "#1d4ed8", "bf16_gemm": "#60a5fa", "attention": "#7c3aed", "our_quant": "#15803d", "inductor": "#f59e0b",
          "layernorm": "#9ca3af", "copy": "#ef4444", "elementwise": "#f97316", "reduce": "#fb7185", "other": "#d1d5db"}


def read_csv(p):
    return list(csv.DictReader(open(p))) if p.exists() else []


def jsonl(name):
    p = R / name
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def order_key(c):
    return ORDER.index(c) if c in ORDER else len(ORDER)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=150)
    plt.close(fig)
    print("wrote", FIG / name)


def latest(rows, keys):
    out = {}
    for r in rows:
        out[tuple(r[k] for k in keys)] = r
    return out


def step_time():
    rows = [r for r in jsonl("step_target.jsonl") if r.get("model") == "pixart_sigma" and "layer" not in r]
    last = latest(rows, ("config", "size"))
    sizes = sorted({k[1] for k in last})
    if not sizes:
        return
    fig, axes = plt.subplots(1, len(sizes), figsize=(5 * len(sizes), 4.5), squeeze=False)
    for ax, s in zip(axes[0], sizes):
        items = sorted([(k[0], v) for k, v in last.items() if k[1] == s], key=lambda kv: order_key(kv[0]))
        names = [i[0] for i in items]
        med = [i[1]["gpu_ms"]["median"] for i in items]
        err = [[m - i[1]["gpu_ms"]["p10"] for m, i in zip(med, items)], [i[1]["gpu_ms"]["p90"] - m for m, i in zip(med, items)]]
        ax.bar(range(len(names)), med, yerr=err, color="#60a5fa", label="eager")
        gr = [i[1].get("graph_ms", {}).get("median") for i in items]
        ax.scatter([x for x, g in enumerate(gr) if g], [g for g in gr if g], color="#15803d", zorder=3, label="CUDA Graph")
        if "bf16" in names:
            ax.axhline(med[names.index("bf16")], color="k", ls="--", lw=1)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=70, ha="right", fontsize=8)
        ax.set_title(f"{s}px: step ms (bars p10-p90)")
        ax.legend(fontsize=8)
    save(fig, "step_time_by_config.png")


def nsys_stack():
    rows = read_csv(R / "analysis" / "nsys_summary.csv")
    sizes = sorted({int(r["size"]) for r in rows})
    if not sizes:
        return
    fig, axes = plt.subplots(1, len(sizes), figsize=(5 * len(sizes), 4.8), squeeze=False)
    for ax, s in zip(axes[0], sizes):
        rs = sorted([r for r in rows if int(r["size"]) == s], key=lambda r: order_key(r["config"]))
        bottom = [0] * len(rs)
        for c in CATEGORIES:
            vals = [float(r[f"{c}_ms"]) for r in rs]
            ax.bar(range(len(rs)), vals, bottom=bottom, color=COLORS[c], label=c)
            bottom = [b + v for b, v in zip(bottom, vals)]
        ax.set_xticks(range(len(rs)))
        ax.set_xticklabels([r["config"] for r in rs], rotation=70, ha="right", fontsize=8)
        ax.set_title(f"{s}px: GPU kernel ms per step (nsys)")
    axes[0][0].legend(fontsize=7)
    save(fig, "nsys_kernel_time_by_category.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for s in sizes:
        rs = sorted([r for r in rows if int(r["size"]) == s], key=lambda r: order_key(r["config"]))
        axes[0].plot([r["config"] for r in rs], [float(r["kernels_per_step"]) for r in rs], "o-", label=f"{s}px")
        axes[1].plot([r["config"] for r in rs], [float(r["busy_pct"]) if r.get("busy_pct") else 0 for r in rs], "o-", label=f"{s}px")
    axes[0].set_yscale("log")
    axes[0].set_title("kernel launches per step")
    axes[1].set_title("GPU busy % (100 - launch gaps)")
    for ax in axes:
        ax.tick_params(axis="x", rotation=70, labelsize=8)
        ax.legend()
    save(fig, "kernels_and_gpu_busy.png")


def ncu_plots():
    rows = read_csv(R / "analysis" / "ncu_layers.csv")
    if not rows:
        return
    layers = sorted({(r["layer"], int(r["size"])) for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    cfgs = sorted({r["config"] for r in rows}, key=order_key)
    w = 0.8 / len(cfgs)
    for j, c in enumerate(cfgs):
        vals = []
        for L in layers:
            m = [r for r in rows if r["config"] == c and (r["layer"], int(r["size"])) == L]
            vals.append(float(m[0]["quant_dram_mb"]) if m else 0)
        axes[0].bar([i + j * w for i in range(len(layers))], vals, w, label=c)
    ideal = [next((float(r["ideal_mb"]) for r in rows if (r["layer"], int(r["size"])) == L and r["ideal_mb"]), 0) for L in layers]
    axes[0].plot([i + 0.4 - w / 2 for i in range(len(layers))], ideal, "k_", markersize=18, mew=2, label="ideal")
    axes[0].set_yscale("log")
    axes[0].set_xticks([i + 0.4 - w / 2 for i in range(len(layers))])
    axes[0].set_xticklabels([f"{l}\n{s}px" for l, s in layers], fontsize=8)
    axes[0].set_title("quantize-path DRAM per call (MB, cold cache)")
    axes[0].legend(fontsize=7)
    for c in cfgs:
        rs = [r for r in rows if r["config"] == c]
        axes[1].scatter([float(r["quant_dram_pct_peak"]) for r in rs], [float(r["quant_sm_pct_peak"]) for r in rs], label=c, s=40)
    axes[1].set_xlabel("DRAM throughput % of peak")
    axes[1].set_ylabel("SM throughput % of peak")
    axes[1].set_title("quantize kernels: memory-bound if far right, compute-bound if high")
    axes[1].legend(fontsize=7)
    save(fig, "ncu_dram_and_throughput.png")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for j, c in enumerate(cfgs):
        vals = []
        for L in layers:
            m = [r for r in rows if r["config"] == c and (r["layer"], int(r["size"])) == L]
            vals.append(float(m[0]["quant_time_us"]) if m else 0)
        ax.bar([i + j * w for i in range(len(layers))], vals, w, label=c)
    ax.set_yscale("log")
    ax.set_xticks([i + 0.4 - w / 2 for i in range(len(layers))])
    ax.set_xticklabels([f"{l}\n{s}px" for l, s in layers], fontsize=8)
    ax.set_title("quantize-path GPU time per call (us)")
    ax.legend(fontsize=7)
    save(fig, "ncu_quantize_time.png")


def cuda_vs_triton():
    rows = [r for r in jsonl("cuda_vs_triton.jsonl") if "time_us" in r]
    if not rows:
        return
    kernels = sorted({r["kernel"] for r in rows})
    fig, axes = plt.subplots(1, len(kernels), figsize=(5 * len(kernels), 4.5), squeeze=False)
    for ax, k in zip(axes[0], kernels):
        rs = [r for r in rows if r["kernel"] == k]
        lab = [f"{r['M']}x{r['K']}" for r in rs]
        ax.bar([i - 0.2 for i in range(len(rs))], [r["time_us"]["median"] for r in rs], 0.4, label="CUDA", color="#15803d")
        base = [r.get("triton_us", r.get("triton_plus_amax_plus_swizzle_us"))["median"] for r in rs]
        ax.bar([i + 0.2 for i in range(len(rs))], base, 0.4, label="Triton (+amax+swizzle where listed)", color="#f59e0b")
        ax.set_xticks(range(len(rs)))
        ax.set_xticklabels(lab, rotation=60, ha="right", fontsize=7)
        ax.set_yscale("log")
        ax.set_title(f"{k}: us per call")
        ax.legend(fontsize=7)
    save(fig, "cuda_vs_triton_kernels.png")


def precision():
    for p in sorted((R / "precision").glob("*_*.json")):
        if p.name.startswith("images_"):
            continue
        d = json.loads(p.read_text())
        groups = sorted(d["groups"], key=lambda g: -g["w4a4"])[:12]
        fig, axes = plt.subplots(1, 3, figsize=(17, 5))
        x = range(len(groups))
        for j, (k, c) in enumerate([("w4a16", "#15803d"), ("w16a4", "#f59e0b"), ("w4a4", "#b91c1c")]):
            axes[0].bar([i + (j - 1) * 0.27 for i in x], [g[k] for g in groups], 0.27, label=k, color=c)
        axes[0].set_xticks(list(x))
        axes[0].set_xticklabels([g["group"].replace("transformer_blocks.*.", "") for g in groups], rotation=60, ha="right", fontsize=7)
        axes[0].set_title("local Linear-output error (rel L2)")
        axes[0].legend()
        for v, c in [("w4a16", "#15803d"), ("w16a4", "#f59e0b"), ("w4a4", "#b91c1c")]:
            ts = sorted({r["t"] for r in d["output"][v]}, reverse=True)
            means = [sum(r["rel_l2"] for r in d["output"][v] if r["t"] == t) / sum(1 for r in d["output"][v] if r["t"] == t) for t in ts]
            axes[1].plot(ts, means, "o-", color=c, label=f"{v} (mean of {len(d.get('prompts', [None]))} prompts)")
        axes[1].set_xlabel("timestep")
        axes[1].set_title("transformer output error (rel L2)")
        axes[1].legend()
        axes[2].scatter([g["act_max_over_rms"] for g in d["groups"]], [g["w16a4"] for g in d["groups"]], color="#f59e0b")
        axes[2].set_xlabel("activation max / rms (outlier ratio)")
        axes[2].set_ylabel("w16a4 local error")
        axes[2].set_title("activation outliers vs activation-quant error")
        save(fig, f"precision_{p.stem}.png")
    ip = sorted((R / "precision").glob("images_*.json"))
    for p in ip:
        d = json.loads(p.read_text())
        names = [k for k in d if "mae" in d[k]]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(names, [d[k]["mae"] for k in names], color="#60a5fa")
        ax.set_title(f"{p.stem}: image MAE vs bf16")
        ax.tick_params(axis="x", rotation=30)
        save(fig, f"precision_{p.stem}.png")


def speed_of_light():
    rows = [r for r in jsonl("step_target.jsonl") if r.get("model") == "pixart_sigma" and "layer" not in r]
    last = latest(rows, ("config", "size"))
    sizes = sorted({k[1] for k in last})
    need = ["bf16", "stock", "fused_triton", "free_bound"]
    if not sizes or not all(any(k[0] == n for k in last) for n in need):
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for j, n in enumerate(need):
        ax.bar([i + j * 0.2 for i in range(len(sizes))], [last[(n, s)]["gpu_ms"]["median"] if (n, s) in last else 0 for s in sizes], 0.2, label=n)
    ax.set_xticks([i + 0.3 for i in range(len(sizes))])
    ax.set_xticklabels([f"{s}px" for s in sizes])
    ax.set_title("free_bound = step time if the block quantize cost nothing")
    ax.set_yscale("log")
    ax.legend()
    save(fig, "speed_of_light_free_quantize.png")


if __name__ == "__main__":
    for fn in (step_time, nsys_stack, ncu_plots, cuda_vs_triton, precision, speed_of_light):
        try:
            fn()
        except Exception as e:
            print(f"{fn.__name__}: skipped ({type(e).__name__}: {e})")
