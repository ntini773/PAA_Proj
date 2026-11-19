"""All figures of the report. Every plotted number is read from report/build/facts.json through common.py.

    python report/make_figures.py --data <dir with results/ from results_eval2.tgz>

Figures (report/build/fig/*.png): block diagram, NVFP4 numeric example, kernel classes, speed bars, scaling, ablation waterfall,
traffic, GEMM efficiency, attention backends, DiTFastAttn plan maps, static-scale drift, quality scatter, image grids, error vs timestep.
"""
import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

from common import BEST, CLASS_COLORS, COLORS, FIG, SHORT, SIZES, STEP_ROWS, TOKENS, chain, ext, facts, quality, step, step_runs

F = facts()
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 100, "savefig.dpi": 200,
                     "axes.titlesize": 10, "axes.titleweight": "bold"})


def save(fig, name):
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)


def box(ax, x, y, w, h, text, fc="#ffffff", ec="#444444", fs=8, bold=False, lw=1.2, alpha=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.05", fc=fc, ec=ec, lw=lw, alpha=alpha))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal")


def arrow(ax, a, b, color="#444444", lw=1.3, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=9, color=color, lw=lw, ls=ls))


# ---------------------------------------------------------------------------------------------------------------- block diagram
def fig_block():
    fig, ax = plt.subplots(figsize=(9.2, 6.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7.2)
    ax.axis("off")
    G, Q = "#dff0d8", "#ffe2b8"
    ax.text(0.1, 6.95, "One PixArt-Sigma transformer block (repeated 28 times). Orange = NVFP4 linear layers (10 per block).", fontsize=9, fontweight="bold")
    xs = 0.2
    box(ax, xs, 5.7, 1.5, 0.7, "hidden state\n(tokens x 1152)", "#eeeeee")
    box(ax, 2.1, 5.7, 1.6, 0.7, "LayerNorm +\nmodulate (adaLN)", "#e8eef9")
    box(ax, 4.1, 6.05, 1.1, 0.4, "Q K V proj", Q, fs=7.5)
    box(ax, 4.1, 5.55, 1.1, 0.4, "(3 sites, one input)", "#ffffff", ec="#bbbbbb", fs=6.5)
    box(ax, 5.45, 5.7, 1.55, 0.7, "self-attention\n(tokens x tokens)", "#f8d7da", bold=True, fs=7.6)
    box(ax, 7.25, 5.7, 1.0, 0.7, "out proj", Q)
    box(ax, 8.5, 5.7, 1.2, 0.7, "gate * out\n+ residual", "#e8eef9", fs=7.5)
    for a, b in [((1.7, 6.05), (2.1, 6.05)), ((3.7, 6.05), (4.1, 6.2)), ((5.2, 6.05), (5.45, 6.05)), ((7.0, 6.05), (7.25, 6.05)), ((8.25, 6.05), (8.5, 6.05))]:
        arrow(ax, a, b)
    ax.text(5.0, 5.35, "1. self-attention", fontsize=8, color="#555555", ha="center")
    y2 = 3.9
    box(ax, 0.2, y2, 1.5, 0.7, "hidden state", "#eeeeee")
    box(ax, 2.1, y2 + 0.4, 1.1, 0.35, "Q proj", Q, fs=7.5)
    box(ax, 2.1, y2 - 0.1, 1.1, 0.35, "text 300 tokens\nK V proj", Q, fs=6.5)
    box(ax, 3.7, y2, 2.0, 0.7, "cross-attention\n(tokens x 300 text)", "#f8d7da", bold=True)
    box(ax, 6.2, y2, 1.1, 0.7, "out proj", Q)
    box(ax, 7.6, y2, 1.2, 0.7, "+ residual", "#e8eef9", fs=7.5)
    for a, b in [((1.7, y2 + 0.35), (2.1, y2 + 0.55)), ((3.2, y2 + 0.55), (3.7, y2 + 0.5)), ((3.2, y2 + 0.08), (3.7, y2 + 0.2)), ((5.7, y2 + 0.35), (6.2, y2 + 0.35)), ((7.3, y2 + 0.35), (7.6, y2 + 0.35))]:
        arrow(ax, a, b)
    ax.text(5.0, y2 - 0.35, "2. cross-attention to the prompt (3 NVFP4 sites: Q, K, V; plus out proj)", fontsize=8, color="#555555", ha="center")
    y3 = 2.0
    box(ax, 0.2, y3, 1.5, 0.7, "hidden state", "#eeeeee")
    box(ax, 2.1, y3, 1.6, 0.7, "LayerNorm +\nmodulate (adaLN)", "#e8eef9")
    box(ax, 4.1, y3, 1.2, 0.7, "fc1\n1152 -> 4608", Q)
    box(ax, 5.6, y3, 0.9, 0.7, "GELU", "#e8eef9")
    box(ax, 6.8, y3, 1.2, 0.7, "fc2\n4608 -> 1152", Q)
    box(ax, 8.3, y3, 1.3, 0.7, "gate * out\n+ residual", "#e8eef9", fs=7.5)
    for a, b in [((1.7, y3 + .35), (2.1, y3 + .35)), ((3.7, y3 + .35), (4.1, y3 + .35)), ((5.3, y3 + .35), (5.6, y3 + .35)), ((6.5, y3 + .35), (6.8, y3 + .35)), ((8.0, y3 + .35), (8.3, y3 + .35))]:
        arrow(ax, a, b)
    ax.text(5.0, y3 - 0.35, "3. feed-forward (2 NVFP4 sites)", fontsize=8, color="#555555", ha="center")
    box(ax, 0.2, 0.25, 9.4, 0.85, "", "#fbfbfb", ec="#999999")
    ax.text(0.35, 0.95, "What 'NVFP4 site' means", fontsize=8.5, fontweight="bold")
    ax.text(0.35, 0.55, "Before every orange layer the 16-bit activations are converted to 4-bit numbers + one 8-bit scale per 16 values (this conversion is the 'quantize' step).\n"
                        "The weights were converted once, offline. Counts: 7 distinct inputs per block (Q/K/V share one) -> 10 layers -> 286 layers in the model incl. 6 outside the blocks.",
            fontsize=7.6, va="center")
    save(fig, "fig_block.png")


# ---------------------------------------------------------------------------------------------------------------- NVFP4 numeric example
def nvfp4_example():
    """Worked example with a real quantize call. Returns the table rows used in the report."""
    import torch
    from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor, per_tensor_amax_to_scale
    x = torch.tensor([0.12, -0.31, 0.05, 0.88, -1.90, 0.40, 0.02, -0.07, 1.30, -0.60, 0.25, 0.90, -0.33, 0.08, 2.40, -1.10])
    pts = per_tensor_amax_to_scale(torch.tensor(24.0))   # pretend the whole tensor has amax 24
    q = NVFP4Tensor.to_nvfp4(x.view(1, 16).bfloat16(), per_tensor_scale=pts, is_swizzled_scales=False)
    deq = q.dequantize(torch.float32).view(-1)
    block_scale = float(q.scale.float().view(-1)[0])          # e4m3 block scale (before the per-tensor factor)
    eff = block_scale * float(pts)
    grid = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6.0])
    r = x / eff
    idx = (r.abs().unsqueeze(-1) - grid).abs().argmin(-1)
    code = grid[idx] * torch.sign(r)
    assert torch.allclose(code * eff, deq, atol=1e-6), "worked example does not match torchao's dequantize"
    return dict(x=x.tolist(), pts=float(pts), block_scale=block_scale, eff=eff, r=r.tolist(), code=code.tolist(), deq=deq.tolist(),
                err=(deq - x).tolist(), rel=float((deq - x).norm() / x.norm()))


def fig_nvfp4(ex):
    fig, ax = plt.subplots(figsize=(9, 3.1))
    xs = np.arange(16)
    ax.bar(xs - 0.2, ex["x"], 0.4, color="#8c8c8c", label="original (16-bit)")
    ax.bar(xs + 0.2, ex["deq"], 0.4, color="#2e8b57", label="after 4-bit + scale")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(xs)
    ax.set_xlabel("position in one block of 16 values")
    ax.set_ylabel("value")
    ax.set_title(f"One NVFP4 block: relative error {ex['rel'] * 100:.1f}% (the largest values survive, small ones are rounded to a few levels)")
    ax.legend(fontsize=8, loc="upper left")
    save(fig, "fig_nvfp4_example.png")


# ---------------------------------------------------------------------------------------------------------------- kernel classes
def fig_classes():
    cfgs = [("bf16_compile", "bf16\ncompiled"), ("stock_1024", "stock NVFP4\neager"), ("stock_compile", "stock NVFP4\ncompiled"),
            ("fused_cuda_swizzled_compile", "old 'fused'\ncompiled"), ("module_best_compile", "module stack\ncompiled")]
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.2), gridspec_kw={"width_ratios": [3, 1.6]})
    ax = axs[0]
    order = ["attention", "fp4_gemm", "bf16_gemm", "our_quant", "inductor", "elementwise", "copy", "reduce", "layernorm", "other"]
    for i, (c, lab) in enumerate(cfgs):
        key = c if c.endswith("1024") else c + "_1024"
        d = F["nsys"][key]["class_ms"]
        bottom = 0
        for k in order:
            v = d.get(k, 0)
            if v > 0.05:
                ax.bar(i, v, bottom=bottom, color=CLASS_COLORS[k], label=k, width=0.62)
            bottom += v
        ax.text(i, bottom + 4, f"{F['nsys'][key]['kernels']:.0f} kernels", ha="center", fontsize=7.5)
    ax.set_xticks(range(len(cfgs)))
    ax.set_xticklabels([l for _, l in cfgs], fontsize=8)
    ax.set_ylabel("GPU kernel time per forward at 1024 px (ms)")
    ax.set_title("Where the GPU time goes (nsys, one forward, 1024 px)")
    h, l = ax.get_legend_handles_labels()
    seen = dict(zip(l, h))
    ax.legend(seen.values(), seen.keys(), fontsize=7, ncol=2, loc="upper right")
    ax = axs[1]
    k = F["nsys"]["module_best_compile_1024"]
    vals = {c: k["class_ms"].get(c, 0) for c in order if k["class_ms"].get(c, 0) > 0.05}
    tot = sum(vals.values())
    ax.pie(vals.values(), labels=[f"{n}\n{v:.1f} ms" if v / tot > 0.04 else "" for n, v in vals.items()], colors=[CLASS_COLORS[n] for n in vals],
           startangle=90, counterclock=False, textprops={"fontsize": 7}, wedgeprops={"width": 0.45})
    ax.set_title("module stack: share of kernel time")
    save(fig, "fig_classes.png")


# ---------------------------------------------------------------------------------------------------------------- speed bars
def fig_speed():
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.9))
    for ax, size in zip(axs, SIZES):
        vals = [(SHORT[k], step(k, size), k) for _, k in STEP_ROWS if step(k, size) is not None]
        names = [v[0] for v in vals]
        y = np.arange(len(vals))[::-1]
        ax.barh(y, [v[1] for v in vals], color=[COLORS[v[2]] for v in vals])
        for yy, (n, t, k) in zip(y, vals):
            sp = step("bf16_compile", size) / t
            ax.text(t, yy, f" {t:.1f} ms ({sp:.2f}x)", va="center", fontsize=7.5)
        ax.set_yticks(y)
        ax.set_yticklabels(names if ax is axs[0] else [""] * len(names), fontsize=8)
        ax.set_xlim(0, max(v[1] for v in vals) * 1.45)
        ax.set_title(f"{size} px ({TOKENS[size]} tokens)")
        ax.set_xlabel("one forward, CUDA Graph ms")
    fig.suptitle("Step time (lower is better). Number in brackets: speed-up over bf16 compiled", fontsize=10, fontweight="bold", y=1.03)
    save(fig, "fig_speed.png")


def fig_scaling():
    fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.7))
    ax = axs[0]
    for k in ("bf16_compile", "stock_compile", "module_best", "module_best_selective"):
        xs = [TOKENS[s] for s in SIZES if step(k, s)]
        ax.plot(xs, [step(k, s) for s in SIZES if step(k, s)], "o-", color=COLORS[k], label=SHORT[k])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("tokens per image (log)")
    ax.set_ylabel("ms per forward (log)")
    ax.set_title("Step time against tokens (log-log)")
    ax.legend(fontsize=7.5)
    ax = axs[1]
    for k in ("fp8_compile", "stock_compile", "module_best", "module_best_xcudnn", "module_best_selective"):
        ax.plot([s for s in SIZES if step(k, s)], [step("bf16_compile", s) / step(k, s) for s in SIZES if step(k, s)], "o-", color=COLORS[k], label=SHORT[k])
    ax.axhline(1, color="k", lw=0.7, ls="--")
    ax.set_xticks(SIZES)
    ax.set_xlabel("image size (px)")
    ax.set_ylabel("speed-up over bf16 compiled")
    peak = max(SIZES, key=lambda s: step("bf16_compile", s) / step("module_best_xcudnn", s))
    ax.set_title(f"Speed-up of the best stack peaks at {peak} px")
    ax.legend(fontsize=7.5)
    save(fig, "fig_scaling.png")


def fig_waterfall():
    steps_ = [("stock NVFP4\ncompiled", "stock_compile"), ("module stack\n(CUDA kernel)", "module_cuda"), ("+ static scale\n+ concat", "module_best"),
              ("+ text trim\n+ cuDNN cross", "module_best_xcudnn")]
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.8))
    for ax, size in zip(axs, (512, 1024)):
        prev = None
        for i, (lab, k) in enumerate(steps_):
            t = step(k, size)
            if prev is None:
                ax.bar(i, t, color=COLORS[k])
            else:
                ax.bar(i, prev - t, bottom=t, color="#4fae6a")
                ax.text(i, prev + 0.01 * prev, f"-{prev - t:.1f}", ha="center", fontsize=8, color="#1b5e3a")
            ax.text(i, t / 2, f"{t:.1f}", ha="center", color="white", fontsize=8, fontweight="bold")
            prev = t
        ax.set_xticks(range(len(steps_)))
        ax.set_xticklabels([s[0] for s in steps_], fontsize=7.5)
        ax.set_title(f"{size} px: what each change removed (ms per forward)")
    save(fig, "fig_waterfall.png")


# ---------------------------------------------------------------------------------------------------------------- traffic
def fig_traffic():
    import traffic
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.7))
    ax = axs[0]
    vals = [("stock path\n(measured, ncu)", F["ncu"]["stock_qkv_1024"]["quant_dram_mb"]), ("our CUDA kernel\npath (measured)", F["ncu"]["fused_cuda_swizzled_qkv_1024"]["quant_dram_mb"]),
            ("ideal: read once,\nwrite once", F["ncu"]["stock_qkv_1024"]["ideal_mb"])]
    ax.bar(range(3), [v[1] for v in vals], color=["#e8903a", "#4fae6a", "#8c8c8c"])
    for i, v in enumerate(vals):
        ax.text(i, v[1] * 1.02, f"{v[1]:.0f} MB", ha="center", fontsize=8)
    ax.set_xticks(range(3))
    ax.set_xticklabels([v[0] for v in vals], fontsize=8)
    ax.set_ylabel("DRAM traffic around one layer (MB)")
    ax.set_title("One Q-projection at 1024 px: 8192 x 1152 input")
    ax = axs[1]
    t = traffic.numbers(1024)
    names = ["now\n(amax pass +\nquantize pass)", "static\nscale", "+ shared\nQ/K/V", "+ LN fused\n(possible)", "ideal floor\n(one pass)"]
    vals = [t["now"], t["now"] - t["static"], t["now"] - t["static"] - t["shared"], t["now"] - t["static"] - t["shared"] - t["ln"], t["floor"]]
    ax.bar(range(5), vals, color=["#e8903a", "#9bcf6b", "#6dbd6a", "#4fae6a", "#8c8c8c"])
    for i, v in enumerate(vals):
        ax.text(i, v * 1.02, f"{v:.1f}", ha="center", fontsize=8)
    ax.set_xticks(range(5))
    ax.set_xticklabels(names, fontsize=7)
    ax.set_ylabel("GB per forward (model, 1024 px)")
    ax.set_title("Quantize traffic per forward: where it can go")
    save(fig, "fig_traffic.png")


def fig_gemm():
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    ax = axs[0]
    for site, col in (("qkv_sep (1152->1152)", "#e8903a"), ("qkv_concat (1152->3456)", "#2e8b57"), ("fc2 (4608->1152)", "#5b8def")):
        rows = sorted([r for r in F["fp4_gemm"] if r["site"] == site and r["tokens"] == 4096], key=lambda r: r["images"])
        ax.plot([r["M"] for r in rows], [r["fp4_tflops"] for r in rows], "o-", color=col, label=site)
    ax.axhline(F["peaks"]["fp4_tflops"], color="k", ls="--", lw=0.8)
    ax.text(2200, F["peaks"]["fp4_tflops"] * 1.02, f"best measured: {F['peaks']['fp4_tflops']:.0f} TFLOP/s (8192^3 matrix)", fontsize=7.5)
    ax.set_xscale("log")
    ax.set_xlabel("rows M = 2 x images x tokens (log)")
    ax.set_ylabel("FP4 GEMM TFLOP/s")
    ax.set_title("Small K = 1152 GEMMs are far from the peak")
    ax.legend(fontsize=7.5, loc="lower right")
    ax = axs[1]
    sites = ["qkv_sep (1152->1152)", "qkv_concat (1152->3456)", "kv_concat (1152->2304)", "fc1 (1152->4608)", "fc2 (4608->1152)"]
    vals = [next(r for r in F["fp4_gemm"] if r["site"] == s and r["tokens"] == 4096 and r["images"] == 1) for s in sites]
    ax.bar(range(5), [v["fp4_tflops"] for v in vals], color=["#e8903a", "#2e8b57", "#5faf6a", "#7a9cc6", "#5b8def"])
    for i, v in enumerate(vals):
        ax.text(i, v["fp4_tflops"] + 8, f"{v['fp4_tflops']:.0f}\n({v['speedup']:.1f}x bf16)", ha="center", fontsize=7.5)
    ax.set_xticks(range(5))
    ax.set_xticklabels(["Q only", "Q+K+V\nconcat", "K+V\nconcat", "fc1", "fc2"], fontsize=8)
    ax.set_ylim(0, 800)
    ax.set_title("1024 px, one image (M = 8192): wider N helps")
    save(fig, "fig_gemm.png")


def fig_attention():
    shapes = ["pixart_self_512", "pixart_self_1024", "sd35_medium_joint", "flux_joint"]
    labels = ["PixArt self\n512 px (D=72)", "PixArt self\n1024 px (D=72)", "SD3.5 Medium\n(D=64)", "FLUX\n(D=128)"]
    be = ["default", "flash", "cudnn", "efficient", "sage2"]
    col = {"default": "#5b8def", "flash": "#2e86ab", "cudnn": "#edae49", "efficient": "#bab0ac", "sage2": "#d1495b"}
    fig, ax = plt.subplots(figsize=(9.5, 3.6))
    w = 0.16
    for j, b in enumerate(be):
        vals = []
        for s in shapes:
            r = next((r for r in F["attn_run1"] if r["shape"] == s and r["backend"] == b), None)
            vals.append(r["tflops"] if r and "tflops" in r else 0)
        ax.bar(np.arange(len(shapes)) + (j - 2) * w, vals, w, color=col[b], label=b)
        for i, v in enumerate(vals):
            if v:
                ax.text(i + (j - 2) * w, v + 3, f"{v:.0f}", ha="center", fontsize=6.5)
    ax.set_xticks(range(len(shapes)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("attention TFLOP/s (higher is better)")
    ax.set_title("Self-attention kernels: Sage 2.x wins for head dim 64 and 128, loses for PixArt's 72 (padded to 96)")
    ax.legend(fontsize=7.5, ncol=5)
    save(fig, "fig_attention.png")


def fig_dfa():
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.2))
    cmap = {"full": 0, "asc": 1, "ast": 2, "wars": 3, "wars_asc": 4}
    cols = ["#f2f2f2", "#9bcf6b", "#5b8def", "#edae49", "#d1495b"]
    from matplotlib.colors import ListedColormap
    for ax, d in zip(axs, ("0.05", "0.1", "0.2")):
        p = F["dfa"][f"1024|{d}"]
        arr = np.array([[cmap[t] for t in row] for row in p["plan"]])
        ax.imshow(arr, cmap=ListedColormap(cols), vmin=0, vmax=4, aspect="auto", interpolation="nearest")
        ax.set_title(f"delta {d}: modeled attention cost {p['cost'] * 100:.0f}%")
        ax.set_xlabel("layer (0-27)")
        ax.set_ylabel("denoise step" if d == "0.05" else "")
    axs[0].legend(handles=[mp.Patch(color=c, label=n) for n, c in zip(cmap, cols)], fontsize=7, loc="lower left", ncol=2)
    fig.suptitle("DiTFastAttn plans at 1024 px: which technique runs for each (step, layer)", fontsize=10, fontweight="bold", y=1.04)
    save(fig, "fig_dfa.png")


def fig_drift():
    fig, ax = plt.subplots(figsize=(8, 3.2))
    for s, c in zip(SIZES, ("#bab0ac", "#5b8def", "#d1495b")):
        d = np.sort(list(F["static"][str(s)]["drift"].values()))
        ax.plot(d, np.arange(1, len(d) + 1) / len(d), color=c, label=f"{s} px")
    ax.set_xscale("log")
    ax.set_xlabel("activation range drift across denoise steps: max / min of the per-step amax (log)")
    ax.set_ylabel("fraction of the 286 layers")
    ax.set_title("A static scale fits most layers (median drift 1.5-1.7x) but a few layers drift by more than 100x")
    ax.legend()
    save(fig, "fig_drift.png")


# ---------------------------------------------------------------------------------------------------------------- quality
def fig_quality():
    fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.8), sharey=True)
    pts = [("stock", "stock_compile", "stock NVFP4", "#e8903a"), ("module", "module_cuda", "module stack", "#7bc67b"), ("best", "module_best", "+ static + concat", "#4fae6a"),
           ("trim", "module_best_xcudnn", "+ trim + cuDNN", "#2e8b57"), ("selective", "module_best_selective", "+ selective", "#1b5e3a")]
    for ax, size in zip(axs, SIZES):
        for qv, sk, lab, c in pts:
            q = quality(size, {"stock": "stock", "module": "module+be=cuda_swizzled", "best": BEST, "trim": BEST + "+trim+xcudnn",
                               "selective": BEST + "+selective+trim+xcudnn"}[qv])
            ax.scatter(step(sk, size), q["lpips"][0], s=70, color=c, zorder=3)
            ax.errorbar(step(sk, size), q["lpips"][0], yerr=q["lpips"][1] / np.sqrt(6), color=c, capsize=2, lw=1)
            ax.annotate(lab, (step(sk, size), q["lpips"][0]), fontsize=7, xytext=(4, 4), textcoords="offset points")
        w = quality(size, "w4a16")
        ax.axhline(w["lpips"][0], color="#5b8def", ls="--", lw=1)
        ax.text(ax.get_xlim()[0], w["lpips"][0], " weights-only (w4a16)", color="#5b8def", fontsize=7, va="bottom")
        ax.set_xlabel("step time (ms, lower is faster)")
        ax.set_title(f"{size} px")
    axs[0].set_ylabel("LPIPS to bf16 image (lower is better)")
    fig.suptitle("Speed against image quality for the headline variants (mean of 6 prompts, bar = standard error)", fontsize=10, fontweight="bold", y=1.04)
    save(fig, "fig_quality.png")


def fig_err_t():
    fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.4), sharey=True)
    for ax, s in zip(axs, ("512", "1024")):
        d = F["real_w4a16"][s]["transformer_rel_l2"]
        ts = ["t999", "t750", "t500", "t250", "t50"]
        for k, lab, c in (("bf16_vs_w4a16", "weights only (w4a16)", "#5b8def"), ("bf16_vs_stock", "weights + activations (stock w4a4)", "#e8903a")):
            ax.plot([int(t[1:]) for t in ts], [d[k][t] for t in ts], "o-", color=c, label=lab)
        ax.invert_xaxis()
        ax.set_xlabel("denoise timestep (999 = noisy start, 50 = nearly done)")
        ax.set_title(f"{s} px")
    axs[0].set_ylabel("transformer output error vs bf16 (relative L2)")
    axs[0].legend(fontsize=8)
    save(fig, "fig_error_t.png")


def image_grid(size, rows, prompts_idx, out, title, root):
    """rows: [(label, path_dir, lpips list or None)] ; prompts_idx: list of prompt indices."""
    thumb = 256
    n_r, n_c = len(rows), len(prompts_idx)
    fig, axs = plt.subplots(n_r, n_c, figsize=(1.45 * n_c + 1.5, 1.36 * n_r), squeeze=False)
    fig.subplots_adjust(left=0.30, right=0.99, top=0.93, bottom=0.01, wspace=0.03, hspace=0.04)
    for r, (lab, d, lp) in enumerate(rows):
        for c, pi in enumerate(prompts_idx):
            ax = axs[r][c]
            im = Image.open(pathlib.Path(root) / d / f"p{pi}.png").convert("RGB").resize((thumb, thumb))
            ax.imshow(im)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if lp is not None:
                ax.text(4, thumb - 8, f"LPIPS {lp[pi]:.2f}", color="white", fontsize=6.5, bbox=dict(fc="black", alpha=0.55, pad=1.2, ec="none"))
            if c == 0:
                ax.set_ylabel(lab, fontsize=7, rotation=0, ha="right", va="center", labelpad=6)
    fig.suptitle(title, fontsize=8, fontweight="bold", y=0.985, x=0.5)
    save(fig, out)


def fig_images(root):
    prompts = pathlib.Path(__file__).resolve().parent.parent / "rtx5000_next" / "prompts" / "prompts.json"
    import json
    P = json.loads(prompts.read_text())
    qf = F["quality_final"]["512"]
    names = [("bf16 (reference)", "bf16", "bf16"), ("stock NVFP4", "stock", "stock"), ("weights only (w4a16)", "w4a16", "w4a16"),
             ("module stack", "module_be=cuda_swizzled", "module+be=cuda_swizzled"),
             ("+ static + concat\n+ trim + cuDNN", "module_be=cuda_swizzled_static_concat_trim_xcudnn", BEST + "+trim+xcudnn"),
             ("+ selective", "module_be=cuda_swizzled_static_concat_selective_trim_xcudnn", BEST + "+selective+trim+xcudnn")]
    rows = [(lab, f"quality_final/512/{d}", None if v == "bf16" else qf[v]["per_prompt_lpips"]) for lab, d, v in names]
    image_grid(512, rows, [0, 1, 5], "fig_images_variants.png", "512 px, same seed and prompt. Left: dancer/cherry blossoms. Middle: fisherman portrait. Right: ramen with a sign reading OPEN LATE.", root)
    qe = F["quality_later"]["512"]
    import json as _j
    sj = {r["name"]: [x["lpips"] for x in r["per_prompt"]] for r in _j.loads((pathlib.Path(root) / "quality/512/summary.json").read_text())}
    rows = [("bf16", "quality/512/bf16", None)]
    for lab, d, v in [("bf16 + cuDNN\ncross-attn (noise floor)", "bf16_xcudnn", "bf16+xcudnn"), ("DiTFastAttn\ndelta 0.1", "bf16_dfa0.1", "bf16+dfa0.1"), ("DiTFastAttn\ndelta 0.2", "bf16_dfa0.2", "bf16+dfa0.2"),
                      ("limited-interval CFG\n0.2-0.8", "bf16_licfg0.2-0.8", "bf16+licfg0.2-0.8"), ("K/V pooling 2x2", "bf16_kvds2", "bf16+kvds2"),
                      ("Sage 2.x attention", "bf16_sage2", "bf16+sage2")]:
        rows.append((lab, f"quality/512/{d}", sj[v]))
    image_grid(512, rows, [0, 1, 5], "fig_images_attention.png", "512 px, bf16 model with one attention or sampling change at a time", root)


# ---------------------------------------------------------------------------------------------------------------- story and e2e
def fig_story():
    fig, ax = plt.subplots(figsize=(10, 2.9))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")
    ratio = step("stock_eager", 1024) / step("bf16_eager", 1024)
    old_calls = max(F["nsys"][k]["class_n"].get("our_quant", 0) for k in ("fused_cuda_swizzled_compile_1024", "fused_triton_op_compile_1024"))
    best = F["nsys"]["module_best_compile_1024"]
    items = [("1  Round 1", f"stock NVFP4 {ratio:.1f}x slower\nthan bf16 eager (1024 px);\ncompile closes the gap", "#fdebd0"),
             ("2  Surprise", f"'fused' + compile = same\nas stock + compile.\nOur kernel ran {old_calls:.0f}x, not 286x", "#f8d7da"),
             ("3  Diagnosis", "monkeypatch not taken\ninside torch.compile", "#f8d7da"),
             ("4  Fix", f"FusedNVFP4Linear: kernel\nand GEMM are graph nodes;\ncounter proves {best['class_n']['our_quant']:.0f} calls", "#dff0d8"),
             ("5  Now", f"quantize + glue {chain('stock_compile_1024'):.1f} -> {chain('module_best_compile_1024'):.1f} ms;\nattention = {best['class_ms']['attention'] / best['total_ms'] * 100:.0f}% of the\nstep: next target", "#dff0d8")]
    for i, (h, t, c) in enumerate(items):
        x = 0.15 + i * 1.97
        box(ax, x, 0.5, 1.75, 2.0, "", c)
        ax.text(x + 0.875, 2.2, h, ha="center", fontsize=8.5, fontweight="bold")
        ax.text(x + 0.875, 1.35, t, ha="center", va="center", fontsize=7.2)
        if i < 4:
            arrow(ax, (x + 1.75, 1.5), (x + 1.97, 1.5))
    save(fig, "fig_story.png")


def fig_e2e():
    row = F["e2e_first"][1]
    t5, den, step_ms, vae, idle, total = float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6])
    fig, ax = plt.subplots(figsize=(8, 1.9))
    parts = [("T5 text encoder", t5, "#b07aa1"), ("14 denoise steps (transformer)", den, "#d1495b"), ("VAE decode", vae, "#59a14f"), ("idle", idle, "#cccccc")]
    left = 0
    for n, v, c in parts:
        ax.barh(0, v, left=left, color=c, label=f"{n}: {v:.0f} ms ({v / total * 100:.1f}%)")
        left += v
    ax.set_yticks([])
    ax.set_xlabel("ms for 4 images at 1024 px, batch of 4 (bf16 compiled)")
    ax.legend(fontsize=7.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 2.1))
    save(fig, "fig_e2e.png")
    return dict(t5=t5, den=den, step=step_ms, vae=vae, idle=idle, total=total)


def make_all(root):
    fig_story()
    fig_block()
    ex = nvfp4_example()
    fig_nvfp4(ex)
    fig_classes()
    fig_speed()
    fig_scaling()
    fig_waterfall()
    fig_traffic()
    fig_gemm()
    fig_attention()
    fig_dfa()
    fig_drift()
    fig_quality()
    fig_err_t()
    fig_images(root)
    e2e = fig_e2e()
    return ex, e2e


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    a = ap.parse_args()
    make_all(pathlib.Path(a.data) / "results")
    print("figures in", FIG)
