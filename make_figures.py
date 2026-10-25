"""Draw the per-part contribution charts from the measured breakdown (2026-10-25 run, clean GPU).
    python3 make_figures.py      -> figures/breakdown_share.png, figures/breakdown_ms.png
Numbers: ms per denoise step, from results/breakdown.jsonl (bf16, NVFP4 stock, NVFP4 fused)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)

# part, colour (validated categorical slots 1-6, light mode)
PARTS = [("GEMM (matrix multiply)", "#2a78d6"), ("Attention", "#eb6834"), ("LayerNorm", "#1baf7a"),
         ("Quantize kernel", "#eda100"), ("Quantize overhead", "#e87ba4"), ("Everything else", "#008300")]
# ms per step in the order of PARTS: gemm, attention, layernorm, quantize kernel, quantize overhead, rest
DATA = {
    256:  {"bf16": (7.6, 2.5, 0.7, 0.0, 0.0, 4.4),    "NVFP4 stock": (9.5, 2.8, 1.4, 67.6, 27.6, 10.0),
           "NVFP4 fused": (8.6, 2.8, 1.4, 23.9, 25.6, 9.7)},
    512:  {"bf16": (15.7, 5.3, 0.6, 0.0, 0.0, 2.9),   "NVFP4 stock": (10.6, 6.4, 1.0, 72.6, 21.4, 7.5),
           "NVFP4 fused": (9.1, 6.4, 1.0, 29.1, 19.3, 7.0)},
    1024: {"bf16": (50.2, 42.0, 1.7, 0.0, 0.0, 10.4), "NVFP4 stock": (14.0, 38.4, 1.6, 281.5, 27.3, 10.2),
           "NVFP4 fused": (13.8, 38.1, 1.5, 163.7, 26.5, 10.1)},
}
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8985"
DARK_TEXT = {"#2a78d6": "white", "#008300": "white"}      # label colour on each fill


def draw(share, path, title, subtitle):
    fig, ax = plt.subplots(figsize=(11, 6.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ypos, yticks, ylabels, ycursor = [], [], [], 0.0
    for size, modes in DATA.items():
        for mode, vals in modes.items():
            total = sum(vals)
            left = 0.0
            for (name, colour), v in zip(PARTS, vals):
                w = 100.0 * v / total if share else v
                if w > 0:
                    ax.barh(ycursor, w, left=left, height=0.62, color=colour, edgecolor=SURFACE, linewidth=2)
                    pct = 100.0 * v / total
                    if (pct >= 6.5) if share else (v >= 9.0):
                        lab = f"{pct:.0f}%" if share else f"{v:.0f}"
                        ax.text(left + w / 2, ycursor, lab, ha="center", va="center", fontsize=9.5,
                                color=DARK_TEXT.get(colour, INK), fontweight="bold")
                left += w
            ax.text(left + (1.2 if share else max(sum(x) for m in DATA.values() for x in m.values()) * 0.01),
                    ycursor, f"{total:.0f} ms", va="center", fontsize=10, color=INK2)
            yticks.append(ycursor)
            ylabels.append(mode)
            ycursor += 0.78
        ycursor += 0.55
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=10, color=INK)
    ax.invert_yaxis()
    # image-size group labels
    y0 = 0
    for size, modes in DATA.items():
        n = len(modes)
        mid = yticks[y0 + n // 2]
        ax.text(-0.285, mid, f"{size}px", transform=ax.get_yaxis_transform(), ha="left", va="center",
                fontsize=12.5, fontweight="bold", color=INK)
        y0 += n
    if share:
        ax.set_xlim(0, 112)
        ax.set_xticks(range(0, 101, 25))
        ax.set_xticklabels([f"{t}%" for t in range(0, 101, 25)], color=INK2)
        ax.set_xlabel("share of one denoise step", color=INK2)
    else:
        ax.set_xlim(0, 420)
        ax.set_xlabel("milliseconds per denoise step (GPU time)", color=INK2)
        ax.tick_params(axis="x", colors=INK2)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.grid(True, color="#e6e5e1", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.suptitle(title, x=0.02, ha="left", fontsize=15, fontweight="bold", color=INK)
    fig.text(0.02, 0.915, subtitle, ha="left", va="top", fontsize=10.5, color=INK2, linespacing=1.5)
    ax.legend(handles=[Patch(facecolor=c, label=n) for n, c in PARTS], loc="upper center",
              bbox_to_anchor=(0.45, -0.1), ncol=6, frameon=False, fontsize=9.5, labelcolor=INK2,
              handlelength=1.2, columnspacing=1.2)
    fig.subplots_adjust(left=0.215, right=0.97, top=0.81, bottom=0.17)
    fig.savefig(path, dpi=170, facecolor=SURFACE)
    fig.savefig(path.replace(".png", ".svg"), facecolor=SURFACE)
    plt.close(fig)


draw(True, os.path.join(OUT, "breakdown_share.png"),
     "Where one denoise step goes: quantization is 79 to 83% of stock NVFP4",
     "PixArt-Sigma, RTX PRO 5000, batch 1 (+ guidance), clean GPU. Labels show shares of 6.5% or more; total step time at right.")
draw(False, os.path.join(OUT, "breakdown_ms.png"),
     "The same steps in milliseconds: fusion shrinks the quantize kernel, the overhead stays",
     "At 1024px, stock NVFP4 spends 282 of 373 ms in the quantize kernel (yellow).\nFusing the 1152-wide layers leaves 164 ms: the unfused 4608-wide MLP layer. Fusing that too\ngave a 99 ms step (measured end to end, not broken down here). Labels: ms, segments of 9 ms or more.")
print("done")

# table view (the light palette needs visible labels or a table)
lines = ["# Share of one denoise step (measured 2026-10-25, clean GPU)", "",
         "![share](breakdown_share.png)", "", "![milliseconds](breakdown_ms.png)", "",
         "| image | mode | step ms | " + " | ".join(n for n, _ in PARTS) + " |",
         "|---|---|---|" + "---|" * len(PARTS)]
for size, modes in DATA.items():
    for mode, vals in modes.items():
        tot = sum(vals)
        lines.append(f"| {size}px | {mode} | {tot:.1f} | " + " | ".join(f"{v:.1f} ms ({100 * v / tot:.0f}%)" for v in vals) + " |")
lines += ["", "Quantization = quantize kernel + quantize overhead. Source: `breakdown_step.py` (CUDA events); regenerate with `python3 make_figures.py`."]
open(os.path.join(OUT, "README.md"), "w").write("\n".join(lines) + "\n")
