"""Reviewer point 6: why the NVFP4 speed-up is smaller at 1024 px. CPU only, from the per-class kernel profiles (W202).

    python eval3/amdahl.py [--model pixart_sigma]
Input : results/eval3/roofline/<model>_<size>_<variant>.json from eval3/roofline_profile.py (sizes 256, 512, 1024)
Output: results/eval3/amdahl/report.md, report.json, amdahl_<model>.png
For each size, the bf16 compiled forward is split into classes (gemm, attention, our_quant, glue) by measured kernel time. For each
optimized variant:
  class speed-up      bf16 class time / variant class time (quantize has no bf16 counterpart, it is added cost)
  Amdahl prediction   1 / sum_c (share_c / speedup_c) with shares from bf16; equals the kernel-time ratio by construction
  measured speed-up   bf16 busy time / variant busy time, and bf16 wall / variant wall (eager launch, includes gaps)
  attention-limited bound  the best speed-up possible if every non-attention class took zero time: 1 / attention_share
The table shows how the attention share grows with token count (tokens = (size/16)^2: 256, 1024, 4096) and how that caps the
NVFP4 gain, and what Sage2 changes. Differences between "kernel" and "busy" speed-ups come from overlap and idle gaps; between
busy and wall from launch overhead (largest at 256 px, which is why only CUDA Graph step times are quoted as speed-ups).
"""
import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from common import RESULTS  # noqa: E402

CLASSES = ("gemm", "attention", "our_quant", "glue")
BF16 = "bf16+compile"


def load(model):
    d = RESULTS / "eval3" / "roofline"
    out = {}
    for p in sorted(d.glob(f"{model}_*.json")):
        r = json.loads(p.read_text())
        out.setdefault(r["size"], {})[r["variant"]] = r
    return out


def short(v):
    return v.replace("module+be=cuda_swizzled", "ours").replace("+compile", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="pixart_sigma")
    a = ap.parse_args()
    data = load(a.model)
    if not data:
        sys.exit(f"no roofline profiles for {a.model}: run W202 (roofline_profile.py) first")
    out_dir = RESULTS / "eval3" / "amdahl"
    out_dir.mkdir(parents=True, exist_ok=True)
    L = [f"# Why the speed-up changes with image size ({a.model})\n",
         "Kernel time per class from torch.profiler (leaf kernels only). Speed-ups are against bf16 compiled at the same size.\n"]
    rep, problems = {}, []
    for size in sorted(data):
        runs = data[size]
        if BF16 not in runs:
            problems.append(f"{size}: no {BF16} profile")
            continue
        b = runs[BF16]["measured"]
        bsum = sum(b["sum_ms"].get(c, 0.0) for c in CLASSES)
        share = {c: b["sum_ms"].get(c, 0.0) / bsum for c in CLASSES}
        tokens = (size // 16) ** 2
        L += [f"## {size} px ({tokens} image tokens)\n",
              f"bf16 compiled: kernel {bsum:.2f} ms, busy {b['busy_ms']:.2f} ms, wall {b['wall_ms']:.2f} ms. "
              f"Shares: gemm {share['gemm']:.0%}, attention {share['attention']:.0%}, glue {share['glue']:.0%}. "
              f"Attention-limited bound for any GEMM/glue speed-up: {1 / max(share['attention'], 1e-9):.2f}x.\n",
              "| variant | gemm x | attention x | quant ms (added) | glue x | kernel speed-up | busy speed-up | wall speed-up | attention share after |",
              "|---|---|---|---|---|---|---|---|---|"]
        rows = []
        for v, r in sorted(runs.items()):
            if v == BF16:
                continue
            m = r["measured"]
            vs = m["sum_ms"]
            vsum = sum(vs.get(c, 0.0) for c in CLASSES)

            def x(c):
                return b["sum_ms"].get(c, 0.0) / vs[c] if vs.get(c, 0.0) > 0 else None
            row = dict(variant=v, class_speedup={c: x(c) for c in ("gemm", "attention", "glue")}, quant_ms=vs.get("our_quant", 0.0),
                       kernel_speedup=bsum / vsum, busy_speedup=b["busy_ms"] / m["busy_ms"], wall_speedup=b["wall_ms"] / m["wall_ms"],
                       attention_share_after=vs.get("attention", 0.0) / vsum)
            rows.append(row)
            f = lambda z: "-" if z is None else f"{z:.2f}"  # noqa: E731
            L.append(f"| {short(v)} | {f(row['class_speedup']['gemm'])} | {f(row['class_speedup']['attention'])} | {row['quant_ms']:.2f} | "
                     f"{f(row['class_speedup']['glue'])} | {row['kernel_speedup']:.2f} | {row['busy_speedup']:.2f} | {row['wall_speedup']:.2f} | "
                     f"{row['attention_share_after']:.0%} |")
        L.append("")
        rep[size] = dict(tokens=tokens, bf16=dict(kernel_ms=bsum, share=share, busy_ms=b["busy_ms"], wall_ms=b["wall_ms"]), variants=rows)
    # one-line conclusion per size, generated from the numbers (no hand-written claims)
    L.append("## Generated summary\n")
    for size, r in sorted(rep.items()):
        best = max(r["variants"], key=lambda z: z["busy_speedup"], default=None)
        nos = [z for z in r["variants"] if "sage" not in z["variant"]]
        best_nos = max(nos, key=lambda z: z["busy_speedup"], default=None)
        if best and best_nos:
            L.append(f"- {size} px: bf16 attention share {r['bf16']['share']['attention']:.0%}; best without Sage "
                     f"{short(best_nos['variant'])} {best_nos['busy_speedup']:.2f}x busy; best overall {short(best['variant'])} {best['busy_speedup']:.2f}x busy.")
    if problems:
        L += ["\n## Problems\n", *[f"- {p}" for p in problems]]
    (out_dir / "report.md").write_text("\n".join(L) + "\n")
    (out_dir / "report.json").write_text(json.dumps(rep, indent=1, default=float))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        sizes = sorted(rep)
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
        ax[0].plot(sizes, [rep[s]["bf16"]["share"]["attention"] for s in sizes], "o-", label="attention")
        ax[0].plot(sizes, [rep[s]["bf16"]["share"]["gemm"] for s in sizes], "s-", label="gemm")
        ax[0].plot(sizes, [rep[s]["bf16"]["share"]["glue"] for s in sizes], "^-", label="glue")
        ax[0].set_xlabel("image size (px)")
        ax[0].set_ylabel("share of bf16 kernel time")
        ax[0].set_xscale("log", base=2)
        ax[0].legend()
        ax[0].set_title("bf16 compiled: where the time goes")
        names = sorted({z["variant"] for s in sizes for z in rep[s]["variants"]})
        for n in names:
            ys = [next((z["busy_speedup"] for z in rep[s]["variants"] if z["variant"] == n), None) for s in sizes]
            pts = [(s, y) for s, y in zip(sizes, ys) if y]
            if pts:
                ax[1].plot([p[0] for p in pts], [p[1] for p in pts], "o-", label=short(n)[:40])
        ax[1].set_xscale("log", base=2)
        ax[1].set_xlabel("image size (px)")
        ax[1].set_ylabel("speed-up vs bf16 compiled (GPU busy time)")
        ax[1].legend(fontsize=6)
        ax[1].set_title("speed-up vs size")
        fig.tight_layout()
        fig.savefig(out_dir / f"amdahl_{a.model}.png", dpi=140)
    except ImportError:
        pass
    print("wrote", out_dir / "report.md", "| problems:", len(problems))
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
