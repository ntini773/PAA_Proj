"""Merge results/eval3/roofline/*.json into one table and figure per (model, size).

    python eval3/roofline_report.py
Writes results/eval3/roofline/report.md and roofline_<model>_<size>.png.
Table columns: wall (eager launch), busy (GPU busy per forward, never above span), measured ms per class, SoL ms per modeled class,
efficiency = SoL / measured per class and for the modeled total vs busy time. n/a = no roof for that class (fp8 GEMM, Sage, glue).
Every row with a flag (ROOF_VIOLATION) is listed under the table; its efficiency is printed as "!" and must not be quoted.
Consistency checks (the script exits 1 if any fails, so a bad number cannot reach the report): efficiency <= 1.0 for every modeled
class, busy <= span, no zero in a class column that the variant must have (gemm always; attention always).
"""
import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = pathlib.Path(__file__).resolve().parent.parent / "results" / "eval3" / "roofline"


def eff(sol, meas):
    if sol is None or not meas:
        return None
    return sol / meas


def main():
    recs = [json.loads(p.read_text()) for p in sorted(D.glob("*_*.json")) if p.name != "report.json"]
    if not recs:
        sys.exit(f"no roofline json in {D}: run bash run_eval3.sh --only W202")
    groups = {}
    for r in recs:
        groups.setdefault((r["model"], r["size"]), []).append(r)
    L, bad = ["# Roofline: measured vs speed of light, per class\n"], []
    for (model, size), rs in sorted(groups.items()):
        L += [f"## {model} {size} px, one transformer forward (CFG batch)\n",
              "| variant | wall | busy | gemm ms (SoL, eff) | attention ms (SoL, eff) | quant ms (floor, eff) | glue ms | modeled SoL / busy |",
              "|---|---|---|---|---|---|---|---|"]
        for r in rs:
            m, s = r["measured"], r["sol"]
            sm = m["sum_ms"]

            def cell(cls, key):
                meas = sm.get(cls, 0.0)
                sol = (s.get(key) or {}).get("sol_ms")
                e = eff(sol, meas)
                if e is not None and e > 1.02:
                    bad.append(f"{model} {size} {r['variant']} {cls}: efficiency {e:.2f} > 1")
                    return f"{meas:.2f} ({sol:.2f}, !)"
                return f"{meas:.2f} ({sol:.2f}, {e:.0%})" if e is not None else f"{meas:.2f} (n/a)"
            if sm.get("gemm", 0) == 0 or sm.get("attention", 0) == 0:
                bad.append(f"{model} {size} {r['variant']}: zero gemm or attention time (kernel classification failed?)")
            if m["busy_ms"] > m["span_ms"] + 1e-6:
                bad.append(f"{model} {size} {r['variant']}: busy > span")
            tot = r["sol_modeled_total_ms"]
            L.append(f"| {r['variant']} | {m['wall_ms']:.1f} | {m['busy_ms']:.1f} | {cell('gemm', 'gemm')} | {cell('attention', 'attention')} | "
                     f"{cell('our_quant', 'quant_floor')} | {sm.get('glue', 0):.2f} | {tot:.1f} / {m['busy_ms']:.1f} = {tot / m['busy_ms']:.0%} (covers {', '.join(r['sol_covers']) or 'nothing'}) |")
        L.append("")
        for r in rs:
            for f in r["flags"]:
                L.append(f"- {r['variant']}: {f}")
                bad.append(f"{model} {size} {r['variant']}: {f}")
        L.append("\nTop kernels of the last variant, ms per forward:\n")
        for n, ms in rs[-1]["measured"]["top_kernels"][:6]:
            L.append(f"- {ms:.2f}  `{n}`")
        L.append("")
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))
        names = [r["variant"].replace("module+be=cuda_swizzled", "mod").replace("+compile", "+c")[:34] for r in rs]
        bottom = [0.0] * len(rs)
        for cls, col in (("gemm", "#1f77b4"), ("attention", "#d62728"), ("our_quant", "#2ca02c"), ("glue", "#7f7f7f")):
            vals = [r["measured"]["sum_ms"].get(cls, 0.0) for r in rs]
            ax[0].bar(range(len(rs)), vals, bottom=bottom, label=cls, color=col)
            bottom = [b + v for b, v in zip(bottom, vals)]
        ax[0].plot(range(len(rs)), [r["sol_modeled_total_ms"] for r in rs], "k_", ms=18, mew=2, label="SoL (modeled classes)")
        ax[0].set_xticks(range(len(rs)))
        ax[0].set_xticklabels(names, rotation=50, ha="right", fontsize=7)
        ax[0].set_ylabel("ms per forward (kernel time)")
        ax[0].legend(fontsize=8)
        ax[0].set_title(f"{model} {size}px: where the time goes")
        # roofline scatter: GEMM and attention classes
        peaks = json.loads((D.parent / "peaks_sweep.json").read_text())["best"]
        bw = peaks["copy_gbs"]
        xs = [10 ** (i / 10) for i in range(-10, 40)]
        for nm, pk in (("bf16 GEMM", peaks["bf16_tflops"]), ("FP4 GEMM", peaks["fp4_tflops"])):
            ax[1].plot(xs, [min(bw * x / 1e3, pk) for x in xs], "-", lw=1, color="k", alpha=0.5)
            ax[1].annotate(f"{nm} {pk:.0f}", (xs[-1], pk), ha="right", va="bottom", fontsize=7)
        for r in rs:
            for cls, key, mk in (("gemm", "gemm", "o"), ("attention", "attention", "s")):
                st = r["sol"].get(key)
                ms = r["measured"]["sum_ms"].get(cls, 0)
                if st and ms > 0:
                    ax[1].scatter(st["gflop"] / max(st["gbyte"], 1e-9), st["gflop"] / ms, marker=mk, s=45, edgecolor="k", zorder=3)
        ax[1].set_xscale("log")
        ax[1].set_yscale("log")
        ax[1].set_xlabel("FLOP / ideal byte")
        ax[1].set_ylabel("achieved TFLOP/s (class)")
        ax[1].set_title("o GEMM, s attention, per variant")
        fig.tight_layout()
        fig.savefig(D / f"roofline_{model}_{size}.png", dpi=140)
        plt.close(fig)
    if bad:
        L += ["## CONSISTENCY FAILURES (numbers above must not be quoted)\n", *[f"- {b}" for b in bad]]
    (D / "report.md").write_text("\n".join(L) + "\n")
    print("wrote", D / "report.md", "| failures:", len(bad))
    for b in bad:
        print("  ", b)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
