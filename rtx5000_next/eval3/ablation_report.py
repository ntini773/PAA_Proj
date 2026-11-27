"""Reviewer points 1 and 2: the gain of each lever, from the ladder and leave-one-out runs (W300) and their quality (W301). CPU only.

    python eval3/ablation_report.py [--since "2026-11-28 10:00:00"] [--quality ladder_512]
Inputs : results/step_target.jsonl (CUDA Graph rows, config names from profiling/configs.sh EVAL3_LADDER / EVAL3_LOO / EVAL3_STATIC),
         results/eval3/quality/<tag>/rows.jsonl (per prompt and seed, from eval3/quality_grid.py)
Outputs: results/eval3/ablation/report.md, report.json, waterfall_<size>.png
Rules:
  - Only rows since CLEAN_SINCE (results/eval3/CLEAN_SINCE, or --since) count: older rows may come from runs that shared the GPU.
  - Each config needs >= 3 repeats; fewer is reported and fails the consistency check.
  - A difference is "measurable" only if the min-max ranges of the two configs do not overlap. Otherwise it is printed as
    "no measurable effect", never as a gain.
  - Ladder deltas are differences of consecutive medians, so they sum exactly to (stock - final); the check recomputes the sum.
  - Quality deltas are paired (same prompt and seed) against the final stack F, with a 95% bootstrap CI (eval3/stats.py).
Exit code 1 if a consistency check fails (missing rows, fewer than 3 repeats, sum mismatch).
"""
import argparse
import json
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from common import RESULTS  # noqa: E402
from eval3.stats import paired_diff  # noqa: E402

LADDER = [("stock_compile", "stock torchao NVFP4 (compiled)"), ("lad1_triton_noshare_compile", "+ our module stack, Triton kernel"),
          ("mod_noshare_compile", "+ our CUDA kernel"), ("mod_base_compile", "+ shared Q/K/V quantize"), ("mod_concat_compile", "+ concat GEMM"),
          ("lad5_sel_compile", "+ selective bf16 layers"), ("lad6_trim_compile", "+ text trim"), ("lad7_xcudnn_compile", "+ cuDNN cross-attention"),
          ("mod_final_compile", "+ Sage2 attention (final stack F)")]
LOO = [("loo_noshare_compile", "F without shared quantize (and concat)"), ("loo_noconcat_compile", "F without concat"),
       ("loo_nosel_compile", "F without selective"), ("loo_notrim_compile", "F without trim"), ("loo_noxcudnn_compile", "F without cuDNN cross-attn"),
       ("lad7_xcudnn_compile", "F without Sage2"), ("loo_triton_compile", "F with the Triton kernel instead of CUDA"),
       ("lad9_static_compile", "F + static activation scale (ablation)")]
BASE = [("bf16_compile", "bf16 compiled"), ("fp8_compile", "fp8 compiled")]
M = "module+be=cuda_swizzled"
F_VARIANT = f"{M}+concat+selective+trim+xcudnn+sage2+compile"


def load_steps(since):
    p = RESULTS / "step_target.jsonl"
    by = {}
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("time", "") < since or "layer" in r or "graph_ms" not in r or not r.get("config"):
            continue
        by.setdefault((r["config"], r["size"]), []).append(r["graph_ms"]["median"])
    return by


def summ(v):
    return dict(median=statistics.median(v), lo=min(v), hi=max(v), n=len(v)) if v else None


def measurable(a, b):
    return a["hi"] < b["lo"] or b["hi"] < a["lo"]


def quality(tag):
    p = RESULTS / "eval3" / "quality" / tag / "rows.jsonl"
    if not p.exists():
        return None
    by = {}
    for line in p.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            by.setdefault(r["variant"], {})[(r["seed"], r["prompt"])] = r["lpips"]
    if F_VARIANT not in by:
        return dict(error=f"final stack {F_VARIANT} missing in {p}")
    out = {}
    for v, d in by.items():
        ks = sorted(set(d) & set(by[F_VARIANT]))
        if v == F_VARIANT or not ks:
            continue
        m, lo, hi = paired_diff([d[k] for k in ks], [by[F_VARIANT][k] for k in ks])
        out[v] = dict(mean_lpips=statistics.mean(d.values()), diff_vs_F=m, lo=lo, hi=hi, n=len(ks), significant=lo > 0 or hi < 0)
    out["_F"] = dict(mean_lpips=statistics.mean(by[F_VARIANT].values()), n=len(by[F_VARIANT]))
    return out


def waterfall(size, steps, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(11, 4.8))
    prev = None
    for i, (cfg, label, s) in enumerate(steps):
        if prev is None:
            ax.bar(i, s["median"], color="#7f7f7f")
        else:
            d = s["median"] - prev
            ax.bar(i, abs(d), bottom=min(prev, s["median"]), color="#2ca02c" if d < 0 else "#d62728")
        ax.errorbar(i, s["median"], yerr=[[s["median"] - s["lo"]], [s["hi"] - s["median"]]], color="k", capsize=3)
        ax.text(i, s["median"], f"{s['median']:.1f}", ha="center", va="bottom", fontsize=7)
        prev = s["median"]
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels([l for _, l, _ in steps], rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("ms per transformer forward (CUDA Graph, CFG batch 2)")
    ax.set_title(f"PixArt-Sigma {size}px: what each lever buys (green = faster); bars show min-max over repeats")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default=None)
    ap.add_argument("--quality", default="ladder_512", help="quality tag; empty string skips the quality section")
    ap.add_argument("--sizes", default="512 1024")
    ap.add_argument("--min-reps", type=int, default=3, help="repeats each config needs (the smoke run uses 1)")
    a = ap.parse_args()
    if a.since is None:
        f = RESULTS / "eval3" / "CLEAN_SINCE"
        a.since = f.read_text().strip() if f.exists() else "2026-11-27"
    by = load_steps(a.since)
    out_dir = RESULTS / "eval3" / "ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    problems, rep = [], dict(since=a.since, sizes={})
    L = [f"# Ablation: ladder and leave-one-out (rows since {a.since})\n",
         "Step time = one transformer forward, CUDA Graph replay, CFG batch 2; median [min-max] over repeats. "
         "'no measurable effect' = the min-max ranges overlap.\n"]
    for size in [int(x) for x in a.sizes.split()]:
        steps = []
        for cfg, label in LADDER:
            s = summ(by.get((cfg, size), []))
            if s is None:
                problems.append(f"{size}: no rows for {cfg}")
                continue
            if s["n"] < a.min_reps:
                problems.append(f"{size}: {cfg} has {s['n']} repeats (< {a.min_reps})")
            steps.append((cfg, label, s))
        L += [f"## {size} px: ladder\n", "| step | ms | vs previous | measurable | speed-up vs stock |", "|---|---|---|---|---|"]
        ladder_rows, prev = [], None
        stock = steps[0][2]["median"] if steps and steps[0][0] == "stock_compile" else None
        for cfg, label, s in steps:
            d = None if prev is None else s["median"] - prev[2]["median"]
            meas = None if prev is None else measurable(s, prev[2])
            ladder_rows.append(dict(config=cfg, label=label, **s, delta_ms=d, measurable=meas))
            d_txt = "-" if d is None else f"{d:+.2f}"
            meas_txt = "-" if meas is None else ("yes" if meas else "no measurable effect")
            sp_txt = "-" if not stock else f"{stock / s['median']:.2f}x"
            L.append(f"| {label} | {s['median']:.2f} [{s['lo']:.2f}-{s['hi']:.2f}] ({s['n']}) | {d_txt} | {meas_txt} | {sp_txt} |")
            prev = (cfg, label, s)
        if len(steps) >= 2:
            total = steps[-1][2]["median"] - steps[0][2]["median"]
            sum_d = sum(r["delta_ms"] for r in ladder_rows if r["delta_ms"] is not None)
            if abs(total - sum_d) > 1e-6:
                problems.append(f"{size}: ladder deltas {sum_d:.3f} do not sum to the total {total:.3f}")
            L.append(f"\nTotal stock -> final: {total:+.2f} ms (sum of steps {sum_d:+.2f}).\n")
            waterfall(size, steps, out_dir / f"waterfall_{size}.png")
        F = summ(by.get(("mod_final_compile", size), []))
        L += [f"## {size} px: leave-one-out from the final stack F\n", "| F without ... | ms | cost of removing it | measurable |", "|---|---|---|---|"]
        loo_rows = []
        for cfg, label in LOO:
            s = summ(by.get((cfg, size), []))
            if s is None or F is None:
                if cfg != "lad9_static_compile":
                    problems.append(f"{size}: no rows for {cfg} or F")
                L.append(f"| {label} | not run | | |")
                continue
            if s["n"] < a.min_reps:
                problems.append(f"{size}: {cfg} has {s['n']} repeats (< {a.min_reps})")
            d, meas = s["median"] - F["median"], measurable(s, F)
            loo_rows.append(dict(config=cfg, label=label, **s, cost_ms=d, measurable=meas))
            L.append(f"| {label} | {s['median']:.2f} [{s['lo']:.2f}-{s['hi']:.2f}] ({s['n']}) | {d:+.2f} | {'yes' if meas else 'no measurable effect'} |")
        base_rows = {cfg: summ(by.get((cfg, size), [])) for cfg, _ in BASE}
        L += ["", "Baselines (same settings): " + ", ".join(f"{lbl} {b['median']:.2f} ms ({b['n']})" for (cfg, lbl), b in
                                                       zip(BASE, base_rows.values()) if b) + "\n"]
        rep["sizes"][size] = dict(ladder=ladder_rows, loo=loo_rows, baselines=base_rows)
    q = quality(a.quality) if a.quality else None
    rep["quality"] = q
    if not a.quality:
        L.append("## Quality\n\nskipped (--quality '').\n")
    elif q is None:
        L.append(f"## Quality\n\nresults/eval3/quality/{a.quality}/rows.jsonl not found (run W301).\n")
    elif "error" in q:
        L.append(f"## Quality\n\n{q['error']}\n")
        problems.append(q["error"])
    else:
        L += [f"## Quality ({a.quality}): paired LPIPS difference to the final stack F (same prompt and seed)\n",
              f"F mean LPIPS to bf16: {q['_F']['mean_lpips']:.3f} (n={q['_F']['n']}). Positive diff = further from bf16 than F.\n",
              "| variant | mean LPIPS | diff vs F [95% CI] | significant |", "|---|---|---|---|"]
        for v, r in sorted((k, v) for k, v in q.items() if k != "_F"):
            L.append(f"| {v} | {r['mean_lpips']:.3f} | {r['diff_vs_F']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}] | {'yes' if r['significant'] else 'no'} |")
    if problems:
        L += ["\n## CONSISTENCY PROBLEMS (do not quote the affected rows)\n", *[f"- {p}" for p in problems]]
    rep["problems"] = problems
    (out_dir / "report.md").write_text("\n".join(L) + "\n")
    (out_dir / "report.json").write_text(json.dumps(rep, indent=1, default=float))
    print("wrote", out_dir / "report.md", "| problems:", len(problems))
    for p in problems:
        print("  ", p)
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
