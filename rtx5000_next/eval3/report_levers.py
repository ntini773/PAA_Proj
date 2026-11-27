"""One compact table of step time (CUDA Graph) and quality per lever, so a run can be read in one paste.

    python eval3/report_levers.py [--quality levers_512]
Step time: median of results/step_target.jsonl rows per (config, size), using graph_ms (falls back to gpu_ms), over all repeats; shows
the min-max range over repeats. Quality: LPIPS to bf16 eager (mean [CI]) from results/eval3/quality/<tag>/summary.json.
Rows with a graph_error are listed at the end (a graph time that failed to record is not silently replaced).
"""
import argparse
import json
import pathlib
import statistics
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from common import RESULTS as RES  # noqa: E402  (honours EVAL3_RESULTS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quality", nargs="*", default=["noise_floor_512", "levers_512", "round2_512", "round2_1024"])
    ap.add_argument("--since", default=None, help="only step rows after this time; default: results/eval3/CLEAN_SINCE, else 2026-11-27")
    a = ap.parse_args()
    if a.since is None:
        f = RES / "eval3" / "CLEAN_SINCE"
        a.since = f.read_text().strip() if f.exists() else "2026-11-27"
    rows = [json.loads(l) for l in (RES / "step_target.jsonl").read_text().splitlines() if l.strip()]
    by, errs = {}, []
    for r in rows:
        if r.get("time", "") < a.since or "layer" in r or not r.get("config"):
            continue
        if "graph_error" in r:
            errs.append((r["config"], r["size"], r["graph_error"][:80]))
        ms = (r.get("graph_ms") or {}).get("median")
        if ms is None:
            continue
        by.setdefault((r["config"], r["size"]), []).append(ms)
    print("STEP TIME, one transformer forward, batch 2 (CFG), CUDA Graph, ms: median [min-max over repeats] (n repeats)")
    cfgs = sorted({c for c, _ in by})
    print(f"{'config':40s} {'512':>26s} {'1024':>26s}")
    ref = {s: statistics.median(by[("bf16_compile", s)]) for s in (512, 1024) if ("bf16_compile", s) in by}
    for c in cfgs:
        cells = []
        for s in (512, 1024):
            v = by.get((c, s))
            if not v:
                cells.append("-")
                continue
            m = statistics.median(v)
            sp = f" {ref[s] / m:.2f}x" if s in ref else ""
            cells.append(f"{m:.1f} [{min(v):.1f}-{max(v):.1f}] ({len(v)}){sp}")
        print(f"{c:40s} {cells[0]:>26s} {cells[1]:>26s}")
    print("(speedup is vs bf16_compile at the same size)")
    for tag in a.quality:
        p = RES / "eval3" / "quality" / tag / "summary.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        print(f"\nQUALITY {tag}: LPIPS to bf16 eager, mean [95% CI], s/img")
        for n, s in d["summary"].items():
            m, lo, hi = s["lpips"]
            print(f"{n:62s} {m:.3f} [{lo:.3f}, {hi:.3f}]  {s['sec_per_image']:.2f}")
        for n, e in d.get("failed", []):
            print(f"FAILED {n}: {e}")
    if errs:
        print("\ngraph errors:", *sorted(set(errs)), sep="\n  ")
    md = RES / "eval3" / "quality" / (a.quality[-1]) / "summary.md"
    if md.exists():
        print("\n--- paired differences vs stock (from summary.md) ---")
        txt = md.read_text()
        print(txt[txt.find("Paired"):] if "Paired" in txt else "(none)")


if __name__ == "__main__":
    main()
