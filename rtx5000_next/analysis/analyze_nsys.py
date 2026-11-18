"""nsys CSVs -> results/analysis/nsys_summary.csv and nsys_top_kernels.md

    python analysis/analyze_nsys.py
Per step: kernel count, kernel time by category, GPU busy share (union of kernel intervals / wall span).
"""
import collections
import csv
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
from kernel_classes import CATEGORIES, QUANT_PATH, classify

NS = ROOT / "results" / "nsys"
OUT = ROOT / "results" / "analysis"


def read_csv(path, header_key):
    lines = path.read_text(errors="replace").splitlines()
    i = next((k for k, l in enumerate(lines) if header_key in l), None)
    if i is None:
        return []
    return list(csv.DictReader(lines[i:]))


def num(x):
    try:
        return float(str(x).replace(",", ""))
    except ValueError:
        return 0.0


def busy_and_span(trace_rows):
    iv = sorted((num(r["Start (ns)"]), num(r["Start (ns)"]) + num(r["Duration (ns)"])) for r in trace_rows if r.get("GrdX"))
    if not iv:
        return 0.0, 0.0, 0
    busy, cur_s, cur_e = 0.0, iv[0][0], iv[0][1]
    for s, e in iv[1:]:
        if s > cur_e:
            busy += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    busy += cur_e - cur_s
    return busy, iv[-1][1] - iv[0][0], len(iv)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, top = [], []
    for kern in sorted(NS.glob("*_cuda_gpu_kern_sum.csv")):
        name = kern.name[: -len("_cuda_gpu_kern_sum.csv")]
        cfg, size = name.rsplit("_", 1)
        meta_p = ROOT / "results" / "meta" / f"{name}.json"
        iters = json.loads(meta_p.read_text())["iters"] if meta_p.exists() else 5
        krows = read_csv(kern, "Time (%)")
        cat_ms = collections.defaultdict(float)
        cat_n = collections.defaultdict(int)
        for r in krows:
            c = classify(r["Name"])
            cat_ms[c] += num(r["Total Time (ns)"]) / 1e6 / iters
            cat_n[c] += int(num(r["Instances"])) / iters
        rec = dict(config=cfg, size=int(size), iters=iters,
                   kernels_per_step=round(sum(cat_n.values()), 1),
                   kernel_ms_per_step=round(sum(cat_ms.values()), 3))
        for c in CATEGORIES:
            rec[f"{c}_ms"] = round(cat_ms[c], 3)
            rec[f"{c}_n"] = round(cat_n[c], 1)
        rec["quant_path_ms"] = round(sum(cat_ms[c] for c in QUANT_PATH), 3)
        tr = NS / f"{name}_cuda_gpu_trace.csv"
        if tr.exists():
            busy, span, n = busy_and_span(read_csv(tr, "Start (ns)"))
            rec.update(busy_ms_per_step=round(busy / 1e6 / iters, 3), span_ms_per_step=round(span / 1e6 / iters, 3),
                       busy_pct=round(100 * busy / span, 1) if span else None)
        rows.append(rec)
        top.append((name, sorted(krows, key=lambda r: -num(r["Total Time (ns)"]))[:8], iters))

    if not rows:
        sys.exit(f"no *_cuda_gpu_kern_sum.csv in {NS}")
    cols = list(rows[0].keys())
    with open(OUT / "nsys_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["size"], r["config"])))
    with open(OUT / "nsys_top_kernels.md", "w") as f:
        for name, krows, iters in top:
            f.write(f"\n### {name}\n| ms/step | calls/step | class | kernel |\n|---|---|---|---|\n")
            for r in krows:
                f.write(f"| {num(r['Total Time (ns)']) / 1e6 / iters:.2f} | {num(r['Instances']) / iters:.0f} | {classify(r['Name'])} | `{r['Name'][:90]}` |\n")
    print(f"wrote {OUT / 'nsys_summary.csv'} ({len(rows)} runs) and nsys_top_kernels.md")


if __name__ == "__main__":
    main()
