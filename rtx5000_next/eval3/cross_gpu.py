"""Reviewer point 5: do the results hold on another Blackwell GPU? CPU only; merges results copied from several machines.

    python eval3/cross_gpu.py --runs server=results_server laptop=results_laptop
Each argument is name=<results folder> (a copy of rtx5000_next/results from that machine; untar results_eval3.tgz into it).
Reads from each folder: eval3/peaks_sweep.json (roofs), step_target.jsonl (CUDA Graph step times; rows carry a "gpu" field from
eval3 on), and eval3/quality/*/summary.json (optional).
Output: results/eval3/cross_gpu/report.md and report.json:
  roofs side by side (copy GB/s, bf16 and FP4 GEMM TFLOP/s, attention TFLOP/s per head dim) and the ridge point (FP4 / bandwidth);
  step time and speed-up vs bf16 compiled for every config measured on all machines at the same size;
  a roofline prediction for a GPU that was not measured (--predict name:copy_gbs:fp4_tflops:bf16_tflops, e.g. a B200 spec),
  which scales each class of the measured profile by the roof ratio. It is labelled PREDICTION, never a measurement.
Speed-ups are compared only for configs present on every machine, at the same size, with >= 2 repeats each.
"""
import argparse
import json
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from common import RESULTS  # noqa: E402


def steps(folder, since):
    p = folder / "step_target.jsonl"
    by, gpus = {}, set()
    if not p.exists():
        return by, gpus
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("time", "") < since or "graph_ms" not in r or not r.get("config"):
            continue
        by.setdefault((r["config"], r["size"]), []).append(r["graph_ms"]["median"])
        if r.get("gpu"):
            gpus.add(r["gpu"])
    return by, gpus


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="name=results_folder ...")
    ap.add_argument("--since", default="2026-11-27")
    ap.add_argument("--predict", nargs="*", default=[], help="name:copy_gbs:fp4_tflops:bf16_tflops (spec numbers, labelled prediction)")
    a = ap.parse_args()
    runs = {}
    for spec in a.runs:
        name, _, folder = spec.partition("=")
        f = pathlib.Path(folder)
        peaks = f / "eval3" / "peaks_sweep.json"
        st, gpus = steps(f, a.since)
        runs[name] = dict(folder=str(f), peaks=json.loads(peaks.read_text())["best"] if peaks.exists() else None, steps=st, gpus=sorted(gpus))
    out_dir = RESULTS / "eval3" / "cross_gpu"
    out_dir.mkdir(parents=True, exist_ok=True)
    names = list(runs)
    L = ["# Cross-GPU comparison\n", "| machine | GPU(s) in rows | copy GB/s | bf16 TFLOP/s | FP4 TFLOP/s | ridge FLOP/B (FP4) | attn d=72 | attn d=64 | attn d=128 |",
         "|---|---|---|---|---|---|---|---|---|"]
    for n in names:
        p = runs[n]["peaks"]
        if not p:
            L.append(f"| {n} | {', '.join(runs[n]['gpus']) or '?'} | missing peaks_sweep.json | | | | | | |")
            continue
        hd = p.get("attn_tflops_by_head_dim", {})
        g = lambda d: f"{hd.get(str(d)) or hd.get(d) or 0:.0f}"  # noqa: E731
        L.append(f"| {n} | {', '.join(runs[n]['gpus']) or '?'} | {p['copy_gbs']:.0f} | {p['bf16_tflops']:.0f} | {p['fp4_tflops']:.0f} | "
                 f"{p['fp4_tflops'] * 1e3 / p['copy_gbs']:.0f} | {g(72)} | {g(64)} | {g(128)} |")
    rep = dict(runs={n: dict(gpus=r["gpus"], peaks=r["peaks"]) for n, r in runs.items()}, compare={})
    for size in (256, 512, 1024):
        common_cfgs = None
        for n in names:
            have = {c for (c, s), v in runs[n]["steps"].items() if s == size and len(v) >= 2}
            common_cfgs = have if common_cfgs is None else common_cfgs & have
        if not common_cfgs or "bf16_compile" not in common_cfgs:
            continue
        L += [f"\n## {size} px: step time (ms, median) and speed-up vs bf16 compiled\n",
              "| config | " + " | ".join(f"{n} ms | {n} x" for n in names) + " |", "|---|" + "---|---|" * len(names)]
        rep["compare"][size] = {}
        for c in sorted(common_cfgs):
            cells, row = [], {}
            for n in names:
                ms = statistics.median(runs[n]["steps"][(c, size)])
                bf = statistics.median(runs[n]["steps"][("bf16_compile", size)])
                row[n] = dict(ms=ms, speedup=bf / ms)
                cells.append(f"{ms:.2f} | {bf / ms:.2f}")
            rep["compare"][size][c] = row
            L.append(f"| {c} | " + " | ".join(cells) + " |")
    if a.predict:
        L += ["\n## PREDICTION for unmeasured GPUs (roofline scaling, not a measurement)\n",
              "Each machine's roofs are compared with the spec numbers given; memory-bound classes scale with bandwidth, GEMM with FP4/bf16 peak.\n",
              "| target | bandwidth ratio vs first machine | FP4 ratio | bf16 ratio |", "|---|---|---|---|"]
        ref = runs[names[0]]["peaks"]
        for spec in a.predict:
            nm, bw, f4, b16 = spec.split(":")
            if ref:
                L.append(f"| {nm} | {float(bw) / ref['copy_gbs']:.2f} | {float(f4) / ref['fp4_tflops']:.2f} | {float(b16) / ref['bf16_tflops']:.2f} |")
    (out_dir / "report.md").write_text("\n".join(L) + "\n")
    (out_dir / "report.json").write_text(json.dumps(rep, indent=1, default=float))
    print("wrote", out_dir / "report.md")


if __name__ == "__main__":
    main()
