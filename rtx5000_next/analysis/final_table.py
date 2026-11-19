"""One table with the headline numbers per size: step time, speed-up, and real-prompt quality. No GPU needed.

    python analysis/final_table.py

Reads  results/step_target.jsonl (rows with a `variant` or `config`; graph time if present, else eager time)
       results/quality_final/<size>/summary.json (from quality/run_quality.py --extra-ref stock)
Writes results/final_table.md and prints it.
Rows are matched by variant string; the speed-up is against bf16+compile at the same size.
"""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
BEST = "module+be=cuda_swizzled+static+concat"
ROWS = [  # (label, step-time keys in order of preference, quality variant name)
    ("bf16 compiled", ["bf16+compile"], "bf16"),
    ("fp8 compiled", ["fp8+compile"], None),
    ("stock NVFP4 compiled", ["stock_compile"], "stock"),
    ("NVFP4 weights only (w4a16)", [], "w4a16"),
    ("module stack (CUDA kernel, compiled)", ["module+be=cuda_swizzled+compile"], "module+be=cuda_swizzled"),
    ("+ static scale + concat", [BEST + "+compile"], BEST),
    ("+ trim + cuDNN cross-attn", [BEST + "+trim+xcudnn+compile"], BEST + "+trim+xcudnn"),
    ("+ selective quantization", [BEST + "+selective+trim+xcudnn+compile"], BEST + "+selective+trim+xcudnn"),
]


def step_times():
    best = {}
    p = ROOT / "results" / "step_target.jsonl"
    if not p.exists():
        return best
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if "layer" in r:
            continue
        key = (r["size"], r.get("variant") or r.get("config"))
        ms = r["graph_ms"]["median"] if "graph_ms" in r else r["gpu_ms"]["median"]
        best[key] = ms  # later rows win
    return best


def quality(size):
    p = ROOT / "results" / "quality_final" / str(size) / "summary.json"
    if not p.exists():
        return {}
    return {r["name"]: r for r in json.loads(p.read_text())}


def main():
    st = step_times()
    sizes = sorted({s for s, _ in st} | {int(d.name) for d in (ROOT / "results" / "quality_final").glob("*") if d.name.isdigit()})
    out = []
    for size in sizes:
        q = quality(size)
        base = st.get((size, "bf16+compile"))
        out.append(f"\n### {size} px\n\n| setup | step ms | speed-up vs bf16+compile | LPIPS to bf16 | LPIPS to stock NVFP4 | CLIP score |\n|---|---|---|---|---|---|")
        for label, keys, qname in ROWS:
            t = next((st[(size, k)] for k in keys if (size, k) in st), None)
            qr = q.get(qname) if qname else None
            lp = f"{qr['summary']['lpips'][0]:.3f}" if qr else "-"
            ls = f"{qr['vs_extra']['lpips'][0]:.3f}" if qr and "vs_extra" in qr else "-"
            cs = f"{qr['summary']['clip_score'][0]:.1f}" if qr and "clip_score" in qr["summary"] else "-"
            if t is None and not qr:
                continue
            out.append(f"| {label} | {'-' if t is None else format(t, '.1f')} | "
                       f"{'-' if (t is None or not base) else format(base / t, '.2f') + 'x'} | {lp} | {ls} | {cs} |")
    text = "\n".join(out) + "\n"
    (ROOT / "results" / "final_table.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
