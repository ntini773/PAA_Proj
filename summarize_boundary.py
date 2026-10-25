"""Stage tables: latency and DRAM traffic of each stage, unfused vs fused. Reads results/boundary/*.csv
and results/boundary_time.jsonl.   python summarize_boundary.py"""
import os, csv, io, json, glob, re
HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "results", "boundary")


def num(s):
    return float(s.replace(",", ""))


def read(size, stage):
    f = os.path.join(D, f"ncu_{size}_{stage}.csv")
    if not os.path.exists(f):
        return None
    t = open(f).read()
    i = t.find('"ID"')
    if i < 0:
        return None
    rows = [r for r in csv.DictReader(io.StringIO(t[i:]))
            if re.fullmatch(r"[\d,.]+", r.get("gpu__time_duration.sum", ""))]
    if not rows:
        return None
    return dict(kernels=len(rows),
                us=sum(num(r["gpu__time_duration.sum"]) for r in rows) / 1e3,       # ns -> us
                rd=sum(num(r["dram__bytes_read.sum"]) for r in rows) / 1e6,         # bytes -> MB
                wr=sum(num(r["dram__bytes_write.sum"]) for r in rows) / 1e6)


times = {}
p = os.path.join(HERE, "results", "boundary_time.jsonl")
if os.path.exists(p):
    for l in open(p):
        r = json.loads(l); times[r["size"]] = r

sizes = sorted({int(re.findall(r"ncu_(\d+)_", f)[0]) for f in glob.glob(os.path.join(D, "ncu_*.csv"))})
rows_def = {
    "UNFUSED (stock model)": [("LayerNorm", "ln"), ("AdaLN modulate", "mod"),
                              ("Quantize: per-tensor max", "amax"), ("Quantize: block-quant + scale swizzle", "quant_unfused")],
    "FUSED QUANTIZE (what the model run uses)": [("LayerNorm", "ln"), ("AdaLN modulate", "mod"),
                                                 ("Quantize: per-tensor max", "amax"), ("Quantize: fused kernel + scale swizzle", "quant_fused")],
    "FULL FUSION (LN + modulate + quantize in one kernel, idealized)": [("one kernel", "full_fused")],
}
for size in sizes:
    t = times.get(size, {})
    print(f"\n### {size}px: rows = {t.get('rows', '?')} (2 x {t.get('N', '?')} tokens) x 1152, one boundary call\n")
    if "diff_packed_bytes" in t:
        print(f"Fused quantize vs torchao quantize: {t['diff_packed_bytes']} differing packed bytes, "
              f"{t['diff_scale_bytes']} differing scale bytes.\n")
    for title, stages in rows_def.items():
        print(f"**{title}**\n")
        print("| stage | kernels | GPU time us | share of time | DRAM read MB | DRAM write MB | DRAM total MB |")
        print("|---|---|---|---|---|---|---|")
        data = [(n, read(size, s)) for n, s in stages]
        tot = sum(d["us"] for _, d in data if d) or 1.0
        T = dict(kernels=0, us=0.0, rd=0.0, wr=0.0)
        for n, d in data:
            if not d:
                print(f"| {n} | no data | | | | | |"); continue
            for k in T: T[k] += d[k]
            print(f"| {n} | {d['kernels']} | {d['us']:.1f} | {100 * d['us'] / tot:.0f}% | {d['rd']:.2f} | {d['wr']:.2f} | {d['rd'] + d['wr']:.2f} |")
        if len(stages) > 1:
            print(f"| **total** | **{T['kernels']}** | **{T['us']:.1f}** | | **{T['rd']:.2f}** | **{T['wr']:.2f}** | **{T['rd'] + T['wr']:.2f}** |")
        print()
    if t:
        c = t["chain_ms"]
        print(f"CUDA-event latency of the whole chain (includes launch gaps): unfused {c['chain_unfused']:.3f} ms, "
              f"fused quantize {c['chain_fused_quant']:.3f} ms, full fusion {c['chain_full_fused']:.3f} ms\n")
