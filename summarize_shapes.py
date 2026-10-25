"""Per layer shape: unfused vs fused quantize (per-tensor max + block-quant), GPU time and DRAM. Reads results/shapes/."""
import os, re, csv, io, json, glob
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "shapes")
num = lambda s: float(s.replace(",", ""))
ok = lambda v: re.fullmatch(r"[\d,.]+", v or "") is not None


def read(label, stage):
    f = os.path.join(D, f"ncu_{label}_{stage}.csv")
    if not os.path.exists(f):
        return None
    t = open(f).read(); i = t.find('"ID"')
    if i < 0:
        return None
    rows = [r for r in csv.DictReader(io.StringIO(t[i:])) if ok(r.get("gpu__time_duration.sum"))]
    if not rows:
        return None
    dram = sum(num(r["dram__bytes.sum"]) for r in rows) / 1e6 if all(ok(r.get("dram__bytes.sum")) for r in rows) else None
    return dict(k=len(rows), us=sum(num(r["gpu__time_duration.sum"]) for r in rows) / 1e3, mb=dram)


print("| layer shape | rows x width | correctness (fused vs stock) | stock: kernels, GPU us, DRAM MB | fused: kernels, GPU us, DRAM MB | ideal DRAM MB | GPU time speedup | DRAM reduction |")
print("|---|---|---|---|---|---|---|---|")
for tf in sorted(glob.glob(os.path.join(D, "time_*.json"))):
    label = os.path.basename(tf)[5:-5]
    try:
        t = json.loads(open(tf).read().strip().splitlines()[-1])
    except Exception:
        continue
    a1, a2, f2 = read(label, "amax"), read(label, "quant_unfused"), read(label, "quant_fused")
    if not (a1 and a2 and f2):
        print(f"| {label} | {t['rows']} x {t['hidden']} | no ncu data | | | | | |"); continue
    st = dict(k=a1["k"] + a2["k"], us=a1["us"] + a2["us"], mb=(a1["mb"] or 0) + (a2["mb"] or 0))
    fu = dict(k=a1["k"] + f2["k"], us=a1["us"] + f2["us"], mb=(a1["mb"] or 0) + (f2["mb"] or 0))
    ideal = t["rows"] * t["hidden"] * (2 + 0.5 + 1 / 16) / 1e6
    eq = f"{t.get('diff_packed_bytes', '?')} + {t.get('diff_scale_bytes', '?')} bytes differ" if "diff_packed_bytes" in t else t.get("diff_error", "?")
    print(f"| {label} | {t['rows']} x {t['hidden']} | {eq} | {st['k']}, {st['us']:.1f}, {st['mb']:.1f} | {fu['k']}, {fu['us']:.1f}, {fu['mb']:.1f} | {ideal:.1f} | {st['us'] / fu['us']:.1f}x | {st['mb'] / max(fu['mb'], 1e-9):.1f}x |")
print("\nStock = per-tensor max + torchao block-quant (with scale swizzle). Fused = per-tensor max + our kernel (with scale swizzle). ncu flushes caches (cold DRAM).")
