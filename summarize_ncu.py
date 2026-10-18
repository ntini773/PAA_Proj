"""Sum ncu raw CSVs: per path and token count -> kernels/iter, GPU us/iter, DRAM MB/iter."""
import sys, glob, os, csv, io, re
d = sys.argv[1]; ITERS = 3
print(f"{'tokens':>7} {'path':>8} {'kern/iter':>9} {'us/iter':>9} {'DRAM MB/iter':>13}")
for f in sorted(glob.glob(os.path.join(d, "ncu_*_*.csv")),
                key=lambda p: (int(re.findall(r'_(\d+)\.csv', p)[0]), p)):
    path, tok = re.findall(r'ncu_(\w+?)_(\d+)\.csv', f)[0]
    txt = open(f).read()
    i = txt.find('"ID"')
    if i < 0:
        print(f"{tok:>7} {path:>8}  (no data: see {f[:-4]}.err)"); continue
    rows = list(csv.DictReader(io.StringIO(txt[i:])))
    rows = [r for r in rows if r.get("gpu__time_duration.sum", "").replace(".", "").replace(",", "").isdigit()]
    t = dm = 0.0
    for r in rows:
        v = float(r["gpu__time_duration.sum"].replace(",", ""))
        t += v
        dm += float(r["dram__bytes.sum"].replace(",", ""))
    print(f"{tok:>7} {path:>8} {len(rows)/ITERS:9.1f} {t/ITERS/1e3:9.1f} {dm/ITERS/1e6:13.2f}")
