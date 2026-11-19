"""ncu CSVs (raw page) -> results/analysis/ncu_layers.csv

    python analysis/analyze_ncu.py
Per config x layer x size: quantize-path kernels vs GEMM, DRAM bytes against the ideal (read bf16 once, write codes and scales once).
"""
import csv
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
from kernel_classes import classify, QUANT_PATH

NCU = ROOT / "results" / "ncu"
OUT = ROOT / "results" / "analysis"
# the raw page uses the short time units (ns, us, ms, s); to nanoseconds and bytes
UNIT = {"ns": 1, "us": 1e3, "ms": 1e6, "s": 1e9, "nsecond": 1, "usecond": 1e3, "msecond": 1e6, "second": 1e9,
        "byte": 1, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9, "KB": 1e3, "MB": 1e6, "GB": 1e9}


def parse(path):
    lines = path.read_text(errors="replace").splitlines()
    i = next((k for k, l in enumerate(lines) if l.startswith('"ID"')), None)
    if i is None:
        return []
    rdr = list(csv.reader(lines[i:]))
    head, units, data = rdr[0], rdr[1], rdr[2:]
    out = []
    for r in data:
        if len(r) != len(head) or not r[0].strip().isdigit():
            continue
        d = {}
        for h, u, v in zip(head, units, r):
            if h == "Kernel Name":
                d[h] = v
                continue
            try:
                d[h] = float(v.replace(",", "")) * UNIT.get(u, 1)
            except ValueError:
                d[h] = v
        out.append(d)
    return out


def ratio(ks, num, den):
    n, d = sum(k.get(num, 0) for k in ks), sum(k.get(den, 0) for k in ks)
    return round(n / d, 2) if d else None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in sorted(NCU.glob("*.csv")):
        cfg, layer, size = p.stem.rsplit("_", 2)
        ks = parse(p)
        if not ks:
            continue
        missing = [m for m in ("gpu__time_duration.sum", "dram__bytes_op_read.sum", "dram__bytes_op_write.sum") if m not in ks[0]]
        if missing:  # ncu drops unknown metric names without an error
            sys.exit(f"{p.name} lacks {missing}: run profiling/ncu_reexport.sh")
        meta_p = ROOT / "results" / "meta" / f"{cfg}_{layer}_{size}.json"
        meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
        shape = meta.get("layer", {}).get("input_shape")
        ideal = None
        if shape:
            M = 1
            for s in shape[:-1]:
                M *= s
            K = shape[-1]
            ideal = M * K * 2 + M * K / 2 + M * K / 16
        quant = [k for k in ks if classify(k["Kernel Name"]) in QUANT_PATH]
        gemm = [k for k in ks if k not in quant]
        dram = lambda k: k.get("dram__bytes_op_read.sum", 0) + k.get("dram__bytes_op_write.sum", 0)
        t = lambda k: k.get("gpu__time_duration.sum", 0)
        qt = sum(map(t, quant))
        wavg = lambda key: sum(k.get(key, 0) * t(k) for k in quant) / qt if qt else None
        rows.append(dict(
            config=cfg, layer=layer, size=int(size), kernels_total=len(ks), quant_kernels=len(quant),
            quant_time_us=round(qt / 1e3, 1), gemm_time_us=round(sum(map(t, gemm)) / 1e3, 1),
            quant_dram_mb=round(sum(map(dram, quant)) / 1e6, 2), gemm_dram_mb=round(sum(map(dram, gemm)) / 1e6, 2),
            ideal_mb=round(ideal / 1e6, 2) if ideal else None,
            dram_vs_ideal=round(sum(map(dram, quant)) / ideal, 1) if ideal else None,
            quant_dram_pct_peak=round(wavg("dram__throughput.avg.pct_of_peak_sustained_elapsed") or 0, 1),
            quant_sm_pct_peak=round(wavg("sm__throughput.avg.pct_of_peak_sustained_elapsed") or 0, 1),
            quant_occupancy_pct=round(wavg("sm__warps_active.avg.pct_of_peak_sustained_active") or 0, 1),
            quant_l2_hit_pct=round(wavg("lts__t_sector_hit_rate.pct") or 0, 1),
            quant_l1_hit_pct=round(wavg("l1tex__t_sector_hit_rate.pct") or 0, 1),
            quant_l2_mb=round(sum(k.get("lts__t_bytes.sum", 0) for k in quant) / 1e6, 2),
            quant_ld_sectors_per_req=ratio(quant, "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum", "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum"),
            quant_st_sectors_per_req=ratio(quant, "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum", "l1tex__t_requests_pipe_lsu_mem_global_op_st.sum")))
    if not rows:
        sys.exit(f"no usable CSV in {NCU}")
    with open(OUT / "ncu_layers.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT / 'ncu_layers.csv'} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
