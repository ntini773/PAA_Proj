"""Audit every test from its output files, not from the markers.

    python check_status.py            # print a table: marker vs outputs found
    python check_status.py --mark     # write results/.done/<ID> for tests whose outputs are complete, clear a stale .fail, refresh PROGRESS.md
Use it after running scripts by hand (no marker) or after a partial run.
"""
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent
R = ROOT / "results"
L = ROOT / "logs"


def jsonl(name):
    p = R / name
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def exists(*paths):
    miss = [str(p.relative_to(ROOT)) for p in paths if not p.exists() or p.stat().st_size == 0]
    return (not miss), ("missing " + ", ".join(miss[:3]) + (" ..." if len(miss) > 3 else "")) if miss else ""


def kernels_ok(kernels):
    rows = [r for r in jsonl("cuda_vs_triton.jsonl") if r.get("kernel") in kernels]
    if not rows:
        return False, "no rows in cuda_vs_triton.jsonl"
    bad = [r for r in rows if "error" in r]
    return (not bad), (f"{len(bad)}/{len(rows)} rows with error: {bad[0]['error'][:80]}" if bad else f"{len(rows)} rows")


def step_ok():
    rows = jsonl("step_target.jsonl")
    cfgs = {(r.get("config"), r.get("size")) for r in rows}
    n_logs = len(list((L / "matrix").glob("*.log")))
    return len(rows) > 0, f"{len(rows)} rows, {len(cfgs)} config/size, {n_logs} matrix logs"


def paired(log_dir, out_dir, exts):
    names = sorted(p.stem for p in log_dir.glob("*.log"))
    if not names:
        return False, f"no logs in {log_dir.relative_to(ROOT)}"
    miss = [n for n in names if not all((out_dir / f"{n}{e}").exists() for e in exts)]
    return (not miss), (f"{len(names) - len(miss)}/{len(names)} runs have outputs; missing: {', '.join(miss[:4])}" if miss else f"{len(names)} runs")


def nsys_ok():
    return paired(L / "nsys", R / "nsys", [".nsys-rep", "_cuda_gpu_kern_sum.csv"])


def ncu_ok():
    return paired(L / "ncu", R / "ncu", [".ncu-rep", ".csv"])


def count(pattern, folder, minimum):
    n = len(list(folder.glob(pattern)))
    return n >= minimum, f"{n} files {pattern}"


def images_ok():
    ok, msg = exists(R / "precision" / "images_1024.json")
    if not ok:
        return ok, msg
    d = json.loads((R / "precision" / "images_1024.json").read_text())
    n = len(list((R / "precision" / "images").glob("p*_1024.png")))
    return bool(d.get("mean")) and n > 0, f"{len(d.get('prompts', []))} prompts, {n} pngs"


CHECKS = {
    "N01": lambda: exists(R / "hf_check.json"),
    "N02": lambda: exists(R / "hf_probe.json"),
    "K01": lambda: kernels_ok({"cuda_row"}),
    "K02": lambda: kernels_ok({"cuda_swizzled", "amax_swizzled"}),
    "K03": lambda: kernels_ok({"ln_mod_quant"}),
    "S01": step_ok,
    "P01": nsys_ok,
    "P02": ncu_ok,
    "P03": lambda: exists(R / "analysis" / "nsys_summary.csv"),
    "P04": lambda: exists(R / "analysis" / "ncu_layers.csv"),
    "R01": lambda: exists(R / "peaks.json"),
    "R02": lambda: exists(R / "analysis" / "roofline_points.csv"),
    "Q01": lambda: exists(R / "precision" / "pixart_sigma_512.json"),
    "Q02": lambda: exists(R / "precision" / "pixart_sigma_1024.json"),
    "Q03": images_ok,
    "T01": lambda: exists(ROOT / "prompts" / "embeds_pixart.pt"),
    "V01": lambda: count("*.png", R / "figures", 1),
}


def main():
    mark = "--mark" in sys.argv
    print(f"{'ID':4} {'marker':8} {'outputs':8} detail")
    for tid, fn in CHECKS.items():
        marker = "FAIL" if (R / ".fail" / tid).exists() else "done" if (R / ".done" / tid).exists() else "-"
        try:
            ok, detail = fn()
        except Exception as e:  # a broken output file must not stop the audit
            ok, detail = False, f"{type(e).__name__}: {e}"
        print(f"{tid:4} {marker:8} {'OK' if ok else 'MISSING':8} {detail}")
        if mark and ok:
            (R / ".done").mkdir(parents=True, exist_ok=True)
            (R / ".done" / tid).touch()
            (R / ".fail" / tid).unlink(missing_ok=True)
    print("\nM01-M03 (other models) are not audited here.")
    if mark:
        subprocess.run([sys.executable, str(ROOT / "progress_report.py")], check=False)


if __name__ == "__main__":
    main()
