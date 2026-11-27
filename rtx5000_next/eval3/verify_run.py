"""Checks that a stage really produced valid results, and the end-of-run verification + packaging (stage W399).

    python eval3/verify_run.py --mode slot1        # all checks for the stages that ran, reports, VERIFY.md, results_eval3.tgz
    python eval3/verify_run.py --mode smoke        # same, with the smoke knobs (results in $EVAL3_RESULTS)
    python eval3/verify_run.py --stage W300        # one stage's checker (used by eval3/orchestrate.py after every stage)
A stage counts as PASS only if its run_eval3.sh marker says done AND its checker below passes. A checker reads result files only;
it never trusts an exit code alone (bug 24: a runner that cannot fail). Knobs come from the same environment variables as
run_eval3.sh (MODELS, STEP_SIZES, REPS, ROOF_SIZES, STATIC_SIZES, SIZE_Q, QHI, TAG_SFX, ...); the defaults are the real run.
Outputs (whole-run mode): $RESULTS/eval3/VERIFY.md and VERIFY.json; results_eval3<TAG_SFX>.tgz next to the results folder
(json, md, csv, png, logs; image folders only as contact sheets, so the archive stays small).
"""
import argparse
import json
import os
import pathlib
import subprocess
import sys
import tarfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from common import RESULTS  # noqa: E402

E3 = RESULTS / "eval3"
PIX_VARIANTS = ["bf16+compile", "fp8+compile", "stock+compile", "module+be=cuda_swizzled+compile",
                "module+be=cuda_swizzled+concat+selective+trim+xcudnn+compile", "bf16+sage2+compile",
                "module+be=cuda_swizzled+concat+selective+trim+xcudnn+sage2+compile"]
OTHER_VARIANTS = ["bf16+compile", "fp8+compile", "stock+compile", "module+compile"]


def knob(name, default):
    return os.environ.get(name, default)


def K():
    return dict(models=knob("MODELS", "pixart_sigma:1024 sd35_medium:1024 flux_schnell:1024").split(),
                step_sizes=[int(x) for x in knob("STEP_SIZES", "512 1024").split()], reps=int(knob("REPS", "3")),
                head_reps=int(knob("HEAD_REPS", "2")), roof_sizes=[int(x) for x in knob("ROOF_SIZES", "256 512 1024").split()],
                static_sizes=[int(x) for x in knob("STATIC_SIZES", "512 1024").split()], size_q=int(knob("SIZE_Q", "512")),
                qhi=int(knob("QHI", "1024")), sfx=knob("TAG_SFX", ""), attn_sizes=[int(x) for x in knob("ATTN_SIZES", "1024").split()],
                xgpu_sizes=[int(x) for x in knob("XGPU_SIZES", "256 512").split()])


def config_list(name):
    out = subprocess.run(["bash", "-c", f"source profiling/configs.sh; echo ${name}"], cwd=ROOT, capture_output=True, text=True)
    return out.stdout.split()


def since():
    f = E3 / "CLEAN_SINCE"
    return f.read_text().strip() if f.exists() else "2000"


def step_counts():
    p = RESULTS / "step_target.jsonl"
    c = {}
    if not p.exists():
        return c
    s = since()
    for line in p.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("time", "") >= s and "graph_ms" in r and r.get("config"):
                c[(r["config"], r["size"])] = c.get((r["config"], r["size"]), 0) + 1
    return c


def need_rows(cfgs, sizes, reps):
    c = step_counts()
    short = [f"{x}@{s}:{c.get((x, s), 0)}" for x in cfgs for s in sizes if c.get((x, s), 0) < reps]
    return (not short, "" if not short else f"too few CUDA Graph rows since CLEAN_SINCE (need {reps}): {short[:8]}")


def fname(v):
    return v.replace("+", "_").replace("=", "")


def quality_ok(tag, min_variants):
    p = E3 / "quality" / tag / "summary.json"
    if not p.exists():
        return False, f"missing {p}"
    d = json.loads(p.read_text())
    if d.get("failed"):
        return False, f"failed variants: {[n for n, _ in d['failed']]}"
    n = len(d.get("summary", {}))
    return (n >= min_variants, "" if n >= min_variants else f"only {n} variants (expected >= {min_variants})")


def run_ok(cmd):
    r = subprocess.run([sys.executable] + cmd, cwd=ROOT, capture_output=True, text=True)
    return r.returncode == 0, " / ".join((r.stdout + r.stderr).strip().splitlines()[-3:]) if r.returncode else ""


def chk(stage, k):
    """(ok, message) for one stage."""
    if stage == "W100":
        p = E3 / "preflight.json"
        if not p.exists():
            return False, "missing preflight.json"
        fails = [r["name"] for r in json.loads(p.read_text()) if r["status"] == "FAIL"]
        return not fails, f"preflight FAIL lines: {fails}" if fails else ""
    if stage == "W104":
        p = E3 / "peaks_sweep.json"
        if not p.exists():
            return False, "missing peaks_sweep.json"
        b = json.loads(p.read_text())["best"]
        miss = [x for x in ("copy_gbs", "bf16_tflops", "fp4_tflops") if not b.get(x)]
        return not miss, f"roof missing: {miss}" if miss else ""
    if stage == "W201":
        bad = []
        for pr in k["models"]:
            m, s = pr.split(":")
            p = E3 / "assume" / f"{m}_{s}.json"
            if not p.exists():
                bad.append(f"{pr}: missing")
                continue
            d = json.loads(p.read_text())
            if d.get("fatal"):
                bad.append(f"{pr}: fatal {d['fatal'][:80]}")
            unk = [n for n, v in d.get("premises", {}).items() if isinstance(v, dict) and v.get("verdict") == "unknown"]
            if unk:
                bad.append(f"{pr}: unknown {unk}")
        return not bad, "; ".join(bad)
    if stage == "W202":
        bad = []
        for s in k["roof_sizes"]:
            for v in PIX_VARIANTS:
                p = E3 / "roofline" / f"pixart_sigma_{s}_{fname(v)}.json"
                if not p.exists():
                    bad.append(f"pixart {s} {v}: missing")
                elif json.loads(p.read_text()).get("flags"):
                    bad.append(f"pixart {s} {v}: {json.loads(p.read_text())['flags'][0][:60]}")
        for pr in k["models"]:
            m, s = pr.split(":")
            if m == "pixart_sigma":
                continue
            for v in OTHER_VARIANTS:
                p = E3 / "roofline" / f"{m}_{s}_{fname(v)}.json"
                if not p.exists():
                    bad.append(f"{m} {s} {v}: missing")
        return not bad, "; ".join(bad[:6])
    if stage == "W203":
        p = E3 / "roofline" / "report.md"
        if not p.exists():
            return False, "missing roofline report"
        return ("CONSISTENCY FAILURES" not in p.read_text(), "roofline report lists consistency failures")
    if stage == "W299":
        miss = [s for s in k["static_sizes"] if not (RESULTS / f"static_scales_{s}.json").exists()]
        return not miss, f"missing static scales for {miss}" if miss else ""
    if stage == "W300":
        ok, msg = run_ok(["eval3/ablation_report.py", "--quality", "", "--sizes", " ".join(map(str, k["step_sizes"])),
                          "--min-reps", str(k["reps"])])
        return ok, msg
    if stage == "W301":
        return quality_ok(f"ladder_{k['size_q']}{k['sfx']}", 12)
    if stage == "W302":
        p = E3 / "kernel_isolation.json"
        if not p.exists():
            return False, "missing kernel_isolation.json"
        d = json.loads(p.read_text())
        ours = [r for r in d["rows"] if r["backend"] == "ours_cuda"]
        if not ours or any("error" in r for r in ours):
            return False, "ours_cuda did not run on every shape"
        return (not d["not_bit_exact"], f"not bit-exact: {d['not_bit_exact']}" if d["not_bit_exact"] else "")
    if stage == "W303":
        if not (RESULTS / "attention_backends.md").exists():
            return False, "missing attention_backends.md"
        return need_rows(config_list("EVAL3_ATTN"), k["attn_sizes"], k["reps"])
    if stage == "W304":
        return quality_ok(f"key_{k['qhi']}{k['sfx']}", 6)
    if stage == "W305":
        return need_rows(config_list("EVAL3_HEAD"), k["step_sizes"], k["head_reps"])
    if stage == "W306":
        return need_rows(config_list("EVAL3_XGPU"), k["xgpu_sizes"], k["reps"])
    if stage == "W399":
        return (E3 / "VERIFY.json").exists(), ""
    return True, "no checker"


def package(sfx):
    out = RESULTS.parent / f"results_eval3{sfx}.tgz"
    keep_ext = {".json", ".jsonl", ".md", ".csv", ".png", ".txt", ".log"}

    def filt(ti):
        p = pathlib.Path(ti.name)
        if ti.isdir():
            return ti
        if "images" in p.parts and p.name != "contact_sheet.png":
            return None
        return ti if p.suffix in keep_ext or p.name == "CLEAN_SINCE" else None
    with tarfile.open(out, "w:gz") as t:
        for rel in ("eval3", "step_target.jsonl", "attention_backends.md", "attention_backends.jsonl"):
            if (RESULTS / rel).exists():
                t.add(RESULTS / rel, arcname=f"{RESULTS.name}/{rel}", filter=filt)
        for p in sorted((ROOT / "logs" / "eval3").glob(f"*{sfx}*.log")) if sfx else sorted((ROOT / "logs" / "eval3").glob("*.log")):
            t.add(p, arcname=f"logs/eval3/{p.name}")
        for p in sorted(RESULTS.glob("static_scales_*.json")):
            t.add(p, arcname=f"{RESULTS.name}/{p.name}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "slot1"], default="slot1")
    ap.add_argument("--stage", default=None, help="check one stage and exit 0/1")
    a = ap.parse_args()
    k = K()
    if a.stage:
        ok, msg = chk(a.stage, k)
        print(("PASS " if ok else "FAIL ") + a.stage + (f": {msg}" if msg else ""))
        sys.exit(0 if ok else 1)
    status_p = E3 / "STATUS.json"
    ran = json.loads(status_p.read_text())["stages"] if status_p.exists() else []
    rows = []
    for st in ran:
        if st["id"] == "W399" or st["status"] == "SKIPPED":
            continue
        ok, msg = chk(st["id"], k)
        rows.append(dict(id=st["id"], prio=st["prio"], status=st["status"], checker_ok=ok, msg=msg))
    reports = {}
    for name, cmd in (("ablation", ["eval3/ablation_report.py", "--sizes", " ".join(map(str, k["step_sizes"])), "--min-reps", str(k["reps"]),
                                    "--quality", f"ladder_{k['size_q']}{k['sfx']}"]),
                      ("amdahl", ["eval3/amdahl.py"]), ("roofline", ["eval3/roofline_report.py"])):
        ok, msg = run_ok(cmd)
        reports[name] = dict(ok=ok, msg=msg)
    if a.mode == "slot1":
        ok, msg = run_ok(["eval3/check_tests.py"])
        reports["check_tests"] = dict(ok=ok, msg=msg)
    tgz = package(k["sfx"])
    p1_ok = all(r["checker_ok"] and r["status"] == "PASS" for r in rows if r["prio"] <= 1)
    all_ok = p1_ok and all(r["checker_ok"] and r["status"] == "PASS" for r in rows)
    verdict = "GREEN" if all_ok and all(v["ok"] for v in reports.values()) else ("YELLOW" if p1_ok else "RED")
    (E3 / "VERIFY.json").write_text(json.dumps(dict(mode=a.mode, verdict=verdict, stages=rows, reports=reports, archive=str(tgz)), indent=1))
    L = [f"# Verification ({a.mode}): {verdict}\n", "GREEN = every stage passed its checker and every report is consistent. "
         "YELLOW = all P0/P1 stages passed, something lower failed or was skipped. RED = a P0/P1 stage failed.\n",
         "| stage | prio | run status | checker | message |", "|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['id']} | P{r['prio']} | {r['status']} | {'ok' if r['checker_ok'] else 'FAIL'} | {str(r['msg'])[:120]} |")
    L += ["", "| report | ok | last lines |", "|---|---|---|"]
    for n, v in reports.items():
        L.append(f"| {n} | {'ok' if v['ok'] else 'FAIL'} | {v['msg']} |")
    L.append(f"\nArchive: `{tgz}`")
    (E3 / "VERIFY.md").write_text("\n".join(L) + "\n")
    print(f"VERDICT {verdict}; wrote {E3 / 'VERIFY.md'} and {tgz}")
    sys.exit(0)


if __name__ == "__main__":
    main()
