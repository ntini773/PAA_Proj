"""Render docs/EVAL3_TESTS.md from eval3/tests.json. A box is ticked [x] ONLY if the result file exists AND its check passes.

    python eval3/check_tests.py
Nothing is ticked by hand and nothing is ticked from expectation. Checks named "pending" have no checker yet, so they stay [ ].
A test whose result exists but whose check fails shows [!] with the reason.
"""
import glob
import json
import pathlib

import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from common import RESULTS as RES  # noqa: E402  (honours EVAL3_RESULTS)

DOC = RES / "eval3" / "EVAL3_TESTS.md"   # generated on the server, untracked (a tracked copy conflicted on git pull)


def _load(p):
    return json.loads(pathlib.Path(p).read_text())


def preflight_no_fail(p):
    n = [r for r in _load(p) if r["status"] == "FAIL"]
    return (not n, f"{len(n)} FAIL lines" if n else "")


def peaks_complete(p):
    b = _load(p)["best"]
    miss = [k for k in ("copy_gbs", "bf16_tflops", "fp4_tflops") if not b.get(k)]
    return (not miss, f"missing {miss}" if miss else "")


def quality_has_controls(p):
    s = _load(p)["summary"]
    need = ["bf16+compile", "bf16@seedshift", "stock"]
    miss = [n for n in need if n not in s]
    return (not miss, f"missing controls {miss}" if miss else "")


def baseline_repeats(_):
    rows = [json.loads(l) for l in (RES / "step_target.jsonl").read_text().splitlines() if l.strip()]
    from collections import Counter
    c = Counter((r.get("config"), r.get("size")) for r in rows if r.get("config") in ("bf16_compile", "fp8_compile", "stock_compile", "stock_tk_compile") and "graph_ms" in r)
    short = [k for k in [(cfg, s) for cfg in ("bf16_compile", "fp8_compile", "stock_compile") for s in (512, 1024)] if c[k] < 3]
    return (not short, f"fewer than 3 repeats: {short}" if short else "")


def exists(_):
    return True, ""


MODELS = ("pixart_sigma", "sd35_medium", "flux_schnell")
R = RES / "eval3"


def _by_model(sub):
    out = {}
    for p in (R / sub).glob("*.json"):
        d = _load(p)
        out.setdefault(d.get("model"), []).append(d)
    return out


def lever_rows(_):
    rows = [json.loads(l) for l in (RES / "step_target.jsonl").read_text().splitlines() if l.strip()]
    need = ["mod_base_compile", "mod_concat_compile", "mod_trim_xcudnn_sage2_compile", "bf16_sage2_compile"]
    have = {(r.get("config"), r.get("size")) for r in rows if "graph_ms" in r}
    miss = [f"{c}@{s}" for c in need for s in (512, 1024) if (c, s) not in have]
    return (not miss, f"missing {miss}" if miss else "")


def quality_no_failures(p):
    d = _load(p)
    return (not d["failed"] and len(d["summary"]) > 1, f"failed variants: {[n for n, _ in d['failed']]}" if d["failed"] else "")


def round2_repeats(_):
    from collections import Counter
    rows = [json.loads(l) for l in (RES / "step_target.jsonl").read_text().splitlines() if l.strip()]
    f = R / "CLEAN_SINCE"  # written by hand at the start of the clean run: rows before it ran beside another GPU job
    since = f.read_text().strip() if f.exists() else "2026-11-27"
    c = Counter((r.get("config"), r.get("size")) for r in rows if "graph_ms" in r and r.get("time", "") >= since)
    need = ["mod_final_compile", "fp8_sage2_compile", "mod_trim_xcudnn_sage2_compile", "stock_compile"]
    short = [f"{k}@{s}" for k in need for s in (512, 1024) if c[(k, s)] < 3]
    return (not short, f"fewer than 3 repeats: {short}" if short else "")


def assume_all_models(_):
    by = _by_model("assume")
    miss = [m for m in MODELS if m not in by]
    unk = [f"{d['model']}:{k}" for ds in by.values() for d in ds for k, v in d.get("premises", {}).items() if isinstance(v, dict) and v.get("verdict") == "unknown"]
    fatal = [d["model"] for ds in by.values() for d in ds if d.get("fatal")]
    ok = not miss and not unk and not fatal
    return ok, f"missing {miss}; unknown {unk[:4]}; fatal {fatal}" if not ok else ""


def convert_all_pass(_):
    by = _by_model("convert")
    miss = [m for m in MODELS if m not in by]
    bad = [f"{d['model']}@{d['size']}" for ds in by.values() for d in ds if not d.get("pass")]
    return (not miss and not bad, f"missing {miss}; not passed {bad}" if (miss or bad) else "")


def cover_reported(_):
    by = _by_model("convert")
    miss = [m for m in MODELS if not any(d["steps"].get("module", {}).get("param_fraction") is not None for d in by.get(m, []))]
    return (not miss, f"no coverage for {miss}" if miss else "")


def sage_profiled(_):
    d = R / "roofline"
    need = [f"pixart_sigma_{s}_{v}.json" for s in (512, 1024) for v in ("bf16_compile", "bf16_sage2_compile")]
    miss = [n for n in need if not (d / n).exists()]
    bad = [n for n in need if (d / n).exists() and _load(d / n)["measured"]["sum_ms"].get("attention", 0) <= 0]
    return (not miss and not bad, f"missing {miss}; zero attention time {bad}" if (miss or bad) else "")


def roofline_report_ok(p):
    t = pathlib.Path(p).read_text()
    return ("CONSISTENCY FAILURES" not in t and "pixart_sigma 1024" in t and "pixart_sigma 512" in t,
            "consistency failures listed in report.md" if "CONSISTENCY FAILURES" in t else "")


def roofline_new_models(p):
    t = pathlib.Path(p).read_text()
    miss = [m for m in ("sd35_medium", "flux_schnell") if m not in t]
    return (not miss and "CONSISTENCY FAILURES" not in t, f"missing {miss}" if miss else "")


def attention_share_new_models(_):
    by = _by_model("assume")
    miss = [m for m in ("sd35_medium", "flux_schnell") if not any("attention_share_of_bf16_forward" in d.get("premises", {}).get("P3_attention_dominates", {}) for d in by.get(m, []))]
    return (not miss, f"no attention share for {miss}" if miss else "")


CHECKS = {f.__name__: f for f in (preflight_no_fail, peaks_complete, quality_has_controls, baseline_repeats, exists, lever_rows, quality_no_failures,
                                  round2_repeats, assume_all_models, convert_all_pass, cover_reported, sage_profiled, roofline_report_ok,
                                  roofline_new_models, attention_share_new_models)}


def status(t):
    if t["check"] == "pending":
        return "[ ]", "no checker yet"
    if t["check"].startswith("verify:"):  # the same per-stage checker the unattended run uses (eval3/verify_run.py)
        import eval3.verify_run as vr
        try:
            ok, why = vr.chk(t["check"].split(":", 1)[1], vr.K())
        except Exception as e:  # noqa: BLE001
            return "[!]", f"check error {type(e).__name__}: {str(e)[:80]}"
        not_run = any(w in why for w in ("missing", "too few", "no rows"))
        return ("[x]", "") if ok else ("[ ]" if not_run else "[!]", why[:120])
    hits = glob.glob(str(ROOT.parent / "rtx5000_next" / t["result"])) or glob.glob(str(ROOT / t["result"]))
    if not hits:
        return "[ ]", "no result yet"
    try:
        ok, why = CHECKS[t["check"]](hits[0])
    except Exception as e:  # noqa: BLE001
        return "[!]", f"check error {type(e).__name__}: {str(e)[:80]}"
    return ("[x]", "") if ok else ("[!]", why)


def main():
    tests = _load(ROOT / "eval3" / "tests.json")
    L = ["# Eval 3 test registry", "",
         "Generated by `python eval3/check_tests.py`. `[x]` = result file exists and its check passed. `[ ]` = not run or no checker yet. `[!]` = ran, check failed.",
         "Nothing here is ticked by hand. Tests marked wave 2 / wave 3 are not implemented yet.", "",
         "| Done | ID | Group | Model | What it proves | Command | Note |", "|---|---|---|---|---|---|---|"]
    n = 0
    for t in tests:
        box, note = status(t)
        n += box == "[x]"
        L.append(f"| {box} | {t['id']} | {t['group']} | {t['model']} | {t['what']} | `{t['cmd']}` | {note} |")
    L.append(f"\n{n} of {len(tests)} ticked.")
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text("\n".join(L) + "\n")
    print(f"{n} of {len(tests)} ticked -> {DOC}")


if __name__ == "__main__":
    main()
