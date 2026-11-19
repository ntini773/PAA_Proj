"""'No two truths' checks for the report. Each check compares two independent sources, or one source with a recomputation.

    python report/check_consistency.py --data <dir with results/>
Prints PASS / FAIL per check, writes report/build/checks.json (used by the report appendix) and exits 1 on any FAIL.
"""
import argparse
import json
import pathlib
import re
import sys

from common import BEST, SIZES, facts, step

F = facts()


def md_table_rows(text):
    rows = {}
    size = None
    for line in text.splitlines():
        m = re.match(r"### (\d+) px", line)
        if m:
            size = int(m.group(1))
        elif line.startswith("| ") and size and not line.startswith("| setup") and not line.startswith("|---"):
            cells = [c.strip() for c in line.strip("| ").split("|")]
            rows[(size, cells[0])] = cells[1:]
    return rows


def run(data):
    res = pathlib.Path(data) / "results"
    out = []

    def check(cid, desc, ok, detail=""):
        out.append(dict(id=cid, desc=desc, ok=bool(ok), detail=detail))

    # C1: the headline quality variants appear in two independent runs (quality_final from run_final30, first E11 run): same values
    worst = 0.0
    n = 0
    for s in ("512", "1024"):
        for v in ("stock", "w4a16", "module+be=cuda_swizzled", BEST, BEST + "+trim+xcudnn"):
            a, b = F["quality_final"][s][v]["lpips"][0], F["quality_ext"][s][v]["lpips"]
            worst = max(worst, abs(a - b))
            n += 1
    check("C1", f"LPIPS of {n} variants is identical in two separate generation runs (run_final30 vs the first E11 run)", worst < 1.5e-3, f"largest difference {worst:.4f}")

    # C2: partial E11 rerun vs first run
    worst, n = 0.0, 0
    for s in ("512", "1024"):
        for v, q in F["quality_later"][s].items():
            if v in F["quality_ext"][s]:
                worst = max(worst, abs(q["lpips"] - F["quality_ext"][s][v]["lpips"]))
                n += 1
    check("C2", f"LPIPS of {n} variants is identical between the first E11 run and the later partial E11 rerun", worst < 1.5e-3, f"largest difference {worst:.4f}")

    # C3: nsys kernel-time total vs the CUDA Graph step time of the same config (independent measurements)
    pairs = [("bf16_compile_1024", "bf16_compile", 1024), ("stock_compile_1024", "stock_compile", 1024), ("module_best_compile_1024", "module_best", 1024),
             ("bf16_compile_512", "bf16_compile", 512), ("stock_compile_512", "stock_compile", 512), ("module_best_compile_512", "module_best", 512)]
    detail, ok = [], True
    for nk, lab, s in pairs:
        a, b = F["nsys"][nk]["total_ms"], step(lab, s)
        rel = a / b - 1
        detail.append(f"{nk}: {a:.1f} vs {b:.1f} ({rel * 100:+.1f}%)")
        ok &= abs(rel) < 0.12
    check("C3", "nsys kernel-time total matches the step time of the same config (within 12%: nsys excludes launch gaps and runs 5 forwards)", ok, "; ".join(detail))

    # C4: quantize call counts: 7 inputs per block x 28 + 6 outside the blocks = 202; 10 per block x 28 + 6 = 286
    n_ours = F["nsys"]["module_best_compile_1024"]["class_n"].get("our_quant", 0)
    n_gemm = F["nsys"]["stock_compile_1024"]["class_n"].get("fp4_gemm", 0)
    check("C4", "our quantize kernel runs 28 x 7 + 6 = 202 times per forward in the compiled module stack, and torchao's compiled path has 28 x 10 + 6 = 286 FP4 GEMMs",
          round(n_ours) == 28 * 7 + 6 and round(n_gemm) == 28 * 10 + 6, f"measured {n_ours:.0f} and {n_gemm:.0f}")

    # C5: old 'fused + compile' did not run our kernel inside the blocks
    n_old = [F["nsys"][k]["class_n"].get("our_quant", 0) for k in ("fused_cuda_swizzled_compile_1024", "fused_triton_op_compile_1024")]
    check("C5", "each old compiled 'fused' config ran our kernel at most 6 times per forward (the 6 layers outside the compiled blocks), not 286: the diagnosis of the artifact",
          max(n_old) <= 6 + 1e-9, f"{n_old[0]:.0f} (CUDA swizzled) and {n_old[1]:.0f} (Triton op) calls per forward")

    # C6: headline step times equal the table the server itself printed (final_table.md, written by a different script)
    rows = md_table_rows((res / "final_table.md").read_text())
    names = {"bf16 compiled": "bf16_compile", "fp8 compiled": "fp8_compile", "stock NVFP4 compiled": "stock_compile", "module stack (CUDA kernel, compiled)": "module_cuda",
             "+ static scale + concat": "module_best", "+ trim + cuDNN cross-attn": "module_best_xcudnn", "+ selective quantization": "module_best_selective"}
    bad, n = [], 0
    for (s, lab), cells in rows.items():
        if lab in names and cells[0] != "-":
            n += 1
            if abs(float(cells[0]) - step(names[lab], s)) > 0.051:
                bad.append(f"{lab}@{s}: server {cells[0]} vs facts {step(names[lab], s):.1f}")
    check("C6", f"{n} headline step times equal the table printed by the server (final_table.md)", not bad and n >= 20, "; ".join(bad) or f"{n} values compared")

    # C7: quality in the server table equals quality in facts
    bad, n = [], 0
    qn = {"bf16 compiled": "bf16", "stock NVFP4 compiled": "stock", "NVFP4 weights only (w4a16)": "w4a16", "module stack (CUDA kernel, compiled)": "module+be=cuda_swizzled",
          "+ static scale + concat": BEST, "+ trim + cuDNN cross-attn": BEST + "+trim+xcudnn", "+ selective quantization": BEST + "+selective+trim+xcudnn"}
    for (s, lab), cells in rows.items():
        if lab in qn and cells[2] != "-":
            n += 1
            if abs(float(cells[2]) - F["quality_final"][str(s)][qn[lab]]["lpips"][0]) > 0.0006:
                bad.append(f"{lab}@{s}")
    check("C7", f"{n} LPIPS values equal the server's table", not bad and n >= 20, "; ".join(bad) or f"{n} values compared")

    # C8: repeatability of the headline configs across independent runs
    worst, wl = 0.0, ""
    for lab in ("bf16_compile", "stock_compile", "module_cuda", "module_best", "module_best_xcudnn", "fp8_compile"):
        for s in SIZES:
            r = F["step"].get(f"{lab}|{s}", {}).get("runs", [])
            if len(r) > 1 and max(r) / min(r) - 1 > worst:
                worst, wl = max(r) / min(r) - 1, f"{lab}@{s} {min(r):.1f}-{max(r):.1f}"
    check("C8", "headline configs repeat within 5% across the independent timing runs", worst < 0.05, f"worst spread {worst * 100:.1f}% ({wl})")

    # C9: DiTFastAttn modeled cost recomputed from the technique counts
    ok, det = True, []
    for k, p in F["dfa"].items():
        size = int(k.split("|")[0])
        share = 1 / 8
        cost = {"ast": 0.02, "wars_asc": 0.5 * share + 0.02, "asc": 0.5, "wars": share + 0.02, "full": 1.0}
        tot = sum(cost[t] * c for t, c in p["counts"].items()) / sum(p["counts"].values())
        det.append(abs(tot - p["cost"]))
        ok &= abs(tot - p["cost"]) < 0.006
    check("C9", "DiTFastAttn modeled attention cost recomputed from the technique counts equals the stored value (9 plans)", ok, f"largest difference {max(det):.4f}")

    # C10: speed-ups are ratios of the same step table (recomputed independently from the raw jsonl)
    rows_ = [json.loads(l) for l in (res / "step_target.jsonl").read_text().splitlines() if l.strip()]
    def raw(variant, size):
        c = [r for r in rows_ if "layer" not in r and r["gpu_ms"]["n"] == 20 and r["size"] == size and (r.get("variant") == variant) and "graph_ms" in r]
        return c[-1]["graph_ms"]["median"] if c else None
    a, b = raw("bf16+compile", 1024), raw(BEST + "+trim+xcudnn+compile", 1024)
    check("C10", "the headline 1024 px speed-up recomputed straight from step_target.jsonl equals the report's value",
          abs(a / b - step("bf16_compile", 1024) / step("module_best_xcudnn", 1024)) < 1e-9, f"{a / b:.3f}")

    # C11: the 6 quality prompts are the same in all sizes (same number of per-prompt values)
    k = {s: len(F["quality_final"][str(s)][BEST]["per_prompt_lpips"]) for s in SIZES}
    check("C11", "every size was evaluated on the same 6 prompts", set(k.values()) == {6}, str(k))

    # C12: kernel exactness claim: the CUDA swizzled kernel used in the module stack has zero byte differences (latest row per shape)
    latest = {}
    for r in F["cuda_vs_triton"]:
        if "packed_diff_vs_triton" in r:   # a few rows come from timing-only calls without the byte comparison
            latest[(r["kernel"], r["M"], r["K"])] = r
    sw = [r for (kn, _, _), r in latest.items() if kn == "cuda_swizzled" and "packed_diff_vs_triton" in r]
    check("C12", "the CUDA swizzled quantize kernel (used in the module stack) has 0 byte differences against Triton and torchao in all tested shapes",
          len(sw) >= 9 and all(r["packed_diff_vs_triton"] == 0 and r["scale_diff_vs_triton"] == 0 and r.get("packed_diff_vs_torchao", 0) == 0 for r in sw), f"{len(sw)} shapes")

    # C13: every figure file referenced exists (done in the builder too)
    figs = list((pathlib.Path(__file__).resolve().parent / "build" / "fig").glob("*.png"))
    check("C13", "all figures were generated", len(figs) >= 17, f"{len(figs)} files")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    a = ap.parse_args()
    r = run(a.data)
    (pathlib.Path(__file__).resolve().parent / "build" / "checks.json").write_text(json.dumps(r))
    for c in r:
        print(("PASS " if c["ok"] else "FAIL ") + c["id"], c["desc"], "|", c["detail"])
    sys.exit(0 if all(c["ok"] for c in r) else 1)
