"""T-assume: do the PixArt premises hold on this model? Decides which levers are worth testing on it.

    python eval3/assume_check.py --model sd35_medium --size 1024
Premises (each gets value, verdict true/false/unknown, and what to do):
  P1 quantize and elementwise ops are memory bound   -> per-GEMM arithmetic intensity vs the FP4 ridge point (peak / bandwidth), and the
                                                        measured share of non-GEMM, non-attention kernel time in a bf16 forward
  P2 FP4 GEMM beats bf16 at the real shapes          -> time bf16 and FP4 GEMM at each distinct (M, K, N), weighted by call count
  P3 attention dominates at high token count         -> measured attention share of a bf16 forward (kernel time, profiler)
  P4 a text mask / padded text exists                -> encoder_attention_mask in the model inputs (trim / xcudnn apply only then)
  P5 CFG batch of two                                -> catalog cfg_batch (limited-interval CFG applies only then)
  P6 NVFP4 error is large vs fp8 (so LPIPS needs a noise floor) -> relative output error stock NVFP4 vs fp8, both against bf16
Output: results/eval3/assume/<model>_<size>.json. Needs the idle GPU, about 3-6 min per model. A failing premise does not fail the
run: it is a result. A premise that could not be measured is "unknown" with the error text.
"""
import argparse
import collections
import gc
import json
import pathlib
import sys
import traceback

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "analysis"))
import torch
import torch.nn.functional as F

from common import RESULTS, cuda_ms, stats, write_json
from eval3.roofline_profile import profile_kernels, record
from model_adapters import CATALOG, build

BF = torch.bfloat16


def free():
    gc.collect()
    torch.cuda.empty_cache()


def gemm_times(shapes):
    """Median ms of bf16 matmul and NVFP4 GEMM (activation pre-quantized, as peaks.py) per (m,k,n)."""
    from bench.peaks_sweep import fp4_row
    out = {}
    for (m, k, n) in shapes:
        a, b = torch.randn(m, k, device="cuda", dtype=BF), torch.randn(k, n, device="cuda", dtype=BF)
        t_bf = stats(cuda_ms(lambda: a @ b, iters=20, warmup=5))["median"]
        try:
            t_fp = fp4_row(m, k, n)
        except Exception as e:  # noqa: BLE001
            t_fp = None
            print("fp4 gemm failed", (m, k, n), repr(e)[:100])
        out[(m, k, n)] = (t_bf, t_fp)
        del a, b
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--size", type=int, required=True)
    a = ap.parse_args()
    peaks = json.loads((RESULTS / "eval3" / "peaks_sweep.json").read_text())["best"]
    ridge = peaks["fp4_tflops"] * 1e12 / (peaks["copy_gbs"] * 1e9)  # FLOP per byte where FP4 GEMM stops being memory bound
    res = dict(model=a.model, size=a.size, ridge_flop_per_byte=ridge, premises={})
    P = res["premises"]

    def prem(name, fn):
        try:
            P[name] = fn()
        except BaseException as e:  # noqa: BLE001
            P[name] = dict(verdict="unknown", error=f"{type(e).__name__}: {str(e)[:250]}")
            traceback.print_exc()
            free()
        print(name, json.dumps(P[name], default=str)[:300], flush=True)

    ctx = {}
    with torch.no_grad():
        def bf16_profile():
            r = build(a.model, a.size, 1, "bf16")
            for _ in range(2):
                r.forward()
            torch.cuda.synchronize()
            with record(r.transformer) as (gemms, attns):
                ctx["ref"] = r.forward().float()
            ctx["gemms"], ctx["attns"] = gemms, attns
            ctx["inputs"] = list(r.inputs)
            ctx["prof"] = profile_kernels(r.forward, 2)
            ctx["tokens"] = r.tokens
            del r
            free()
        try:
            bf16_profile()
        except BaseException as e:  # noqa: BLE001
            res["fatal"] = f"{type(e).__name__}: {str(e)[:300]}"
            write_json(f"eval3/assume/{a.model}_{a.size}.json", res)
            traceback.print_exc()
            sys.exit(1)

        def p1():
            agg = collections.Counter(ctx["gemms"])
            ai = {f"{m}x{k}x{n}": dict(count=c, intensity=(2 * m * k * n) / (0.5625 * (m * k + k * n) + 2 * m * n)) for (m, k, n), c in agg.items()}
            fl_tot = sum(2 * m * k * n * c for (m, k, n), c in agg.items())
            below = sum(2 * m * k * n * c for (m, k, n), c in agg.items() if ai[f"{m}x{k}x{n}"]["intensity"] < ridge)
            sm = ctx["prof"]["sum_ms"]
            tot = sum(sm.values())
            glue = sm.get("glue", 0) / tot
            return dict(verdict="true" if glue > 0.15 or below / fl_tot > 0.3 else "false", ridge=ridge,
                        gemm_flops_below_ridge_fraction=below / fl_tot, glue_share_of_bf16_forward=glue,
                        note="glue = LayerNorm, modulate, GELU, residual, copies in the bf16 forward; if small, quantize traffic levers cannot matter",
                        intensity_by_shape=dict(list(ai.items())[:8]))
        prem("P1_memory_bound_quant_and_elementwise", p1)

        def p2():
            agg = collections.Counter(ctx["gemms"])
            t = gemm_times(list(agg))
            tb = sum(t[s][0] * c for s, c in agg.items())
            tf = sum(t[s][1] * c for s, c in agg.items() if t[s][1])
            ok = all(t[s][1] for s in agg)
            return dict(verdict="true" if ok and tf < tb else ("false" if ok else "unknown"), bf16_gemm_ms=tb, fp4_gemm_ms=tf if ok else None,
                        predicted_gemm_speedup=tb / tf if ok else None,
                        per_shape=[dict(m=m, k=k, n=n, count=c, bf16_ms=t[(m, k, n)][0], fp4_ms=t[(m, k, n)][1]) for (m, k, n), c in agg.most_common(8)],
                        note="FP4 time excludes the quantize step; the real gain is lower by the quantize cost")
        prem("P2_fp4_gemm_beats_bf16", p2)

        def p3():
            sm = ctx["prof"]["sum_ms"]
            tot = sum(sm.values())
            share = sm.get("attention", 0) / tot
            n_attn = collections.Counter(ctx["attns"])
            return dict(verdict="true" if share > 0.35 else "false", attention_share_of_bf16_forward=share, gemm_share=sm.get("gemm", 0) / tot,
                        tokens=ctx["tokens"], busy_ms=ctx["prof"]["busy_ms"],
                        attention_shapes=[dict(b=b, h=h, lq=lq, lk=lk, d=d, count=c) for (b, h, lq, lk, d), c in n_attn.most_common(4)],
                        note="head dims decide which attention kernel applies: 64 and 128 are native for Sage, 72 is padded to 96")
        prem("P3_attention_dominates", p3)

        prem("P4_text_mask_or_padding", lambda: dict(verdict="true" if "encoder_attention_mask" in ctx["inputs"] else "false",
                                                     inputs=ctx["inputs"], note="trim / xcudnn apply only when a padded text mask is an input"))
        prem("P5_cfg_batch_two", lambda: dict(verdict="true" if CATALOG[a.model]["cfg_batch"] == 2 else "false", cfg_batch=CATALOG[a.model]["cfg_batch"],
                                              note="limited-interval CFG applies only with classifier-free guidance"))

        def p6():
            out = {}
            for q in ("fp8", "nvfp4"):
                r = build(a.model, a.size, 1, q)
                y = r.forward().float()
                out[q] = float((y - ctx["ref"]).norm() / ctx["ref"].norm())
                del r
                free()
            return dict(verdict="true" if out["nvfp4"] > 1.5 * out["fp8"] else "false", rel_err_vs_bf16=out,
                        note="true = NVFP4 error well above fp8: image metrics need a noise floor and many prompts")
        prem("P6_nvfp4_error_vs_fp8", p6)

    P["consequence"] = dict(
        traffic_levers_worth_testing=P["P1_memory_bound_quant_and_elementwise"].get("verdict") == "true",
        nvfp4_worth_testing=P["P2_fp4_gemm_beats_bf16"].get("verdict") in ("true", "unknown"),
        attention_levers_priority="high" if P["P3_attention_dominates"].get("verdict") == "true" else "low",
        trim_xcudnn_apply=P["P4_text_mask_or_padding"].get("verdict") == "true",
        licfg_applies=P["P5_cfg_batch_two"].get("verdict") == "true")
    write_json(f"eval3/assume/{a.model}_{a.size}.json", res)
    print(json.dumps(P["consequence"], indent=1))


if __name__ == "__main__":
    main()
