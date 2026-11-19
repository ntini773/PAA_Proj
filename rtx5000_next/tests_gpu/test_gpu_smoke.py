"""GPU smoke test of the eval2 stack on the real PixArt-Sigma weights (default 256 px, 2 prompts, a few steps).

Run on any sm_120 GPU (the laptop RTX 5060 or the server):  python tests_gpu/test_gpu_smoke.py [--size 256]
It needs the PixArt weights in the HF cache and prompts/embeds_pixart.pt. Uses the Triton quantize backend, so no CUDA
extension build is needed. Checks, in order of how much they matter for the experiments:
  1. the module stack equals torchao's NVFP4 output (fast GEMM path correctness) and its quantize op runs once per site
  2. the same under torch.compile (the bug that made 'fused + compile' measure the stock graph)
  3. real weights-only (w4a16) differs from stock and sits closer to bf16
  4. DiTFastAttn plan search + run, limited-interval CFG batches, image metrics
Timings printed here come from whatever GPU this runs on and are not results.
"""
import argparse
import sys
import time

sys.path.insert(0, __import__("pathlib").Path(__file__).resolve().parent.parent.as_posix() + "/tests_cpu")
import _boot
from _boot import check, torch

import fused_linear as fl
import variants as vr
from common import load_prompt_embeds
from precision.real_w4a16_vs_w4a4 import rel, transformer_outputs
from quality import eval_images as ev

A = argparse.ArgumentParser()
A.add_argument("--size", type=int, default=256)
A.add_argument("--steps", type=int, default=4)
A.add_argument("--only", nargs="*", default=None, help="test function names; run_gpu_checks.sh runs each in its own process")
ARGS, _ = A.parse_known_args()
ITEMS = load_prompt_embeds("prompts/embeds_pixart.pt")[:2]
EXPECT_SITES = 28 * 7 + 6


def test_layer_exact_vs_torchao():
    """Per layer, on identical input, our module must reproduce torchao's NVFP4 linear bit for bit."""
    from torchao.prototype.mx_formats.inference_workflow import NVFP4DynamicActivationNVFP4WeightConfig
    from torchao.quantization import quantize_
    torch.manual_seed(0)
    ok = True
    for M, K, N in [(512, 1152, 1152), (600, 1152, 1152), (512, 4608, 1152), (2048, 1152, 4608)]:
        L = torch.nn.Linear(K, N).cuda().bfloat16()
        x = (torch.randn(M, K, device="cuda") * 1.5).bfloat16()
        q = torch.nn.Linear(K, N).cuda().bfloat16()
        q.load_state_dict(L.state_dict())
        quantize_(q, NVFP4DynamicActivationNVFP4WeightConfig(use_dynamic_per_tensor_scale=True, use_triton_kernel=False))
        ref = q(x)
        for be in ("triton", "torchref"):
            ok &= torch.equal(fl.FusedNVFP4Linear([L], backend=be, fast=True)(x), ref)
    check("layer level: module output is bit-identical to torchao NVFP4 (both backends, 4 shapes)", ok)


def test_model_error_matches_stock_and_compile_engages():
    """Whole-model outputs are NOT expected to agree between two NVFP4 implementations: quantization is a discontinuous
    function, so a bf16-rounding-level difference upstream flips some FP4 roundings and the noise realization changes
    (stock vs our bit-exact layers still differ by about the quantization error). What must hold is that each is about
    equally far from bf16."""
    outs = {}
    for name in ("bf16", "stock", "module+be=triton"):
        b = vr.build(name, ARGS.size)
        outs[name] = transformer_outputs(b, ITEMS[:1], ARGS.size)
        if name == "module+be=triton":
            fl.reset_counts()
            transformer_outputs(b, ITEMS[:1], ARGS.size)
            sites = fl.COUNTS["quant:triton"] // len(_T)
            check("module stack: one quantize op per site per forward", sites == EXPECT_SITES, f"{sites} vs {EXPECT_SITES}")
        del b
        torch.cuda.empty_cache()
    comp = vr.build("module+be=triton+compile", ARGS.size)
    transformer_outputs(comp, ITEMS[:1], ARGS.size)  # compile happens here
    fl.reset_counts()
    outs["compiled"] = transformer_outputs(comp, ITEMS[:1], ARGS.size)
    sites_c = fl.COUNTS["quant:triton"] // len(_T)
    check("compiled module stack: quantize op runs once per site per forward", sites_c == EXPECT_SITES, f"{sites_c} vs {EXPECT_SITES}")
    del comp
    torch.cuda.empty_cache()
    err = {n: sum(rel(outs[n][k], outs["bf16"][k]) for k in outs["bf16"]) / len(outs["bf16"]) for n in ("stock", "module+be=triton", "compiled")}
    print("      mean rel L2 vs bf16:", {n: round(e, 4) for n, e in err.items()})
    for n in ("module+be=triton", "compiled"):
        check(f"{n}: error vs bf16 within 30% of stock NVFP4's", 0.7 <= err[n] / err["stock"] <= 1.3, f"{err[n]:.4f} vs {err['stock']:.4f}")


def test_w4a16_real():
    bf = vr.build("bf16", ARGS.size)
    o_bf = transformer_outputs(bf, ITEMS[:1], ARGS.size)
    del bf
    torch.cuda.empty_cache()
    w = vr.build("w4a16", ARGS.size)
    o_w = transformer_outputs(w, ITEMS[:1], ARGS.size)
    n = w.report["weights_only_layers"]
    del w
    torch.cuda.empty_cache()
    s = vr.build("stock", ARGS.size)
    o_s = transformer_outputs(s, ITEMS[:1], ARGS.size)
    del s
    torch.cuda.empty_cache()
    check("w4a16: weights-only branch set on NVFP4 layers", n > 100, f"{n} layers")
    e_w = max(rel(o_w[k], o_bf[k]) for k in o_bf)
    e_s = max(rel(o_s[k], o_bf[k]) for k in o_bf)
    check("w4a16 differs from stock w4a4", max(rel(o_w[k], o_s[k]) for k in o_s) > 1e-3)
    print(f"      rel L2 vs bf16: w4a16 {e_w:.4f}  w4a4 {e_s:.4f}")
    check("w4a16 is no farther from bf16 than w4a4", e_w <= e_s * 1.05, f"{e_w:.4f} vs {e_s:.4f}")


def test_dfa_licfg_metrics():
    import attention_patch as ap
    import cfg_interval
    import ditfast.attention as dfa
    import ditfast.plan_search as ps
    b = vr.build("bf16", ARGS.size)
    ps.enable(b.pipe, ap.AttnConfig(), use_dfa=True)
    plan = ps.search_plan(b.pipe, ITEMS, ARGS.size, steps=ARGS.steps + 2, delta=0.1)
    print("      plan counts", plan["counts"], "modeled attention cost", round(plan["modeled_attention_cost"], 2))
    check("plan has one entry per step and layer", len(plan["plan"]) == ARGS.steps + 2 and len(plan["plan"][0]) == 28)
    ap.install_attention(b.pipe.transformer, ap.AttnConfig(), use_dfa=False)  # plain attention for the reference images
    ref = cfg_interval.sample(b.pipe, ITEMS, ARGS.size, steps=ARGS.steps + 2)
    ap.install_attention(b.pipe.transformer, ap.AttnConfig(), use_dfa=True)
    dfa.set_plan(plan)
    comp = cfg_interval.sample(b.pipe, ITEMS, ARGS.size, steps=ARGS.steps + 2, dfa_active=True)
    check("DiTFastAttn run produces finite images", all(__import__("numpy").isfinite(i).all() for i in comp["images"]))
    li = cfg_interval.sample(b.pipe, ITEMS, ARGS.size, steps=6, interval=(1 / 3, 2 / 3), output="latent")
    check("limited-interval CFG halves the batch outside the band", li["model_batches"] == [2, 2, 4, 4, 2, 2], str(li["model_batches"]))
    clip = ev.Clip("random-tiny")
    rows, summary = ev.evaluate(comp["images"], ref["images"], ["a", "b"], clip)
    check("metrics: LPIPS finite and below 1", 0 <= summary["lpips"][0] < 1.0, str(summary["lpips"]))
    rows0, s0 = ev.evaluate(ref["images"], ref["images"], ["a", "b"], clip)
    check("metrics: identical images give LPIPS 0, SSIM 1", s0["lpips"][0] < 1e-4 and s0["ssim"][0] > 0.999, str(s0))
    print("      DFA vs full attention: LPIPS", round(summary["lpips"][0], 4), "SSIM", round(summary["ssim"][0], 3))


_T = [999, 750, 500, 250, 50]

if __name__ == "__main__":
    t0 = time.time()
    ALL = [test_layer_exact_vs_torchao, test_model_error_matches_stock_and_compile_engages, test_w4a16_real, test_dfa_licfg_metrics]
    _boot.run(*[t for t in ALL if ARGS.only is None or t.__name__ in ARGS.only])
    print(f"{time.time() - t0:.0f}s")
