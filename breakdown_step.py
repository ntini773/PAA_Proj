"""Per-part time breakdown of ONE PixArt-Sigma denoise step (the transformer forward).

Answers: in the stock NVFP4 pipeline, how much of the step is
  LayerNorm | attention core | NVFP4 GEMM | activation quantize (the part we fuse)
  | quantize overhead around it (per-tensor max, scale re-layout, ...) | everything else
and how does each part change with the fused kernel (--mode fused) and with bf16.

Method (no profiler needed for the main table): CUDA events around
  - the whole forward                      -> step
  - every nn.LayerNorm                     -> layernorm
  - F.scaled_dot_product_attention         -> attn_core
  - torch._scaled_mm (the FP4 matmul)      -> gemm      (bf16 mode: gemm = all Linear time)
  - NVFP4Tensor.to_nvfp4 (activation quant)-> quant_fn  (stock: torchao path, fused: our kernel)
  - every nn.Linear                        -> linear_total
  quant_overhead = linear_total - gemm - quant_fn
  rest           = step - layernorm - attn_core - linear_total   (modulate, residual, GELU, copies)
Events measure GPU timeline between launch points, so they include GPU idle gaps.
A second pass with torch.profiler lists the top GPU kernels by name.

    pixi run python breakdown_step.py --size 512 --mode stock
Appends to results/breakdown.jsonl
"""
import os, sys, json, argparse, statistics, collections
os.environ.setdefault("HF_HUB_OFFLINE", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import PixArtTransformer2DModel, TorchAoConfig
from torchao.prototype.mx_formats.inference_workflow import NVFP4DynamicActivationNVFP4WeightConfig
import torchao.prototype.mx_formats.nvfp4_tensor as nvfp4_mod
from scale_e2e import REPO
from tier2_end_to_end import install_fused_quant

items = []          # (category, start_event, end_event)
state = {"on": False}


def ev():
    e = torch.cuda.Event(enable_timing=True)
    e.record()
    return e


def timed(cat, fn):
    def wrapper(*a, **k):
        if not state["on"]:
            return fn(*a, **k)
        s = ev()
        out = fn(*a, **k)
        items.append((cat, s, ev()))
        return out
    return wrapper


def hook_module(mod, cat):
    stack = []
    def pre(m, args, kwargs):
        if state["on"]:
            stack.append(ev())
    def post(m, args, kwargs, out):
        if state["on"] and stack:
            items.append((cat, stack.pop(), ev()))
    mod.register_forward_pre_hook(pre, with_kwargs=True)
    mod.register_forward_hook(post, with_kwargs=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--mode", choices=["bf16", "stock", "fused"], required=True)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--passes", type=int, default=5)
    p.add_argument("--no-profiler", action="store_true")
    a = p.parse_args()

    kw = {}
    if a.mode != "bf16":
        kw["quantization_config"] = TorchAoConfig(NVFP4DynamicActivationNVFP4WeightConfig(
            use_dynamic_per_tensor_scale=True, use_triton_kernel=False))
    tr = PixArtTransformer2DModel.from_pretrained(
        REPO[a.size], subfolder="transformer", torch_dtype=torch.bfloat16, **kw).to("cuda").eval()
    counter = install_fused_quant() if a.mode == "fused" else None

    # ---- timing wrappers (installed after the fused patch so they wrap it) ----
    calls = collections.Counter()
    F.scaled_dot_product_attention = timed("attn_core", F.scaled_dot_product_attention)
    _mm = torch._scaled_mm
    def mm(*args, **kwargs):
        calls["scaled_mm"] += 1
        return _mm(*args, **kwargs)
    torch._scaled_mm = timed("gemm", mm)
    cur = nvfp4_mod.NVFP4Tensor.to_nvfp4
    nvfp4_mod.NVFP4Tensor.to_nvfp4 = staticmethod(timed("quant_fn", cur))

    for _, m in tr.named_modules():
        if isinstance(m, nn.LayerNorm):
            hook_module(m, "layernorm")
        elif isinstance(m, nn.Linear):
            hook_module(m, "linear_total")
    hook_module(tr, "step")

    B2 = 2 * a.batch
    lat = a.size // 8
    g = torch.Generator("cuda").manual_seed(0)
    x = torch.randn(B2, 4, lat, lat, device="cuda", dtype=torch.bfloat16, generator=g)
    txt = torch.randn(B2, 300, 4096, device="cuda", dtype=torch.bfloat16, generator=g) * 0.5
    msk = torch.ones(B2, 300, dtype=torch.long, device="cuda")
    t = torch.full((B2,), 500.0, device="cuda")

    def fwd():
        return tr(x, encoder_hidden_states=txt, timestep=t, encoder_attention_mask=msk,
                  added_cond_kwargs={"resolution": None, "aspect_ratio": None},
                  return_dict=False)[0]

    with torch.no_grad():
        for _ in range(4):
            fwd()
        torch.cuda.synchronize()
        calls.clear()
        state["on"] = True
        for _ in range(a.passes):
            fwd()
        torch.cuda.synchronize()
        state["on"] = False

        tot = collections.Counter()
        for cat, s, e in items:
            tot[cat] += s.elapsed_time(e)
        n = a.passes
        ms = {k: v / n for k, v in tot.items()}
        step = ms["step"]
        if a.mode == "bf16":
            ms["gemm"] = ms.get("linear_total", 0.0)     # all Linear time is the matmul
            ms["quant_fn"] = 0.0
            ms["quant_overhead"] = 0.0
        else:
            ms["quant_overhead"] = ms["linear_total"] - ms.get("gemm", 0.0) - ms.get("quant_fn", 0.0)
        ms["rest"] = step - ms.get("layernorm", 0.0) - ms.get("attn_core", 0.0) - ms["linear_total"]

        if a.mode != "bf16" and calls["scaled_mm"] == 0:
            print("WARNING: torch._scaled_mm was never called; gemm time is missing and "
                  "quant_overhead includes it. Read the top-kernels list instead.")
        order = ["layernorm", "attn_core", "gemm", "quant_fn", "quant_overhead", "rest"]
        res = dict(mode=a.mode, size=a.size, batch=a.batch, tokens=(a.size // 16) ** 2,
                   step_ms=step, parts_ms={k: round(ms.get(k, 0.0), 3) for k in order},
                   parts_pct={k: round(100 * ms.get(k, 0.0) / step, 1) for k in order},
                   scaled_mm_calls_per_step=calls["scaled_mm"] / n,
                   fused_calls=counter["n"] if counter else None,
                   fallback_calls=counter["fallback"] if counter else None)

        # ---- second pass: top GPU kernels by name ----
        if not a.no_profiler:
            from torch.profiler import profile, ProfilerActivity
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                for _ in range(2):
                    fwd()
                torch.cuda.synchronize()
            rows = []
            for e in prof.key_averages():
                if str(getattr(e, "device_type", "")).endswith("CUDA"):
                    t_us = getattr(e, "self_device_time_total", None)
                    if t_us is None:
                        t_us = getattr(e, "self_cuda_time_total", 0)
                    rows.append((t_us, e.count, e.key))
            total = sum(r[0] for r in rows) or 1.0
            rows.sort(reverse=True)
            res["top_kernels"] = [dict(pct=round(100 * r[0] / total, 1), us_per_step=round(r[0] / 2, 1),
                                       count_per_step=r[1] / 2, name=r[2][:110]) for r in rows[:15]]

    print(json.dumps({k: v for k, v in res.items() if k != "top_kernels"}))
    if "top_kernels" in res:
        print("top kernels (% of GPU kernel time):")
        for k in res["top_kernels"]:
            print(f"  {k['pct']:5.1f}%  {k['us_per_step']:9.1f} us/step  x{k['count_per_step']:.0f}  {k['name']}")
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "breakdown.jsonl"), "a") as f:
        f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
