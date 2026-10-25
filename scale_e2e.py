"""Scaling experiment: PixArt-Sigma DiT, one mode per process.

  --mode bf16   : no quantization (reference)
  --mode fp16   : no quantization, float16 weights/activations
  --mode fp8    : torchao FP8 (per-row dynamic activation + weight)
  --mode stock  : torchao NVFP4 (unfused quant path)
  --mode fused  : torchao NVFP4 + our fused quant kernel spliced in (1152-wide layers only)
  --mode fused_wide : same, also the 4608-wide MLP layer and 4096-wide caption layer

Reports GPU time per transformer forward (CUDA events, = 1 denoise step incl.
CFG batch of 2), wall time per image, peak memory, fused/fallback call counts.
Each size uses its own checkpoint (256 / 512-MS / 1024-MS), NVFP4-quantized at
load time by torchao. Text embeddings are SYNTHETIC (seeded randn, shape [B,300,4096]): speed and
stock-vs-fused equality do not depend on prompt content, and this avoids the
19GB T5 download. Image content is not a real prompt; compare stock vs fused only.

    pixi run python scale_e2e.py --size 512 --mode fused
"""
import os, sys, time, json, argparse, statistics
os.environ.setdefault("HF_HUB_OFFLINE", "1")  # models come from `pixi run download`
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch
import numpy as np
from diffusers import PixArtSigmaPipeline, PixArtTransformer2DModel, TorchAoConfig
from torchao.prototype.mx_formats.inference_workflow import (
    NVFP4DynamicActivationNVFP4WeightConfig,
)
from tier2_end_to_end import install_fused_quant


REPO = {256: "PixArt-alpha/PixArt-Sigma-XL-2-256x256",
        512: "PixArt-alpha/PixArt-Sigma-XL-2-512-MS",
        1024: "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS"}


def load_pipe(mode, size):
    kw = {}
    dt = torch.float16 if mode == "fp16" else torch.bfloat16
    if mode in ("stock", "fused", "fused_wide"):
        cfg = NVFP4DynamicActivationNVFP4WeightConfig(
            use_dynamic_per_tensor_scale=True, use_triton_kernel=False)
        kw["quantization_config"] = TorchAoConfig(cfg)
    elif mode == "fp8":
        from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, PerRow
        kw["quantization_config"] = TorchAoConfig(
            Float8DynamicActivationFloat8WeightConfig(granularity=PerRow()))
    tr = PixArtTransformer2DModel.from_pretrained(
        REPO[size], subfolder="transformer",
        torch_dtype=dt, **kw)
    pipe = PixArtSigmaPipeline.from_pretrained(
        "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS", transformer=tr,
        text_encoder=None, tokenizer=None, torch_dtype=dt).to("cuda")
    return pipe


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--mode", choices=["bf16", "fp16", "fp8", "stock", "fused", "fused_wide"], required=True)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--steps", type=int, default=14)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--profile", action="store_true", help="wrap timed reps in cudaProfilerStart/Stop for nsys")
    p.add_argument("--tag", default="")
    p.add_argument("--out", default=os.path.join(HERE, "results"))
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    B = a.batch
    g0 = torch.Generator().manual_seed(1234)
    mk = lambda: (torch.randn(B, 300, 4096, generator=g0) * 0.5).to("cuda", torch.bfloat16)
    ones = lambda: torch.ones(B, 300, dtype=torch.long, device="cuda")
    emb = {"pe": mk(), "pm": ones(), "ne": mk(), "nm": ones()}

    pipe = load_pipe(a.mode, a.size)
    if a.mode == "fused_wide":      # also fuse the 4608-wide MLP layer and the 4096-wide caption layer
        os.environ["EXTRA_OK_K"] = "4608,4096"
    counter = install_fused_quant() if a.mode in ("fused", "fused_wide") else None

    fwd_ms, ev = [], []
    def pre(m, args, kwargs=None):
        s = torch.cuda.Event(enable_timing=True); s.record(); ev.append([s, None])
    def post(m, args, kwargs, out=None):
        e = torch.cuda.Event(enable_timing=True); e.record(); ev[-1][1] = e
    pipe.transformer.register_forward_pre_hook(pre, with_kwargs=True)
    pipe.transformer.register_forward_hook(post, with_kwargs=True)
    if a.compile:
        # Compile only the 28 blocks. Compiling the whole transformer failed in the
        # timestep embedder (NVFP4Tensor linear on fake tensors, "data pointer" error).
        for blk in pipe.transformer.transformer_blocks:
            blk.compile(fullgraph=False)

    def gen(steps):
        g = torch.Generator(device="cuda").manual_seed(a.seed)
        return pipe(prompt=None, negative_prompt=None,
                    prompt_embeds=emb["pe"], prompt_attention_mask=emb["pm"],
                    negative_prompt_embeds=emb["ne"],
                    negative_prompt_attention_mask=emb["nm"],
                    height=a.size, width=a.size, num_inference_steps=steps,
                    generator=g, output_type="np").images

    gen(2); torch.cuda.synchronize()          # warmup: Triton JIT / compile
    if counter: counter["n"] = counter["fallback"] = 0
    ev.clear(); torch.cuda.reset_peak_memory_stats()
    walls = []
    if a.profile: torch.cuda.profiler.start()
    for _ in range(a.reps):
        torch.cuda.synchronize(); t0 = time.time()
        img = gen(a.steps)
        torch.cuda.synchronize(); walls.append(time.time() - t0)
    if a.profile: torch.cuda.profiler.stop()
    fwd_ms = [s.elapsed_time(e) for s, e in ev]

    res = dict(mode=a.mode, size=a.size, batch=B, steps=a.steps, compile=a.compile,
               tokens=(a.size // 16) ** 2,
               step_gpu_ms_median=statistics.median(fwd_ms),
               step_gpu_ms_min=min(fwd_ms),
               step_gpu_ms_all=[round(v, 2) for v in fwd_ms],   # every denoise step, all reps
               wall_s_median=statistics.median(walls),
               peak_mem_gb=torch.cuda.max_memory_allocated() / 2**30,
               fused_calls=counter["n"] if counter else None,
               fallback_calls=counter["fallback"] if counter else None)
    print(json.dumps(res))
    name = f"{a.mode}_{a.size}_b{B}{'_compile' if a.compile else ''}{a.tag}"
    np.save(os.path.join(a.out, name + ".npy"), img)
    with open(os.path.join(a.out, "e2e.jsonl"), "a") as f:
        f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
