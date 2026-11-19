"""End-to-end generation time per variant: text encoder, every denoise step, VAE, and the GPU idle time between them.

    python bench/pipeline_e2e.py --size 1024 --variants bf16+compile stock_compile module+be=cuda_swizzled+static+concat+compile \\
        --images 4 [--t5]

Why: step times alone hide what a user waits for. This measures the whole path with real prompts: T5 encode (optional, once per
prompt batch), the denoise loop (per-step CUDA events, so launch-bound steps and limited-interval CFG steps show up), the VAE
decode, and host wall time. "idle" is wall time minus GPU-measured phases: the cost of Python and launch overhead.
Variant strings: variants.py. A variant with licfg<lo>-<hi> runs half-size batches outside the guided band; one with dfa<delta>
needs its plan (ditfast/make_plans.py).
Output: results/pipeline_e2e.jsonl and results/pipeline_e2e.md.
"""
import argparse
import json
import pathlib
import statistics
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import torch

import variants as vr
from common import RESULTS, load_prompt_embeds


def t5_ms(prompts, repo="PixArt-alpha/PixArt-Sigma-XL-2-1024-MS"):
    from transformers import T5EncoderModel, T5Tokenizer
    tok = T5Tokenizer.from_pretrained(repo, subfolder="tokenizer", legacy=True)
    te = T5EncoderModel.from_pretrained(repo, subfolder="text_encoder", dtype=torch.bfloat16).cuda().eval()
    ids = tok(prompts, padding="max_length", max_length=300, truncation=True, return_tensors="pt").to("cuda")
    with torch.no_grad():
        te(input_ids=ids.input_ids, attention_mask=ids.attention_mask)
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        te(input_ids=ids.input_ids, attention_mask=ids.attention_mask)
        e.record()
        torch.cuda.synchronize()
    del te
    torch.cuda.empty_cache()
    return s.elapsed_time(e)


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--size", type=int, default=1024)
    a.add_argument("--variants", nargs="+", required=True)
    a.add_argument("--images", type=int, default=4, help="prompts generated together as one batch")
    a.add_argument("--steps", type=int, default=14)
    a.add_argument("--reps", type=int, default=3)
    a.add_argument("--embeds", default="prompts/embeds_pixart.pt")
    a.add_argument("--t5", action="store_true")
    a = a.parse_args()
    items = load_prompt_embeds(a.embeds)[:a.images]
    t5 = t5_ms([it.get("prompt", "") for it in items]) if a.t5 else None
    lines = ["| variant | T5 ms | denoise ms | median step ms | VAE ms | idle ms | total ms | ms per image |", "|---|---|---|---|---|---|---|---|"]
    out = []
    for name in a.variants:
        b = vr.build(name, a.size)
        b.run(items, steps=2, output="latent")  # warm-up: compile, caches, allocator
        runs = []
        for _ in range(a.reps):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            r = b.run(items, steps=a.steps, timing=True)
            torch.cuda.synchronize()
            wall = (time.perf_counter() - t0) * 1e3
            runs.append(dict(denoise_ms=r["denoise_ms"], vae_ms=r["vae_ms"], step_ms=r["step_ms"], wall_ms=wall, batches=r["model_batches"]))
        best = min(runs, key=lambda x: x["wall_ms"])
        idle = best["wall_ms"] - best["denoise_ms"] - best["vae_ms"]
        total = best["wall_ms"] + (t5 or 0.0)
        rec = dict(variant=name, size=a.size, images=len(items), steps=a.steps, t5_ms=t5, denoise_ms=best["denoise_ms"], vae_ms=best["vae_ms"],
                   median_step_ms=statistics.median(best["step_ms"]), idle_ms=idle, total_ms=total, ms_per_image=total / len(items),
                   step_ms=best["step_ms"], model_batches=best["batches"], report=b.report)
        out.append(rec)
        lines.append(f"| {name} | {t5 or 0:.0f} | {rec['denoise_ms']:.0f} | {rec['median_step_ms']:.1f} | {rec['vae_ms']:.0f} | {idle:.0f} | {total:.0f} | {rec['ms_per_image']:.0f} |")
        print(lines[-1])
        del b
        torch.cuda.empty_cache()
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "pipeline_e2e.jsonl", "a") as f:
        f.write("\n".join(json.dumps(r, default=float) for r in out) + "\n")
    (RESULTS / f"pipeline_e2e_{a.size}.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
