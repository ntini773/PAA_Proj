"""What does quantizing the activations cost on top of quantized weights? Real NVFP4 only, no emulation.

    python precision/real_w4a16_vs_w4a4.py --size 512 --embeds prompts/embeds_pixart.pt

Three real models, same real prompts and seeds:
    bf16    reference
    w4a16   torchao NVFP4 weights, activations left in bf16. Built from the stock model by setting
            act_quant_kwargs=None on every NVFP4 weight, which makes torchao's nvfp4_linear take its weight-only branch
            (dequantize the weight, bf16 F.linear). So "skip the activation quantization step" is the real library path.
    w4a4    the unchanged stock NVFP4 model (weights and activations in NVFP4)
The gap between w4a16 and w4a4 is the quality price of the activation quantize step, and so also the most any
activation-quantization fusion could ever change.

Outputs (results/precision/real_w4a16_vs_w4a4_<size>.json and .md)
    transformer level  relative L2 of the transformer output at 5 timesteps, per prompt (CFG batch [uncond, cond]):
                       bf16 vs w4a16, bf16 vs w4a4, w4a16 vs w4a4
    image level        LPIPS / PSNR / SSIM / CLIP for the same three pairs from the full 14-step pipeline
"""
import argparse
import gc
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import numpy as np
import torch

import variants as vr
from common import RESULTS, load_prompt_embeds
from quality import eval_images as ev

TIMESTEPS = [999, 750, 500, 250, 50]


def rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm().clamp(min=1e-12))


@torch.no_grad()
def transformer_outputs(b, items, size, seed=0):
    """Transformer output per (prompt, timestep) on fixed noise, with the real prompt in the CFG batch."""
    tr = b.pipe.transformer
    out = {}
    for pi, it in enumerate(items):
        g = torch.Generator("cuda").manual_seed(seed + pi)
        x = torch.randn(1, 4, size // 8, size // 8, generator=g, device="cuda", dtype=torch.bfloat16).repeat(2, 1, 1, 1)
        enc = torch.cat([it["ne"], it["pe"]]).to("cuda", torch.bfloat16)
        mask = torch.cat([it["nm"], it["pm"]]).to("cuda")
        for t in TIMESTEPS:
            y = tr(x, encoder_hidden_states=enc, encoder_attention_mask=mask, timestep=torch.full((2,), t, device="cuda"),
                   added_cond_kwargs={"resolution": None, "aspect_ratio": None}, return_dict=False)[0]
            out[pi, t] = y.float().cpu()
    return out


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--size", type=int, default=512)
    a.add_argument("--embeds", default="prompts/embeds_pixart.pt")
    a.add_argument("--max-prompts", type=int, default=6)
    a.add_argument("--steps", type=int, default=14)
    a.add_argument("--clip", default="openai/clip-vit-large-patch14")
    a.add_argument("--no-clip", action="store_true")
    a = a.parse_args()
    items = load_prompt_embeds(a.embeds)[:a.max_prompts]
    prompts = [it.get("prompt", "") for it in items]

    outs, imgs, report = {}, {}, {}
    for name in ("bf16", "w4a16", "stock"):
        b = vr.build(name, a.size)
        report[name] = b.report
        outs[name] = transformer_outputs(b, items, a.size)
        imgs[name] = b.run(items, steps=a.steps)["images"]
        del b
        gc.collect()
        torch.cuda.empty_cache()
        print("done", name)

    pairs = [("bf16", "w4a16"), ("bf16", "stock"), ("w4a16", "stock")]
    tl = {f"{x}_vs_{y}": {f"t{t}": float(np.mean([rel(outs[y][pi, t], outs[x][pi, t]) for pi in range(len(items))])) for t in TIMESTEPS}
          for x, y in pairs}
    clip = None if a.no_clip else ev.Clip(a.clip)
    il = {}
    for x, y in pairs:
        rows, summary = ev.evaluate(imgs[y], imgs[x], prompts, clip)
        il[f"{x}_vs_{y}"] = dict(summary=summary, per_prompt=rows)
    res = dict(size=a.size, steps=a.steps, prompts=len(items), weights_only_layers=report["w4a16"].get("weights_only_layers"),
               transformer_rel_l2=tl, image=il)
    out = RESULTS / "precision"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"real_w4a16_vs_w4a4_{a.size}.json").write_text(json.dumps(res, default=float))
    lines = ["| pair | " + " | ".join(f"rel L2 t={t}" for t in TIMESTEPS) + " | LPIPS | SSIM | PSNR |", "|---|" + "---|" * (len(TIMESTEPS) + 3)]
    for k, v in tl.items():
        s = il[k]["summary"]
        lines.append(f"| {k} | " + " | ".join(f"{v[f't{t}']:.4f}" for t in TIMESTEPS) +
                     f" | {s['lpips'][0]:.4f} | {s['ssim'][0]:.3f} | {s['psnr'][0]:.1f} |")
    (out / f"real_w4a16_vs_w4a4_{a.size}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
