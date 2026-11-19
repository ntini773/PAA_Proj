"""Record per-layer activation amax over a real sampling run, for static NVFP4 activation scales.

    python calibrate_static_scales.py --size 512 --embeds prompts/embeds_calib.pt

Why
    Dynamic NVFP4 computes amax(x) for every activation on every call: an extra full pass over x (one more read of the
    tensor from DRAM) before the quantize pass. A static per-layer amax removes that pass, which is the main memory-traffic
    saving of the module stack ("module+static"). The price is a possible mismatch when the activation range drifts across
    denoise steps, so this script also reports the drift (max/min over steps) per layer.

How
    bf16 model, full sampling loop with CFG (both halves of the batch are recorded), forward pre-hooks on every nn.Linear
    store the running max of |x| per (layer, denoise step). Layer names are the original module names, which
    fused_linear.convert_pixart() uses as keys.
Output  results/static_scales.json  {"amax": {layer: max over steps}, "per_step": {layer: [amax per step]},
                                      "drift": {layer: max/min}, "steps", "size", "prompts": n}
Use margin (variants.load_static_amax(margin)) > 1 to leave head-room; a too-small scale saturates the largest blocks.
"""
import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import torch

import cfg_interval
from common import RESULTS, load_prompt_embeds
from variants import STATIC_PATH, _load_pipe


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--size", type=int, default=512)
    a.add_argument("--steps", type=int, default=14)
    a.add_argument("--embeds", default="prompts/embeds_calib.pt")
    a.add_argument("--n", type=int, default=8)
    a.add_argument("--allow-eval-prompts", action="store_true")
    a.add_argument("--out", default=None, help="default: results/static_scales_<size>.json")
    a = a.parse_args()
    if a.out is None:
        a.out = str(STATIC_PATH.with_name(f"static_scales_{a.size}.json"))

    emb = pathlib.Path(a.embeds)
    if not emb.exists():
        if not a.allow_eval_prompts:
            sys.exit(f"{emb} missing: run prompts/make_prompt_embeds.py --prompts prompts/prompts_calib.json --out {emb}")
        emb = pathlib.Path("prompts/embeds_pixart.pt")
    items = load_prompt_embeds(emb)[:a.n]
    pipe = _load_pipe(a.size, "bf16")
    tr = pipe.transformer
    buf, cur = {}, [0]
    hooks = []
    for name, m in tr.named_modules():
        if isinstance(m, torch.nn.Linear):
            buf[name] = torch.zeros(a.steps, device="cuda")

            def pre(mod, args, name=name):
                i = cur[0]
                buf[name][i] = torch.maximum(buf[name][i], args[0].detach().abs().max().float())

            hooks.append(m.register_forward_pre_hook(pre))
    cfg_interval.sample(pipe, items, a.size, steps=a.steps, output="latent", on_step=lambda i: cur.__setitem__(0, i))
    for h in hooks:
        h.remove()
    per_step = {k: v.tolist() for k, v in buf.items()}
    amax = {k: max(v) for k, v in per_step.items()}
    drift = {k: max(v) / max(min(x for x in v if x > 0), 1e-12) for k, v in per_step.items() if any(x > 0 for x in v)}
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(amax=amax, per_step=per_step, drift=drift, steps=a.steps, size=a.size, prompts=len(items))))
    worst = sorted(drift.items(), key=lambda kv: -kv[1])[:5]
    print(f"{len(amax)} layers; wrote {out}")
    print("largest amax drift across denoise steps (max/min):", [(k, round(d, 1)) for k, d in worst])


if __name__ == "__main__":
    main()
