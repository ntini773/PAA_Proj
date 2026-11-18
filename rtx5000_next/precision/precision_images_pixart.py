"""PixArt images from the bf16 model with emulated NVFP4 variants, plus the real torchao NVFP4 model.
Same seed and embeddings for every variant. Real embeddings: python prompts/make_prompt_embeds.py, then --embeds.
One image set per prompt: results/precision/images/p<i>_<variant>_<size>.png. Metrics are per prompt and the mean.

    python precision/precision_images_pixart.py --size 1024 --embeds prompts/embeds_pixart.pt
"""
import argparse
import functools
import gc
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import numpy as np
import torch
from PIL import Image

from common import RESULTS, load_prompt_embeds, write_json
from precision.nvfp4_emulation import patch_linears
from scale_e2e import load_pipe

VARIANTS = {"bf16": (False, False), "w4a16": (True, False), "w16a4": (False, True), "w4a4_emulated": (True, True)}


@functools.lru_cache(maxsize=1)
def lpips_net():
    import lpips  # first use downloads the AlexNet backbone (torchvision); prefetch: python -c "import lpips; lpips.LPIPS(net='alex')"
    return lpips.LPIPS(net="alex", verbose=False).cuda()


def metrics(x, ref):
    mse = float(((x - ref) ** 2).mean())
    out = dict(mae=float(np.abs(x - ref).mean()), psnr=float("inf") if mse == 0 else 10 * np.log10(1 / mse))
    try:
        t = lambda v: torch.from_numpy(v).permute(2, 0, 1)[None].cuda().float() * 2 - 1
        out["lpips"] = float(lpips_net()(t(x), t(ref)))
    except Exception as e:
        print("lpips skipped:", repr(e))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--embeds", default=None)
    ap.add_argument("--max-prompts", type=int, default=6)
    ap.add_argument("--skip-stock", action="store_true")
    a = ap.parse_args()

    if a.embeds:
        items = load_prompt_embeds(a.embeds)[:a.max_prompts]
    else:
        g0 = torch.Generator().manual_seed(1234)
        mk = lambda: (torch.randn(1, 300, 4096, generator=g0) * 0.5).bfloat16()
        ones = lambda: torch.ones(1, 300, dtype=torch.long)
        items = [dict(prompt="synthetic", pe=mk(), pm=ones(), ne=mk(), nm=ones())]
        print("WARNING: synthetic embeddings, images are not meaningful")

    def gen(pipe, item):
        emb = {k: item[k].cuda() for k in ("pe", "pm", "ne", "nm")}
        g = torch.Generator(device="cuda").manual_seed(a.seed)
        return pipe(prompt=None, negative_prompt=None, prompt_embeds=emb["pe"], prompt_attention_mask=emb["pm"],
                    negative_prompt_embeds=emb["ne"], negative_prompt_attention_mask=emb["nm"],
                    height=a.size, width=a.size, num_inference_steps=a.steps, generator=g, output_type="np").images[0]

    out_dir = RESULTS / "precision" / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    imgs = {}
    pipe = load_pipe("bf16", a.size)
    with torch.no_grad():
        for name, (w, act) in VARIANTS.items():
            undo = patch_linears(pipe.transformer, w, act, True) if (w or act) else (lambda: None)
            for pi, item in enumerate(items):
                imgs[pi, name] = gen(pipe, item)
            undo()
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    if not a.skip_stock:
        pipe = load_pipe("stock", a.size)
        with torch.no_grad():
            for pi, item in enumerate(items):
                imgs[pi, "stock_torchao"] = gen(pipe, item)
        del pipe

    res = dict(prompts=[it["prompt"] for it in items], per_prompt=[])
    for pi in range(len(items)):
        row = {}
        for name in [n for (i, n) in imgs if i == pi]:
            img = imgs[pi, name]
            Image.fromarray((img * 255).round().astype("uint8")).save(out_dir / f"p{pi}_{name}_{a.size}.png")
            if name != "bf16":
                row[name] = metrics(img.astype("f4"), imgs[pi, "bf16"].astype("f4"))
        if (pi, "stock_torchao") in imgs:
            row["stock_vs_emulated"] = metrics(imgs[pi, "stock_torchao"].astype("f4"), imgs[pi, "w4a4_emulated"].astype("f4"))
        res["per_prompt"].append(row)
    names = res["per_prompt"][0].keys()
    res["mean"] = {n: {k: float(np.mean([r[n][k] for r in res["per_prompt"]])) for k in res["per_prompt"][0][n]} for n in names}
    write_json(f"precision/images_{a.size}.json", res)
    print(json.dumps(res["mean"], indent=1))


if __name__ == "__main__":
    main()
