"""Real-prompt quality of every speed variant, next to its generation time.

    python quality/run_quality.py --size 512 --variants stock w4a16 module module+static+concat dfa... --embeds prompts/embeds_pixart.pt

What it does
    1. Builds the reference (first variant, default bf16) and every listed variant (grammar in variants.py).
    2. Generates the real prompts (same seeds, 14 steps, same guidance) as one batch per variant.
    3. Compares each variant with the reference per prompt (quality/eval_images.py) and records the denoise time per image.
    4. Writes images, a contact sheet, results/quality/<size>/summary.{csv,md,json} and a speed-vs-quality plot.
Why: every approximation (static scales, DiTFastAttn, Sage, K/V pooling, limited-interval CFG) must be shown with its quality
cost. A variant counts as acceptable only when its LPIPS is not worse than stock NVFP4's plus the margin chosen in eval2-pipeline.md.
The time here is the whole denoise loop (CUDA events, batch of all prompts, after a warm-up run); step-level speed comes from step_target.py.
"""
import argparse
import csv
import gc
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import numpy as np
import torch
from PIL import Image

from common import RESULTS, load_prompt_embeds
from quality import eval_images as ev
import variants as vr


def save(imgs, d):
    d.mkdir(parents=True, exist_ok=True)
    for i, im in enumerate(imgs):
        Image.fromarray((np.clip(im, 0, 1) * 255).round().astype("uint8")).save(d / f"p{i}.png")


def generate(name, size, items, steps, warm=True):
    """Build a variant, warm it up (compile, caches), then generate all prompts. Returns images, ms, report."""
    b = vr.build(name, size)
    if warm:
        b.run(items, steps=2, output="latent")
        torch.cuda.synchronize()
    res = b.run(items, steps=steps, timing=True)
    out = dict(images=res["images"], denoise_ms=res["denoise_ms"], vae_ms=res.get("vae_ms"), step_ms=res["step_ms"],
               batches=res["model_batches"], report=b.report)
    del b
    gc.collect()
    torch.cuda.empty_cache()
    return out


def contact_sheet(rows, path, thumb=192, n_prompts=3):
    """rows: {variant: [images]}; one row per variant, columns are the first prompts."""
    names = list(rows)
    W = thumb * n_prompts
    sheet = Image.new("RGB", (W, thumb * len(names)))
    for r, n in enumerate(names):
        for c in range(min(n_prompts, len(rows[n]))):
            im = Image.fromarray((np.clip(rows[n][c], 0, 1) * 255).round().astype("uint8")).resize((thumb, thumb))
            sheet.paste(im, (c * thumb, r * thumb))
    sheet.save(path)


def pareto(table, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib missing: skipping the Pareto plot")
        return
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for k, (kind, ylab) in enumerate((("lpips", "LPIPS to bf16 reference (lower is better)"), ("clip_score", "CLIP score (higher is better)"))):
        for r in table:
            if kind in r["summary"]:
                ax[k].scatter(r["ms_per_image"], r["summary"][kind][0], s=60)
                ax[k].annotate(r["name"], (r["ms_per_image"], r["summary"][kind][0]), fontsize=7, rotation=20)
        ax[k].set_xlabel("denoise time per image (ms, batch of all prompts)")
        ax[k].set_ylabel(ylab)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--size", type=int, default=512)
    a.add_argument("--variants", nargs="+", required=True)
    a.add_argument("--reference", default="bf16")
    a.add_argument("--embeds", default="prompts/embeds_pixart.pt")
    a.add_argument("--prompts", default="prompts/prompts.json")
    a.add_argument("--max-prompts", type=int, default=6)
    a.add_argument("--steps", type=int, default=14)
    a.add_argument("--clip", default="openai/clip-vit-large-patch14")
    a.add_argument("--no-clip", action="store_true")
    a.add_argument("--out", default=None)
    a = a.parse_args()

    items = load_prompt_embeds(a.embeds)[:a.max_prompts]
    prompts = [it.get("prompt", "") for it in items]
    out = pathlib.Path(a.out) if a.out else RESULTS / "quality" / str(a.size)
    clip = None if a.no_clip else ev.Clip(a.clip)

    ref = generate(a.reference, a.size, items, a.steps)
    save(ref["images"], out / a.reference.replace("+", "_"))
    table, sheet = [], {a.reference: ref["images"]}
    for name in [a.reference] + [v for v in a.variants if v != a.reference]:
        g = ref if name == a.reference else generate(name, a.size, items, a.steps)
        rows, summary = ev.evaluate(g["images"], ref["images"], prompts, clip)
        if name != a.reference:
            save(g["images"], out / name.replace("+", "_"))
            sheet[name] = g["images"]
        table.append(dict(name=name, ms_per_image=g["denoise_ms"] / len(items), denoise_ms=g["denoise_ms"], vae_ms=g["vae_ms"],
                          model_batches=g["batches"], report=g["report"], summary=summary, per_prompt=rows))
        s = summary
        print(f"{name:45s} {g['denoise_ms'] / len(items):8.1f} ms/img  LPIPS {s['lpips'][0]:.4f}  SSIM {s['ssim'][0]:.3f}"
              + (f"  CLIP {s['clip_score'][0]:.1f}" if "clip_score" in s else ""))

    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(table, default=float))
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        keys = list(table[0]["summary"])
        w.writerow(["variant", "ms_per_image"] + [f"{k}_{s}" for k in keys for s in ("mean", "std")])
        for r in table:
            w.writerow([r["name"], round(r["ms_per_image"], 1)] + [round(r["summary"][k][i], 4) for k in keys for i in (0, 1)])
    with open(out / "summary.md", "w") as f:
        f.write(f"Real prompts ({len(items)}), {a.size} px, {a.steps} steps, reference `{a.reference}`. Mean over prompts.\n\n"
                "| variant | ms/img | LPIPS | PSNR | SSIM | CLIP score | CLIP sim to ref |\n|---|---|---|---|---|---|---|\n")
        for r in table:
            s = r["summary"]
            g = lambda k: f"{s[k][0]:.3f}±{s[k][1]:.3f}" if k in s else "-"
            f.write(f"| {r['name']} | {r['ms_per_image']:.0f} | {g('lpips')} | {s['psnr'][0]:.1f} | {g('ssim')} | "
                    f"{s['clip_score'][0]:.1f} | {g('clip_sim')} |\n" if "clip_score" in s else
                    f"| {r['name']} | {r['ms_per_image']:.0f} | {g('lpips')} | {s['psnr'][0]:.1f} | {g('ssim')} | - | {g('clip_sim')} |\n")
    contact_sheet(sheet, out / "contact_sheet.png")
    pareto(table, out / "pareto.png")
    print("wrote", out)


if __name__ == "__main__":
    main()
