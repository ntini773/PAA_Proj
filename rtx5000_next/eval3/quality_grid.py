"""Quality grid with controls and paired confidence intervals (eval3 T-qual). Also the noise-floor run (default variants).

    python eval3/quality_grid.py --size 512 --embeds prompts/embeds_showcase.pt --n-prompts 32 --seeds 2          # noise floor
    python eval3/quality_grid.py --size 512 --variants stock module+be=cuda_swizzled ... --controls none        # a real grid
Per variant: images for every (prompt, seed) pair, saved as PNG. Metrics against the reference variant (default bf16 eager, the
same prompt and seed): LPIPS, SSIM, PSNR, CLIP score, CLIP similarity. Each row is one (prompt, seed) pair, so differences between
variants are PAIRED, and the summary gives mean and 95% bootstrap CI.
Controls (default) put the NVFP4 numbers on a scale:
  bf16+compile        same weights, different kernels: the numerical floor of "same model"
  bf16@seedshift      bf16 with other noise: the ceiling for "a different image of the same quality"
  fp8+compile, stock  the usual cheaper baselines
One variant at a time; pipeline freed in between; host RSS logged after each (bug 27). A failing variant is logged and skipped.
Output: results/eval3/quality/<tag>/rows.jsonl, summary.json, summary.md, images/<variant>/s<seed>_p<i>.png
"""
import argparse
import gc
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import numpy as np
import torch
from PIL import Image

from common import RESULTS, load_prompt_embeds
from eval3.stats import boot_ci, fmt, paired_diff
from quality import eval_images as ev
import variants as vr

SEED_SHIFT = 1000
DEFAULT_CONTROLS = ["bf16+compile", "bf16@seedshift", "fp8+compile", "stock"]


def rss_gb():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 2**30
    except Exception:  # noqa: BLE001
        return float("nan")


def to_u8(im):
    return (np.clip(im, 0, 1) * 255).round().astype("uint8")


def generate(name, size, items, seeds, steps, batch, img_dir):
    """All (seed, prompt) images of one variant, saved to img_dir. Returns {(seed, prompt_idx): float image}, seconds per image."""
    shift = name.endswith("@seedshift")
    base = name.replace("@seedshift", "")
    b = vr.build(base, size)
    chunk = items[:batch]
    b.run(chunk, steps=2, seeds=list(range(len(chunk))), output="latent")  # warm-up: compile, caches
    torch.cuda.synchronize()
    out, t0, n = {}, time.perf_counter(), 0
    img_dir.mkdir(parents=True, exist_ok=True)
    for s in seeds:
        for i in range(0, len(items), batch):
            ch = items[i:i + batch]
            sd = [(s + (SEED_SHIFT if shift else 0)) * 100003 + i + j for j in range(len(ch))]
            imgs = b.run(ch, steps=steps, seeds=sd)["images"]
            for j, im in enumerate(imgs):
                out[(s, i + j)] = im
                Image.fromarray(to_u8(im)).save(img_dir / f"s{s}_p{i + j}.png")
            n += len(imgs)
    torch.cuda.synchronize()
    sec = (time.perf_counter() - t0) / max(n, 1)
    report = b.report
    del b
    gc.collect()
    torch.cuda.empty_cache()
    return out, sec, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--embeds", default="prompts/embeds_showcase.pt")
    ap.add_argument("--n-prompts", type=int, default=32)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--steps", type=int, default=14)
    ap.add_argument("--batch", type=int, default=8, help="prompts per generation batch (memory)")
    ap.add_argument("--reference", default="bf16")
    ap.add_argument("--variants", nargs="*", default=None, help="default: the controls")
    ap.add_argument("--controls", choices=["default", "none"], default="default", help="add the control variants to the list")
    ap.add_argument("--clip", default="openai/clip-vit-large-patch14")
    ap.add_argument("--no-clip", action="store_true")
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()

    items = load_prompt_embeds(a.embeds)[:a.n_prompts]
    prompts = [it.get("prompt", "") for it in items]
    seeds = list(range(a.seeds))
    names = list(a.variants or [])
    if a.controls == "default" or not names:
        names = DEFAULT_CONTROLS + [n for n in names if n not in DEFAULT_CONTROLS]
    out = RESULTS / "eval3" / "quality" / (a.tag or f"{a.size}")
    out.mkdir(parents=True, exist_ok=True)
    clip = None if a.no_clip else ev.Clip(a.clip)
    print(f"{len(items)} prompts x {len(seeds)} seeds, {a.size}px, reference {a.reference}, variants {names}", flush=True)

    ref, ref_sec, _ = generate(a.reference, a.size, items, seeds, a.steps, a.batch, out / "images" / a.reference.replace("+", "_"))
    print(f"reference done, {ref_sec:.2f} s/img, host RSS {rss_gb():.1f} GB", flush=True)
    keys = sorted(ref)
    rows_all, failed, summary = [], [], {}
    rows_f = open(out / "rows.jsonl", "w")
    for name in names:
        if name == a.reference:
            continue
        try:
            g, sec, report = generate(name, a.size, items, seeds, a.steps, a.batch, out / "images" / name.replace("+", "_").replace("@", "_"))
            rows = []
            for s in seeds:  # evaluate per seed so CLIP text features line up with the prompt list
                ks = [k for k in keys if k[0] == s]
                r, _ = ev.evaluate([g[k] for k in ks], [ref[k] for k in ks], prompts[:len(ks)], clip)
                for k, row in zip(ks, r):
                    rows.append(dict(variant=name, seed=k[0], prompt=k[1], **{m: v for m, v in row.items() if m != "prompt"}))
        except (Exception, SystemExit) as e:  # noqa: BLE001
            failed.append((name, f"{type(e).__name__}: {str(e)[:200]}"))
            print(f"{name:50s} FAILED {failed[-1][1]}", flush=True)
            gc.collect()
            torch.cuda.empty_cache()
            continue
        for r in rows:
            rows_f.write(json.dumps(r) + "\n")
        rows_f.flush()
        rows_all += rows
        lp = [r["lpips"] for r in rows]
        summary[name] = dict(sec_per_image=sec, rss_gb=rss_gb(), n=len(rows), lpips=boot_ci(lp),
                             **{m: boot_ci([r[m] for r in rows if m in r]) for m in ("clip_sim", "clip_score", "ssim", "psnr") if any(m in r for r in rows)})
        print(f"{name:50s} LPIPS {fmt(summary[name]['lpips'])}  {sec:.2f} s/img  RSS {rss_gb():.1f} GB", flush=True)
        (out / "summary.json").write_text(json.dumps(dict(summary=summary, failed=failed, args=vars(a)), default=float, indent=1))
    rows_f.close()
    write_md(out, a, summary, failed, rows_all)


def write_md(out, a, summary, failed, rows):
    L = [f"Quality vs `{a.reference}` (same prompt, same seed). {a.size} px, {a.steps} steps, {a.n_prompts} prompts x {a.seeds} seeds. "
         "Mean [95% bootstrap CI over (prompt, seed) pairs].\n",
         "| variant | n | LPIPS | SSIM | CLIP sim | CLIP score | s/img |", "|---|---|---|---|---|---|---|"]
    for n, s in summary.items():
        g = lambda k: fmt(s[k]) if k in s else "-"  # noqa: E731
        L.append(f"| {n} | {s['n']} | {g('lpips')} | {g('ssim')} | {g('clip_sim')} | {g('clip_score')} | {s['sec_per_image']:.2f} |")
    L.append("")
    if "stock" in summary:  # paired differences against stock NVFP4, the question "does this lever cost quality"
        L += ["Paired LPIPS difference to stock NVFP4 distance (variant minus stock, same prompt+seed; negative = closer to bf16 than stock):\n",
              "| variant | diff [95% CI] | significant |", "|---|---|---|"]
        by = {}
        for r in rows:
            by.setdefault(r["variant"], {})[(r["seed"], r["prompt"])] = r["lpips"]
        for n in by:
            if n == "stock" or n not in by or "stock" not in by:
                continue
            ks = sorted(set(by[n]) & set(by["stock"]))
            m, lo, hi = paired_diff([by[n][k] for k in ks], [by["stock"][k] for k in ks])
            L.append(f"| {n} | {m:.3f} [{lo:.3f}, {hi:.3f}] | {'yes' if lo > 0 or hi < 0 else 'no'} |")
        L.append("")
    if failed:
        L += ["Failed variants:", *[f"- {n}: {e}" for n, e in failed]]
    (out / "summary.md").write_text("\n".join(L) + "\n")
    print("wrote", out / "summary.md")


if __name__ == "__main__":
    main()
