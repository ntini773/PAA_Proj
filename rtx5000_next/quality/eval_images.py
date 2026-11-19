"""Image-quality metrics for comparing a variant's images with a reference, per prompt.

Metrics (all on float32 HxWx3 images in 0..1)
    lpips       perceptual distance (AlexNet), 0 = identical. Main quality number; used for the quality gate.
    psnr, ssim  pixel and structural similarity to the reference (own torch SSIM, 11x11 Gaussian window)
    mae         mean absolute pixel difference
    clip_score  100 * cosine(CLIP image, CLIP prompt text): does the image still match the prompt (not a distance to the reference)
    clip_sim    cosine(CLIP image, CLIP reference image): semantic closeness to the reference
Why these: with 6 prompts FID is meaningless; per-image distances to the same-seed reference show how far a speed
optimization moves the output, and CLIP score shows whether the prompt is still followed.
"""
import functools

import numpy as np
import torch
import torch.nn.functional as F


@functools.lru_cache(maxsize=1)
def lpips_net():
    import lpips  # first use downloads the AlexNet backbone (prefetch: python -c "import lpips; lpips.LPIPS(net='alex')")
    return lpips.LPIPS(net="alex", verbose=False).to(_dev())


def _dev():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _t(img):
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)[None].float().to(_dev())


def psnr(x, ref):
    mse = float(((x - ref) ** 2).mean())
    return 99.0 if mse == 0 else float(10 * np.log10(1 / mse))


def _gauss(size=11, sigma=1.5):
    g = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(g ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None]


def ssim(x, ref):
    X, Y = _t(x), _t(ref)
    w = _gauss().to(X.device).repeat(3, 1, 1, 1)
    conv = lambda t: F.conv2d(t, w, groups=3)
    mx, my = conv(X), conv(Y)
    sxx, syy, sxy = conv(X * X) - mx ** 2, conv(Y * Y) - my ** 2, conv(X * Y) - mx * my
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx ** 2 + my ** 2 + c1) * (sxx + syy + c2))
    return float(s.mean())


def lpips_dist(x, ref):
    with torch.no_grad():
        return float(lpips_net()(_t(x) * 2 - 1, _t(ref) * 2 - 1))


class Clip:
    """CLIP image/text features. name='random-tiny' builds a random model for tests (image features only)."""

    def __init__(self, name="openai/clip-vit-large-patch14"):
        from transformers import CLIPConfig, CLIPModel, CLIPProcessor
        self.dev = _dev()
        if name == "random-tiny":
            from transformers import CLIPTextConfig, CLIPVisionConfig
            cfg = CLIPConfig.from_text_vision_configs(
                CLIPTextConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2, vocab_size=100, projection_dim=16),
                CLIPVisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2, image_size=224, patch_size=32, projection_dim=16),
                projection_dim=16)
            self.model, self.proc = CLIPModel(cfg).eval().to(self.dev), None
        else:
            self.model = CLIPModel.from_pretrained(name).eval().to(self.dev)
            self.proc = CLIPProcessor.from_pretrained(name)
        self.mean = torch.tensor([0.4815, 0.4578, 0.4082]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.2686, 0.2613, 0.2758]).view(1, 3, 1, 1)

    @torch.no_grad()
    def image_features(self, imgs):
        x = torch.cat([_t(i) for i in imgs]).cpu()
        x = F.interpolate(x, size=224, mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
        x = ((x - self.mean) / self.std).to(self.dev)
        return F.normalize(self.model.get_image_features(pixel_values=x), dim=-1)

    @torch.no_grad()
    def text_features(self, texts):
        t = self.proc(text=texts, return_tensors="pt", padding=True, truncation=True, max_length=77).to(self.dev)
        return F.normalize(self.model.get_text_features(**t), dim=-1)


def evaluate(imgs, ref_imgs, prompts=None, clip=None):
    """Per-prompt metric rows and a mean/std summary. imgs and ref_imgs are lists of HxWx3 float arrays."""
    rows = []
    fi = clip.image_features(imgs) if clip else None
    fr = clip.image_features(ref_imgs) if clip else None
    ft = clip.text_features(prompts) if (clip and prompts and clip.proc is not None) else None
    for i, (x, r) in enumerate(zip(imgs, ref_imgs)):
        row = dict(prompt=i, lpips=lpips_dist(x, r), psnr=psnr(x, r), ssim=ssim(x, r), mae=float(np.abs(x - r).mean()))
        if clip:
            row["clip_sim"] = float((fi[i] * fr[i]).sum())
            if ft is not None:
                row["clip_score"] = 100 * max(float((fi[i] * ft[i]).sum()), 0.0)
        rows.append(row)
    keys = [k for k in rows[0] if k != "prompt"]
    summary = {k: (float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows]))) for k in keys}
    return rows, summary
