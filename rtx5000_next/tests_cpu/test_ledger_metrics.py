"""CPU tests for the traffic ledger (hand-computed bytes) and the image metrics (sanity of LPIPS, SSIM, PSNR, CLIP features)."""
import _boot
from _boot import check, torch

import numpy as np

import analysis.traffic_ledger as tl
from quality import eval_images as ev


def test_ledger_matches_hand_computation():
    size, batch = 512, 2
    n = (size // 16) ** 2
    M, T = batch * n, batch * tl.TEXT
    D, FF, L, BF = tl.D, tl.FF, tl.LAYERS, tl.BF
    cur, qc, _, _ = tl.ledger(size, batch)
    # x read for the amax pass: qkv 3x, attn1 out, attn2 q, attn2 kv 2x (text rows), attn2 out, fc1 (K=1152), fc2 (K=4608)
    hand = L * BF * (3 * M * D + M * D + M * D + 2 * T * D + M * D + M * D + M * FF)
    check("ledger: amax-pass bytes equal the hand formula", abs(cur["x_read_amax"] - hand) < 1, f"{cur['x_read_amax']} vs {hand}")
    check("ledger: quantize calls 10 per block now, 7 with shared Q/K/V and K/V", qc == {"current": 10 * L, "shared": 7 * L}, str(qc))


def test_metrics_sanity():
    rng = np.random.default_rng(0)
    img = rng.random((64, 64, 3)).astype("float32")
    check("psnr of identical images is capped at 99", ev.psnr(img, img) == 99.0)
    check("ssim of identical images is 1", abs(ev.ssim(img, img) - 1) < 1e-4)
    noisy1 = np.clip(img + rng.normal(0, 0.05, img.shape), 0, 1).astype("float32")
    noisy2 = np.clip(img + rng.normal(0, 0.3, img.shape), 0, 1).astype("float32")
    check("ssim falls as noise grows", ev.ssim(img, noisy1) > ev.ssim(img, noisy2))
    check("psnr falls as noise grows", ev.psnr(img, noisy1) > ev.psnr(img, noisy2))
    try:
        d0, d1, d2 = (ev.lpips_dist(img, x) for x in (img, noisy1, noisy2))
        check("lpips: identical is 0 and grows with noise", d0 < 1e-4 and d1 < d2, f"{d0:.4f} {d1:.4f} {d2:.4f}")
    except Exception as e:  # AlexNet backbone download needs internet the first time
        print(f"SKIP  lpips ({type(e).__name__}: {str(e)[:80]})")
    clip = ev.Clip("random-tiny")
    f = clip.image_features([img, img, noisy2])
    check("clip features are unit length", torch.allclose(f.norm(dim=-1), torch.ones(3), atol=1e-4))
    check("clip: identical images have similarity 1", float((f[0] * f[1]).sum()) > 0.9999)


if __name__ == "__main__":
    _boot.run(test_ledger_matches_hand_computation, test_metrics_sanity)
