"""How far do the other precisions drift from bf16? (same noise, same synthetic embeddings; no GPU needed)
Reads results/<mode>_<size>_b1.npy written by scale_e2e.py. Numerical drift, not a perceptual-quality score.
    python quality_compare.py"""
import os, numpy as np
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
print("| image | mode | MAE vs bf16 | PSNR vs bf16 (dB) |\n|---|---|---|---|")
for size in (256, 512, 1024):
    ref = os.path.join(D, f"bf16_{size}_b1.npy")
    if not os.path.exists(ref):
        continue
    r = np.load(ref).astype("f4")
    for mode in ("fp16", "fp8", "stock", "fused"):
        f = os.path.join(D, f"{mode}_{size}_b1.npy")
        if not os.path.exists(f):
            continue
        x = np.load(f).astype("f4")
        mse = float(((x - r) ** 2).mean())
        print(f"| {size}px | {mode} | {abs(x - r).mean():.4f} | {'inf' if mse == 0 else f'{10 * np.log10(1.0 / mse):.1f}'} |")
