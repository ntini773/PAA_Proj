"""Stock vs fused output equality per size (0.0 = identical images)."""
import glob, os, numpy as np
d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
for f in sorted(glob.glob(d + "/stock_*.npy")):
    g = f.replace("/stock_", "/fused_")
    if not os.path.exists(g): continue
    a, b = np.load(f).astype(np.float32), np.load(g).astype(np.float32)
    print(f"{os.path.basename(f)[6:-4]:14s} mean|diff|={np.abs(a-b).mean():.6f} max|diff|={np.abs(a-b).max():.6f}")
