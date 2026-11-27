"""Small statistics helpers for the quality tests: paired differences with a bootstrap confidence interval."""
import numpy as np


def boot_ci(x, n_boot=5000, alpha=0.05, seed=0):
    """Mean and (1-alpha) bootstrap CI of a 1-D sample."""
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return float(x.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_diff(a, b, **kw):
    """Mean of (a - b) over the same (prompt, seed) items, with CI. Negative = a is lower (better for LPIPS)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    assert a.shape == b.shape, "paired samples need the same items"
    return boot_ci(a - b, **kw)


def fmt(ci, digits=3):
    m, lo, hi = ci
    return f"{m:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"
