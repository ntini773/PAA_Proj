"""NVFP4 quantize-dequantize in plain PyTorch (two-level scaling, 16-value blocks), plus Linear patching.
Used to separate weight error from activation error. Ties round down, so a few codes can differ from torchao.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
E4M3_MAX, E2M1_MAX = 448.0, 6.0
_cache = {}


def _grid(dev):
    if dev not in _cache:
        g = torch.tensor(GRID, device=dev)
        _cache[dev] = (g, (g[1:] + g[:-1]) / 2)
    return _cache[dev]


def nvfp4_fakequant(x, per_tensor_scale=None):
    shp, dt = x.shape, x.dtype
    assert shp[-1] % 16 == 0
    xf = x.float().reshape(-1, 16)
    if per_tensor_scale is None:
        per_tensor_scale = xf.abs().max() / (E4M3_MAX * E2M1_MAX)
    pts = per_tensor_scale.clamp(min=torch.finfo(torch.float32).tiny)
    bs = (xf.abs().amax(1, keepdim=True) / E2M1_MAX / pts).clamp(min=torch.finfo(torch.float8_e4m3fn).tiny, max=E4M3_MAX)
    bs = bs.to(torch.float8_e4m3fn).float() * pts
    grid, mid = _grid(x.device)
    a = (xf / bs).clamp(-E2M1_MAX, E2M1_MAX)
    q = grid[torch.bucketize(a.abs(), mid)] * torch.sign(a)
    return (q * bs).reshape(shp).to(dt)


def rel_l2(a, ref):
    return float((a.float() - ref.float()).norm() / ref.float().norm().clamp(min=1e-12))


def patch_linears(model, weights=False, acts=False, cache_weights=True):
    """Replace nn.Linear.forward with a fake-quantized version. Returns an undo function."""
    wcache, patched = {}, []
    for m in model.modules():
        if not isinstance(m, nn.Linear) or m.in_features % 16:
            continue

        def fwd(x, m=m):
            w = m.weight
            if weights:
                if m not in wcache or not cache_weights:
                    wcache[m] = nvfp4_fakequant(w)
                w = wcache[m]
            if acts:
                x = nvfp4_fakequant(x)
            return F.linear(x, w, m.bias)

        m.forward = fwd
        patched.append(m)

    def undo():
        for m in patched:
            m.__dict__.pop("forward", None)
        wcache.clear()

    return undo
