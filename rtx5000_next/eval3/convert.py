"""Generic NVFP4 conversion of any diffusers transformer: every eligible nn.Linear becomes a FusedNVFP4Linear.

    from eval3.convert import convert_linears
    report = convert_linears(transformer, backend="cuda_swizzled")

Why generic: fused_linear.convert_pixart knows PixArt's block layout (shared Q/K/V, cross K/V). SD3.5 and FLUX have other layouts,
so this converts each linear on its own. That is the same coverage as torchao's stock config (all Linear layers whose K and N are
multiples of 16), so a stock-vs-module comparison on a new model compares the same set of layers.
Not carried over from PixArt: shared Q/K/V quantize, concat, static scales (dropped from the method), selective list.
`skip` is a list of regexes on the module name; a matching layer stays bf16 (used for the sensitivity-based selective variant).
Returns dict(converted=[...], skipped=[(name, reason)], param_fraction, ...). Layer names are the original module names.
"""
import re

import torch.nn as nn

import fused_linear as fl


def convert_linears(transformer, backend="cuda_swizzled", skip=(), min_features=16, fast=None):
    skip_rx = [re.compile(s) for s in skip]
    converted, skipped = [], []
    params_total = sum(m.weight.numel() for m in transformer.modules() if isinstance(m, nn.Linear))
    params_conv = 0
    targets = [(n, m) for n, m in transformer.named_modules() if isinstance(m, nn.Linear)]
    for name, lin in targets:
        k, n_out = lin.in_features, lin.out_features
        if k % fl.BLOCK or n_out % fl.BLOCK:
            skipped.append((name, f"K={k} or N={n_out} not a multiple of {fl.BLOCK}"))
            continue
        if min(k, n_out) < min_features:
            skipped.append((name, "too small"))
            continue
        if any(r.search(name) for r in skip_rx):
            skipped.append((name, "skip list"))
            continue
        parent_name, _, child = name.rpartition(".")
        parent = transformer.get_submodule(parent_name) if parent_name else transformer
        parent._modules[child] = fl.FusedNVFP4Linear([lin], backend=backend, fast=fast)  # works for ModuleList / Sequential indices too
        converted.append(name)
        params_conv += lin.weight.numel()
    return dict(converted=converted, skipped=skipped, n_converted=len(converted), n_skipped=len(skipped),
                param_fraction=params_conv / max(params_total, 1), params_total=params_total)
