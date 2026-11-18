"""Build the transformer on the meta device from config.json only (no weights) and list its Linear layers.
Shows which input widths need the fused kernel, how much is Linear vs conv, and the block count.

    python hf/probe_model_arch.py --models dit_xl2 sana_600m --size 512
"""
import argparse
import collections
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from hf_common import CATALOG

ROOT = pathlib.Path(__file__).resolve().parent.parent


def probe(key, size):
    import diffusers
    import torch.nn as nn
    from accelerate import init_empty_weights

    spec = CATALOG[key]
    repo = spec["repos"].get(str(size)) or next(iter(spec["repos"].values()))
    cls = getattr(diffusers, next(c for c in spec["classes"] if hasattr(diffusers, c)))
    cfg = cls.load_config(repo, subfolder="transformer")
    with init_empty_weights():
        model = cls.from_config(cfg)

    total = sum(p.numel() for p in model.parameters())
    shapes = collections.OrderedDict()
    lin_params = 0
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            k = (m.in_features, m.out_features)
            s = shapes.setdefault(k, dict(count=0, example=name))
            s["count"] += 1
            lin_params += m.in_features * m.out_features
    conv_params = sum(p.numel() for m in model.modules() if isinstance(m, (nn.Conv1d, nn.Conv2d)) for p in m.parameters(recurse=False))
    widths = collections.Counter()
    for (i, o), s in shapes.items():
        widths[i] += s["count"]
    blocks = {n: len(getattr(model, n)) for n in ("transformer_blocks", "single_transformer_blocks") if hasattr(model, n)}
    ok_k = sorted(k for k in widths if k % 16 == 0 and k <= 8192)

    print(f"\n== {key} ({repo}) {cls.__name__}")
    print(f"params {total / 1e9:.2f} B | linear {100 * lin_params / total:.0f}% | conv {100 * conv_params / total:.0f}% | blocks {blocks}")
    print("| in x out | layers | example |\n|---|---|---|")
    for (i, o), s in sorted(shapes.items(), key=lambda kv: -kv[1]["count"]):
        print(f"| {i} x {o} | {s['count']} | {s['example']} |")
    print(f"quantizable linear layers: {sum(widths[k] for k in ok_k)} of {sum(widths.values())}")
    print(f"most common input width (hidden): {widths.most_common(1)[0][0]}")
    print(f"width list for the fused kernel: {','.join(map(str, ok_k))}")
    return dict(model=key, repo=repo, params_b=total / 1e9, linear_share=lin_params / total, conv_share=conv_params / total,
                blocks=blocks, widths=dict(widths), ok_k=ok_k, n_linear=sum(widths.values()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["pixart_sigma", "dit_xl2", "sana_600m", "sd35_medium", "flux_schnell"])
    ap.add_argument("--size", type=int, default=512)
    a = ap.parse_args()
    out, fails = [], []
    for k in a.models:
        try:
            out.append(probe(k, a.size))
        except Exception as e:
            print(f"\n== {k}: FAILED {type(e).__name__}: {str(e)[:200]}")
            fails.append(k)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "hf_probe.json").write_text(json.dumps(out, indent=1))
    print(f"\nprobed {len(out)}, failed {fails}")


if __name__ == "__main__":
    main()
