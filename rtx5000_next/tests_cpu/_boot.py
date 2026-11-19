"""Shared helpers for the CPU tests: import path, a tiny PixArt pipeline, and a mini test runner.

The tiny pipeline (2 blocks, hidden 64, 16 tokens) has the same structure as PixArt-Sigma, so code paths
(adaln_single, shared QKV, cross-attention masks, learned-sigma split, CFG batching) are exercised on a CPU in seconds.
"""
import logging
import os
import pathlib
import sys
import traceback

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
logging.getLogger("torch.utils._pytree").setLevel(logging.ERROR)

import torch  # noqa: E402

SIZE = 64          # pixels -> 8x8 latent -> 4x4 = 16 tokens with patch 2
TEXT = 16          # text tokens in the tiny model


def tiny_pipe(layers=2, heads=4, head_dim=16, seed=0, dtype=torch.float32):
    from diffusers import AutoencoderKL, DPMSolverMultistepScheduler, PixArtSigmaPipeline, PixArtTransformer2DModel
    torch.manual_seed(seed)
    tr = PixArtTransformer2DModel(num_attention_heads=heads, attention_head_dim=head_dim, in_channels=4, out_channels=8,
                                  num_layers=layers, cross_attention_dim=heads * head_dim, caption_channels=32,
                                  sample_size=SIZE // 8, patch_size=2)
    vae = AutoencoderKL(in_channels=3, out_channels=3, down_block_types=("DownEncoderBlock2D",) * 4,
                        up_block_types=("UpDecoderBlock2D",) * 4, block_out_channels=(16, 16, 16, 16), layers_per_block=1,
                        latent_channels=4, norm_num_groups=4, sample_size=SIZE)
    pipe = PixArtSigmaPipeline(tokenizer=None, text_encoder=None, vae=vae, transformer=tr,
                               scheduler=DPMSolverMultistepScheduler())
    pipe = pipe.to(dtype=dtype)
    pipe.transformer.eval()
    pipe.vae.eval()
    return pipe


def tiny_items(n=2, seed=0, valid=(5, 7, 3, 6)):
    g = torch.Generator().manual_seed(seed)
    items = []
    for i in range(n):
        pm = torch.zeros(1, TEXT, dtype=torch.long)
        pm[0, :valid[i % len(valid)]] = 1
        nm = torch.zeros(1, TEXT, dtype=torch.long)
        nm[0, :1] = 1
        items.append(dict(pe=torch.randn(1, TEXT, 32, generator=g), pm=pm, ne=torch.randn(1, TEXT, 32, generator=g) * 0.1, nm=nm))
    return items


_results = []


def check(name, cond, detail=""):
    _results.append((name, bool(cond)))
    print(("PASS  " if cond else "FAIL  ") + name + (f"   {detail}" if detail and not cond else ""))
    return bool(cond)


def run(*tests):
    for t in tests:
        try:
            t()
        except Exception:
            _results.append((t.__name__, False))
            print(f"FAIL  {t.__name__} raised:\n{traceback.format_exc()}")
    bad = [n for n, ok in _results if not ok]
    print(f"\n{len(_results) - len(bad)}/{len(_results)} checks passed" + (f"; failed: {bad}" if bad else ""))
    sys.exit(1 if bad else 0)
