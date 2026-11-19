"""Microbenchmark of attention kernels at the shapes of PixArt-Sigma, SD3.5 Medium and FLUX. Needs an idle GPU, about 1 min.

    python bench/attention_backends.py [--skip-large-ref]

Why: attention is 57% of the compiled PixArt step at 1024 px, and the flash kernel runs with head dim 72 padded to 96.
This shows, per shape, what each backend achieves (TFLOP/s) and how far its output is from an fp32 reference, before any
end-to-end run. Backends: torch SDPA default / flash / cudnn / efficient, and sage2 / sage3 when installed
(SageAttention INT8-FP8 and FP4 kernels); a backend that is missing or rejects a shape is reported with its error.
Shapes (B, H, Nq, Nk, D, masked):
    PixArt self      2, 16, 1024|4096, same, 72      PixArt cross  2, 16, 4096, 300 and 64, 72, masked (padding mask)
    SD3.5 Medium     2, 24, 4429, 4429, 64 (joint image+text)           FLUX  2, 24, 4608, 4608, 128
Output: results/attention_backends.jsonl and results/attention_backends.md.
"""
import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import torch

import attention_patch as ap
from common import RESULTS, cuda_ms, stats

SHAPES = [("pixart_self_512", 2, 16, 1024, 1024, 72, False), ("pixart_self_1024", 2, 16, 4096, 4096, 72, False),
          ("pixart_cross_T300", 2, 16, 4096, 300, 72, True), ("pixart_cross_T64", 2, 16, 4096, 64, 72, True),
          ("sd35_medium_joint", 2, 24, 4429, 4429, 64, False), ("flux_joint", 2, 24, 4608, 4608, 128, False)]
BACKENDS = ["default", "flash", "cudnn", "efficient", "sage2", "sage3"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--skip-large-ref", action="store_true", help="skip the fp32 reference for the joint shapes (memory)")
    p.add_argument("--iters", type=int, default=30)
    a = p.parse_args()
    out, lines = [], ["| shape | backend | ms | TFLOP/s | rel L2 vs fp32 | note |", "|---|---|---|---|---|---|"]
    g = torch.Generator("cuda").manual_seed(0)
    for name, B, H, Nq, Nk, D, masked in SHAPES:
        q, k, v = (torch.randn(B, H, n, D, device="cuda", dtype=torch.bfloat16, generator=g) for n in (Nq, Nk, Nk))
        mask = None
        if masked:
            valid = torch.tensor([Nk // 6, 2], device="cuda")  # a cond row with ~1/6 valid tokens and an uncond row with 2
            mask = torch.where(torch.arange(Nk, device="cuda")[None, None, None, :] >= valid[:, None, None, None], -10000.0, 0.0).to(torch.bfloat16)
        ref = None
        if not (a.skip_large_ref and Nq > 4200):
            ref = torch.nn.functional.scaled_dot_product_attention(q.float(), k.float(), v.float(), attn_mask=None if mask is None else mask.float())
        flops = 4 * B * H * Nq * Nk * D
        for be in BACKENDS:
            cfg = ap.AttnConfig(backend=be, sage_on_cross=False)
            rec = dict(shape=name, backend=be)
            try:
                fn = lambda: ap.sdpa(q, k, v, mask, cfg, not masked)
                y = fn()
                t = stats(cuda_ms(fn, iters=a.iters, warmup=5))["median"]
                rec.update(ms=t, tflops=flops / t / 1e9, rel_l2=None if ref is None else float((y.float() - ref).norm() / ref.norm()))
                note = "mask: torch fallback" if (masked and be.startswith("sage")) else ""
            except Exception as e:  # missing package, unsupported head dim or mask
                rec["error"] = f"{type(e).__name__}: {str(e)[:120]}"
                note = rec["error"]
            out.append(rec)
            lines.append(f"| {name} | {be} | {rec.get('ms', float('nan')):.3f} | {rec.get('tflops', float('nan')):.0f} | "
                         f"{'-' if rec.get('rel_l2') is None else format(rec['rel_l2'], '.4f')} | {note} |")
        del q, k, v, ref
        torch.cuda.empty_cache()
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "attention_backends.jsonl").write_text("\n".join(json.dumps(r) for r in out) + "\n")
    (RESULTS / "attention_backends.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
