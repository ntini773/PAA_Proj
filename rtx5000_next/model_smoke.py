"""Does the transformer load, quantize and run a forward? Run once per new model before any profiling.

    python model_smoke.py --model dit_xl2 --size 512
"""
import argparse
import json

import torch

import quant_backends as qb
from common import append_jsonl
from model_adapters import build, linear_widths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--size", type=int, required=True)
    ap.add_argument("--no-nvfp4", action="store_true")
    a = ap.parse_args()

    res = dict(model=a.model, size=a.size)
    with torch.no_grad():
        r = build(a.model, a.size, 1, "bf16")
        ref = r.forward().float()
        res.update(bf16_ok=bool(torch.isfinite(ref).all()), out_shape=list(ref.shape), tokens=r.tokens,
                   widths=dict(linear_widths(r.transformer)))
        del r
        torch.cuda.empty_cache()
        if not a.no_nvfp4:
            r = build(a.model, a.size, 1, "nvfp4")
            y = r.forward().float()
            res["stock_ok"] = bool(torch.isfinite(y).all())
            res["stock_rel_err_vs_bf16"] = float((y - ref).norm() / ref.norm())
            ok_k = {k for k in linear_widths(r.transformer) if k % 16 == 0}
            c = qb.install("triton", ok_k)
            y2 = r.forward().float()
            res["fused_matches_stock"] = bool(torch.equal(y, y2))
            res["fused_calls"], res["fallback_calls"] = c["n"], c["fallback"]
    print(json.dumps(res, indent=1))
    append_jsonl("model_smoke.jsonl", res)


if __name__ == "__main__":
    main()
