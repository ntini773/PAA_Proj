"""T-conv and T-cover: convert a model to NVFP4 two ways and check the module path against torchao. One model at a time.

    python eval3/model_check.py --model sd35_medium --size 1024
    python eval3/model_check.py --model flux_schnell --size 1024
    python eval3/model_check.py --model pixart_sigma --size 512
Steps (each reported; a failed step is recorded and the later steps still run where possible):
  1. build bf16 (transformer only, adapters in model_adapters.py, fixed random inputs), forward -> reference output, finite check
  2. layer table: every nn.Linear with (K, N), count, which qualify (K, N multiples of 16), param share in FP4
  3. our conversion (eval3.convert.convert_linears) on a fresh bf16 copy: forward, relative error to bf16, quantize-op counter
  4. torchao stock NVFP4 (adapters quant="nvfp4"): forward, relative error to bf16
  5. per-layer bit-exactness: for up to 6 distinct (K, N) layers, FusedNVFP4Linear(bf16 layer) vs torchao's quantized layer on the same input
Result: results/eval3/convert/<model>_<size>.json. Pass criterion (tests.json T11/T12): finite outputs, every sampled layer bit-equal,
our relative error to bf16 within 1.25x of stock's. NOT a pass criterion: whole-model equality with stock (bug 19: rounding is discontinuous).
"""
import argparse
import collections
import copy
import gc
import json
import pathlib
import sys
import traceback

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import torch
import torch.nn as nn

import fused_linear as fl
from common import write_json
from eval3.convert import convert_linears
from model_adapters import build


def rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm())


def free():
    gc.collect()
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--size", type=int, required=True)
    ap.add_argument("--backend", default="cuda_swizzled")
    ap.add_argument("--layers", type=int, default=6, help="distinct (K, N) shapes to test bit-exactly")
    a = ap.parse_args()
    res = dict(model=a.model, size=a.size, backend=a.backend, steps={})

    def step(name, fn):
        try:
            res["steps"][name] = dict(ok=True, **(fn() or {}))
        except BaseException as e:  # noqa: BLE001
            res["steps"][name] = dict(ok=False, error=f"{type(e).__name__}: {str(e)[:300]}")
            traceback.print_exc()
            free()
        print(name, json.dumps(res["steps"][name], default=str)[:400], flush=True)

    ctx = {}
    with torch.no_grad():
        def s_bf16():
            r = build(a.model, a.size, 1, "bf16")
            ctx["ref"] = r.forward().float()
            tr = r.transformer
            lins = [(n, m) for n, m in tr.named_modules() if isinstance(m, nn.Linear)]
            shapes = collections.Counter((m.in_features, m.out_features) for _, m in lins)
            qual = [(n, m) for n, m in lins if m.in_features % 16 == 0 and m.out_features % 16 == 0]
            ctx["samples"] = {}
            for n, m in qual:  # first layer of each distinct shape, copied to CPU (a few tens of MB at most)
                key = (m.in_features, m.out_features)
                if key not in ctx["samples"] and len(ctx["samples"]) < a.layers:
                    ctx["samples"][key] = (n, copy.deepcopy(m).cpu())
            ctx["runner_meta"] = dict(tokens=r.tokens, rows=r.rows)
            ctx["tr"] = tr
            ctx["runner"] = r
            pf = sum(m.weight.numel() for _, m in qual) / sum(m.weight.numel() for _, m in lins)
            return dict(finite=bool(torch.isfinite(ctx["ref"]).all()), out_shape=list(ctx["ref"].shape), n_linears=len(lins),
                        n_qualify=len(qual), qualifying_param_fraction=pf, tokens=r.tokens, rows=r.rows,
                        shapes=[dict(k=k, n=n, count=c) for (k, n), c in sorted(shapes.items(), key=lambda kv: -kv[1])])
        step("bf16", s_bf16)
        if not res["steps"]["bf16"]["ok"]:
            write_json(f"eval3/convert/{a.model}_{a.size}.json", res)
            return

        def s_module():
            fl.reset_counts()
            rep = convert_linears(ctx["tr"], backend=a.backend)
            y = ctx["runner"].forward().float()
            ctx["y_mod"] = y
            return dict(finite=bool(torch.isfinite(y).all()), rel_err_vs_bf16=rel(y, ctx["ref"]), n_converted=rep["n_converted"],
                        n_skipped=rep["n_skipped"], skipped=rep["skipped"][:20], param_fraction=rep["param_fraction"],
                        quant_calls=dict(fl.COUNTS))
        step("module", s_module)
        ctx.pop("runner", None)
        ctx.pop("tr", None)
        free()

        def s_stock():
            r = build(a.model, a.size, 1, "nvfp4")
            y = r.forward().float()
            ctx["stock_tr"] = r.transformer
            return dict(finite=bool(torch.isfinite(y).all()), rel_err_vs_bf16=rel(y, ctx["ref"]), rel_diff_module_vs_stock=rel(ctx["y_mod"], y)
                        if "y_mod" in ctx else None)
        step("stock", s_stock)

        def s_layers():
            rows, g = [], torch.Generator("cuda").manual_seed(1)
            for (k, n), (name, lin_cpu) in ctx["samples"].items():
                x = torch.randn(256, k, device="cuda", dtype=torch.bfloat16, generator=g)
                stock = ctx["stock_tr"].get_submodule(name)
                mine = fl.FusedNVFP4Linear([lin_cpu.cuda()], backend=a.backend)
                ys, ym = stock(x), mine(x)
                rows.append(dict(layer=name, k=k, n=n, bit_equal=bool(torch.equal(ys, ym)), max_abs_diff=float((ys.float() - ym.float()).abs().max())))
            return dict(layers=rows, all_bit_equal=all(r["bit_equal"] for r in rows))
        if "stock_tr" in ctx:
            step("layers", s_layers)
    st = res["steps"]
    res["pass"] = bool(all(st.get(k, {}).get("ok") for k in ("bf16", "module", "stock", "layers"))
                       and st["module"]["finite"] and st["stock"]["finite"] and st["layers"]["all_bit_equal"]
                       and st["module"]["rel_err_vs_bf16"] <= 1.25 * st["stock"]["rel_err_vs_bf16"] + 1e-6)
    write_json(f"eval3/convert/{a.model}_{a.size}.json", res)
    print("PASS" if res["pass"] else "NOT PASSED", a.model, a.size)


if __name__ == "__main__":
    main()
