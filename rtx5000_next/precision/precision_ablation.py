"""How much error does NVFP4 add, and how much of it comes from the activations?

Variants (all emulated on a bf16 model, same inputs):
  w4a16  NVFP4 weights, activations untouched   <- the "skip activation quantization" ablation
  w16a4  bf16 weights, NVFP4 activations
  w4a4   both (what torchao NVFP4 does)

Part A  per-layer local error: every Linear gets the same input, compare outputs in fp32 (no error build-up).
Part B  whole-transformer output error at several timesteps (error build-up through 28 blocks).
Optional --validate-stock compares the emulated w4a4 against the real torchao NVFP4 model.

    python precision/precision_ablation.py --model pixart_sigma --size 512
"""
import argparse
import collections
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import load_prompt_embeds, write_json
from model_adapters import build
from precision.nvfp4_emulation import nvfp4_fakequant, patch_linears, rel_l2

VARIANTS = {"w4a16": (True, False), "w16a4": (False, True), "w4a4": (True, True)}


def group_of(name):
    return re.sub(r"\.\d+(?=\.|$)", ".*", name)


def set_prompt(r, item):
    """PixArt: CFG batch is [negative, prompt]. item=None keeps the synthetic embeddings."""
    if item is None:
        return
    dev = r.inputs["encoder_hidden_states"].device
    r.inputs["encoder_hidden_states"] = torch.cat([item["ne"], item["pe"]]).to(dev)
    r.inputs["encoder_attention_mask"] = torch.cat([item["nm"], item["pm"]]).to(dev)


def local_errors(r, timesteps, prompts):
    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    shapes = {}

    def make_hook(name):
        def hook(mod, args):
            x = args[0].reshape(-1, args[0].shape[-1])
            if x.shape[-1] % 16:
                return
            xf, wf = x.float(), mod.weight.float()
            b = mod.bias.float() if mod.bias is not None else None
            xq, wq = nvfp4_fakequant(x).float(), nvfp4_fakequant(mod.weight).float()
            y = F.linear(xf, wf, b)
            for key, (xi, wi) in dict(w4a16=(xf, wq), w16a4=(xq, wf), w4a4=(xq, wq)).items():
                acc[name][key].append(rel_l2(F.linear(xi, wi, b), y))
            acc[name]["act_rel"].append(rel_l2(xq, xf))
            acc[name]["w_rel"].append(rel_l2(wq, wf))
            acc[name]["act_max_over_rms"].append(float(xf.abs().max() / xf.pow(2).mean().sqrt().clamp(min=1e-12)))
            shapes[name] = (mod.in_features, mod.out_features, x.shape[0])
        return hook

    hooks = [m.register_forward_pre_hook(make_hook(n)) for n, m in r.transformer.named_modules() if isinstance(m, nn.Linear)]
    with torch.no_grad():
        for item in prompts:
            set_prompt(r, item)
            for t in timesteps:
                r.forward(t)
    for h in hooks:
        h.remove()

    layers = []
    for name, d in acc.items():
        row = {k: sum(v) / len(v) for k, v in d.items()}
        row.update(layer=name, group=group_of(name), in_features=shapes[name][0], out_features=shapes[name][1], rows=shapes[name][2])
        row["act_share_of_w4a4_sq_error"] = row["w16a4"] ** 2 / max(row["w16a4"] ** 2 + row["w4a16"] ** 2, 1e-30)
        layers.append(row)
    return layers


def summarize(layers):
    keys = ["w4a16", "w16a4", "w4a4", "act_rel", "w_rel", "act_max_over_rms", "act_share_of_w4a4_sq_error"]
    by_group = collections.defaultdict(list)
    for l in layers:
        by_group[l["group"]].append(l)
    mean = lambda rows, k: sum(r[k] for r in rows) / len(rows)
    groups = [dict(group=g, layers=len(rs), **{k: mean(rs, k) for k in keys}) for g, rs in by_group.items()]
    overall = {k: mean(layers, k) for k in keys}
    return groups, overall


def output_errors(r, timesteps, prompts, ref, cache_weights):
    """Rows per (prompt, t). ref is keyed (prompt index, t)."""
    res = {}
    for name, (w, a) in VARIANTS.items():
        undo = patch_linears(r.transformer, weights=w, acts=a, cache_weights=cache_weights)
        rows = []
        with torch.no_grad():
            for pi, item in enumerate(prompts):
                set_prompt(r, item)
                for t in timesteps:
                    y = r.forward(t).float()
                    d = y - ref[pi, t]
                    rows.append(dict(prompt=pi, t=t, rel_l2=float(d.norm() / ref[pi, t].norm()), mae=float(d.abs().mean()),
                                     cosine=float(F.cosine_similarity(y.flatten(), ref[pi, t].flatten(), dim=0)),
                                     max_abs=float(d.abs().max())))
        undo()
        res[name] = rows
    return res


def mean_by_t(rows, timesteps, key="rel_l2"):
    return [sum(x[key] for x in rows if x["t"] == t) / max(1, sum(1 for x in rows if x["t"] == t)) for t in timesteps]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="pixart_sigma")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--timesteps", type=int, nargs="*", default=[999, 750, 500, 250, 50])
    ap.add_argument("--no-cache-weights", action="store_true")
    ap.add_argument("--validate-stock", action="store_true")
    ap.add_argument("--embeds", default=None, help="prompts/embeds_pixart.pt (PixArt only). Without it: synthetic embeddings")
    ap.add_argument("--max-prompts", type=int, default=6)
    a = ap.parse_args()

    prompts = load_prompt_embeds(a.embeds)[:a.max_prompts] if a.embeds else [None]
    if a.embeds and a.model != "pixart_sigma":
        sys.exit("--embeds only works for pixart_sigma")
    if not a.embeds:
        print("WARNING: synthetic embeddings, errors are indicative only")

    r = build(a.model, a.size, 1, "bf16")
    ref = {}
    with torch.no_grad():
        for pi, item in enumerate(prompts):
            set_prompt(r, item)
            for t in a.timesteps:
                ref[pi, t] = r.forward(t).float()

    layers = local_errors(r, a.timesteps, prompts)
    groups, overall = summarize(layers)
    out_err = output_errors(r, a.timesteps, prompts, ref, not a.no_cache_weights)

    res = dict(model=a.model, size=a.size, timesteps=a.timesteps, prompts=[p["prompt"] if p else "synthetic" for p in prompts],
              overall_local=overall, groups=groups, layers=layers, output=out_err)

    if a.validate_stock:
        with torch.no_grad():
            emu_undo = patch_linears(r.transformer, True, True, True)
            emu = {}
            for pi, item in enumerate(prompts):
                set_prompt(r, item)
                for t in a.timesteps:
                    emu[pi, t] = r.forward(t).float()
            emu_undo()
        del r
        torch.cuda.empty_cache()
        rs = build(a.model, a.size, 1, "nvfp4")
        stock = {}
        with torch.no_grad():
            for pi, item in enumerate(prompts):
                set_prompt(rs, item)
                for t in a.timesteps:
                    stock[pi, t] = rs.forward(t).float()
        res["validate"] = [dict(prompt=pi, t=t, stock_vs_bf16=rel_l2(stock[pi, t], ref[pi, t]), emulated_vs_bf16=rel_l2(emu[pi, t], ref[pi, t]),
                                stock_vs_emulated=rel_l2(stock[pi, t], emu[pi, t])) for (pi, t) in stock]

    write_json(f"precision/{a.model}_{a.size}.json", res)

    print(f"\nlocal layer error (relative L2 of Linear output, mean over {len(layers)} layers and {len(a.timesteps)} timesteps)")
    print(f"  w4a16 (activations kept) {overall['w4a16']:.4f}   w16a4 {overall['w16a4']:.4f}   w4a4 {overall['w4a4']:.4f}")
    print(f"  activation error share of the w4a4 squared error: {overall['act_share_of_w4a4_sq_error']:.2f}")
    print("\n| group | layers | w4a16 | w16a4 | w4a4 | act max/rms |\n|---|---|---|---|---|---|")
    for g in sorted(groups, key=lambda g: -g["w4a4"]):
        print(f"| {g['group']} | {g['layers']} | {g['w4a16']:.4f} | {g['w16a4']:.4f} | {g['w4a4']:.4f} | {g['act_max_over_rms']:.0f} |")
    print(f"\ntransformer output relative L2 vs bf16 (mean over {len(prompts)} prompts)")
    print("| variant | " + " | ".join(f"t={t}" for t in a.timesteps) + " |\n|---|" + "---|" * len(a.timesteps))
    for v, rows in out_err.items():
        print(f"| {v} | " + " | ".join(f"{x:.4f}" for x in mean_by_t(rows, a.timesteps)) + " |")
    if len(prompts) > 1:
        print("\nw4a4 relative L2 per prompt (mean over timesteps)")
        for pi, p in enumerate(res["prompts"]):
            rows = [x["rel_l2"] for x in out_err["w4a4"] if x["prompt"] == pi]
            print(f"  {sum(rows) / len(rows):.4f}  {p[:60]}")
    if "validate" in res:
        print("\nvalidation against real torchao NVFP4:", json.dumps(res["validate"], indent=0))


if __name__ == "__main__":
    main()
