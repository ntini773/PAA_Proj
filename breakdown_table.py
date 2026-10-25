"""Turn results/breakdown.jsonl into a markdown table: share of each part of one denoise step.
    python breakdown_table.py [path/to/breakdown.jsonl]
"""
import sys, os, json
HERE = os.path.dirname(os.path.abspath(__file__))
path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results", "breakdown.jsonl")
rows = [json.loads(l) for l in open(path)] if os.path.exists(path) else []
last = {}
for r in rows:                       # keep the latest run per (size, mode, batch)
    last[(r["size"], r["mode"], r["batch"])] = r
parts = ["layernorm", "attn_core", "gemm", "quant_fn", "quant_overhead", "rest"]
names = {"layernorm": "LayerNorm", "attn_core": "Attention core", "gemm": "GEMM",
         "quant_fn": "Quantize kernel", "quant_overhead": "Quantize overhead", "rest": "Rest (modulate, GELU, copies)"}
mode_name = {"bf16": "bf16", "stock": "NVFP4 stock", "fused": "NVFP4 fused"}
print("| image | mode | step ms | " + " | ".join(names[p] + " ms (%)" for p in parts) + " |")
print("|---|---|---|" + "---|" * len(parts))
for (size, mode, b), r in sorted(last.items(), key=lambda kv: (kv[0][0], ["bf16", "stock", "fused"].index(kv[0][1]), kv[0][2])):
    cells = [f"{r['parts_ms'][p]:.1f} ({r['parts_pct'][p]:.0f}%)" for p in parts]
    print(f"| {size}px B{b} | {mode_name[mode]} | {r['step_ms']:.1f} | " + " | ".join(cells) + " |")

print()
print("### The unfused chain: LayerNorm + AdaLN modulate vs quantize, and GPU idle time")
print()
print("| image | mode | step ms | LN + modulate (2 spots/block) ms (%) | quantize total ms (%) | GPU kernels busy ms (%) | idle / launch gaps % |")
print("|---|---|---|---|---|---|---|")
for (size, mode, b), r in sorted(last.items(), key=lambda kv: (kv[0][0], ["bf16", "stock", "fused"].index(kv[0][1]), kv[0][2])):
    q = r["parts_ms"]["quant_fn"] + r["parts_ms"]["quant_overhead"]
    lm = r.get("ln_mod_ms", 0.0)
    busy = r.get("kernel_busy_ms")
    busy_c = f"{busy:.1f} ({r['kernel_busy_pct']:.0f}%)" if busy is not None else "-"
    idle = f"{100 - r['kernel_busy_pct']:.0f}%" if busy is not None else "-"
    print(f"| {size}px B{b} | {mode_name[mode]} | {r['step_ms']:.1f} | {lm:.1f} ({100*lm/r['step_ms']:.0f}%) | "
          f"{q:.1f} ({100*q/r['step_ms']:.0f}%) | {busy_c} | {idle} |")

print()
print("### Per layer type: where the quantize time sits (ms per denoise step)")
print("Layer types per block: self_qkv = 3 calls (Q,K,V quantize the same input), self_out, cross_q, "
      "cross_kv = 2 calls (text input), cross_out, mlp_fc1, mlp_fc2 (4608 wide). other_linear = caption/timestep layers.")
print()
sites_order = ["self_qkv", "self_out", "cross_q", "cross_kv", "cross_out", "mlp_fc1", "mlp_fc2", "other_linear"]
for size in sorted({k[0] for k in last}):
    st, fu = last.get((size, "stock", 1)), last.get((size, "fused", 1))
    if not st or not fu or "sites" not in st or "sites" not in fu:
        continue
    print(f"**{size}px**")
    print()
    print("| layer type | calls/step | stock quantize | stock overhead | stock GEMM | fused quantize | fused overhead | calls fused / step | quantize saved |")
    print("|---|---|---|---|---|---|---|---|---|")
    ts = tf = 0.0
    for site in sites_order:
        a_, b_ = st["sites"].get(site), fu["sites"].get(site)
        if not a_ or not b_:
            continue
        saved = a_["quant_ms"] - b_["quant_ms"]
        print(f"| {site} | {a_['calls_per_step']:.0f} | {a_['quant_ms']:.1f} | {a_['overhead_ms']:.1f} | {a_['gemm_ms']:.1f} | "
              f"{b_['quant_ms']:.1f} | {b_['overhead_ms']:.1f} | {b_['fused_calls_per_step']:.0f} | {saved:.1f} |")
        ts += a_["quant_ms"]; tf += b_["quant_ms"]
    print(f"| **total** | | **{ts:.1f}** | | | **{tf:.1f}** | | | **{ts - tf:.1f}** |")
    print()
    for tag, r in (("stock", st), ("fused", fu)):
        lns = r.get("ln_sites", {})
        parts = ", ".join(f"{k.replace('layernorm@', 'LN ').replace('ln_mod@', 'LN+modulate ')}: {v['ms']:.2f} ms x{v['calls_per_step']:.0f}"
                          for k, v in sorted(lns.items()))
        print(f"- {tag}: {parts}")
    print()
