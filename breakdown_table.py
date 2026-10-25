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
