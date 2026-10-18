"""Collect everything in results/ into results/REPORT.md (tables + raw logs).
    python make_report.py          (after run_all.sh and run_inspired.sh)
"""
import os, json, csv
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
def rd(n):
    p = os.path.join(R, n); return open(p).read().strip() if os.path.exists(p) else "(not run)"
def jl(n):
    p = os.path.join(R, n)
    return [json.loads(l) for l in open(p) if l.startswith("{")] if os.path.exists(p) else []
out = ["# Results (auto-generated)\n"]

e2e = [r for r in jl("e2e.jsonl") if r["batch"] == 1 and not r["compile"]]
out.append("## 1. Full model: time per denoise step (GPU ms, CFG batch 2 inside), batch 1\n")
out.append("| image | tokens | bf16 | NVFP4 stock | NVFP4 fused | fused vs stock | stock vs bf16 | peak GB (fused) |\n|---|---|---|---|---|---|---|---|")
by = {}
for r in e2e: by[(r["size"], r["mode"])] = r
for s in sorted({k[0] for k in by}):
    g = lambda m: by.get((s, m))
    b, st, fu = g("bf16"), g("stock"), g("fused")
    f = lambda r: f"{r['step_gpu_ms_median']:.1f}" if r else "-"
    sp = f"{st['step_gpu_ms_median']/fu['step_gpu_ms_median']:.2f}x" if st and fu else "-"
    sb = f"{b['step_gpu_ms_median']/st['step_gpu_ms_median']:.2f}x" if b and st else "-"
    out.append(f"| {s}px | {(s//16)**2} | {f(b)} | {f(st)} | {f(fu)} | {sp} | {sb} | {fu['peak_mem_gb']:.1f} |" if fu else f"| {s}px | | {f(b)} | {f(st)} | - | {sp} | {sb} | |")
out.append("\n('x' > 1 means the first is faster. stock vs bf16 > 1 means NVFP4 beats bf16.)\n")
out.append("## 2. Image equality stock vs fused\n```\n" + rd("images.log") + "\n```\n")
out.append("## 3. Boundary microbenchmark (Triton), ms per call\n```\n" + rd("tier1.log") + "\n```\n")
out.append("## 4. ncu: GPU time + DRAM per boundary\n```\n" + rd("ncu_summary.log") + "\n```\n")
out.append("## 5. nsys: where the GPU time goes\n```\n" + rd("nsys_summary.log") + "\n```\n")
out.append("## 6. CUDA C++: fusion vs launch cost\n```\n" + rd("launch_vs_fusion.log") + "\n```\n")
out.append("## 7. Real model, eager vs CUDA Graph\n")
out.append("| image | mode | eager GPU ms | eager CPU launch ms | graph GPU ms | graph speedup | same output |\n|---|---|---|---|---|---|---|")
for r in jl("graph_step.jsonl"):
    out.append(f"| {r['size']} | {r['mode']} | {r['eager_gpu_ms']:.1f} | {r['eager_cpu_launch_ms']:.1f} | "
               f"{r.get('graph_gpu_ms', float('nan')):.1f} | {r.get('graph_speedup', float('nan')):.2f}x | "
               f"{r.get('graph_matches_eager', r.get('graph_error'))} |")
out.append("\n## 8. Extra runs (batch / torch.compile)\n```\n" + "\n".join(
    json.dumps(r) for r in jl("e2e.jsonl") if r["batch"] != 1 or r["compile"]) + "\n```\n")
open(os.path.join(R, "REPORT.md"), "w").write("\n".join(out))
print("wrote results/REPORT.md")
