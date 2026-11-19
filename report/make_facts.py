"""Single source of truth for the eval2 report: read the rsynced server results once and write report/build/facts.json.

    python report/make_facts.py --data <dir that holds results/ logs/ from results_eval2.tgz> [--hist <dir with first-run E11/E12 logs>]

Every number that appears in the PDF comes from facts.json (the text is generated from it), so the same quantity cannot appear
with two different values. Rules for choosing a value when a quantity was measured more than once are fixed here, in one place:

  step time      CUDA Graph median ms, only runs with >= 20 timed iterations (the 5-iteration rows are nsys captures and are not used).
                 Headline value = the dedicated final run (20 iterations, run_final30.sh) if the config has one, otherwise the single
                 30-iteration run. All >= 20-iteration runs of a config are kept as `runs` to show repeatability.
  kernel classes nsys 5-iteration captures, classified with rtx5000_next/analysis/kernel_classes.py (with the quant_kernel fix).
  quality        results/quality_final (the 6 headline variants, all sizes); the longer variant list at 512 and 1024 px comes from the
                 first E11 run's log (hist/), cross-checked against the later partial E11 summary.json files.
  ncu            re-exported here from the saved .ncu-rep files with the correct GB202 metric names.
"""
import argparse
import collections
import csv
import json
import pathlib
import re
import statistics
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
NEXT = REPO / "rtx5000_next"
sys.path.insert(0, str(NEXT / "analysis"))
sys.path.insert(0, str(NEXT))
from kernel_classes import QUANT_PATH, classify  # noqa: E402

SIZES = (256, 512, 1024)
NSYS_ITERS = 5
BEST = "module+be=cuda_swizzled+static+concat"
# label -> names used in step_target.jsonl (variant string for eval2 runs, config name for round-1 runs)
STEP_KEYS = {
    "bf16_eager": ["bf16"], "bf16_compile": ["bf16+compile"], "bf16_trim_compile": ["bf16+trim+compile"],
    "fp8_eager": ["fp8"], "fp8_compile": ["fp8+compile"],
    "stock_eager": ["stock"], "stock_compile": ["stock_compile"],
    "old_fused_triton_narrow": ["fused_triton_narrow"], "old_fused_triton": ["fused_triton"], "old_fused_cuda_row": ["fused_cuda_row"],
    "old_fused_cuda_swizzled": ["fused_cuda_swizzled"], "old_fused_triton_op_compile": ["fused_triton_op_compile"],
    "old_fused_cuda_swizzled_compile": ["fused_cuda_swizzled_compile"], "free_bound": ["free_bound"],
    "module_triton_eager": ["module+be=triton"], "module_triton_compile": ["module+be=triton+compile"],
    "module_triton_noshare": ["module+be=triton+noshare+compile"], "module_triton_static": ["module+be=triton+static+compile"],
    "module_triton_concat": ["module+be=triton+concat+compile"], "module_triton_selective": ["module+be=triton+selective+compile"],
    "module_cuda": ["module+be=cuda_swizzled+compile"], "module_best": [BEST + "+compile"],
    "module_best_trim": [BEST + "+trim+compile"], "module_best_xcudnn": [BEST + "+trim+xcudnn+compile"],
    "module_best_cudnn_self": [BEST + "+trim+cudnn+xcudnn+compile"], "module_best_kvds2": [BEST + "+trim+kvds2+compile"],
    "module_best_selective": [BEST + "+selective+trim+xcudnn+compile"],
}


def step_facts(res):
    rows = [json.loads(l) for l in (res / "step_target.jsonl").read_text().splitlines() if l.strip()]
    rows = [r for r in rows if "layer" not in r and r["gpu_ms"]["n"] >= 20 and "graph_ms" in r]
    out = {}
    for label, names in STEP_KEYS.items():
        for size in SIZES:
            cand = [r for r in rows if r["size"] == size and (r.get("variant") or r["config"]) in names]
            if not cand:
                continue
            final = [r for r in cand if r["gpu_ms"]["n"] == 20]
            head = final[-1] if final else cand[-1]
            out[f"{label}|{size}"] = dict(graph=head["graph_ms"]["median"], eager=head["gpu_ms"]["median"], cpu_launch=head["cpu_launch_ms"],
                                          iters=head["gpu_ms"]["n"], runs=[r["graph_ms"]["median"] for r in cand],
                                          peak_mem_gb=head.get("peak_mem_gb"))
    return out


def nsys_facts(res):
    out = {}
    for rep in sorted((res / "nsys").glob("*_cuda_gpu_kern_sum.csv")):
        name = rep.name[: -len("_cuda_gpu_kern_sum.csv")]
        cfg, size = name.rsplit("_", 1)
        # nsys_step.sh always captures 5 forwards. results/meta/<name>.json cannot be used for this: run_step_matrix.sh writes a
        # meta file with the same name (20 iterations) after the nsys run, so it holds the wrong iteration count.
        iters = NSYS_ITERS
        lines = rep.read_text(errors="replace").splitlines()
        i = next(k for k, l in enumerate(lines) if "Time (%)" in l)
        cls_ms, cls_n, tops = collections.defaultdict(float), collections.defaultdict(float), []
        for r in csv.DictReader(lines[i:]):
            c = classify(r["Name"])
            ms = float(r["Total Time (ns)"].replace(",", "")) / 1e6 / iters
            n = float(r["Instances"].replace(",", "")) / iters
            cls_ms[c] += ms
            cls_n[c] += n
            tops.append((ms, n, c, r["Name"]))
        tops.sort(reverse=True)
        out[name] = dict(config=cfg, size=int(size), iters=iters, class_ms=dict(cls_ms), class_n=dict(cls_n), total_ms=sum(cls_ms.values()),
                         kernels=sum(cls_n.values()), top=[dict(ms=a, calls=b, cls=c, name=d[:110]) for a, b, c, d in tops[:8]])
    return out


def ncu_facts(res):
    metrics = ("gpu__time_duration.sum,dram__bytes_op_read.sum,dram__bytes_op_write.sum,lts__t_bytes.sum,lts__t_sector_hit_rate.pct,"
               "l1tex__t_sector_hit_rate.pct,dram__throughput.avg.pct_of_peak_sustained_elapsed")
    out = {}
    unit = {"ns": 1e-3, "us": 1.0, "ms": 1e3, "s": 1e6}
    for rep in sorted((res / "ncu").glob("*.ncu-rep")):
        key = rep.stem
        cfg, layer, size = key.rsplit("_", 2)
        txt = subprocess.run(["ncu", "-i", str(rep), "--csv", "--page", "raw", "--metrics", metrics], capture_output=True, text=True).stdout
        lines = txt.splitlines()
        i = next((k for k, l in enumerate(lines) if l.startswith('"ID"')), None)
        if i is None:
            continue
        rd = list(csv.reader(lines[i:]))
        head, units, data = rd[0], rd[1], rd[2:]
        meta = json.loads((res / "meta" / f"{cfg}_{layer}_{size}.json").read_text())
        shp = meta["layer"]["input_shape"]
        M = 1
        for s in shp[:-1]:
            M *= s
        K = shp[-1]
        ideal = M * K * (2 + 0.5 + 1 / 16)
        q_us = q_dram = q_l2 = g_us = 0.0
        qn = 0
        l2w = l1w = tw = 0.0
        for r in data:
            d = dict(zip(head, r))
            if not d.get("ID", "").strip().isdigit():
                continue
            u = dict(zip(head, units))
            t = float(d["gpu__time_duration.sum"].replace(",", "")) * unit[u["gpu__time_duration.sum"]]
            rb = float(d["dram__bytes_op_read.sum"].replace(",", ""))
            wb = float(d["dram__bytes_op_write.sum"].replace(",", ""))
            scale = {"byte": 1, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9}
            dram = rb * scale[u["dram__bytes_op_read.sum"]] + wb * scale[u["dram__bytes_op_write.sum"]]
            l2 = float(d["lts__t_bytes.sum"].replace(",", "")) * scale[u["lts__t_bytes.sum"]]
            if classify(d["Kernel Name"]) in QUANT_PATH:
                q_us += t
                q_dram += dram
                q_l2 += l2
                qn += 1
                l2w += float(d["lts__t_sector_hit_rate.pct"]) * t
                tw += t
            else:
                g_us += t
        out[key] = dict(config=cfg, layer=layer, size=int(size), rows=M, k=K, ideal_mb=ideal / 1e6, quant_kernels=qn, quant_us=q_us,
                        quant_dram_mb=q_dram / 1e6, quant_l2_mb=q_l2 / 1e6, quant_l2_hit_pct=l2w / tw if tw else None, gemm_us=g_us)
    return out


def quality_facts(res, hist):
    final = {}
    for s in SIZES:
        p = res / "quality_final" / str(s) / "summary.json"
        if p.exists():
            final[s] = {r["name"]: dict(lpips=r["summary"]["lpips"], ssim=r["summary"]["ssim"], psnr=r["summary"]["psnr"],
                                        clip=r["summary"].get("clip_score"), clip_sim=r["summary"].get("clip_sim"),
                                        vs_stock=r["vs_extra"]["lpips"] if "vs_extra" in r else None,
                                        per_prompt_lpips=[x["lpips"] for x in r["per_prompt"]]) for r in json.loads(p.read_text())}
    ext = {}
    pat = re.compile(r"^(\S+)\s+([\d.]+) ms/img\s+LPIPS ([\d.]+)\s+SSIM ([\d.]+)(?:\s+CLIP ([\d.]+))?")
    for s in (512, 1024):
        f = hist / f"E11_{s}_first.log"
        d = {}
        for line in f.read_text(errors="replace").splitlines():
            m = pat.match(line.strip())
            if m:
                d[m.group(1)] = dict(ms_per_image=float(m.group(2)), lpips=float(m.group(3)), ssim=float(m.group(4)),
                                     clip=float(m.group(5)) if m.group(5) else None)
        ext[s] = d
    later = {}
    for s in (512, 1024):
        p = res / "quality" / str(s) / "summary.json"
        if p.exists():
            later[s] = {r["name"]: dict(lpips=r["summary"]["lpips"][0], ssim=r["summary"]["ssim"][0],
                                        clip=r["summary"].get("clip_score", [None])[0]) for r in json.loads(p.read_text())}
    return final, ext, later


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--hist", required=True)
    a = ap.parse_args()
    data, hist = pathlib.Path(a.data), pathlib.Path(a.hist)
    res = data / "results"
    F = {}
    F["step"] = step_facts(res)
    F["nsys"] = nsys_facts(res)
    F["ncu"] = ncu_facts(res)
    F["quality_final"], F["quality_ext"], F["quality_later"] = quality_facts(res, hist)
    F["peaks"] = json.loads((res / "peaks.json").read_text())
    F["real_w4a16"] = {s: json.loads((res / "precision" / f"real_w4a16_vs_w4a4_{s}.json").read_text()) for s in (512, 1024)}
    F["fp4_gemm"] = [json.loads(l) for l in (res / "fp4_gemm_shapes.jsonl").read_text().splitlines() if l.strip()]
    F["attn_run1"] = [json.loads(l) for l in (res / "attention_backends.jsonl").read_text().splitlines() if l.strip()]
    F["hf_check"] = json.loads((res / "hf_check.json").read_text())
    F["dfa"] = {}
    for s in SIZES:
        for d in ("0.05", "0.1", "0.2"):
            p = json.loads((res / "dfa_plans" / f"pixart_{s}_d{d}.json").read_text())
            F["dfa"][f"{s}|{d}"] = dict(counts=p["counts"], cost=p["modeled_attention_cost"], plan=p["plan"], steps=p["steps"], layers=p["layers"])
    F["static"] = {}
    for s in SIZES:
        p = json.loads((res / f"static_scales_{s}.json").read_text())
        F["static"][s] = dict(amax=p["amax"], drift=p["drift"], per_step=p["per_step"], steps=p["steps"])
    F["cuda_vs_triton"] = [json.loads(l) for l in (res / "cuda_vs_triton.jsonl").read_text().splitlines() if l.strip()]
    smoke = (data / "logs" / "smoke" / "gpu.log").read_text(errors="replace")
    m = re.search(r"mean rel L2 vs bf16: (\{.*?\})", smoke)
    F["smoke_rel_l2"] = eval(m.group(1)) if m else None   # server GPU smoke test, 256 px, {'stock','module+be=triton','compiled'}
    m = re.search(r"rel L2 vs bf16: w4a16 ([\d.]+)\s+w4a4 ([\d.]+)", smoke)
    F["smoke_w4a16"] = dict(w4a16=float(m.group(1)), w4a4=float(m.group(2))) if m else None
    F["smoke_layer_exact"] = "layer level: module output is bit-identical to torchao NVFP4" in smoke and "PASS  layer level" in smoke
    e12 = [l for l in (hist / "E12_first.log").read_text().splitlines() if l.startswith("| bf16")]
    F["e2e_first"] = [[c.strip() for c in l.strip("| ").split("|")] for l in e12]
    out = HERE / "build"
    out.mkdir(exist_ok=True)
    (out / "facts.json").write_text(json.dumps(F, default=float))
    print("wrote", out / "facts.json", {k: (len(v) if hasattr(v, "__len__") else v) for k, v in F.items()})


if __name__ == "__main__":
    main()
