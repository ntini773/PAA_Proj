"""T-roof / T-attn: measured GPU time per class for one forward, next to the speed-of-light (SoL) time from the roofs. One run = one variant.

    python eval3/roofline_profile.py --model pixart_sigma --size 1024 --variants bf16+compile stock+compile \
        module+be=cuda_swizzled+concat+selective+trim+xcudnn+sage2+compile
    python eval3/roofline_profile.py --model sd35_medium --size 1024 --variants bf16+compile stock+compile module+compile
Output: results/eval3/roofline/<model>_<size>_<variant>.json  (merge and plot with eval3/roofline_report.py)

Measurement (fixes bug 26): torch.profiler CUDA KERNEL events only (leaf events, no CPU-op parents). Per forward we report
  sum_ms   sum of kernel durations, per class
  busy_ms  union of kernel intervals (never above span_ms; the script asserts busy <= span)
  wall_ms  CUDA-event time of one eager-launched forward (includes launch gaps; CUDA Graph numbers are in step_target.jsonl)
Classes: gemm (fp4 + bf16/fp8 GEMM), attention (flash/cuDNN/efficient/Sage incl. its int8 prep), our_quant (our quantize kernel),
glue (everything else: LayerNorm, modulate, GELU, residual, torchao's own quantize ops, copies).
Workload (analytic FLOPs/bytes) is recorded from a bf16 run of the same model and size (hooks on every nn.Linear and on
F.scaled_dot_product_attention), so the shapes are the real ones for that model, not hard-coded. 'trim' is honoured for PixArt.
SoL per class = max(FLOPs / compute roof, bytes / bandwidth), with the roofs from results/eval3/peaks_sweep.json (best achieved,
per (K, N) for GEMMs and per head dim for attention). Not modeled (marked n/a): fp8 GEMM, Sage attention (int8/fp8 compute), glue.
Whole-forward SoL = gemm + attention + quantize floor (one read of bf16 x, one write of codes + scales; no amax pass).
A class whose measured time is below its SoL is flagged ROOF_VIOLATION: the roof or the workload is wrong, do not quote the efficiency.
"""
import argparse
import collections
import contextlib
import gc
import json
import math
import pathlib
import sys
import traceback

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "analysis"))
import torch
import torch.nn as nn
import torch.nn.functional as F

import fused_linear as fl
from common import RESULTS, write_json
from kernel_classes import classify

FP4_B = 0.5 + 1 / 16  # bytes per element: packed code plus e4m3 block scale
CLASS = {"fp4_gemm": "gemm", "bf16_gemm": "gemm", "attention": "attention", "our_quant": "our_quant"}


def free():
    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------- workload
@contextlib.contextmanager
def record(tr):
    """Record (M, K, N) of every nn.Linear call and (B, H, Lq, Lk, d) of every SDPA call during the with-block."""
    gemms, attns, hooks = [], [], []
    orig = F.scaled_dot_product_attention

    def pre(m, args):
        x = args[0]
        gemms.append((int(x.numel() // x.shape[-1]), int(m.in_features), int(m.out_features)))

    def sdpa(*a, **kw):  # diffusers calls it with query=, key=, value= keywords; PixArt's patch passes them positionally
        q = a[0] if len(a) > 0 else kw["query"]
        k = a[1] if len(a) > 1 else kw["key"]
        attns.append((int(q.shape[0]), int(q.shape[1]), int(q.shape[2]), int(k.shape[2]), int(q.shape[3])))
        return orig(*a, **kw)

    for m in tr.modules():
        if isinstance(m, nn.Linear):
            hooks.append(m.register_forward_pre_hook(pre))
    F.scaled_dot_product_attention = sdpa
    try:
        yield gemms, attns
    finally:
        F.scaled_dot_product_attention = orig
        for h in hooks:
            h.remove()


# ---------------------------------------------------------------- builders
def parse_tokens(variant):
    toks = variant.split("+")
    return toks[0], set(toks[1:])


def build_pixart(variant, size):
    import variants as vr
    from common import load_prompt_embeds
    items = load_prompt_embeds("prompts/embeds_pixart.pt")
    built = vr.build(variant, size)
    return built.pipe.transformer, vr.make_forward(built, items, size, 1), built


def build_other(model, variant, size):
    from eval3.convert import convert_linears
    from model_adapters import build
    base, mods = parse_tokens(variant)
    if base not in ("bf16", "fp8", "stock", "module"):
        raise ValueError(f"{model}: variant base {base} not supported (bf16, fp8, stock, module; modifier compile)")
    unknown = mods - {"compile"} - {m for m in mods if m.startswith("be=")}
    if unknown:
        raise ValueError(f"{model}: modifiers {sorted(unknown)} are PixArt-only; use bf16/fp8/stock/module [+compile]")
    be = next((m[3:] for m in mods if m.startswith("be=")), "cuda_swizzled")
    r = build(model, size, 1, "nvfp4" if base == "stock" else ("fp8" if base == "fp8" else "bf16"))
    if base == "module":
        convert_linears(r.transformer, backend=be)
    if "compile" in mods:
        for b in r.blocks:
            b.compile()
    return r.transformer, r.forward, r


# ---------------------------------------------------------------- profiling
def profile_kernels(fwd, iters):
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fwd()
        torch.cuda.synchronize()
    ks = []
    for e in prof.events():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            ks.append((e.time_range.start, e.time_range.end, e.name))
    if not ks:
        raise RuntimeError("profiler returned no CUDA kernel events")
    ks.sort()
    by_cls, by_cat = collections.defaultdict(float), collections.defaultdict(float)
    for s, e, n in ks:
        cat = classify(n)
        by_cls[CLASS.get(cat, "glue")] += (e - s) / 1e3
        by_cat[cat] += (e - s) / 1e3
    busy, cs, ce = 0.0, ks[0][0], ks[0][1]
    for s, e, _ in ks[1:]:
        if s > ce:
            busy += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    busy += ce - cs
    span = ks[-1][1] - ks[0][0]
    span = max(span, max(e for _, e, _ in ks) - ks[0][0])
    assert busy <= span + 1e-6, f"busy {busy} > span {span}: bad measurement (bug 26 type)"
    top = collections.Counter()
    for s, e, n in ks:
        top[n[:90]] += (e - s) / 1e3
    return dict(sum_ms={k: v / iters for k, v in by_cls.items()}, category_ms={k: v / iters for k, v in by_cat.items()},
                busy_ms=busy / 1e3 / iters, span_ms=span / 1e3 / iters, kernels_per_forward=len(ks) / iters,
                top_kernels=[(n, ms / iters) for n, ms in top.most_common(8)])


def wall_ms(fwd, iters=5):
    ts = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fwd()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return sorted(ts)[len(ts) // 2]


# ---------------------------------------------------------------- speed of light
def load_peaks():
    p = RESULTS / "eval3" / "peaks_sweep.json"
    if not p.exists():
        sys.exit(f"{p} missing: run bash run_eval3.sh --only W104 first")
    return json.loads(p.read_text())


def shape_roof(peaks, k, n, kind):
    rows = [r[f"{kind}_tflops"] for r in peaks["gemm"] if r["k"] == k and r["n"] == n and f"{kind}_tflops" in r]
    glob = peaks["best"][f"{kind}_tflops"]
    return (max(rows), "shape") if rows else (glob, "global")


def sol(peaks, fp4_gemms, bf16_gemms, attns, base, mods, quant_sites):
    """fp4_gemms / bf16_gemms: lists of (M, K, N) as the variant really runs them (concat = one GEMM with the summed N;
    selective layers stay in bf16_gemms). quant_sites: (M, K) once per quantize call (shared Q/K/V = one site)."""
    bw = peaks["best"]["copy_gbs"] * 1e9
    out = dict(modeled={})
    if base in ("bf16", "stock", "module", "w4a16"):
        t_g, fl_g, by_g, src = 0.0, 0.0, 0.0, collections.Counter()
        for kind, gemms in (("fp4", fp4_gemms), ("bf16", bf16_gemms)):
            for (m, k, n), cnt in collections.Counter(gemms).items():
                roof, how = shape_roof(peaks, k, n, kind)
                src[f"{kind}:{how}"] += cnt
                flops = 2 * m * k * n
                byts = (2 * (m * k + k * n + m * n)) if kind == "bf16" else (FP4_B * (m * k + k * n) + 2 * m * n)
                t_g += cnt * max(flops / (roof * 1e12), byts / bw)
                fl_g += cnt * flops
                by_g += cnt * byts
        out["gemm"] = dict(sol_ms=t_g * 1e3, gflop=fl_g / 1e9, gbyte=by_g / 1e9, roof_source=dict(src), kind="mixed" if fp4_gemms and bf16_gemms else ("fp4" if fp4_gemms else "bf16"))
    else:
        out["gemm"] = None  # fp8: no roof measured
    sage = any(t.startswith("sage") for t in mods)
    hd = peaks["best"]["attn_tflops_by_head_dim"]
    if not sage and attns:
        t_a = fl_a = by_a = 0.0
        for (b, h, lq, lk, d), cnt in collections.Counter(attns).items():
            roof = hd.get(str(d)) or hd.get(d) or max(v for v in hd.values() if v)
            flops = 4 * b * h * lq * lk * d
            byts = 2 * b * h * (2 * lq * d + 2 * lk * d)
            t_a += cnt * max(flops / (roof * 1e12), byts / bw)
            fl_a += cnt * flops
            by_a += cnt * byts
        out["attention"] = dict(sol_ms=t_a * 1e3, gflop=fl_a / 1e9, gbyte=by_a / 1e9)
    else:
        out["attention"] = None
    if base in ("stock", "module") and quant_sites:
        qb = sum(cnt * m * k * (2 + FP4_B) for (m, k), cnt in collections.Counter(quant_sites).items())
        out["quant_floor"] = dict(sol_ms=qb / bw * 1e3, gbyte=qb / 1e9, with_amax_pass_ms=(qb + sum(cnt * m * k * 2 for (m, k), cnt in collections.Counter(quant_sites).items())) / bw * 1e3)
    else:
        out["quant_floor"] = None
    return out


# ---------------------------------------------------------------- main
def run_one(model, size, variant, iters):
    base, mods = parse_tokens(variant)
    pixart = model == "pixart_sigma"
    peaks = load_peaks()
    # workload from the bf16 twin of this variant (same trim, no quantization, default attention)
    twin = "bf16" + ("+trim" if "trim" in mods else "")
    with torch.no_grad():
        if pixart:
            tr0, fwd0, _ = build_pixart(twin, size)
        else:
            tr0, fwd0, _ = build_other(model, "bf16", size)
        with record(tr0) as (gemms, attns):
            fwd0()
        del tr0, fwd0
        free()
        quant_sites, fp4_gemms, bf16_gemms = [], [], []
        if base == "module":
            # The module stack changes the GEMM list (concat merges Q/K/V into one GEMM, selective keeps layers in bf16) and the number
            # of quantize calls (shared Q/K/V quantize once). Hooks do not fire inside compiled blocks, so record the workload on an
            # UNCOMPILED copy of the same variant with the default attention (Sage bypasses F.sdpa; attention comes from the twin).
            wl = "+".join(t for t in variant.split("+") if t not in ("compile", "xcudnn", "cudnn", "xefficient") and not t.startswith("sage"))
            tw, fw, bw_ = build_pixart(wl, size) if pixart else build_other(model, wl, size)
            hs = []

            def qh(m, args):
                x = args[0]
                M = int(x.numel() // x.shape[-1])
                quant_sites.append((M, int(m.K)))
                if m.concat:
                    fp4_gemms.append((M, int(m.K), int(sum(m.sizes))))
                else:
                    fp4_gemms.extend((M, int(m.K), int(n)) for n in m.sizes)

            def lh(m, args):
                x = args[0]
                bf16_gemms.append((int(x.numel() // x.shape[-1]), int(m.in_features), int(m.out_features)))
            for m in tw.modules():
                if isinstance(m, fl.FusedNVFP4Linear):
                    hs.append(m.register_forward_pre_hook(qh))
                elif isinstance(m, nn.Linear):
                    hs.append(m.register_forward_pre_hook(lh))
            fw()
            for h in hs:
                h.remove()
            del tw, fw, bw_
            free()
        elif base == "stock":
            fp4_gemms = list(gemms)
            quant_sites = [(m, k) for (m, k, _n) in gemms]
        elif base == "bf16":
            bf16_gemms = list(gemms)
        tr, fwd, built = build_pixart(variant, size) if pixart else build_other(model, variant, size)
        for _ in range(3):  # warm-up: compile, caches
            fwd()
        torch.cuda.synchronize()
        prof = profile_kernels(fwd, iters)
        prof["wall_ms"] = wall_ms(fwd)
    s = sol(peaks, fp4_gemms, bf16_gemms, attns, base, mods, quant_sites)
    flags = []
    cls_sum = prof["sum_ms"]
    for key, cls in (("gemm", "gemm"), ("attention", "attention"), ("quant_floor", "our_quant")):
        if s.get(key) and cls_sum.get(cls, 0) > 0 and cls_sum[cls] < s[key]["sol_ms"] * 0.98:
            flags.append(f"ROOF_VIOLATION {key}: measured {cls_sum[cls]:.2f} ms < SoL {s[key]['sol_ms']:.2f} ms")
    modeled = sum(v["sol_ms"] for v in (s["gemm"], s["attention"], s["quant_floor"]) if v)
    rec = dict(model=model, size=size, variant=variant, base=base, modifiers=sorted(mods), n_gemm_calls=len(gemms), n_attn_calls=len(attns),
               gemm_shapes=[dict(m=m, k=k, n=n, count=c) for (m, k, n), c in collections.Counter(gemms).most_common(12)],
               attn_shapes=[dict(b=b, h=h, lq=lq, lk=lk, d=d, count=c) for (b, h, lq, lk, d), c in collections.Counter(attns).most_common(6)],
               measured=prof, sol=s, sol_modeled_total_ms=modeled,
               sol_covers=[k for k in ("gemm", "attention", "quant_floor") if s.get(k)], flags=flags)
    name = variant.replace("+", "_").replace("=", "")
    write_json(f"eval3/roofline/{model}_{size}_{name}.json", rec)
    sm = prof["sum_ms"]
    print(f"{variant:60s} wall {prof['wall_ms']:7.2f}  busy {prof['busy_ms']:7.2f}  gemm {sm.get('gemm', 0):6.2f}  attn {sm.get('attention', 0):6.2f}  "
          f"quant {sm.get('our_quant', 0):5.2f}  glue {sm.get('glue', 0):6.2f}   SoL(modeled) {modeled:6.2f}  {'; '.join(flags)}", flush=True)
    del tr, fwd, built
    free()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="pixart_sigma")
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--variants", nargs="+", required=True)
    ap.add_argument("--iters", type=int, default=3)
    a = ap.parse_args()
    failed = []
    for v in a.variants:
        try:
            run_one(a.model, a.size, v, a.iters)
        except BaseException as e:  # noqa: BLE001
            failed.append(v)
            print(f"{v}: FAILED {type(e).__name__}: {str(e)[:300]}", flush=True)
            traceback.print_exc()
            free()
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
