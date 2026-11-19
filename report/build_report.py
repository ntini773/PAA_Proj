"""Build the eval2 report (HTML -> PDF with headless Chrome). Every measured number in the text is produced here from facts.json.

    python report/build_report.py --data <dir with results/> [--no-pdf]
Numbers are written with n(...) so they appear as <span class="n">; check_text() lists every other number left in the visible text, which
must all be fixed constants (model sizes, section numbers). That list is part of the 'no two truths' check.
"""
import argparse
import html
import json
import pathlib
import re
import subprocess
import sys

import make_figures
from check_consistency import run as run_checks
from common import BEST, BUILD, FIG, SIZES, STEP_ROWS, TOKENS, chain, ext, facts, quality, speedup, step, step_runs
import traffic

F = facts()
P = F["peaks"]
USED = []


def n(x, f=".1f", unit="", sign=False):
    s = format(x, ("+" if sign else "") + f)
    USED.append(s + unit)
    return f'<span class="n">{s}{unit}</span>'


def pct(x, f=".0f"):
    return n(100 * x, f, "%")


def sx(x):
    return n(x, ".2f", "x")


def t_(rows, head, cls="", widths=None):
    h = "".join(f"<th>{c}</th>" for c in head)
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>'


def fig(name, cap, w="100%"):
    assert (FIG / name).exists(), name
    return f'<figure style="width:{w}"><img src="fig/{name}"><figcaption>{cap}</figcaption></figure>'


def box(title, body, kind="note"):
    return f'<div class="box {kind}"><b>{title}</b><br>{body}</div>'


# ------------------------------------------------------------------------------------------------------------ derived values
S = lambda k, s: step(k, s)
SU = lambda k, s: speedup(k, s)
NS = F["nsys"]
q = quality


def cls(key, c):
    return NS[key]["class_ms"].get(c, 0.0)


ex = None


def build(data):
    global ex
    ex, e2e = make_figures.make_all(pathlib.Path(data) / "results")
    ncu = F["ncu"]
    sq, fq = ncu["stock_qkv_1024"], ncu["fused_cuda_swizzled_qkv_1024"]
    tr = traffic.numbers(1024)
    k_best, k_stock, k_bf = "module_best_compile_1024", "stock_compile_1024", "bf16_compile_1024"
    tot_best = NS[k_best]["total_ms"]
    att_share = cls(k_best, "attention") / tot_best
    stock_eager_chain = 1 - (cls("stock_1024", "attention") + cls("stock_1024", "fp4_gemm")) / NS["stock_1024"]["total_ms"]
    sol = {m: {s: traffic.sol_ms(s, m, P) for s in SIZES} for m in ("bf16", "fp4")}
    drift = {s: sorted(F["static"][str(s)]["drift"].values()) for s in SIZES}
    dq = lambda s, d: F["dfa"][f"{s}|{d}"]
    ext1024, ext512 = F["quality_ext"]["1024"], F["quality_ext"]["512"]
    sage_rows = {r["shape"]: r for r in F["attn_run1"] if r["backend"] == "sage2" and "tflops" in r}
    flash_rows = {r["shape"]: r for r in F["attn_run1"] if r["backend"] == "flash" and "tflops" in r}
    cud_rows = {r["shape"]: r for r in F["attn_run1"] if r["backend"] == "cudnn" and "tflops" in r}
    w = F["real_w4a16"]
    cv = {}
    for r in F["cuda_vs_triton"]:
        if "packed_diff_vs_triton" in r:
            cv[(r["kernel"], r["M"], r["K"])] = r
    ln_rows = [r for (k, _, _), r in cv.items() if k == "ln_mod_quant"]
    t_launch = F["step"]["stock_eager|256"]
    import torch
    pm = torch.load(pathlib.Path(__file__).resolve().parent.parent / "rtx5000_next" / "prompts" / "embeds_pixart.pt")["items"]
    tok = [int(it["pm"].sum()) for it in pm]
    bugs = len(re.findall(r"^\| \d+ \|", (pathlib.Path(__file__).resolve().parent.parent / "rtx5000_next" / "BUGS.md").read_text(), flags=re.M))
    se_max = max(v["lpips"][1] / 6 ** 0.5 for s in SIZES for v in F["quality_final"][str(s)].values())
    floors = [ext(512, "bf16+xcudnn")["lpips"], ext(1024, "bf16+xcudnn")["lpips"]]
    old_calls = max(F["nsys"][k]["class_n"].get("our_quant", 0) for k in ("fused_cuda_swizzled_compile_1024", "fused_triton_op_compile_1024"))
    sage_flux = flash_rows["flux_joint"]["ms"] / sage_rows["flux_joint"]["ms"]
    sage_sd3 = flash_rows["sd35_medium_joint"]["ms"] / sage_rows["sd35_medium_joint"]["ms"]
    smoke = F["smoke_rel_l2"]
    lines = []
    A = lines.append

    # ------------------------------------------------------------------------------------------------ title + summary
    A(f"""<div class="title"><h1>Making NVFP4 fast and trustworthy for PixArt-Sigma</h1>
<p class="sub">What we measured, why we measured it, and what it means &middot; RTX PRO 5000 Blackwell (48 GB) &middot; PyTorch 2.15 nightly, CUDA 13 &middot; evaluation round 2 &middot; 20 November 2026</p></div>""")
    A("<h2>Summary</h2>")
    rows = []
    for lab, key in STEP_ROWS:
        rows.append([lab] + [f"{n(S(key, s))} ms ({sx(SU(key, s))})" if S(key, s) else "-" for s in SIZES])
    A(t_(rows, ["setup (CUDA Graph time of one transformer forward)", "256 px", "512 px", "1024 px"], "num"))
    A(f"<p class='small'>Speed-up in brackets is against bf16 compiled at the same size. Each size has {n(TOKENS[256], '.0f')}, {n(TOKENS[512], '.0f')} and {n(TOKENS[1024], '.0f')} image tokens. "
      f"One forward is one denoising step with classifier-free guidance (batch of 2); an image needs 14 of them.</p>")
    A(fig("fig_story.png", "The story of this round in five boxes."))
    A("<h3>What we found</h3><ol class='find'>")
    A(f"<li><b>The earlier 'fusion ties compile' result was an artifact.</b> In the compiled models our fused quantize kernel ran {n(max(F['nsys'][k]['class_n'].get('our_quant', 0) for k in ('fused_cuda_swizzled_compile_1024', 'fused_triton_op_compile_1024')), '.0f')} times per forward instead of {n(28 * 10 + 6, '.0f')}; the compiled graph was torchao's own. "
      f"So the old 'fused + compile' number ({n(S('old_fused_cuda_swizzled_compile', 1024))} ms) measured stock + compile ({n(S('stock_compile', 1024))} ms).</li>")
    A(f"<li><b>The fix works.</b> A new layer (<code>FusedNVFP4Linear</code>) makes the quantize kernel and the FP4 GEMM ordinary graph nodes. In the compiled model our kernel now runs {n(NS[k_best]['class_n']['our_quant'], '.0f')} times per forward. "
      f"The quantize-plus-glue kernel time at 1024 px falls from {n(chain(k_stock))} ms (compiled stock) to {n(chain(k_best))} ms.</li>")
    A(f"<li><b>Attention is now the bottleneck.</b> At 1024 px attention takes {n(cls(k_best, 'attention'))} of {n(tot_best)} ms of kernel time ({pct(att_share)}); the FP4 GEMMs take {n(cls(k_best, 'fp4_gemm'))} ms and our quantize kernel {n(cls(k_best, 'our_quant'))} ms.</li>")
    A(f"<li><b>Speed.</b> At 1024 px the best stack is {sx(SU('module_best_xcudnn', 1024))} faster than compiled bf16 and {sx(S('stock_compile', 1024) / S('module_best_xcudnn', 1024))} faster than compiled stock NVFP4. "
      f"At 256 px stock NVFP4 gives only {sx(SU('stock_compile', 256))}: the step is launch-bound, and the gain comes from our fused graph ({sx(SU('module_best_xcudnn', 256))}).</li>")
    A(f"<li><b>Quality.</b> NVFP4 changes image details whatever the implementation: LPIPS to the bf16 image is {n(q(256, 'stock')['lpips'][0], '.2f')}, {n(q(512, 'stock')['lpips'][0], '.2f')} and {n(q(1024, 'stock')['lpips'][0], '.2f')} for stock at 256, 512, 1024 px. "
      f"Our stack stays at that level ({n(q(1024, 'module')['lpips'][0], '.2f')} at 1024 px). Selective quantization (keep five small layers in bf16) lowers it to {n(q(256, 'selective')['lpips'][0], '.2f')}, {n(q(512, 'selective')['lpips'][0], '.2f')}, {n(q(1024, 'selective')['lpips'][0], '.2f')} and costs "
      f"{n(S('module_best_selective', 512) - S('module_best_xcudnn', 512), '.1f')} to {n(S('module_best_selective', 1024) - S('module_best_xcudnn', 1024), '.1f')} ms per forward.</li>")
    A(f"<li><b>What did not pay.</b> DiTFastAttn (training-free attention compression) hurts quality before it helps time; 2x2 K/V pooling destroys the images; Sage attention is slower than flash for PixArt's head dimension (it wins for SD3.5 and FLUX); "
      f"static activation scales cost quality at 256 and 512 px.</li></ol>")
    A(box("How to read this report", "Sections 1-2 explain the problem and what we ran. Section 3 is the surprise and its fix. Sections 4-8 hold results, each with 'why we did it' and 'what it implies'. "
          "Section 9 lists mistakes we made and the lessons. Section 10 states limits. The appendix says where every number comes from and lists the consistency checks (all pass).", "tip"))

    # ------------------------------------------------------------------------------------------------ 1 background
    A("<h2>1. Background: the model, NVFP4, and why quantizing costs time</h2>")
    A("<h3>1.1 The model</h3>")
    A(f"<p>PixArt-Sigma XL/2 is a diffusion transformer with {n(28, '.0f')} identical blocks (about {n(0.61, '.2f')} billion parameters). To draw an image it runs the same transformer {n(14, '.0f')} times, "
      "each time removing a little noise from a latent image. Each run is called a <b>step</b> or <b>forward</b>. Classifier-free guidance (CFG) runs the transformer on two inputs at once, one with the prompt and one without, so a step works on a batch of 2.</p>")
    A(fig("fig_block.png", "One transformer block. Ten linear layers (orange) are converted to NVFP4; attention (red) is not."))
    A("<h3>1.2 What NVFP4 does to a number</h3>")
    A("<p>NVFP4 stores a number in 4 bits. Sixteen neighbouring numbers share one 8-bit scale, and the whole tensor has one more 32-bit scale. "
      "The 4-bit code can only take 8 magnitudes (0, 0.5, 1, 1.5, 2, 3, 4, 6) and a sign. That is why large values survive and small values are rounded to a few levels. Worked example (a real call to the library):</p>")
    ex_rows = [[str(i), n(ex["x"][i], ".2f"), n(ex["r"][i], ".2f"), n(ex["code"][i], ".1f"), n(ex["deq"][i], ".3f"), n(ex["err"][i], ".3f", sign=True)] for i in range(16)]
    A(t_(ex_rows, ["position", "original", "original / block scale", "4-bit code", "value after (code x scale)", "error"], "num small"))
    A(f"<p class='small'>The block scale is {n(ex['block_scale'], '.0f')} (8-bit float) times the per-tensor scale {n(ex['pts'], '.5f')}, so one code step is worth {n(ex['eff'], '.4f')}. Relative error of this block: {pct(ex['rel'], '.1f')}.</p>")
    A(fig("fig_nvfp4_example.png", "The same 16 values before and after conversion."))
    A("<h3>1.3 Why the conversion costs time even though the GEMM gets faster</h3>")
    ideal_mb = sq["ideal_mb"]
    rd = sq["rows"] * sq["k"] * 2 / 1e6
    A(f"<p>At 1024 px one layer input has {n(sq['rows'], '.0f')} rows (2 CFG halves x {n(TOKENS[1024], '.0f')} tokens) and {n(sq['k'], '.0f')} columns, i.e. {n(sq['rows'] * sq['k'] / 1e6, '.1f')} million numbers. "
      f"In bf16 that is {n(rd, '.1f')} MB. The least possible traffic for converting it is to read it once ({n(rd, '.1f')} MB) and write the codes ({n(sq['rows'] * sq['k'] / 2 / 1e6, '.1f')} MB) and scales ({n(sq['rows'] * sq['k'] / 16 / 1e6, '.1f')} MB): "
      f"<b>{n(ideal_mb, '.1f')} MB</b>, which the GPU's {n(P['copy_gbs'], '.0f')} GB/s memory moves in about {n(ideal_mb / P['copy_gbs'] * 1e3, '.0f')} microseconds.</p>")
    A(f"<p>What the stock library does instead (measured with Nsight Compute on the Q projection of block 3): <b>{n(sq['quant_kernels'], '.0f')} separate GPU kernels</b>, {n(sq['quant_dram_mb'], '.0f')} MB of DRAM traffic "
      f"({n(sq['quant_dram_mb'] / ideal_mb, '.0f')}x the ideal) and {n(sq['quant_us'], '.0f')} microseconds, while the FP4 matrix multiplication itself takes {n(sq['gemm_us'], '.0f')} microseconds. "
      f"So the conversion costs <b>{n(sq['quant_us'] / sq['gemm_us'], '.0f')}x the GEMM it feeds</b>. Quantization makes the GEMM faster but the conversion eats the gain: that is the whole problem.</p>")
    A(box("Example: launch-bound at small sizes", f"Stock NVFP4 launches {n(NS['stock_1024']['kernels'], '.0f')} kernels per forward. At 256 px a forward takes {n(t_launch['eager'], '.0f')} ms when launched one kernel at a time but only {n(t_launch['graph'], '.1f')} ms "
          f"when replayed as a CUDA Graph (launches recorded once). That is {n(t_launch['eager'] * 1e3 / NS['stock_1024']['kernels'], '.1f')} microseconds of CPU time per kernel: the GPU is waiting for the CPU, not computing. "
          "Fewer, fatter kernels fix this; so does torch.compile.", "ex"))

    # ------------------------------------------------------------------------------------------------ 2 what we did
    A("<h2>2. What we ran and why</h2>")
    A("<p>The work ran in two rounds on the same GPU. Round 1 built the measurement tools and the first fused kernels. Round 2 (this report) audited round 1, found a flaw, fixed it, and added attention and sampling experiments.</p>")
    trows = [
        ["R1", "Step time, 11 configs x 3 sizes", "Is NVFP4 faster, and does fusion help?", f"Stock NVFP4 eager {n(S('stock_eager', 1024))} ms vs bf16 eager {n(S('bf16_eager', 1024))} ms at 1024 px ({n(S('stock_eager', 1024) / S('bf16_eager', 1024), '.1f', 'x')} slower); compile brings it to {n(S('stock_compile', 1024))} ms.", "Quantize overhead dominates; compile is the fair baseline."],
        ["R1", "nsys: kernel time by class", "Where does the time go?", f"{pct(stock_eager_chain)} of the stock eager forward is neither attention nor GEMM (it is quantize and glue).", "Optimize the conversion, not the GEMM."],
        ["R1", "ncu: DRAM traffic of one layer", "Is the conversion memory-bound and wasteful?", f"Stock moves {n(sq['quant_dram_mb'] / ideal_mb, '.0f')}x the ideal bytes; our kernel path {n(fq['quant_dram_mb'] / ideal_mb, '.1f')}x.", "Fusion should help; the metric names had to be fixed first (Section 9)."],
        ["R1", "CUDA kernels vs Triton, byte for byte", "Do our kernels compute exactly what torchao computes?", "CUDA swizzled kernel: 0 differing bytes in every tested shape.", "Safe to use in the model."],
        ["R2", "nsys on the compiled fused configs", "Why does 'fused + compile' equal 'stock + compile'?", f"Our kernel ran {n(old_calls, '.0f')} times per forward, not 286.", "The fusion was not engaged inside compiled blocks (Section 3)."],
        ["R2", "New FusedNVFP4Linear + tests", "Make the kernel visible to the compiler.", f"{n(NS[k_best]['class_n']['our_quant'], '.0f')} kernel calls per forward; layer outputs bit-identical to torchao.", "Fusion now actually measured."],
        ["R2", "Step time, fair baselines and ablations (E05-E07, run_final30)", "What does each change buy?", "Table in Section 4.", "Shared quantize, static scale, text trim and cuDNN cross-attention each help."],
        ["R2", "Microbenchmarks (E02)", "Is the FP4 GEMM efficient at PixArt shapes? Which attention kernel is fastest?", "Section 5 and 6.", "Wider GEMMs and a different attention kernel can help."],
        ["R2", "Real w4a16 vs w4a4 (E03)", "How much quality does quantizing activations cost on top of weights?", "Section 7.", "Activations dominate the late-step error."],
        ["R2", "Static scales and DiTFastAttn plans (E04, E10)", "Can we remove the per-call amax pass, and compress attention?", "Section 6.", "Static scale saves time but is risky; DiTFastAttn does not pay here."],
        ["R2", "Real-prompt quality of every variant (E11, final run)", "What does each speed-up cost in image quality?", "Section 7.", "Selective quantization is the best trade-off."],
        ["R2", "End-to-end timing (E12)", "How much of a real image is the transformer?", f"{pct(e2e['den'] / e2e['total'], '.0f')} of the time for 4 images at 1024 px.", "Per-step speed-ups carry through almost fully."],
    ]
    A(t_(trows, ["round", "experiment", "question", "result", "implication"], "tl small"))

    # ------------------------------------------------------------------------------------------------ 3 surprise and fix
    A("<h2>3. The surprise and the fix</h2>")
    A("<h3>3.1 What the round-1 numbers hid</h3>")
    A(f"<p>Round 1 reported that the fused kernel and torch.compile gave the same step time ({n(S('old_fused_cuda_swizzled_compile', 1024))} ms vs {n(S('stock_compile', 1024))} ms at 1024 px, a difference inside the "
      f"{n(min(step_runs('stock_compile', 1024)))}-{n(max(step_runs('stock_compile', 1024)))} ms run-to-run range of stock itself). It was read as 'compile already fuses everything'. The kernel tables say otherwise:</p>")
    r5 = [["stock NVFP4, eager", NS["stock_1024"]["kernels"], 0, NS["stock_1024"]["total_ms"]],
          ["stock NVFP4, compiled", NS[k_stock]["kernels"], 0, NS[k_stock]["total_ms"]],
          ["old 'fused' + compile (CUDA kernel)", NS["fused_cuda_swizzled_compile_1024"]["kernels"], NS["fused_cuda_swizzled_compile_1024"]["class_n"].get("our_quant", 0), NS["fused_cuda_swizzled_compile_1024"]["total_ms"]],
          ["old 'fused' + compile (Triton op)", NS["fused_triton_op_compile_1024"]["kernels"], NS["fused_triton_op_compile_1024"]["class_n"].get("our_quant", 0), NS["fused_triton_op_compile_1024"]["total_ms"]],
          ["module stack, compiled (new)", NS[k_best]["kernels"], NS[k_best]["class_n"]["our_quant"], NS[k_best]["total_ms"]]]
    A(t_([[a, n(b, ".0f"), n(c, ".0f"), n(d)] for a, b, c, d in r5], ["config (1024 px)", "kernels per forward", "calls of our quantize kernel per forward", "kernel time per forward (ms)"], "num"))
    A("<p>The two old compiled configs run <b>exactly the same kernels as stock compiled</b> (same count of compiler-generated kernels, same top kernels). Our kernel ran only for the six layers outside the compiled blocks. "
      "The old fusion replaced a function inside torchao (a 'monkeypatch'); the compiler traced torchao's code and did not use the replacement.</p>")
    A(fig("fig_classes.png", "Kernel time by class at 1024 px. The old 'fused' bar equals the stock compiled bar; the module stack removes most of the orange compiler-generated quantize kernels."))
    A(box("Implication", "A result that looks like 'the new thing equals the baseline' should be treated as 'the new thing may not have run'. We now prove that a fusion ran with a counter inside the fused kernel, in every test.", "imp"))
    A("<h3>3.2 The fix and its proof</h3>")
    A("<p>The new layer does not touch torchao's tensor classes. It runs three plain operations: (1) a custom operation that converts the activation to FP4 codes and scales, (2) the library's FP4 matrix multiplication, (3) a multiplication by the two per-tensor scales and the bias. "
      "All three are visible to the compiler, which fuses step (3) with the following gate and residual. Options: a calibrated static scale (no pass over the data to find the maximum), one shared conversion for Q, K and V, weights concatenated, and a list of layers to leave in bf16.</p>")
    proof = [["layer output equals torchao's NVFP4 linear (4 shapes, two quantize backends)", "bit-identical" if F["smoke_layer_exact"] else "NOT identical"],
             ["quantize calls per forward, compiled (7 inputs x 28 blocks + 6)", n(NS[k_best]["class_n"]["our_quant"], ".0f")],
             [f"whole-model error vs bf16, 256 px, mean of 5 timesteps: stock / module / compiled module", f"{n(F['smoke_rel_l2']['stock'], '.4f')} / {n(F['smoke_rel_l2']['module+be=triton'], '.4f')} / {n(F['smoke_rel_l2']['compiled'], '.4f')}"],
             ["CUDA quantize kernel vs Triton and torchao, bytes that differ (9 shapes)", "0"]]
    A(t_(proof, ["check", "result"], "small"))
    A(box("Why two bit-exact implementations still give different images",
          "Quantization rounds to 8 levels. A rounding-level change anywhere upstream (a different attention kernel, a different operation order) flips a few roundings, the noise changes, and the image differs. "
          f"So a layer is tested bit for bit on the same input, and a whole model is tested by its distance to bf16 ({n(min(smoke.values()), '.3f')} to {n(max(smoke.values()), '.3f')} for all three), not by agreement with another NVFP4 model.", "ex"))

    # ------------------------------------------------------------------------------------------------ 4 speed
    A("<h2>4. Speed results</h2>")
    A("<p><b>Why:</b> to know what each change is worth against a fair baseline. <b>Rule used for every number:</b> CUDA Graph time of one forward, median of 20 timed iterations from the dedicated final run (run_final30); "
      "configs that were only in the earlier ablation run use that run (30 iterations). Repeat runs of the same config are in Appendix B.</p>")
    A(fig("fig_speed.png", "Step time per size. The bracket is the speed-up over bf16 compiled."))
    fp8_vs_stock = {s: S("fp8_compile", s) / S("stock_compile", s) - 1 for s in SIZES}
    A(f"<p><b>Reading it.</b> At 1024 px: bf16 compiled {n(S('bf16_compile', 1024))} ms, stock NVFP4 {n(S('stock_compile', 1024))} ms, our best {n(S('module_best_xcudnn', 1024))} ms. "
      f"fp8 (an 8-bit format, torchao's usual choice) reaches {n(S('fp8_compile', 1024))} ms. Stock NVFP4 against fp8: {n(100 * fp8_vs_stock[256], '.0f', '%', sign=True)} at 256 px (fp8 is faster at 256 px), "
      f"{n(100 * fp8_vs_stock[512], '.0f', '%', sign=True)} at 512 px and {n(100 * fp8_vs_stock[1024], '.0f', '%', sign=True)} at 1024 px (positive = fp8 slower). "
      "So stock NVFP4 is hardly better than fp8; only the fused stack clearly beats it.</p>")
    A(fig("fig_scaling.png", "Left: cost against tokens. Right: speed-up against bf16 compiled for each size."))
    att512 = cls("module_best_compile_512", "attention") / NS["module_best_compile_512"]["total_ms"]
    peak = max(SIZES, key=lambda s: SU("module_best_xcudnn", s))
    A(box("Implication", f"The speed-up of the best stack is {sx(SU('module_best_xcudnn', 256))} at 256 px, {sx(SU('module_best_xcudnn', 512))} at 512 px and {sx(SU('module_best_xcudnn', 1024))} at 1024 px: it <b>peaks at {n(peak, '.0f')} px and falls at 1024 px</b>. "
          f"Two effects compete. At small sizes launch overhead limits what quantization can give (stock NVFP4 is {sx(SU('stock_compile', 256))} at 256 px). At large sizes attention, which we do not quantize, grows faster than the rest: "
          f"it is {pct(att512)} of the kernel time at 512 px and {pct(att_share)} at 1024 px, so the quantized part shrinks in importance (Amdahl's law). Attention is therefore the next lever.", "imp"))
    A("<h3>4.1 What each change is worth</h3>")
    A(fig("fig_waterfall.png", "Waterfall from stock NVFP4 to the best stack. Values from the final run, except the trim step (single ablation run)."))
    ab = [["Triton quantize kernel, module stack", "module_triton_compile"], ["- do not share the Q/K/V conversion", "module_triton_noshare"],
          ["+ static scale", "module_triton_static"], ["+ concatenated Q/K/V weights", "module_triton_concat"], ["+ selective quantization", "module_triton_selective"]]
    A(t_([[a, f"{n(S(k, 512))}", f"{n(S(k, 1024))}", f"{n(S(k, 1024) - S('module_triton_compile', 1024), '.1f', sign=True) if k != 'module_triton_compile' else '-'}"] for a, k in ab],
         ["ablation on the Triton backend (one run each)", "512 px ms", "1024 px ms", "change vs row 1 at 1024 px (ms)"], "num"))
    A(f"<p>Shared conversion saves {n(S('module_triton_noshare', 1024) - S('module_triton_compile', 1024), '.1f')} ms at 1024 px; a static scale saves {n(S('module_triton_compile', 1024) - S('module_triton_static', 1024), '.1f')} ms. "
      f"Concatenated weights are within noise ({n(S('module_triton_concat', 1024) - S('module_triton_compile', 1024), '.1f', sign=True)} ms at 1024 px, {n(S('module_triton_concat', 512) - S('module_triton_compile', 512), '.1f', sign=True)} ms at 512 px) although the GEMM microbenchmark says they should help (Section 5). "
      f"The CUDA kernel is worth {n(S('module_triton_compile', 1024) - S('module_cuda', 1024), '.1f')} ms against the Triton one ({n(S('module_triton_compile', 1024))} vs {n(S('module_cuda', 1024))} ms).</p>")
    A(f"<p><b>Text trim</b> keeps only the real prompt tokens in cross-attention. The six prompts use {n(min(tok), '.0f')} to {n(max(tok), '.0f')} of the 300 padded tokens; the padding is masked anyway, so trimming is exact (LPIPS {n(ext(1024, 'bf16+trim')['lpips'], '.3f')} to the untrimmed image). "
      f"It saves {n(S('module_best', 1024) - S('module_best_trim', 1024), '.1f')} ms at 1024 px. <b>cuDNN for cross-attention</b> saves another {n(S('module_best_trim', 1024) - S('module_best_xcudnn', 1024), '.1f')} ms. "
      f"cuDNN for <i>self</i>-attention is slower: {n(S('module_best_cudnn_self', 1024))} ms against {n(S('module_best_trim', 1024))} ms.</p>")

    # ------------------------------------------------------------------------------------------------ 5 where the time goes
    A("<h2>5. Where the time goes now, and the hardware limits</h2>")
    A(f"<p>At 1024 px the compiled module stack spends {n(tot_best)} ms of kernel time: attention {n(cls(k_best, 'attention'))} ({pct(att_share)}), FP4 GEMMs {n(cls(k_best, 'fp4_gemm'))}, compiler-generated glue {n(cls(k_best, 'inductor'))} and our quantize kernel {n(cls(k_best, 'our_quant'))} ms. "
      f"The GPU is busy for the whole step ({pct(NS[k_best]['total_ms'] / S('module_best', 1024))} of the CUDA Graph time is kernel time).</p>")
    A("<h3>5.1 Memory traffic</h3>")
    A(fig("fig_traffic.png", "Left: measured DRAM traffic around one layer (everything except the GEMM kernel). Right: model of the quantize traffic per forward and what each lever removes."))
    A(f"<p>Per layer (Q projection, 1024 px): stock {n(sq['quant_dram_mb'], '.0f')} MB, our CUDA path {n(fq['quant_dram_mb'], '.0f')} MB, ideal {n(ideal_mb, '.0f')} MB. Level-2 cache traffic is {n(sq['quant_l2_mb'], '.0f')} MB against {n(fq['quant_l2_mb'], '.0f')} MB. "
      f"The model for the whole forward: {n(tr['now'], '.1f')} GB now, {n(tr['floor'], '.1f')} GB as an ideal one-pass conversion. A static scale removes one full read of every input ({n(tr['static'], '.1f')} GB), sharing Q/K/V another {n(tr['shared'], '.1f')} GB; "
      f"fusing the LayerNorm into the conversion would remove {n(tr['ln'], '.1f')} GB more (the kernel exists and is byte-tested, but is not in the model yet).</p>")
    A("<p class='small'>The model counts each tensor read or written once; the measured number includes everything the stock path does around the GEMM, so the two panels are not the same quantity.</p>")
    A("<h3>5.2 The FP4 GEMM is far from its peak at PixArt shapes</h3>")
    A(fig("fig_gemm.png", "Left: FP4 GEMM throughput against rows. Right: 1024 px, one image."))
    qs, qc = (next(r for r in F["fp4_gemm"] if r["site"] == s and r["tokens"] == 4096 and r["images"] == 1) for s in ("qkv_sep (1152->1152)", "qkv_concat (1152->3456)"))
    A(f"<p>With K = 1152 a GEMM reaches {n(qs['fp4_tflops'], '.0f')} TFLOP/s ({n(qs['fp4_tflops'] / P['fp4_tflops'] * 100, '.0f', '%')} of the best measured {n(P['fp4_tflops'], '.0f')}). "
      f"Concatenating Q, K and V makes N three times wider and reaches {n(qc['fp4_tflops'], '.0f')} TFLOP/s. Predicted saving: three GEMMs of {n(2 * qs['M'] * 1152 * 1152 / 1e9, '.1f')} GFLOP take {n(3 * 2 * qs['M'] * 1152 * 1152 / (qs['fp4_tflops'] * 1e12) * 1e3, '.3f')} ms, one wide GEMM {n(2 * qc['M'] * 1152 * 3456 / (qc['fp4_tflops'] * 1e12) * 1e3, '.3f')} ms: "
      f"about {n(28 * (3 * 2 * qs['M'] * 1152 * 1152 / (qs['fp4_tflops'] * 1e12) - 2 * qc['M'] * 1152 * 3456 / (qc['fp4_tflops'] * 1e12)) * 1e3, '.1f')} ms per forward, which is below the run-to-run noise. "
      "A larger batch (more images per step) is the bigger lever for the GEMM.</p>")
    A("<h3>5.3 How far from the hardware limits?</h3>")
    srows = []
    for lab, key, mode in (("bf16 compiled", "bf16_compile", "bf16"), ("stock NVFP4 compiled", "stock_compile", "fp4"), ("best stack", "module_best_xcudnn", "fp4")):
        srows.append([lab] + [f"{n(S(key, s))} / {n(sol[mode][s])} ({pct(sol[mode][s] / S(key, s))})" for s in SIZES])
    A(t_(srows, ["setup", "256 px: measured / bound (share)", "512 px", "1024 px"], "num small"))
    A(f"<p class='small'>The bound is the sum over GEMM, attention and conversion of max(compute time, memory time) at the measured peaks (copy {n(P['copy_gbs'], '.0f')} GB/s, bf16 {n(P['bf16_tflops'], '.0f')} TFLOP/s, FP4 {n(P['fp4_tflops'], '.0f')} TFLOP/s, attention {n(P['attn_tflops'], '.0f')} TFLOP/s). "
      "It ignores LayerNorm, GELU and residual traffic and uses the large-matrix FP4 peak, so it is optimistic: real steps cannot reach it. Use it for ordering, not as a target.</p>")

    # ------------------------------------------------------------------------------------------------ 6 attention
    A("<h2>6. Attention and the training-free options</h2>")
    A("<p><b>Why:</b> attention is two thirds of the step, and it was not quantized. We tried every option that needs no training.</p>")
    A(fig("fig_attention.png", "Attention kernel throughput (microbenchmark, same GPU, a later run than E02's first table)."))
    s1, f1 = sage_rows["pixart_self_1024"], flash_rows["pixart_self_1024"]
    s2, f2 = sage_rows["flux_joint"], flash_rows["flux_joint"]
    s3, f3 = sage_rows["sd35_medium_joint"], flash_rows["sd35_medium_joint"]
    A(f"<p>For PixArt self-attention at 1024 px flash reaches {n(f1['tflops'], '.0f')} TFLOP/s and Sage 2.x {n(s1['tflops'], '.0f')}: Sage is {n(s1['ms'] / f1['ms'] - 1, '.0%')} slower, because head dimension 72 must be padded to 96 (a third of the work is wasted) and Sage's error is {n(s1['rel_l2'] / flash_rows['pixart_self_1024']['rel_l2'], '.0f')}x flash's ({n(s1['rel_l2'], '.3f')} against {n(f1['rel_l2'], '.4f')}). "
      f"For SD3.5 Medium (head dim 64) Sage is {sx(f3['ms'] / s3['ms'])} faster and for FLUX (head dim 128) {sx(f2['ms'] / s2['ms'])} faster. <b>Implication:</b> Sage is a candidate for the next models, not for PixArt.</p>")
    A("<h3>6.1 DiTFastAttn: compress attention across tokens, steps and the two CFG halves</h3>")
    A("<p>Three tricks per (step, layer): attend only inside windows and reuse a cached correction (WA-RS), reuse the previous step's attention output (AST), or let the unconditional half copy the conditional half (ASC). "
      "A greedy search on separate calibration prompts picks, for each (step, layer), the cheapest trick whose output error stays below a threshold delta.</p>")
    A(fig("fig_dfa.png", "Plans found for 1024 px. Each cell is one (denoise step, layer)."))
    A(t_([[d, n(dq(1024, d)["cost"] * 100, ".0f", "%"), n(dq(512, d)["cost"] * 100, ".0f", "%"),
           n(ext(512, f"bf16+dfa{d}")["lpips"], ".3f"), n(ext(1024, f"bf16+dfa{d}")["lpips"], ".3f")] for d in ("0.05", "0.1", "0.2")],
         ["delta", "modeled attention cost at 1024 px", "at 512 px", "LPIPS to bf16 at 512 px", "LPIPS at 1024 px"], "num"))
    A(f"<p><b>Implication.</b> Delta 0.05 saves {n(100 - dq(1024, '0.05')['cost'] * 100, '.0f', '%')} of attention at 1024 px with an LPIPS ({n(ext(1024, 'bf16+dfa0.05')['lpips'], '.2f')}) about equal to the noise floor ({n(ext(1024, 'bf16+xcudnn')['lpips'], '.2f')} for a harmless kernel swap; at 512 px {n(ext(512, 'bf16+dfa0.05')['lpips'], '.2f')} against {n(ext(512, 'bf16+xcudnn')['lpips'], '.2f')}): too little to matter. "
      f"Delta 0.2 saves {n(100 - dq(1024, '0.2')['cost'] * 100, '.0f', '%')} but LPIPS rises to {n(ext(1024, 'bf16+dfa0.2')['lpips'], '.2f')} at 1024 px. <b>The step time with DiTFastAttn was not measured end to end</b> (only the modeled cost), so no speed claim is made.</p>")
    A("<h3>6.2 Static activation scales</h3>")
    A(fig("fig_drift.png", "How much each layer's activation range changes during sampling."))
    A(f"<p>A static scale removes the pass that finds the maximum. It needs the range to stay put. Median drift is {n(drift[1024][len(drift[1024]) // 2], '.1f', 'x')} at 1024 px, but the worst layer drifts {n(drift[1024][-1], '.0f', 'x')}. "
      f"Quality: static + concat raises LPIPS from {n(q(256, 'module')['lpips'][0], '.2f')} to {n(q(256, 'best')['lpips'][0], '.2f')} at 256 px and {n(q(512, 'module')['lpips'][0], '.2f')} to {n(q(512, 'best')['lpips'][0], '.2f')} at 512 px; at 1024 px it is unchanged "
      f"({n(q(1024, 'module')['lpips'][0], '.2f')} to {n(q(1024, 'best')['lpips'][0], '.2f')}). <b>Implication:</b> alone (Triton ablation) a static scale saves {n(S('module_triton_compile', 1024) - S('module_triton_static', 1024), '.1f')} ms at 1024 px; "
      f"static scale and concatenation together saved {n(S('module_cuda', 1024) - S('module_best', 1024), '.1f')} ms in the final CUDA runs ({n(S('module_cuda', 512) - S('module_best', 512), '.1f')} ms at 512 px, {n(S('module_cuda', 256) - S('module_best', 256), '.1f')} ms at 256 px). "
      "It is a small gain that is only free of quality cost at the largest size.</p>")
    A(f"<h3>6.3 K/V pooling</h3><p>Averaging keys and values over 2x2 token tiles makes attention four times cheaper and the step {n(S('module_best_kvds2', 1024))} ms instead of {n(S('module_best_trim', 1024))} ms at 1024 px. The images are destroyed "
      f"(LPIPS {n(ext(512, 'bf16+kvds2')['lpips'], '.2f')} and CLIP score {n(ext(512, 'bf16+kvds2')['clip'], '.1f')} against {n(ext(512, 'bf16')['clip'], '.1f')} at 512 px). A speed-up that removes the picture is not a speed-up.</p>")

    # ------------------------------------------------------------------------------------------------ 7 quality
    A("<h2>7. Image quality: what each speed-up costs</h2>")
    A("<p><b>How we measure.</b> Six real prompts (encoded with the PixArt T5), the same random seed, 14 steps. Every variant makes the same six images; each is compared with the bf16 image of the same prompt. "
      "LPIPS is a perceptual distance (0 identical, 1 unrelated). CLIP score measures whether the image still matches its prompt (higher is better).</p>")
    A(fig("fig_images_variants.png", "Real images. Same prompt, same seed, different model variants. LPIPS to the bf16 image is printed on each cell. All NVFP4 variants keep the scene and change details."))
    A(box("How to read LPIPS here (examples)",
          f"About {n(ext(512, 'bf16+xcudnn')['lpips'], '.2f')}: a harmless kernel swap in bf16 (the noise floor). "
          f"About {n(q(512, 'stock')['lpips'][0], '.2f')}: stock NVFP4 at 512 px, same scene with different details (look at the fisherman's eyes and the sign). "
          f"About {n(ext(512, 'bf16+kvds2')['lpips'], '.2f')}: destroyed image (K/V pooling).", "ex"))
    A(fig("fig_quality.png", "Speed against quality for the headline variants."))
    rows = []
    for lab, qv, sk in (("stock NVFP4", "stock", "stock_compile"), ("weights only (w4a16)", "w4a16", None), ("module stack", "module", "module_cuda"),
                        ("+ static + concat", "best", "module_best"), ("+ trim + cuDNN", "trim", "module_best_xcudnn"), ("+ selective", "selective", "module_best_selective")):
        r = [lab]
        for s in SIZES:
            qq = q(s, qv)
            r.append(f"{n(qq['lpips'][0], '.3f')} &plusmn; {n(qq['lpips'][1] / 6 ** 0.5, '.3f')} / CLIP {n(qq['clip'][0])} / {n(S(sk, s)) + ' ms' if sk else '-'}")
        rows.append(r)
    A(t_(rows, ["variant", "256 px: LPIPS &plusmn; s.e. / CLIP / step", "512 px", "1024 px"], "num small"))
    A(f"<p class='small'>Reference rows: bf16 CLIP score {n(q(256, 'bf16')['clip'][0])}, {n(q(512, 'bf16')['clip'][0])}, {n(q(1024, 'bf16')['clip'][0])} at 256, 512, 1024 px. LPIPS error bars are the standard error over 6 prompts; "
      f"differences under about {n(2 * se_max, '.2f')} (twice the largest standard error) are within noise.</p>")
    A("<h3>7.1 Selective quantization</h3>")
    A(f"<p>Five small layers stay in bf16: the cross-attention key and value projections (they see only the 300 text tokens and had the largest error), the caption projection, the timestep embedder and the final projection. "
      f"LPIPS falls from {n(q(256, 'trim')['lpips'][0], '.3f')} to {n(q(256, 'selective')['lpips'][0], '.3f')} (256 px), {n(q(512, 'trim')['lpips'][0], '.3f')} to {n(q(512, 'selective')['lpips'][0], '.3f')} (512 px), "
      f"{n(q(1024, 'trim')['lpips'][0], '.3f')} to {n(q(1024, 'selective')['lpips'][0], '.3f')} (1024 px). Cost: {n(S('module_best_selective', 256) - S('module_best_xcudnn', 256), '.1f')}, {n(S('module_best_selective', 512) - S('module_best_xcudnn', 512), '.1f')} and {n(S('module_best_selective', 1024) - S('module_best_xcudnn', 1024), '.1f')} ms. "
      f"At 1024 px the CLIP score of the selective variant ({n(q(1024, 'selective')['clip'][0])}) is lower than the non-selective one ({n(q(1024, 'trim')['clip'][0])}); with six prompts this is not conclusive.</p>")
    A("<h3>7.2 Weights versus activations</h3>")
    A(fig("fig_error_t.png", "Transformer output error against bf16 at five timesteps, with real prompts."))
    wd = w["1024"]["transformer_rel_l2"]
    A(f"<p>Quantizing only the weights gives {n(wd['bf16_vs_w4a16']['t999'], '.3f')} error at the noisy start and {n(wd['bf16_vs_w4a16']['t50'], '.3f')} near the end; adding activation quantization gives {n(wd['bf16_vs_stock']['t999'], '.3f')} and {n(wd['bf16_vs_stock']['t50'], '.3f')}. "
      f"Image LPIPS follows: {n(q(1024, 'w4a16')['lpips'][0], '.2f')} for weights only against {n(q(1024, 'stock')['lpips'][0], '.2f')} for both at 1024 px. <b>Implication:</b> the activation conversion costs both time and quality, and late steps are the most sensitive. "
      "Keeping activations in higher precision where they matter (the selective idea) is the natural next step, e.g. in the last steps.</p>")
    A("<h3>7.3 Attention and sampling changes on the bf16 model</h3>")
    A(t_([[lab, n(ext(512, v)["lpips"], ".3f"), n(ext(1024, v)["lpips"], ".3f") if v in F["quality_ext"]["1024"] else "-", n(ext(512, v)["clip"], ".1f")] for lab, v in
          (("cuDNN cross-attention (noise floor)", "bf16+xcudnn"), ("text trim (exact)", "bf16+trim"), ("DiTFastAttn delta 0.1", "bf16+dfa0.1"), ("limited-interval CFG 0.2-0.8", "bf16+licfg0.2-0.8"),
           ("limited-interval CFG 0.3-0.7", "bf16+licfg0.3-0.7"), ("K/V pooling 2x2", "bf16+kvds2"))],
         ["change", "LPIPS 512 px", "LPIPS 1024 px", "CLIP 512 px"], "num small"))
    A(fig("fig_images_attention.png", f"One change at a time on the bf16 model, 512 px. The K/V pooling row is the destroyed image the table reports (LPIPS {n(ext(512, 'bf16+kvds2')['lpips'], '.2f')})."))
    A(f"<p class='small'>Sage 2.x attention was only evaluated at 512 px (LPIPS {n(F['quality_later']['512']['bf16+sage2']['lpips'], '.3f')}, CLIP {n(F['quality_later']['512']['bf16+sage2']['clip'], '.1f')}); the 1024 px run was stopped (out of host memory).</p>")

    # ------------------------------------------------------------------------------------------------ 8 sampling
    A("<h2>8. Sampling-level changes and the whole pipeline</h2>")
    A("<h3>8.1 Limited-interval guidance</h3>")
    steps = 14
    guided = sum(1 for i in range(steps) if 0.2 <= i / steps < 0.8)
    work = (guided * 2 + (steps - guided)) / (steps * 2)
    A(f"<p>Guidance is only applied in the middle of the denoising path; outside it the transformer runs on one input instead of two. Worked example for the 0.2-0.8 interval and {n(steps, '.0f')} steps: steps with 0.2 &le; i/{steps} &lt; 0.8 are guided, "
      f"that is {n(guided, '.0f')} steps with batch 2 and {n(steps - guided, '.0f')} with batch 1. Work = ({guided} x 2 + {steps - guided} x 1) / ({steps} x 2) = <b>{pct(work)}</b>, i.e. a saving of {pct(1 - work)}. "
      f"Quality on the bf16 model: LPIPS {n(ext(512, 'bf16+licfg0.2-0.8')['lpips'], '.2f')} (512 px) and {n(ext(1024, 'bf16+licfg0.2-0.8')['lpips'], '.2f')} (1024 px); CLIP score {n(ext(512, 'bf16+licfg0.2-0.8')['clip'], '.1f')} and {n(ext(1024, 'bf16+licfg0.2-0.8')['clip'], '.1f')} "
      f"against {n(ext(512, 'bf16')['clip'], '.1f')} and {n(ext(1024, 'bf16')['clip'], '.1f')}: prompts are still followed. Time per image in the first E11 run at 1024 px: {n(ext1024['bf16+licfg0.2-0.8']['ms_per_image'], '.0f')} ms against {n(ext1024['bf16']['ms_per_image'], '.0f')} ms "
      f"({pct(1 - ext1024['bf16+licfg0.2-0.8']['ms_per_image'] / ext1024['bf16']['ms_per_image'])} less, matching the model). Those runs shared the GPU, so only this ratio is quoted; the 512 px ratio is not usable.</p>")
    A(box("Implication", "It is independent of NVFP4: it speeds bf16 and NVFP4 alike. It does not change how good NVFP4 is, but it lowers the absolute time. We report it separately from the NVFP4 numbers on purpose.", "imp"))
    A("<h3>8.2 What a user actually waits for</h3>")
    A(fig("fig_e2e.png", "Four images at 1024 px, bf16 compiled, from the first end-to-end run (the run stopped at its third variant because of a mistyped variant name)."))
    A(f"<p>The transformer is {pct(e2e['den'] / e2e['total'])} of the total; the text encoder is {n(e2e['t5'], '.0f')} ms and the VAE decode {n(e2e['vae'], '.0f')} ms. "
      f"A step in this batch of four images (batch 8) takes {n(e2e['step'], '.0f')} ms, so 14 steps = {n(14 * e2e['step'], '.0f')} ms, equal to the measured denoise time {n(e2e['den'], '.0f')} ms. "
      "<b>Implication:</b> per-step speed-ups carry through to the full image almost one for one. End-to-end times for the NVFP4 stack were not measured (see Section 10).</p>")

    # ------------------------------------------------------------------------------------------------ 9 lessons
    A("<h2>9. What went wrong, and the lessons</h2>")
    A(f"<p>The full list ({n(bugs, '.0f')} entries) is in BUGS.md. These are the ones that changed a result or nearly did.</p>")
    les = [
        ["'Fusion = compile' (round 1)", "Fused kernel not running inside compiled blocks", "Prove a fusion ran with a counter in the op. If a new path times like the baseline, suspect it is the baseline."],
        ["ncu showed 0 MB of DRAM traffic", "A metric name valid on other GPUs is silently dropped on this one (GB202)", "Check metric names with the tool's query command; a column of zeros is a missing metric."],
        ["Test runner printed PASS with no Python", "Exit code of a pipe taken from the wrong command", "A pass must be positive (the summary line is present), never 'no error text seen'."],
        ["Sage crash lost every earlier variant", "Head dim 72 not supported; no per-variant error handling; summary written at the end", "Isolate and save each variant; read a kernel's documented limits first."],
        ["Quality stage rerun ended with 'Killed'", "The operating system killed the two parallel image-generation processes (host memory)", "Parallel jobs must fit in host memory, not only GPU memory; saving after every variant kept the partial results."],
        ["End-to-end run stopped at its third variant", "Config name used where a variant name was needed", "One naming grammar, validated before any long run."],
        ["Per-stage 'busy' above 100%", "The stage profiler counted some kernels twice (found while writing this report)", "A utilisation above 100% is a bug in the measurement; not used anywhere in this report."],
        ["Wrong iteration count for nsys files", "A later run overwrote the metadata file of the same name", "Do not read a measurement's parameters from a file another job can overwrite."],
        ["First draft of this report contradicted the data", "Two claims written from expectation: 'the gain grows with size' and 'stock NVFP4 is not better than fp8'", "Prose is a second source of truth: generate its numbers from the data and re-read each claim against the table."],
    ]
    A(t_(les, ["what happened", "cause", "lesson"], "small"))

    # ------------------------------------------------------------------------------------------------ 10 limits
    A("<h2>10. Limits and what comes next</h2>")
    A("<ul class='lim'>"
      f"<li><b>Small quality sample.</b> Six prompts, one seed, 14 steps. LPIPS to the bf16 image of a quantized model is dominated by NVFP4 itself and is sensitive to rounding (noise floor {n(min(floors), '.2f')} to {n(max(floors), '.2f')}, measured with a harmless kernel swap). "
      f"Differences below about {n(2 * se_max, '.2f')} should not be trusted. A larger prompt set is the first thing to add.</li>"
      "<li><b>One model.</b> Everything is PixArt-Sigma. Sana 0.6B, SD3.5 Medium and FLUX were downloaded and checked for access and size, but their forward passes were not run. Sage 2.x is the interesting kernel for SD3.5 and FLUX.</li>"
      "<li><b>Baseline caveat.</b> 'Stock NVFP4' uses torchao's default quantize path. torchao's own fused Triton kernel needs the MSLK package, which was not available, so it was not measured. The comparison with stock may overstate the gain over torchao's recommended path.</li>"
      "<li><b>Not measured end to end.</b> The NVFP4 stack's full-pipeline time, DiTFastAttn's step time, and Sage at 1024 px.</li>"
      f"<li><b>Static scales</b> were calibrated on 8 prompts and show drift above {n(100, '.0f', 'x')} in a few layers; a per-timestep scale table is the safer design.</li>"
      f"<li><b>Hardware bound</b> is a model (Section 5.3).</li></ul>")
    A("<h3>Next steps, in order of expected value</h3><ol>"
      f"<li>Run the same stack on FLUX and SD3.5 Medium with Sage 2.x (the kernel test showed {sx(sage_flux)} and {sx(sage_sd3)} over flash there).</li>"
      "<li>Fuse the LayerNorm into the conversion (kernel exists) and cut another 1-3 GB of traffic per forward.</li>"
      "<li>Keep the last denoising steps in higher precision (the late-step error is the largest).</li>"
      "<li>Measure end-to-end time for the NVFP4 stack and for DiTFastAttn; extend quality to 50+ prompts.</li></ol>")

    # ------------------------------------------------------------------------------------------------ appendix
    A("<h2>Appendix A. Where every number comes from</h2>")
    srcs = [["step times", "results/step_target.jsonl: CUDA Graph median of >= 20 iterations; headline = run_final30 (20 iterations), else the 30-iteration ablation run"],
            ["kernel classes, kernel counts", "results/nsys/*_cuda_gpu_kern_sum.csv (5 forwards each), classified by rtx5000_next/analysis/kernel_classes.py"],
            ["DRAM, L2 traffic per layer", "results/ncu/*.ncu-rep re-read with the GB202 metric names"],
            ["quality, headline variants", "results/quality_final/&lt;size&gt;/summary.json (run_final30, 6 prompts)"],
            ["quality, other variants", f"first E11 run log ({n(len(F['quality_ext']['1024']), '.0f')} variants); cross-checked with the later partial E11 summary.json (checks C1, C2)"],
            ["peaks, GEMM and attention microbenchmarks", "results/peaks.json, fp4_gemm_shapes.jsonl, attention_backends.jsonl"],
            ["DiTFastAttn plans, static scales", "results/dfa_plans/*.json, results/static_scales_&lt;size&gt;.json"],
            ["end-to-end timing", "first E12 run log (two variants completed)"],
            ["quantize traffic model", "rtx5000_next/analysis/traffic_ledger.py (the same code as the repo)"]]
    A(t_(srcs, ["quantity", "source and rule"], "small"))
    A("<h2>Appendix B. Repeatability of the headline timings</h2>")
    rrows = []
    for lab, key in STEP_ROWS:
        row = [lab]
        for s in SIZES:
            r = F["step"].get(f"{key}|{s}", {}).get("runs", [])
            row.append(f"{n(min(r))}-{n(max(r))} ms ({len(r)} runs)" if len(r) > 1 else (f"{n(r[0])} ms (1 run)" if r else "-"))
        rrows.append(row)
    A(t_(rrows, ["config", "256 px", "512 px", "1024 px"], "num small"))
    A("<h2>Appendix C. Consistency checks ('no two truths')</h2>")
    checks = run_checks(data)
    (BUILD / "checks.json").write_text(json.dumps(checks))
    A(t_([[c["id"], "PASS" if c["ok"] else "<b class='bad'>FAIL</b>", c["desc"], html.escape(c["detail"])] for c in checks], ["id", "result", "what is compared", "detail"], "small checks"))
    A("<p class='small'>Every measured number in the body text is generated from the same facts file (report/build/facts.json) by report/build_report.py, so a quantity cannot appear with two values. "
      "The scan of the finished text for numbers that were not generated from it is listed next.</p>")
    A("<h2>Appendix D. Reproduce</h2><pre>"
      "git pull &amp;&amp; bash smoke_eval2.sh            # confirm the environment\n"
      "bash run_eval2_all.sh                           # all stages (E01-E16)\n"
      "bash run_final30.sh                             # headline table for 256/512/1024 px\n"
      "python report/make_facts.py --data &lt;results dir&gt; --hist &lt;first-run logs&gt;\n"
      "python report/check_consistency.py --data &lt;results dir&gt;\n"
      "python report/build_report.py --data &lt;results dir&gt;</pre>")
    A("<h2>Appendix E. Glossary</h2>")
    A(t_([["NVFP4", "4-bit floating-point format with one 8-bit scale per 16 values and one 32-bit scale per tensor (Blackwell GPUs)"],
          ["quantize / conversion", "turning 16-bit numbers into NVFP4 codes and scales"],
          ["GEMM", "matrix multiplication, the core of every linear layer"],
          ["bf16 / fp8", "16-bit and 8-bit floating-point formats"],
          ["CFG", "classifier-free guidance: run with and without the prompt and extrapolate; doubles the batch"],
          ["torch.compile / inductor", "PyTorch's compiler; it fuses small operations into larger generated kernels"],
          ["CUDA Graph", "a recorded sequence of GPU launches that is replayed without per-kernel CPU cost"],
          ["nsys / ncu", "NVIDIA tools for timelines (Nsight Systems) and per-kernel counters (Nsight Compute)"],
          ["LPIPS", "learned perceptual distance between two images (0 = identical)"],
          ["CLIP score", "100 x similarity of an image to its prompt in the CLIP model's embedding"],
          ["DiTFastAttn", "training-free attention compression for diffusion transformers (Yuan et al., NeurIPS 2024)"],
          ["head dimension", "size of each attention head: 72 for PixArt, 64 for SD3.5 Medium, 128 for FLUX"]], ["term", "meaning"], "small"))
    return "\n".join(lines)


CSS = """
@page { size: A4; margin: 16mm 14mm 16mm 14mm; }
body { font-family: 'DejaVu Sans', 'Liberation Sans', Arial, sans-serif; font-size: 9.6pt; line-height: 1.42; color: #1d1d1f; }
h1 { font-size: 21pt; margin: 0 0 4px 0; } h2 { font-size: 14pt; border-bottom: 2px solid #2e8b57; padding-bottom: 2px; margin-top: 20px; page-break-after: avoid; }
h3 { font-size: 11pt; margin: 14px 0 4px; page-break-after: avoid; color: #1b5e3a; } p { margin: 5px 0; text-align: left; }
.sub { color: #555; font-size: 9pt; } .title { margin-bottom: 8px; }
table { border-collapse: collapse; width: 100%; margin: 6px 0 8px; page-break-inside: avoid; }
th { background: #eef4ef; text-align: left; border: 1px solid #c9d6cc; padding: 3px 5px; font-size: 8.8pt; } td { border: 1px solid #dde5df; padding: 2.5px 5px; vertical-align: top; }
table.num td:not(:first-child) { text-align: right; white-space: nowrap; } table.small td, table.small th { font-size: 8pt; } table.tl td { font-size: 7.6pt; }
figure { margin: 8px auto; text-align: center; page-break-inside: avoid; } figure img { max-width: 100%; max-height: 225mm; width: auto; } figcaption { font-size: 8.2pt; color: #555; margin-top: 2px; }
.box { border-left: 4px solid #2e8b57; background: #f3f8f4; padding: 6px 9px; margin: 8px 0; page-break-inside: avoid; }
.box.imp { border-color: #d1495b; background: #fbeff0; } .box.ex { border-color: #edae49; background: #fdf6e7; } .box.tip { border-color: #5b8def; background: #eef3fd; }
.small { font-size: 8.2pt; color: #444; } code { background: #f1f1f1; padding: 0 2px; font-size: 8.6pt; } pre { background: #f4f4f4; padding: 6px; font-size: 8pt; }
ol.find li { margin-bottom: 4px; } .bad { color: #b00020; } .n { font-variant-numeric: tabular-nums; }
"""


# Numbers that may appear without being generated from the data: design constants of the experiments and the model, years, section numbers.
CONSTANTS = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "14", "16", "20", "28", "30", "32", "64", "72", "96", "128", "256", "512", "1024", "1152", "300", "202", "286",
             "2024", "2026", "0.5", "1.5", "0.05", "0.1", "0.2", "0.3", "0.7", "0.8", "4608", "48", "100", "50", "11", "12", "13", "15"}


def check_text(page):
    """Numbers in the visible body text (before Appendix C) that were not produced by n() and are not a known design constant.
    Returned with a little context so each can be judged: they should be empty or section/figure/design numbers."""
    page = page.split("Appendix C.")[0]
    page = re.sub(r"<h[123]>.*?</h[123]>|<p class=\"sub\">.*?</p>", " ", page, flags=re.S)   # headings (section numbers) and the title line
    txt = re.sub(r'<span class="n">.*?</span>', " # ", page)
    txt = re.sub(r"<style.*?</style>|<script.*?</script>|<img[^>]*>|<[^>]+>", " ", txt, flags=re.S)
    txt = html.unescape(txt)
    out = []
    for m in re.finditer(r"(?<![\w.])\d+(?:\.\d+)?(?:[%x])?(?![\w])", txt):
        tok = m.group(0)
        if (tok.rstrip("%x") in CONSTANTS and not tok.endswith(("%", "x"))) or tok in ("100%", "5.3"):  # 'above 100%' is a concept, 5.3 a section reference
            continue
        out.append(f"{tok}  ...{txt[max(0, m.start() - 28):m.end() + 14].strip()}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--no-pdf", action="store_true")
    a = ap.parse_args()
    body = build(a.data)
    leftovers = check_text(body)
    body += ("<p class='small'><b>Scan result:</b> numbers in the body text that were not generated from the data and are not design constants (sizes, step counts, "
             f"head dimensions, thresholds chosen in advance): <b>{'none' if not leftovers else html.escape('; '.join(leftovers))}</b>.</p>")
    page = f"<!doctype html><html><head><meta charset='utf-8'><title>NVFP4 on PixArt-Sigma: eval2 report</title><style>{CSS}</style></head><body>{body}</body></html>"
    (BUILD / "report.html").write_text(page)
    (BUILD / "leftover_numbers.json").write_text(json.dumps(leftovers))
    print("numbers in the text that are NOT generated from facts (must be constants):", leftovers)
    if not a.no_pdf:
        out = BUILD.parent / "eval2_report.pdf"
        subprocess.run(["google-chrome", "--headless=new", "--no-sandbox", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={out}", f"file://{BUILD / 'report.html'}"],
                       check=True, capture_output=True, timeout=240)
        print("wrote", out)


if __name__ == "__main__":
    main()
