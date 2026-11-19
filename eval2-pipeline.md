# eval2: making the NVFP4 stack beat compiled bf16 and compiled stock

Living log of the second evaluation round. Code is in `rtx5000_next/`, tracker in `rtx5000_next/PROGRESS_EVAL2.md`, bugs in `rtx5000_next/BUGS.md`, every command in `commands.md`. Update the "Results" section as runs finish.

## 1. Why this round exists

Round 1 (S01 step times, nsys, ncu, roofline, precision) gave, at 1024 px (CUDA Graph ms): bf16 103.7, stock 369.7, stock_compile 75.4, fused_triton_op_compile 74.8, fused_cuda_swizzled_compile 74.8. "Fusion ties compile" looked like a result. The nsys kernel tables say it is not:

1. **The fused kernel did not run inside the compiled blocks.** Compiled-fused runs show our kernel 6 times per step (286 sites), the same 1288-1291 inductor kernels as stock_compile, and identical top kernels. The old fusion monkeypatched `NVFP4Tensor.to_nvfp4`; torch.compile did not take the patch. So fused-vs-compile is untested.
2. **Attention is 57% of the compiled 1024 px step** (flash 30.3 ms, masked cross-attention 10.3 ms, pad/copy 1.95 ms of 75). Flash runs with head dim 72 padded to 96. Cross-attention runs on 300 padded text tokens although real prompts use 26-36.
3. NVFP4 GEMM reaches only 274-302 TFLOP/s on PixArt shapes (K=1152) against 707 at 8192^3.
4. The quantize path is memory-bound: ideal traffic about 6 GB per forward at 1024 px; the compiled path moves about 13 GB.
5. The baseline set was not fair: bf16 and fp8 only eager, torchao's own fused kernel (`use_triton_kernel=True`) never measured, end-to-end time (T5, VAE) never measured.

Goal: remove the real bottlenecks with training-free changes, measure every change with real prompts and image-quality metrics, and report speed next to quality.

## 2. What was built (and why each piece exists)

| Piece | File | Reason |
|---|---|---|
| NVFP4 linear that compile can see | `fused_linear.py` | quantize custom op + `torch._scaled_mm`; both are ordinary graph nodes, a counter inside the op proves every site ran our kernel. Options: static activation scale (no amax pass), shared Q/K/V quantize, concat, selective quantization |
| Attention options | `attention_patch.py` | one `sdpa()` for backend choice (flash, cuDNN, efficient, Sage2/3), cross-attention text trimming (exact), K/V pooling; the PixArt processor uses the fused projections |
| DiTFastAttn | `ditfast/attention.py`, `plan_search.py`, `make_plans.py` | training-free attention compression: window attention with residual sharing, sharing across timesteps, sharing across the CFG halves; greedy plan search on calibration prompts; one opaque op keeps the compiled graph identical at every step |
| Limited-interval CFG sampler | `cfg_interval.py` | reproduces diffusers' pipeline exactly (test) and drops the unconditional half outside a guided band; batching of several prompts |
| Variant registry | `variants.py` | one string describes a setup (`module+be=cuda_swizzled+static+concat+trim+xcudnn+dfa0.1+compile`) so speed, quality, timing and calibration all build the same thing |
| Static scale calibration | `calibrate_static_scales.py` | per-layer amax over a real sampling run, with the drift across denoise steps |
| Real-prompt quality | `quality/eval_images.py`, `run_quality.py` | LPIPS, SSIM, PSNR, CLIP score, CLIP similarity to the reference, per prompt; images, contact sheet, speed-vs-quality plot |
| Real w4a16 vs w4a4 | `precision/real_w4a16_vs_w4a4.py` | weights-only NVFP4 via `act_quant_kwargs=None` on the real torchao model; the gap is the quality price of the activation quantize step (no emulation) |
| Timing per stage | `timing/stages.py`, `profile_stages.py`, `analysis/analyze_stages.py`, `nsys_step.sh` | GPU busy, GPU gaps, kernel counts, CPU launch time per stage; nsys `nvtx_gpu_proj_sum`, `nvtx_kern_sum`, `cuda_kern_exec_sum` as the independent check |
| Memory ledger | `analysis/traffic_ledger.py` | bytes per site for the current path and for each lever, in ms at measured bandwidth |
| Compute-side microbenchmarks | `bench/fp4_gemm_shapes.py`, `bench/attention_backends.py` | does concat help the FP4 GEMM, at what batch does it saturate, which attention kernel is fastest |
| End-to-end timing | `bench/pipeline_e2e.py` | T5, each denoise step, VAE, host idle time |
| Orchestration | `smoke_eval2.sh`, `run_eval2.sh` | one confirm command (about 10-15 min), then every stage in order with resume |
| Tests | `tests_cpu/` (71 checks), `tests_gpu/` | each module has a test that states what it must mean |

## 3. Design decisions and reasons

- **Module stack instead of fixing the monkeypatch.** The patch lives inside torchao's tensor-subclass dispatch, which AOT tracing handles in ways we do not control. The module stack avoids it and gives the quantize op a counter that works under compile.
- **Layer-level bit-exactness is the correctness test, not whole-model agreement.** On identical input our module equals torchao's NVFP4 linear bit for bit (both quantize backends, 4 shapes). Whole models differ between two exact implementations by about the quantization error (see Findings): NVFP4 rounding is discontinuous, so any bf16-level difference upstream changes the rounding noise realization. What must hold is that each implementation is about equally far from bf16 (0.1195 stock, 0.1202 module, 0.1203 compiled module in the local check).
- **Calibration prompts differ from evaluation prompts** (`prompts_calib.json` vs `prompts.json`) for static scales and DiTFastAttn plans, so the quality numbers are not in-sample.
- **Trim text tokens, cuDNN for cross-attention are exact or near-exact "small hacks"**; they run as toggles on the best stack, after the large items.
- **DiTFastAttn at attention-core level only.** The unconditional half's Q/K/V/out projections still run; skipping them would also save NVFP4 quantize and GEMM work but needs module-level changes. Possible next step.
- **Limited-interval CFG and DiTFastAttn's CFG sharing both cut unconditional work.** The quality grid includes each alone and both together.
- **TeaCache-style step caching is not implemented**: lower payoff per effort for a 14-step schedule and a real quality risk (reported 2.12x at LPIPS 0.215 on FLUX schnell, 4.02x at 0.396).

## 4. Findings so far (verified locally on the RTX 5060; laptop timings are not results)

- Module stack: quantize op runs once per site per forward, eager and compiled (202 sites with shared Q/K/V and K/V on PixArt-Sigma; stock quantizes 280+6 times). This is the test the old fusion failed.
- Layer outputs are bit-identical to torchao's NVFP4 linear (triton and torchref backends).
- Whole-model error vs bf16 (256 px, mean over 5 timesteps): stock 0.1195, module 0.1202, compiled module 0.1203.
- Real weights-only w4a16 (256 px, worst timestep): 0.1448 vs w4a4 0.2155 relative L2 to bf16.
- The earlier Q01 mismatch "emulated vs real torchao differ by 0.06-0.17" is mostly the discontinuity effect above, not necessarily an emulation bug; judge an emulation per layer on identical inputs.
- Block 0 self-attention sees identical unconditional and conditional inputs (same latent, same timestep; text enters at cross-attention), so CFG sharing is exact there at no quality cost.
- Stage table (laptop, 256 px, eager): stock 15454 kernels and 38% GPU busy; module stack 2694 kernels; every stage is launch-bound at 256 px. The `quant` stage shows 474 kernels for 202 calls with the Triton backend: the `to_blocked` swizzle runs as separate small kernels, which the CUDA swizzled backend removes.
- Static-scale calibration found layers whose amax changes up to 46x across denoise steps (`attn2.to_out` at some blocks). Static scales are risky for those; the quality run decides.
- Masked cross-attention: cuDNN is faster than the default (efficient) kernel in the laptop microbenchmark (0.80 vs 1.93 ms at 300 tokens, 0.26 vs 0.65 ms at 64); self-attention favoured flash there. Needs the server GPU to confirm.
- K/V pooling at 256 px destroyed quality (LPIPS 0.80): high risk; kept as a measured option.

## 5. Quality gate

A variant is acceptable when its mean LPIPS to the bf16 same-seed reference is not worse than stock NVFP4's LPIPS by more than the margin set here: **margin = 0.02 (change if you disagree)**. CLIP-score change is reported, not gated. Every speed claim in the final table carries its LPIPS and CLIP score.

## 6. Plan for the server (3 GPU hours)

```bash
git pull
bash smoke_eval2.sh        # confirms every step below will run
bash run_eval2.sh          # E01-E15
```

Rough stage times in `PROGRESS_EVAL2.md` / `bash run_eval2.sh --list` (guesses). Stages run to completion; resume with `--from`.

## 7. Memory-movement levers and compute-side drafts

Quantize traffic (1024 px, batch 2, ledger): now 13.4 GB (12.1 ms at 1113 GB/s) against a one-pass floor of 6.1 GB (5.5 ms). Levers: static scale (5.9 GB), shared Q/K/V and K/V quantize (2.5 GB), LN+modulate fused into quantize (3.2 GB, kernel exists: K03), GELU+quantize in the fc1 GEMM epilogue (4.2 GB, draft: needs a CUTLASS SM120 block-scaled epilogue, build risk). The output-scale and bias passes (23.6 GB eager) are already folded by inductor when compiled.

Compute-side drafts (not implemented): attention FLOP cuts (DiTFastAttn, Sage FP4/INT8, K/V pooling), FP4 GEMM efficiency (concat, larger batch; E02 measures it), CFG interval, CUDA Graph for sizes up to 512 px.

## 8. Results

(filled in after the server run: step-time table with compiled baselines, stage tables, quality table with Pareto plot, end-to-end times, DiTFastAttn plan summaries.)

## 9. Open questions

- Does the compiled module stack beat compiled stock by at least 3% at 1024 px once the fused kernel really runs? (E05/E06)
- Is static scaling safe given the amax drift? (E04, E11)
- Does Sage (head dim 72) run, and at what quality? (E02, E11)
- How much does DiTFastAttn compress at 1024 px at an acceptable LPIPS? (E10, E11)
- New-model adapters (Sana, SD3.5, FLUX) from memory: E14.
