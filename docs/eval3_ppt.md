# Eval 3 deck notes (poster / demo / viva)

Status: SKELETON. Numbers and claims are filled only from result files after the runs (`eval3/make_ppt_md.py`, not written yet).
Rule: a box is `[x]` only when its result file exists and its checker passes. Prose is written after the numbers exist.
Source of the plan: `docs/EVAL3_HANDOFF_PLAN.md`.

## A. Things to explain (no numbers needed)
- **NVFP4 W4A4.** FP4 (e2m1) codes, one e4m3 scale per 16 values, one fp32 per-tensor scale. Weights and activations are both FP4. The activation quantize step runs every forward and costs memory traffic.
- **Our method.** A torch.compile-visible module stack: quantize custom op (CUDA swizzled or Triton kernel) plus `torch._scaled_mm`. Bit-identical to torchao per layer. Levers: shared Q/K/V quantize, concat GEMM, selective bf16 layers, text trim, cuDNN cross-attention, Sage2 attention, limited-interval CFG, torch.compile.
- **What stays bf16.** Attention (except Sage2: INT8/FP8), LayerNorm, GELU, residuals, VAE, selective layers.
- **Why static activation scale was dropped.** Per-layer amax drifts up to 46x across denoise steps; a fixed scale clips or wastes range. Kept only as an ablation row.
- **Why LPIPS needs a noise floor.** NVFP4 rounding is discontinuous: a tiny upstream difference flips roundings and gives a different but equally valid image. So we measure bf16 compiled vs eager (floor), bf16 with another seed (ceiling), fp8 and stock NVFP4 on the same scale, with paired confidence intervals.
- **Why speed-up shrinks at 1024 px.** (Amdahl) Attention is unaffected by NVFP4 and grows with token count. See section C6.
- **Roofline method.** Roofs are measured (best of a shape sweep), workload shapes are recorded from the real model, efficiency above 100% means the model is wrong and is never quoted.

## B. Method and baselines at a glance
| item | value | source file | status |
|---|---|---|---|
| model, steps, CFG | PixArt-Sigma, 14 steps, CFG batch 2 | `variants.py` | [ ] FILL |
| baselines | bf16c, fp8c, stock NVFP4 c (stock_tk: not measurable unless MSLK builds) | step_target.jsonl | [ ] FILL |
| final stack F | module cuda_swizzled + concat + selective + trim + xcudnn + sage2 + compile | `variants.py` | [ ] FILL |

## C. Reviewer feedback (Eval 2) -> evidence
### C1. Separate the gain of each optimisation, and of the kernel itself  [ ]
- What we did: cumulative ladder, leave-one-out, op-level kernel isolation (torchao quantize vs ours, GB/s vs copy roof).
- Evidence: FILL (waterfall, leave-one-out table with repeat ranges) from `results/eval3/ablation/*.json`.
- Claim: FILL (generated; "no measurable effect" when inside the repeat range).
- Caveat: FILL.
### C2. Stock torchao under identical CUDA Graph capture  [ ]
- Evidence: one table, same compile/graph/iters, 3+ repeats, incremental gain column. FILL from `step_target.jsonl`.
- `stock_tk`: FILL (measured or "not measurable: MSLK not built").
### C3. Quality of static scale and activation quantization with more prompts and seeds  [ ]
- Evidence: w4a16, stock, module, +static, +selective, leave-one-out; 30 prompts x 3 seeds (512), 20 x 2 (1024); paired CIs. FILL from `results/eval3/quality/*/summary.json`.
- Per-layer amax drift list. FILL.
### C4. Other attention implementations  [ ]
- Evidence: backend table per (head dim, tokens, masked): flash, cuDNN, efficient, Sage2, flash-attn 2, xformers, Sage3 (SD3.5/FLUX), DiTFastAttn; step time and quality at 1024 px. FILL.
- How attention is kept efficient: see `docs/EVAL3_HANDOFF_PLAN.md` section 6.
### C5. Other GPUs  [ ]
- Evidence: RTX PRO 5000 vs RTX 5060 Laptop (both sm120): roofs, speed-ups, quality reproduction. FILL from `eval3/cross_gpu.py`.
- Not measured: sm100 (B200). Roofline prediction only. FILL.
### C6. Why the speed-up falls at 1024 px  [ ]
- Evidence: per-class time (gemm, attention, quantize, glue) for bf16c and ours at 256/512/1024; Amdahl prediction vs measured. FILL from `eval3/amdahl.py`.

## D. Findings already established (re-verify against files before quoting)
- Noise floor at 512 px: bf16 compiled LPIPS 0.126 [0.112, 0.142]; other seed 0.627; fp8c 0.230; stock NVFP4 0.430. (server, clean run; source `results/eval3/quality/noise_floor_512/summary.json`)  [ ]
- Our stack vs stock NVFP4 at 1024 px: 65.4 vs 76.8 ms; same LPIPS (0.427 vs 0.430). (source `step_target.jsonl`, `levers_512`)  [ ]
- Trim, not cuDNN, gives the cross-attention gain (1024 px: concat 66.4 -> trim 60.9 -> +xcudnn 59.5 ms).  [ ]
- Sage2: 59.5 -> 34.9 ms at 1024 px; LPIPS inside the noise floor at 512 px; 1024 px quality pending.  [ ]
- Selective quantization: LPIPS 0.417 -> 0.355 for about 1 ms.  [ ]
- Concat and shared Q/K/V quantize: marginal.  [ ]

## E. Other models (SD3.5 Medium, FLUX schnell)  [ ]
FILL: conversion pass, layers and parameter share in FP4, premises (memory bound, FP4 beats bf16, attention share, mask, CFG), roofline, step time. Quality only if T51 completes.

## F. Limits to state honestly
- NVFP4 LPIPS 0.43 vs fp8 0.23; stock NVFP4 is only about 2% faster than fp8 at 1024 px.
- Prompts are author-picked showcase prompts.
- One GPU family measured (two sm120 parts); sm100 predicted only.
- Image quality for other models: FILL (only if T51 completes).
- `stock_tk` / Sage3 / flash-attn 2 / xformers: FILL (built or not).

## G. Demo and poster checklist
- [ ] demo runs end to end on the laptop (512 px, cached embeddings)
- [ ] demo runs on the server (1024 px)
- [ ] poster figures generated by script from results
- [ ] viva notes written (`docs/VIVA_NOTES.md`), contribution section filled by the student
