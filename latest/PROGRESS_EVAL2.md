# eval2 tracker: NVFP4 module stack, attention, DiTFastAttn, limited-interval CFG, real-prompt quality

Plan, reasons and findings: [../eval2-pipeline.md](../eval2-pipeline.md). Bugs: [BUGS.md](BUGS.md). Commands: [../commands.md](../commands.md).
`run_eval2.sh` writes the markers; `python progress_report.py` refreshes the Status column (this file is separate from PROGRESS.md on purpose, so the server's local edits to PROGRESS.md never block a `git pull`).
Counts: <!-- counts -->3 done, 0 failed, 12 todo<!-- /counts -->

## Order on the server

```bash
bash smoke_eval2.sh        # about 10-15 min: confirms every step below will run. Fix any FAIL first.
bash run_eval2.sh --list   # the stages and rough times
bash run_eval2.sh          # all stages in order, each to completion; resume with --from E06, rerun one with FORCE=1 --only E06
```

## Tests

| ID | Test | Why | Script | Status |
|---|---|---|---|---|
| E01 | Encode 8 calibration prompts with the PixArt T5 | static scales and DiTFastAttn plans must be tuned on prompts other than the evaluation prompts | `prompts/make_prompt_embeds.py --prompts prompts/prompts_calib.json` | DONE 11-20 01:27 |
| E02 | FP4 GEMM efficiency vs shape; attention backends (flash, cuDNN, Sage) at PixArt, SD3.5, FLUX shapes | compute-side limits before any end-to-end run: does concat help the GEMM, which attention kernel is fastest, is Sage usable | `bench/fp4_gemm_shapes.py`, `bench/attention_backends.py` | todo |
| E03 | Real w4a16 vs w4a4: transformer error and image quality | the quality price of the activation quantize step with no emulation | `precision/real_w4a16_vs_w4a4.py` | todo |
| E04 | Static activation scales per size (and amax drift across denoise steps) | removes the amax pass (memory traffic); drift shows where it is unsafe | `calibrate_static_scales.py` | DONE 11-20 01:28 |
| E05 | Step time, fair baselines: bf16, fp8, stock, stock_tk, all compiled | the speedup claim needs compiled baselines and torchao's own fused kernel | `profiling/run_step_matrix.sh` with EVAL2_BASELINES | todo |
| E06 | Step time, module stack: dynamic, no-share, static, concat, selective, CUDA kernel | tests that the fused kernel really runs in the compiled graph and what each memory-traffic lever buys | `profiling/run_step_matrix.sh` with EVAL2_MODULE | todo |
| E07 | Step time, attention options on the best stack | text trim, cuDNN, Sage, K/V pooling, each against the same stack | `profiling/run_step_matrix.sh` with EVAL2_ATTENTION | todo |
| E08 | Per-stage timing: GPU busy, gaps, kernel counts, CPU launch time | tells launch-bound stages from memory-bound ones | `timing/profile_stages.py`, `analysis/analyze_stages.py` | todo |
| E09 | nsys with stage ranges and kernel launch tables | the independent check of E08 (`nvtx_gpu_proj_sum`, `cuda_kern_exec_sum`) | `profiling/nsys_step.sh` | todo |
| E10 | DiTFastAttn plans at three error thresholds | the attention compression itself | `ditfast/make_plans.py` | DONE 11-20 01:29 |
| E11 | Real-prompt quality of every variant, with time per image | the speed-vs-quality trade-off for each optimization | `quality/run_quality.py` | todo |
| E12 | End-to-end time: T5, steps, VAE for the headline variants | what a user waits for | `bench/pipeline_e2e.py` | todo |
| E13 | Traffic ledger and whole-forward roofline | memory-movement budget and distance from the roofs | `analysis/traffic_ledger.py`, `analysis/roofline_whole_forward.py` | todo |
| E14 | Smoke of Sana 0.6B, SD3.5 Medium, FLUX | finds adapter bugs before the next window | `model_smoke.py` | todo |
| E15 | Package results | one file to copy back | `tar` | todo |
