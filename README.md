# PixArt-Sigma NVFP4: fused quantize kernel on RTX PRO 5000

Goal: show how much of a diffusion-transformer step is spent in the unfused
LayerNorm -> AdaLN modulate -> NVFP4 quantize chain, and how much a fused kernel removes.
This file is the single list of what we run, what each run answers, and its status.
Plain-language background: `GUIDE.md` and `CODE_PRIMER.md`.
Older step lists `RUN_NOW.md` and `RUN_30MIN.md` are superseded by this file.

Server: `team13@10.4.25.63`, repo at `~/PAA_Proj`. Python: `.pixi/envs/default/bin/python`.

## 1. Setup (once)

| Command | Purpose |
|---|---|
| `pixi install --locked` | Build the environment from `pixi.lock` |
| `pixi run gpucheck` | Check the GPU is visible |
| copy models with `rsync` (see `RUN_NOW.md` step 2) | The 3 PixArt transformers + VAE; runs are offline |
| `pixi run smoke` | 1-minute check that the model path works |
| `nvidia-smi` | Before every run: make sure no other user is on the GPU |

## 2. Run commands

| Command | What it runs | Time |
|---|---|---|
| `nohup bash run_full_2h.sh > run.log 2>&1 &` then `tail -f results/progress.log` | Every stage below, in order. One copy at a time. | ~10 min on a clean GPU (up to ~2 h worst case) |
| `bash run_breakdown_only.sh` | Only stage B (per-part breakdown, now with per-layer-type tables) | ~5 min |
| `bash boundary_ncu.sh` | Stage K: per-stage GPU time + DRAM traffic of the LN -> modulate -> quantize boundary | ~10-15 min |
| `.pixi/envs/default/bin/python make_report.py` | Builds `results/REPORT.md` from the results files | seconds |
| `tar czf results.tgz results run.log` | Pack results to copy back | seconds |

Single runs (what the stages call):
```bash
PY=.pixi/envs/default/bin/python
$PY scale_e2e.py --size 512 --mode fused --steps 14 --reps 3         # modes: bf16 fp16 fp8 stock fused
$PY breakdown_step.py --size 512 --mode stock                        # per-part breakdown of one step
$PY external_tests_inspired/graph_step_bench.py --size 512 --mode fused
EXTRA_OK_K=4608,4096 $PY scale_e2e.py --size 1024 --mode fused       # widen fusion to the MLP 2nd layer
```

## 3. Stages and status

| Stage | Test | Question | Script | Status |
|---|---|---|---|---|
| A | Whole-model step time, 5 precisions x 256/512/1024px, + stock-vs-fused image diff | How much faster is the whole transformer? | `scale_e2e.py`, `compare_images.py` | Ran (2026-10-25). Check `results/e2e.jsonl`, `images.log` |
| B | Per-part breakdown of one step; LN + modulate vs quantize; GPU idle share | Where does the unfused chain's time go? | `breakdown_step.py`, `breakdown_table.py` | **Ran, results below** |
| C | Batch 4 and 8 at 256px, batch 4 at 512px | Does the gain hold with bigger batches? | `scale_e2e.py --batch` | Ran. Not read yet |
| D | CUDA Graph replay of the real step | Is the step launch-bound? | `external_tests_inspired/graph_step_bench.py` | Ran. Not read yet |
| E | `nsys` kernel-time split | Which kernel categories dominate? | `classify_nsys.py` | Ran. Not read yet |
| G | Triton boundary sweep + `ncu` | Kernel-level speed and DRAM traffic | `sweep_tier1.py`, `ncu_sweep.sh` | Ran. Not read yet |
| H | CUDA C++ fusion vs launch cost | Fusion or launch overhead? | `external_tests_inspired/launch_vs_fusion.cu` | Ran 2026-10-24: fusion 2.9x to 5.0x, graph 1.0x |
| I | `torch.compile` stock vs fused | Does a compiled baseline close the gap? | `scale_e2e.py --compile` | **Failed** (exit 1). Read `results/logs/compile_stock_256.log` |
| K | LayerNorm / AdaLN modulate / quantize, stage by stage: GPU time and DRAM traffic, unfused vs fused | What does each of the 3 stages cost, and what is left after fusion? | `boundary_stages.py`, `boundary_ncu.sh`, `summarize_boundary.py` | **Written, not run.** `bash boundary_ncu.sh` (~10-15 min) |
| J | Widen fusion to K=4608 / 4096 | Does covering the MLP 2nd layer remove the biggest leftover? | `EXTRA_OK_K=...` | **Next, not run** |

## 4. Results so far (clean GPU, 2026-10-25, one denoise step, batch 1)

Step time (ms) and share of the step:

| Image | Mode | Step | LN + modulate | Quantize total | GEMM | Attention | GPU idle |
|---|---|---|---|---|---|---|---|
| 256px | bf16 | 15.1 | 2.3 (15%) | 0 | 7.5 (50%) | 2.5 (17%) | 48% |
| 256px | NVFP4 stock | 118.9 | 3.6 (3%) | 95.3 (80%) | 9.4 (8%) | 2.8 (2%) | 73% |
| 256px | NVFP4 fused | 71.1 | 3.6 (5%) | 48.9 (69%) | 8.6 (12%) | 2.7 (4%) | 78% |
| 512px | bf16 | 24.4 | 1.6 (7%) | 0 | 15.6 (64%) | 5.3 (22%) | 7% |
| 512px | NVFP4 stock | 117.5 | 2.3 (2%) | 92.7 (79%) | 9.9 (8%) | 6.4 (6%) | 40% |
| 512px | NVFP4 fused | 90.3 | 2.6 (3%) | 62.8 (70%) | 10.7 (12%) | 6.7 (7%) | 56% |
| 1024px | bf16 | 103.5 | 4.9 (5%) | 0 | 49.9 (48%) | 41.5 (40%) | 2% |
| 1024px | NVFP4 stock | 372.7 | 4.4 (1%) | 308.7 (83%) | 14.0 (4%) | 38.2 (10%) | 1% |
| 1024px | NVFP4 fused | 253.5 | 4.4 (2%) | 190.1 (75%) | 13.8 (5%) | 38.0 (15%) | 2% |

Fused vs stock, whole step: 1.67x (256px), 1.30x (512px, noisy: another run gave 1.69x), 1.47x (1024px).

What this shows:
1. **LayerNorm + AdaLN modulate cost only 1% to 3% of the stock step.** The unfused overhead is in quantization (79% to 83%), not in LN or modulate.
2. **The FP4 matmul is fast** (14 ms vs 50 ms bf16 at 1024px, 3.6x), but quantization costs more than it saves.
3. **Fusion removes about a third to a half of the quantize time** (95 to 49 ms, 93 to 63 ms, 309 to 190 ms), but the quantize overhead (27 ms) is untouched.
4. **At 256 and 512px the GPU is idle 40% to 78% of the step.** The CPU launching small kernels is the limit there. At 1024px the GPU is full (98%).
5. **At 1024px the fused path still spends 190 ms on quantize,** while the kernel alone should take about 8 ms. The rest is work we do not fuse (estimate: the 28 MLP second-layer calls at K=4608, about 100 ms, plus the scale re-layout). Experiment J tests this.

Caveats: a 2026-10-24 run on a shared GPU gave about 2.2x larger absolute times and should not be quoted. Fused 512px varies run to run (70 to 90 ms) because it is CPU-bound. Text embeddings are random numbers, so images are only comparable between modes.

## 5. Next steps (in order)

1. Paste `tail -25 results/logs/breakdown_fused_1024.log` (top kernels) and `cat results/graph_step.jsonl results/images.log`.
2. **Experiment J:** run the fused 1024 case with `EXTRA_OK_K=4608,4096`, then check the images still match stock and the step time drops.
   ```bash
   EXTRA_OK_K=4608,4096 .pixi/envs/default/bin/python scale_e2e.py --size 1024 --mode fused --steps 14 --reps 3 --tag _k4608
   .pixi/envs/default/bin/python -c "import numpy as np; a=np.load('results/stock_1024_b1.npy').astype('f4'); b=np.load('results/fused_1024_b1_k4608.npy').astype('f4'); print('mean|diff|', abs(a-b).mean(), 'max', abs(a-b).max())"
   ```
   (`compare_images.py` does not pair tagged files, so compare by hand. 0.0 means identical.)
   Check `fallback_calls` falls from about 30 per step to near 0.
3. Read the `torch.compile` error and decide whether to fix it.
4. Read stages C, D, E, G and add their numbers to section 4.
5. Later: move the scale re-layout into the kernel, quantize Q/K/V input once, newer models.
