# Progress tracker: next experiments on the RTX PRO 5000

All code in this folder is written but **never run**. First run on the server will find bugs. Paste the failing `logs/<ID>.log`.
`python progress_report.py` refreshes the Status column from `results/.done/<ID>` and `results/.fail/<ID>`. `run_all.sh` calls it after every test.
Counts: <!-- counts -->0 done, 0 failed, 0 todo<!-- /counts -->

## How to run

Extra packages that `pixi.toml` does not have: `sentencepiece` (T5 tokenizer, needed by `embeds`), `ninja` (needed by `kernels`), `matplotlib` (plots), `lpips` (optional, image metric).
Install: `$PY -m pip install -r rtx5000_next/requirements_next.txt`. If the pixi env has no pip: `pixi run pip install -r requirements_next.txt`, or `uv pip install --python $PY -r requirements_next.txt`.
The `kernels` stage also needs `nvcc` on PATH (the server has 13.3) or `CUDA_HOME` set.

```bash
nvidia-smi                                   # GPU must be idle
export PY=.pixi/envs/default/bin/python      # the pixi python
bash run_all.sh hf                           # N01 N02: model checks, no weights downloaded
bash run_all.sh kernels                      # K01-K03: builds the CUDA kernels, checks bytes against Triton
bash run_all.sh time                         # S01: step time, CPU launch time, CUDA Graph for every config
bash run_all.sh prof                         # P01-P04: nsys, ncu, analysis
bash run_all.sh embeds                       # T01: 6 real prompts -> prompts/embeds_pixart.pt (downloads the T5-XXL encoder from the PixArt-Sigma repo, 10 to 20 GB depending on stored dtype; check disk)
bash run_all.sh precision                    # Q01-Q03 (6 prompts each)
bash run_all.sh models                       # M01-M03: Sana 0.6B, SD3.5 Medium, FLUX.1-schnell (after PixArt; downloads transformers only)
bash run_all.sh plots                        # V01: results/figures/*.png
```
Env knobs: `MODEL`, `SIZES="256 512 1024"`, `CONFIGS="stock fused_triton"`, `ITERS`, `EMBEDS=embeds.pt`, `NVFP4_CODE=<folder with fused_adaln_nvfp4.py>`.
The scripts find the older code (`fused_adaln_nvfp4.py`, `scale_e2e.py`, `tier2_end_to_end.py`) in `../rtx_5000_runs` or `..`.

## Hypotheses

| ID | Hypothesis | Evidence from | Figure |
|---|---|---|---|
| H1 | Stock quantize is memory and launch bound (about 45x ideal DRAM) | P02, P04 | ncu_dram_and_throughput |
| H2 | Fusion removes kernels and DRAM traffic | P01, P02 | kernels_and_gpu_busy, ncu_quantize_time |
| H3 | What is left after fusion is the per-tensor max, the scale swizzle, 3x Q/K/V quantize and Python launch cost | S01, P01, K02 | nsys_kernel_time_by_category |
| H4 | Compiled stock is the fair baseline and may erase the gain at small sizes | S01 (stock_compile vs fused_*_compile) | step_time_by_config |
| H5 | 256 and 512 px are CPU-launch bound; CUDA Graph fixes it | S01, P01 | step_time_by_config, kernels_and_gpu_busy |
| H6 | With free quantization the step is bounded by GEMM + attention + glue | S01 (free_bound) | speed_of_light_free_quantize |
| H7 | Activation quantization, not weight quantization, dominates the error | Q01, Q02, Q03 | precision_* |
| H8 | The result holds on other models | M01-M03 | step_time_by_config |

## Tests

| ID | Test | Hypothesis | Script | Status |
|---|---|---|---|---|
| N01 | Repo access, size, params, fits 48 GB, diffusers class present | H8 | `hf/check_model_repos.py` | todo |
| N02 | Linear layer widths per model (meta device, config only) | H8 | `hf/probe_model_arch.py` | todo |
| K01 | CUDA row kernel: bytes vs Triton and torchao, time | H3 | `bench/cuda_vs_triton.py --only cuda_row` | todo |
| K02 | CUDA swizzled kernel and amax+quantize: bytes, time | H3 | `bench/cuda_vs_triton.py --only cuda_swizzled amax_swizzled` | todo |
| K03 | CUDA LayerNorm + modulate + quantize: bytes, time | H3 | `bench/cuda_vs_triton.py --only ln_mod_quant` | todo |
| S01 | Step time matrix: 11 configs x sizes, eager, CPU launch, CUDA Graph | H3 H4 H5 H6 | `profiling/run_step_matrix.sh` | todo |
| P01 | nsys per config and size: kernels per step, GPU busy, time by category | H2 H3 H5 | `profiling/nsys_step.sh` | todo |
| P02 | ncu per layer (qkv, fc1, fc2, cross K/V): DRAM vs ideal, throughput | H1 H2 | `profiling/ncu_layer.sh` | todo |
| P03 | Summarize nsys | H2 | `analysis/analyze_nsys.py` | todo |
| P04 | Summarize ncu | H1 | `analysis/analyze_ncu.py` | todo |
| Q01 | Precision ablation w4a16 / w16a4 / w4a4 at 512 px, validated against real NVFP4 | H7 | `precision/precision_ablation.py --validate-stock` | todo |
| Q02 | Same at 1024 px | H7 | `precision/precision_ablation.py --size 1024` | todo |
| Q03 | PixArt images for each variant (needs real embeds for a meaningful image) | H7 | `precision/precision_images_pixart.py` | todo |
| M01 | Sana 0.6B 1024 (download 3.7 GB): download, smoke, matrix, precision | H8 | `bench/new_model_suite.sh sana_600m 1024` | todo |
| M02 | SD3.5 Medium 1024 (download 5.1 GB, gated, access confirmed by N01) | H8 | `bench/new_model_suite.sh sd35_medium 1024` | todo |
| M03 | FLUX.1-schnell 1024 (download 24 GB) | H8 | `bench/new_model_suite.sh flux_schnell 1024` | todo |
| V01 | All figures | all | `viz/plot_all.py` | todo |
| T01 | Encode 6 real prompts with T5 once, save `prompts/embeds_pixart.pt` (about 15 MB, keep it). Edit `prompts/prompts.json` to change the prompts. Q01-Q03 use it automatically | H7 | `prompts/make_prompt_embeds.py` | todo |

## Files to copy back for local analysis and GUI

- `results/nsys/*.nsys-rep`: open in Nsight Systems (timeline, kernel gaps, CPU launch). `results/nsys/*.csv`: input for `analysis/analyze_nsys.py`.
- `results/ncu/*.ncu-rep`: open in Nsight Compute (Memory Workload Analysis chart, roofline, sectors per request, source counters). `results/ncu/*.csv`: input for `analysis/analyze_ncu.py`.
- `results/*.jsonl`, `results/meta/`, `results/hf_check.json`. Then run `analysis/*.py` and `viz/plot_all.py` locally (no GPU needed).
- Optional on nsys: `NSYS_GPU_METRICS=1` is not wired. Add `--gpu-metrics-devices=all` by hand if the driver allows it.

## Configs (profiling/configs.sh)

| Config | Meaning |
|---|---|
| bf16, fp8 | references |
| stock | torchao NVFP4 |
| stock_compile | stock with `torch.compile` on the 28 blocks |
| fused_triton_narrow | Triton quantize on 1152-wide inputs only (the old `fused`) |
| fused_triton | Triton quantize on every Linear width (the old `fused_wide`) |
| fused_triton_op_compile | Triton kernel wrapped as a custom op, compiled. This is the fix for the compile failure |
| fused_cuda_row, fused_cuda_swizzled | CUDA kernels (written by agents, in `cuda_kernels/`) |
| fused_cuda_swizzled_compile | same, as a custom op, compiled |
| free_bound | block quantize costs nothing (cached zero buffers). Images are meaningless. Gives the lower bound for any fusion |

## Known risks in code that has not run

- CUDA kernels were written without compiling (local nvcc is 11.5, no sm_120). Build flags in `cuda_ext.py` include `--fmad=false`. If K01 to K03 fail to build, send the log.
- `quant2lvl_swizzled.cu` uses IEEE division for `amax/6`. Triton may lower it differently, so a few scale bytes could differ. K02 reports the count.
- `ln_mod_quant_cuda.cu` can differ from Triton by rare one-ULP LayerNorm ties. K03 reports the count.
- Adapters for DiT, Sana, SD3.5 and FLUX follow diffusers signatures from memory. `model_smoke.py` runs first in `new_model_suite.sh`.
- ncu layer names assume diffusers PixArt names (`ff.net.0.proj`, `ff.net.2`). Other models: `python step_target.py --model X --size S --mode bf16 --list-layers`, then set `LAYERS`.
- Sana's FFN is convolutional, so fewer layers are NVFP4 linears. N02 shows the share.
- Precision runs in bf16, not fp16: fp16 gave NaN images earlier.
- `torch.library.custom_op` wrapping may still hit a torchao tensor-subclass limit under `torch.compile`. S01 will show it.
