# Commands

Every command used so far on the RTX PRO 5000 server, in the order they were run, then the eval2 commands. All run from `rtx_5000_runs/rtx5000_next` unless stated. Times are rough. `pixi run` puts the pixi python first on PATH; it works for any command.

## Round 1 (done)

| Step | Command | Purpose | Time |
|---|---|---|---|
| setup | `git pull && pixi install && pixi run gpucheck` | environment and GPU check | 2 min |
| setup | `pixi run python -m pip install -r requirements_next.txt` | sentencepiece, protobuf, ninja, matplotlib, lpips (not in `pixi.toml`) | 2 min |
| setup | `df -h . ~/.cache/huggingface`; `which nvcc nsys ncu`; `grep -i RmProfilingAdminOnly /proc/driver/nvidia/params` | disk, tools, ncu permission (1 means ncu needs admin) | 1 min |
| downloads | `pixi run download` | PixArt-Sigma transformers for 256/512/1024 and the shared VAE | 8 GB |
| downloads | `pixi run huggingface-cli download PixArt-alpha/PixArt-Sigma-XL-2-1024-MS --include "text_encoder/*" "tokenizer/*"` | T5 prefetch for prompt embeddings | 10-20 GB |
| downloads | `pixi run python hf/download_model.py <sana_600m\|sd35_medium\|flux_schnell> --size 1024` | new-model checkpoints (transformer only) | 3.7 / 5.1 / 24 GB |
| downloads | `pixi run python -c "import lpips; lpips.LPIPS(net='alex')"` | prefetch the AlexNet weights LPIPS needs | seconds |
| N01 N02 | `pixi run bash run_all.sh hf` | repo access, sizes, layer widths (no GPU) | 1 min |
| kernels | `pixi run python -c "import cuda_ext as c; [c.load_ext(n) for n in ('quant2lvl_row','quant2lvl_swizzled','ln_mod_quant_cuda')]"` | pre-build the CUDA kernels (failed first with C++17; fixed to C++20) | 5 min |
| K01-K03 | `pixi run bash run_all.sh kernels` | CUDA kernels vs Triton, byte for byte | 5-10 min |
| S01 | `pixi run bash run_all.sh time` | step time, 11 configs x 256/512/1024, eager, CPU launch, CUDA Graph | 35-50 min |
| P01 | `SIZES="512 1024" pixi run bash profiling/nsys_step.sh` | nsys per config | 25-35 min |
| P01 rerun | `CONFIGS="stock fused_triton_narrow fused_triton fused_cuda_row fused_cuda_swizzled free_bound" SIZES="512 1024" pixi run bash profiling/nsys_step.sh` | the 12 runs that failed on the NVTX hook bug | 15-20 min |
| P03 | `pixi run python analysis/analyze_nsys.py` | nsys summary | 1 min |
| P02 | `SIZES=1024 CONFIGS="stock fused_triton fused_cuda_swizzled" LAYERS='qkv:transformer_blocks\.3\.attn1\.to_q$ fc1:transformer_blocks\.3\.ff\.net\.0\.proj$' pixi run bash profiling/ncu_layer.sh` | ncu with memory sections, reduced to 6 runs | 25-45 min |
| ncu fix | `bash profiling/ncu_reexport.sh` (local, needs ncu) then `python analysis/analyze_ncu.py` | rebuild CSVs from the saved reports with the correct DRAM metric names | 1 min |
| R01 | `pixi run python bench/peaks.py` | measured copy bandwidth, bf16/FP4 GEMM, attention TFLOP/s (idle GPU) | 2 min |
| R02 R03 | `pixi run python analysis/roofline_pixart.py`; `pixi run python analysis/roofline_whole_forward.py` | roofline from nsys and from the step-time table | 1 min |
| T01 | `pixi run bash run_all.sh embeds` | T5-encode the 6 real prompts into `prompts/embeds_pixart.pt` | 10-25 min |
| Q01-Q03 | `pixi run python precision/precision_ablation.py --size 512 --validate-stock --embeds prompts/embeds_pixart.pt`; same with `--size 1024`; `pixi run python precision/precision_images_pixart.py --size 1024 --embeds prompts/embeds_pixart.pt` | emulated w4a16/w16a4/w4a4 error and images | 25-40 min |
| V01 | `pixi run python viz/plot_all.py` | figures | 1 min |
| package | `pixi run python progress_report.py && tar czf results.tgz results logs prompts/embeds_pixart.pt` | copy back | 1 min |

Orchestrator for round 1: `pixi run bash run_all.sh [hf kernels time prof roofline embeds precision plots models]`.

## eval2 (new): run on the server after `git pull`

```bash
git pull                                   # if it complains about local changes to PROGRESS.md: git checkout PROGRESS.md
pixi run python -m pip install -r requirements_next.txt     # unchanged; adds nothing new
bash smoke_eval2.sh                        # 10-15 min: confirms every step below will run; read the FAIL lines
bash run_eval2.sh --list                   # the stages
bash run_eval2.sh                          # all stages E01-E15, each to completion
bash run_eval2.sh --from E06               # resume
FORCE=1 bash run_eval2.sh --only E11       # rerun one stage
bash run_eval2.sh --dry-run                # print what would run
```

Use `pixi run` in front of each if `python` is not the pixi python (or set `PY=.pixi/envs/default/bin/python`).

| ID | Stage | Underlying command | Time |
|---|---|---|---|
| E01 | calibration embeddings | `python prompts/make_prompt_embeds.py --prompts prompts/prompts_calib.json --out prompts/embeds_calib.pt` | 5 min |
| E02 | microbenchmarks | `python bench/fp4_gemm_shapes.py && python bench/attention_backends.py` | 5 min |
| E03 | real w4a16 vs w4a4 | `python precision/real_w4a16_vs_w4a4.py --size 512 --embeds prompts/embeds_pixart.pt` (and 1024) | 15 min |
| E04 | static scales | `python calibrate_static_scales.py --size 512 --embeds prompts/embeds_calib.pt` (and 1024) | 5 min |
| E05 | baselines step time | `CONFIGS="bf16_compile fp8_compile stock_compile stock_tk stock_tk_compile" SIZES="512 1024" bash profiling/run_step_matrix.sh` | 25 min |
| E06 | module stack step time | same script with the module configs in `profiling/configs.sh` (`EVAL2_MODULE`) | 35 min |
| E07 | attention options | same script with `EVAL2_ATTENTION` | 20 min |
| E08 | per-stage timing | `python timing/profile_stages.py --variant <v> --size <s>` then `python analysis/analyze_stages.py` | 10 min |
| E09 | nsys stages and launches | `CONFIGS="bf16_compile stock_compile module_best_compile" bash profiling/nsys_step.sh`; `EXTRA_ARGS="--stage-ranges" CONFIGS=module_triton bash profiling/nsys_step.sh`; `python analysis/analyze_nsys.py` | 25 min |
| E10 | DiTFastAttn plans | `python ditfast/make_plans.py --size 1024 --deltas 0.05 0.1 0.2 --embeds prompts/embeds_calib.pt` | 15 min |
| E11 | real-prompt quality | `python quality/run_quality.py --size 1024 --variants stock w4a16 module+... dfa... licfg...` | 45 min |
| E12 | end-to-end time | `python bench/pipeline_e2e.py --size 1024 --images 4 --t5 --variants ...` | 15 min |
| E13 | ledger, roofline | `python analysis/traffic_ledger.py --size 1024; python analysis/roofline_whole_forward.py` | 1 min |
| E14 | new-model smoke | `python model_smoke.py --model <sana_600m\|sd35_medium\|flux_schnell> --size 1024` | 15 min |
| E15 | package | `tar czf results_eval2.tgz results logs prompts/embeds_calib.pt` | 1 min |

Optional installs for the Sage attention variants (CPU build, may fail with CUDA 13; the stages skip them if absent). **`pip install sageattention` gives only the old 1.0.6 (Triton INT8); Sage 2.x is not on PyPI and must be built from source:**

```bash
python -m pip uninstall -y sageattention
git clone https://github.com/thu-ml/SageAttention.git ~/SageAttention && cd ~/SageAttention
TORCH_CUDA_ARCH_LIST="12.0" python -m pip install -e . --no-build-isolation    # builds for sm_120 only, so it is faster
```
Then `python bench/attention_backends.py` and look at the `pixart_self_1024 sage2` row. Sage FP4 is the `sageattention3_blackwell` folder of the same repo (head dim 64/128 only, so not usable on PixArt's 72). `stock_tk` needs the real MSLK from https://github.com/meta-pytorch/MSLK (the PyPI `mslk` 0.0.0 is empty).

## Saving results while the runs go (so nothing is lost if the GPU session ends)

`run_eval2.sh` saves after every stage by itself (copies small result files to the git branch `results-eval2` and pushes it). Check that pushing works before the long run:

```bash
bash save_results.sh --check      # tests read and push access to origin
bash save_results.sh mylabel      # save now by hand
SAVE_DIR=/some/other/disk bash save_results.sh mylabel   # also copy to another directory
NO_SAVE=1 bash run_eval2.sh       # turn the automatic save off
```

If push access fails, the files stay committed in `../PAA_results_eval2` (a second git worktree) and the next save pushes them. To fix credentials: `git config --global credential.helper store`, then one manual `git push` that asks for a GitHub user name and a personal access token. Big files (`*.nsys-rep`, `*.ncu-rep`, `*.sqlite`, images) are not saved; copy them with the tar from stage E15.
Read the results on your machine: `git fetch origin results-eval2 && git worktree add ../results_eval2 origin/results-eval2` (folder `latest/`).

## Local checks (laptop RTX 5060, no nvcc 13)

```bash
bash tests_cpu/run_cpu_checks.sh              # 71 checks, tiny model on CPU
bash tests_gpu/run_gpu_checks.sh 256          # real PixArt weights, 256 px
SKIP_CUDA=1 SKIP_MODELS=1 bash smoke_eval2.sh # the whole smoke without the CUDA-extension and new-model parts
```
