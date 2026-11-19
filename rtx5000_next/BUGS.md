# Bug log: how not to write systems code

Every bug found so far in `rtx5000_next` and the earlier work. Add a row when you find one. Keep the lesson short and general.
"Found by" tells how the bug surfaced: **server** = a run failed, **review** = caught by reading before any run, **local** = caught while writing.
Most of this code was written without being able to run it (no sm_120 nvcc locally, no GPU in the session). The log shows what that costs.

## Bugs

| # | Date | Where | Symptom | Root cause | Fix | Found by | Lesson |
|---|---|---|---|---|---|---|---|
| 1 | 11-18 | `cuda_ext.py` | `error: #error C++20 or later compatible compiler is required to use PyTorch` before any kernel line compiled | Build flags said `-std=c++17`. The pinned torch nightly headers require C++20 | `-std=c++20` for nvcc and host (commit 726b7a7) | server | Build flags must match the installed library, not the last version you knew. Read the library's own requirement. Compile every native file once as early as possible, even before the logic is right. |
| 2 | 11-18 | env (`pixi.toml`) | T5 tokenizer, kernel build and plots would fail on import or build | `sentencepiece`, `ninja`, `matplotlib`, `lpips` were used but not in the env. I assumed they were there | `requirements_next.txt`, an import check with a clear message in `make_prompt_embeds.py` | review | List every import and every external tool a new script needs, and diff it against the lock file before shipping. Fail early with the install command in the message. |
| 3 | 11-18 | `analysis/roofline_pixart.py` (R02) | `FileNotFoundError: results/analysis/nsys_summary.csv` | R02 depends on P01 and P03. The stage order allowed running it before them, and the error did not say what to run | Check inputs first and exit with "run P01 then P03" (this commit) | server | Declare dependencies between steps in the tracker and in the code. A missing input must name the step that creates it. |
| 4 | 11-18 | `precision_ablation.py` vs `viz/plot_all.py` | Plot would have broken or averaged wrong after output rows became one per (prompt, timestep) | I changed a data format and did not update the reader | Plot averages per timestep | review | When a data format changes, grep every consumer in the same commit. Version the format if it is shared. |
| 5 | 11-18 | `profiling/ncu_layer.sh` vs `analysis/analyze_ncu.py` | The CSV from `ncu -i ... --page details` would not parse: the parser expects the `raw` page (one row per kernel) | Producer and parser were written apart, with different assumptions about the format | `--page raw`, parser docstring names the format | review | The format of a file is part of the interface. Write it once next to the producer and the parser. |
| 6 | 11-18 | `precision_images_pixart.py` | LPIPS net rebuilt and moved to the GPU for every image pair (about 30 times). The AlexNet weights download appeared in the middle of a run. A failure was swallowed by `except Exception: pass` and the metric silently vanished | Expensive object built inside a hot function. A bare `except` hid errors. An external download was not prefetched | `lru_cache` on the net, `print` on failure, prefetch command in the run flow (commit aac2a63) | server | Build expensive objects once. Never swallow an exception without logging it. Prefetch every external download before the timed run. |
| 7 | 11-18 | `hf/hf_common.py` `_pick_weights` | Wrong fallback expression: could return an empty list for repos with only variant weights | Compact one-liner I did not trace by hand | `rest = [...]; return [non-variant] or rest` | local | Trace the empty and fallback paths of a selection function. Do not compress logic to one line. |
| 8 | 11-18 | run plan | ncu default of 32 runs with 9 sections would take hours. Tracker times were far too low | I estimated from kernel count and ignored the replay passes each section adds | Reduced scope: 1 size, 3 configs, 2 layers (about 6 runs) | review | Profilers multiply cost: kernels x passes x runs. Estimate that product before choosing defaults. |
| 9 | 11-18 | run plan | "T5 download about 9 GB" was wrong (10 to 20 GB) | I copied the number from a docstring in an older script about a different download | Corrected to 10-20 GB, disk check added | review | Do not reuse a number from a comment. Ask the tool (`--dry-run`, repo metadata). |
| 10 | 11-18 | `prompts/make_prompt_embeds.py` | Three noisy warnings: `KernelPreference`, T5Tokenizer legacy notice, `torch_dtype` deprecated | Used old argument names and did not set explicit options | `legacy=True`, `dtype=`, quieter logger (commit f038d2b) | server | Treat a warning as a to-do. Pass explicit options so behaviour does not depend on defaults that change. |
| 11 | 10-xx (earlier work) | `fused_adaln_nvfp4.py` under `torch.compile` | `torch.compile` fails on the fused path: Triton launch under FakeTensor | A raw Triton kernel is invisible to the compiler's tracing | Wrap the kernel as a `torch.library.custom_op` with a fake implementation (`quant_backends.py`). Not yet verified | server | A custom kernel needs a registered op and a fake (shape-only) implementation to be compile-safe. |
| 12 | 10-xx (earlier work) | fp16 runs | NaN images | fp16 range too small for these activations | Precision ablation runs in bf16 | server | Pick the dtype from the activation range, not from habit. |
| 13 | 11-18 | N01 output for `dit_xl2` | `params B = 0.0` and `bf16 GB = 0.0` | Parameter count could not be read from this repo's config (open) | Not fixed. DiT-XL/2 was dropped from the plan | server | A zero in a size table is a failed read, not a small model. Print "unknown". |

| 14 | 11-18 | `step_target.py` `--nvtx` block hooks (P01) | 12 of 22 nsys runs failed: `RuntimeError: forward pre-hook must return None or a tuple ..., but got -1`. All non-compiled `stock` and `fused*` configs, including `free_bound`, at both sizes. bf16, fp8 and the compiled configs were fine | The hooks were lambdas that returned the value of `nvtx.range_push` and `range_pop`. A hook's return value is used: a pre-hook's replaces the block arguments, a forward hook's replaces the block output. The code path only runs for `--nvtx` with `stock` or `fused` and no compile, so no other test touched it | Hooks are now functions that return nothing | server | A hook or callback must return `None` unless it means to change data. Never write a lambda whose value is a side-effect call. Run the `--nvtx` path once before launching a 25 minute batch. |

| 15 | 11-19 | `profiling/ncu_layer.sh` METRICS (P02, P04) | `ncu_layers.csv` showed `dram_vs_ideal = 0.0` and DRAM MB 0.0 for every row, with no error anywhere | I asked for `dram__bytes_read.sum` and `dram__bytes_write.sum`. On GB202 the names are `dram__bytes_op_read.sum` and `dram__bytes_op_write.sum`. ncu silently dropped the unknown names, so the CSV had no DRAM byte columns. The saved `.ncu-rep` files held the right counters from the sections | Names fixed. `profiling/ncu_reexport.sh` rebuilds the CSVs from the existing reports (no GPU). `analyze_ncu.py` now stops if a required column is missing | review of results | A profiler may ignore a name it does not know. Check names with `ncu --query-metrics --chip <chip>` and assert that every requested column is in the output. A column of zeros is a missing metric, not a measurement. |
| 16 | 11-19 | `analysis/analyze_ncu.py` units | `quant_time_us` was 0.1 to 1.3 for kernels that run for tens of microseconds | The unit table had `usecond`, but the raw page prints `us`. The unknown unit fell back to a factor of 1, so microseconds were read as nanoseconds | Added `ns`, `us`, `ms`, `s` | review of results | Parse the unit from the file and fail on an unknown time or size unit. Do not default to 1. |

## Open risks (code that has not run)

| Area | Risk | How it will show |
|---|---|---|
| `cuda_kernels/*.cu` (written by agents) | Never compiled with sm_120 until the C++20 fix. Logic and byte-exactness vs Triton unverified | K01 to K03 |
| `model_adapters.py` for Sana, SD3.5, FLUX | Input signatures written from memory | `model_smoke.py` at the start of each M test |
| `quant_backends.py` custom-op wrapper | May still hit a torchao tensor-subclass limit under `torch.compile` | S01, config `fused_triton_op_compile` |
| `bench/peaks.py` NVFP4 GEMM | Uses a pre-quantized input through `F.linear`. Read in torchao source, not run | R01 (`fp4_tflops` null if it fails) |
| `analysis/roofline_pixart.py` | The quantize class time includes all non-GEMM, non-attention kernels, so its efficiency is a lower bound | R02 |
| Emulated NVFP4 vs real torchao (Q01 `--validate-stock`, seen 11-19) | The emulation is not tight. Whole-transformer relative L2 between stock and emulated is 0.06 at t=999..500, 0.09 at t=250 and 0.17 at t=50, while each is about 0.10 to 0.19 from bf16. The emulated error is about half of the real error in size, so the w4a16 / w16a4 split may not transfer to real NVFP4. Cause not yet found (candidates: scale rounding to e4m3, per-tensor scale, rounding ties, weight path) | Write a single-tensor check: emulated fake-quant vs torchao quantize then dequantize on real activations |
| Precision at real timesteps | Latents are fixed noise, not partly denoised, in Q01 and Q02 | Q03 uses the full pipeline |

## Rules distilled

1. Compile and import every file once before building on top of it.
2. Pin and diff dependencies. Fail with the install command.
3. A step that needs another step's output must check for it and name it.
4. A format is an interface: one definition, producer and parser both reference it.
5. No bare `except`. Log what you skip.
6. Build expensive objects once. Prefetch downloads before timed runs.
7. Estimate cost as a product (kernels x passes x runs) before setting defaults.
8. Verify a number with the tool. Do not copy it from a comment.
9. When you change a data shape, grep all its readers in the same commit.
10. A warning is a to-do. Explicit options beat defaults.
11. Run a batch script on its smallest case first (one config, one size). A bug in a shared path then costs seconds, not the whole batch.
12. A callback returns `None` unless it means to change data.
