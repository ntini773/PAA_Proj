# Laptop RTX 5060 (sm120, 8 GB): local env and gate

The laptop is the development and smoke-test machine. The server (RTX PRO 5000, 47 GB) does the real runs.

## Machine limits
- GPU: RTX 5060 Laptop, sm120, 8 GB VRAM. One GPU job at a time.
- RAM: 14-15 GB (about 8-9 GB free). Source builds use `MAX_JOBS=2`.
- Disk: /home is nearly full (8 GB free before 2026-11-28 cleanup, about 18 GB after the fresh env). Stop any step that would leave < 3 GB.
- System nvcc is 11.5 and cannot build sm120. Use the pixi nvcc 13.5 (conda-forge `cuda-nvcc`).

## Python environments
| env | path | built from | use |
|---|---|---|---|
| fresh (preferred) | `rtx_5000_runs/env/url-pins/.pixi/envs/default` (about 9 GB) | `env/url-pins/pixi.toml` + `pixi.lock` | everything, incl. CUDA kernel builds |
| old (fallback) | `nvfp4/.pixi/envs/default` (parent folder) | parent `pixi.toml` (phase 1) | do not modify; lacks nvcc 13, ninja, torchmetrics, torch-fidelity |

Why `env/url-pins/`: `rtx_5000_runs/pixi.toml` cannot be locked any more, because the nightly index no longer lists
torch 2.15.0.dev20260819+cu130, torchao 0.19.0.dev20260920+cu130, triton 3.8.0+gitdf3f91dd and torchvision 0.30.0.dev20260819+cu130.
The wheel files still download from their direct URLs (the ones in `pixi.lock`). `env/url-pins/pixi.toml` is the same manifest with
those four pins written as direct URLs (same versions). Keep it in sync with `../../pixi.toml`.

```bash
cd rtx_5000_runs/env/url-pins && pixi install --locked     # rebuild the fresh env (about 8 min, about 9 GB)
pixi run --manifest-path rtx_5000_runs/env/url-pins/pixi.toml check-env
```

## Running the local gate
```bash
cd rtx_5000_runs/rtx5000_next
E=$PWD/../env/url-pins/.pixi/envs/default
export PY=$E/bin/python CUDA_HOME=$E PATH=$E/bin:$PATH HF_HUB_OFFLINE=1
$PY eval3/check_env.py                       # every import declared in pixi.toml
bash tests_cpu/run_cpu_checks.sh
bash tests_gpu/run_gpu_checks.sh 256
$PY eval3/preflight.py                       # SD3.5 / FLUX FAIL is expected here (not cached, do not fit)
bash run_eval3.sh --list; bash run_eval3.sh --dry-run --only W104
$PY eval3/quality_grid.py --size 256 --embeds prompts/embeds_pixart.pt --n-prompts 2 --seeds 1 --steps 4 --no-clip --controls none --variants stock module+be=triton --tag gate_local
$PY bench/peaks_sweep.py --quick             # overwrites results/eval3/peaks_sweep.json with laptop roofs
$PY eval3/roofline_profile.py --model pixart_sigma --size 256 --variants bf16 --iters 2
$PY bench/cuda_vs_triton.py --only cuda_swizzled   # builds the CUDA kernel with nvcc 13.5; expect diff 0
SKIP_MODELS=1 bash smoke_eval2.sh
```
`CUDA_HOME` must point at the pixi env, else torch picks the system nvcc 11.5.
Note: `bench/cuda_vs_triton.py` exits 0 even when the kernel build fails; read the lines for `error`.

## What fits in 8 GB
- PixArt-Sigma transformer at 256 and 512 px; 1024 px only for single forwards. Cached: the 256, 512 and 1024 transformers,
  and the VAE from the 1024 repo.
- T5-XXL does not fit: copy `prompts/embeds_*.pt` from the server (`embeds_pixart.pt` is present; `embeds_showcase.pt`, `embeds_calib.pt` are not).
- SD3.5 Medium and FLUX schnell: not cached and do not fit; preflight FAIL for them is expected.
- Metric weights (CLIP, PickScore) are not cached; use `--no-clip`.

## Source builds and optional kernels (status 2026-11-28, fresh env)
| package | how | result |
|---|---|---|
| mslk | wheel pin `mslk==2026.8.19+cu130` (nightly index) | works: imports, `stock_tk+compile` runs at 256 px |
| mslk | `build-mslk` (source) | FAIL: build deps missing (`setuptools_git_versioning`, scikit-build, cmake); not needed |
| xformers 0.0.37 | `build-xformers` | builds in 25 s (pure Python, ops come from mslk); attention at head dim 72 runs |
| sageattention 2.x | `build-sage2` | FAIL with `-std=c++17`; `env/build_from_git.sh` now patches to C++20, retry not run yet |
| sageattn3 | `build-sage3` | same as sage2 |
| flash-attn 2.8.3.post2 | `build-flashattn2` | timeout after 20 min (MAX_JOBS=2); needs a longer build or the server |

Build trees live in `rtx_5000_runs/.pixi/build/` (about 1.8 GB, git-ignored, safe to delete after a build).
`uv pip install --no-build-isolation` also installs the package's runtime deps that are not in `pixi.toml`
(xformers pulled the PyPI placeholder `mslk 0.0.0` before the wheel pin existed). Check `pixi list` after a build.

## Known skips on the laptop
- Sage 2/3 and flash-attn rows until their builds pass.
- HF "401 Client Error" lines appear when online without a valid token; the cached weights still load. Use `HF_HUB_OFFLINE=1`.
