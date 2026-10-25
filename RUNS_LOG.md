# Server run log (RTX PRO 5000 Blackwell, 2026-10-18)

Working dir on the server: `~/PAA_Proj`. Add a row for each new run.
Details and full tables: `RESULTS_server_partial.md`.

## Runs

| # | Step | Command | Result | Status |
|---|---|---|---|---|
| 1 | Install env | `pixi install --locked` | Env built. Only warning: `[system-requirements]` is deprecated (harmless). | done |
| 2 | Model download | `pixi run download` | Slow; server timed out before it finished. `smoke` and all model tests blocked. | **not finished** |
| 3 | CUDA C++ test | `cd external_tests_inspired && nvcc -O3 -std=c++17 -arch=native launch_vs_fusion.cu -o launch_vs_fusion && ./launch_vs_fusion` | Compiled on nvcc 13.3. Fused == unfused byte-identical at all sizes. Fusion 2.9x (256px) to 5.0x (2048px). CUDA Graph 1.0x everywhere. | done |
| 4 | Triton boundary sweep | `pixi run tier1` | Fused 10x-42x faster than eager unfused; 1.7x-3.1x faster than `torch.compile`. At most 3+1 bytes differ per million. | done |
| 5 | ncu boundary profile | `TOKENS="256 1024" bash ncu_sweep.sh` | Counters work. 50 kernels -> 1. DRAM 45.3 -> 1.78 MB (256 tok), 179 -> 7.09 MB (1024 tok). GPU time 164.7 -> 4.3 us and 278.8 -> 8.4 us. | done |
| 6 | Smoke test | `pixi run smoke` | Needs the 256 transformer and the VAE. | not run |
| 7 | Full model: stock vs fused (512) | `for M in stock fused; do pixi run python scale_e2e.py --size 512 --mode $M --steps 14 --reps 2; done` | Needs the 512 transformer. | not run |
| 8 | Graph step test | `pixi run python external_tests_inspired/graph_step_bench.py --size 256 --mode fused` | Needs the 256 transformer. | not run |
| 9 | nsys split | `bash run_all.sh nsys` | Needs models. | not run |

## Key numbers so far (boundary only, not the whole image)

| Measure | 256px (256 tokens) | 512px (1024 tokens) |
|---|---|---|
| Kernels per boundary, unfused -> fused | 50 -> 1 | 50 -> 1 |
| ncu GPU time, unfused -> fused | 164.7 -> 4.3 us | 278.8 -> 8.4 us |
| ncu DRAM, unfused -> fused | 45.30 -> 1.78 MB | 178.97 -> 7.09 MB |
| Triton CUDA-event time, eager unfused / fused (B=1) | 0.231 / 0.021 ms | 0.238 / 0.021 ms |
| `torch.compile` unfused / fused (B=1) | 0.036 / 0.021 ms | 0.040 / 0.021 ms |
| CUDA C++ fusion gain, B=2 (56 calls per step) | 2.9x | 3.7x |
| CUDA C++ CUDA-Graph gain | 1.0x | 1.0x |

## Open items

- Download the transformers and VAE, then run `smoke` (rows 6-9). This is the only way to get the full-model speedup.
- No local copy of the models exists. A local download of the 256 transformer reached about 2.1 of 2.44 GB, then I stopped it. Only the VAE (320 MB) is cached locally.
- To resume: `pixi run download` on the server, or run the download locally and copy the HF cache to the server.
