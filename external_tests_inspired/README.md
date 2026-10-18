# external_tests_inspired

Ideas taken from NVlabs/GraspGenX PR #7 ("30x faster inference with CUDA kernels, fusion in a C++ runtime").
That PR is fast mainly because it (1) fuses tiny kernels and (2) replays the whole forward as one CUDA Graph.
We test whether the same two levers matter for PixArt-Sigma + NVFP4.

| File | What it tests | Needs |
|---|---|---|
| `launch_vs_fusion.cu` | Hand-written CUDA C++ version of our fused LN+AdaLN+NVFP4 kernel vs a 3-kernel unfused chain, each launched normally and as a CUDA Graph. Separates "fusion" gain from "launch overhead" gain. Self-checks that fused == unfused byte for byte. | `nvcc` (server has 13.3) |
| `graph_step_bench.py` | The REAL transformer forward (1 denoise step): eager vs CUDA-Graph replay, for bf16 / stock NVFP4 / fused NVFP4. Prints GPU ms and CPU launch ms. | models downloaded |
| `run_inspired.sh` | Runs both. | |

Status: the CUDA device code was compiled with NVRTC for sm_120 and run locally on an RTX 5060:
byte-identical to the unfused chain, and identical to the Triton kernel at M=256/1024 (5 differing bytes of 2.6M at M=4096).
The host timing harness (`main`) and `graph_step_bench.py` have not been run yet.
