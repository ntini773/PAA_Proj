# RTX PRO 5000 Blackwell: partial results (2026-10-18)

Hardware: NVIDIA RTX PRO 5000 Blackwell (sm_120). Stack: pixi env from `pixi.lock`
(torch 2.15.0.dev20260819+cu130, torchao 0.19.0.dev20260920+cu130, triton 3.8.0), nvcc 13.3.

Done: the CUDA C++ test and the Triton boundary sweep (`pixi run tier1`).
Done also: `ncu_sweep.sh` for 256 and 1024 tokens.
Not yet run: full-model tests (`scale_e2e.py`), `nsys`, `graph_step_bench.py`.
All numbers below are the boundary only (LayerNorm -> AdaLN -> NVFP4 quant), not the whole image.

## 1. CUDA C++ test: fusion vs launch cost (`launch_vs_fusion.cu`)

One "step" = 56 boundary calls (28 blocks x 2 places), batch 2 rows per image (CFG). ms per step.

| image | rows M | fused == unfused | unfused eager | unfused graph | fused eager | fused graph | fusion x (eager) | graph x (unfused) |
|---|---|---|---|---|---|---|---|---|
| 256 | 512 | identical | 0.711 | 0.688 | 0.242 | 0.233 | 2.9x | 1.0x |
| 512 | 2048 | identical | 2.093 | 2.066 | 0.573 | 0.563 | 3.7x | 1.0x |
| 1024 | 8192 | identical | 7.159 | 7.136 | 1.743 | 1.735 | 4.1x | 1.0x |
| 2048 | 32768 | identical | 34.072 | 34.196 | 6.783 | 6.784 | 5.0x | 1.0x |

What it shows:
- Fusion alone gives 2.9x to 5.0x, and the gain grows with image size (less DRAM traffic).
- CUDA Graphs give 1.0x at every size. In this chain the CPU already launches fast enough to keep the GPU busy, so launch cost is not the limit here.
- Limit of this test: it launches CUDA kernels directly. The real model also pays PyTorch/Python dispatch cost, which this test does not include. `graph_step_bench.py` tests that on the real model and has not run yet.
- Correctness: the fused and unfused kernels give byte-identical output at all four sizes.

## 2. Triton boundary sweep (`pixi run tier1`)

GPU time in ms per single call. `unfused3` = eager LayerNorm+modulate then torchao quant. `compiled` = `torch.compile` of the same unfused chain. `q_old` / `q_new` = torchao quant vs our fused quant on a ready tensor (what the end-to-end run replaces).

| size | B | rows M | bytes differing (packed+scale) | unfused3 | fused | unfused3 / fused | q_old | q_new | q_old / q_new | compiled | compiled / fused |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 256 | 1 | 256 | 0+0 | 0.231 | 0.021 | 11.0x | 0.174 | 0.029 | 6.0x | 0.036 | 1.7x |
| 256 | 2 | 512 | 0+0 | 0.241 | 0.022 | 11.0x | 0.178 | 0.028 | 6.4x | 0.040 | 1.8x |
| 512 | 1 | 1024 | 0+0 | 0.238 | 0.021 | 11.3x | 0.182 | 0.028 | 6.5x | 0.040 | 1.9x |
| 512 | 2 | 2048 | 1+0 | 0.240 | 0.023 | 10.4x | 0.181 | 0.028 | 6.5x | 0.040 | 1.7x |
| 768 | 1 | 2304 | 1+0 | 0.246 | 0.023 | 10.7x | 0.185 | 0.029 | 6.4x | 0.042 | 1.8x |
| 768 | 2 | 4608 | 0+0 | 0.428 | 0.023 | 18.6x | 0.285 | 0.031 | 9.2x | 0.045 | 2.0x |
| 1024 | 1 | 4096 | 2+1 | 0.377 | 0.023 | 16.4x | 0.258 | 0.030 | 8.6x | 0.044 | 1.9x |
| 1024 | 2 | 8192 | 2+0 | 0.941 | 0.032 | 29.4x | 0.617 | 0.030 | 20.6x | 0.072 | 2.3x |
| 1536 | 1 | 9216 | 3+1 | 1.154 | 0.036 | 32.1x | 0.768 | 0.031 | 24.8x | 0.083 | 2.3x |
| 1536 | 2 | 18432 | 2+1 | 4.160 | 0.112 | 37.1x | 2.843 | 0.061 | 46.6x | 0.299 | 2.7x |
| 2048 | 1 | 16384 | 3+1 | 3.292 | 0.099 | 33.3x | 2.222 | 0.054 | 41.1x | 0.205 | 2.1x |
| 2048 | 2 | 32768 | 2+0 | 8.180 | 0.195 | 41.9x | 5.575 | 0.149 | 37.4x | 0.611 | 3.1x |

What it shows:
- Against eager PyTorch, the fused kernel is 10x to 42x faster. This is the large number, and it uses a weak baseline.
- Against `torch.compile`, the gap is 1.7x to 3.1x. A compiled baseline gets part of the fusion for free. Report the compiled number as the fair one.
- Output equality: a few bytes differ at larger sizes (at most 3 packed bytes and 1 scale byte out of millions). This is the known fp32 reduction-order tie, not a logic error. 256px is exact.
- The ratio grows with size because the unfused chain is memory-bound and the fused kernel is not.

## 3. ncu: kernel time and DRAM traffic per boundary (`ncu_sweep.sh`)

Per iteration, summed over all kernels of one boundary call. 256 tokens = 256px B=1. 1024 tokens = 512px B=1.

| tokens | path | kernels | GPU time (us) | DRAM (MB) |
|---|---|---|---|---|
| 256 | fused | 1 | 4.3 | 1.78 |
| 256 | unfused | 50 | 164.7 | 45.30 |
| 1024 | fused | 1 | 8.4 | 7.09 |
| 1024 | unfused | 50 | 278.8 | 178.97 |

Ratios (unfused / fused): kernels 50x; GPU time 38x at 256 tokens and 33x at 1024 tokens; DRAM 25x at both sizes.

What it shows:
- The unfused chain is 50 kernels and moves 25x more DRAM. The DRAM ratio is the same at both sizes, so it is a fixed property of the chain, not of image size.
- The fused kernel itself takes only 4.3 us at 256 tokens. The CUDA-event time in section 2 was 21 us for the same call. The gap (about 17 us) is probably Python/Triton launch cost, not GPU work. This is a hypothesis; a CUDA Graph test of the fused path would confirm it.
- ncu flushes caches and times each kernel alone, so its DRAM bytes are cold-cache numbers and its times exclude gaps between kernels. The earlier "24 MB / 13.6x" figure in the old notes is superseded by these measurements (45 MB / 25x).

## 4. Reading all together

- The boundary is a small part of a full step. 0.03 ms (fused) or 0.04 ms (compiled) per call is tiny next to GEMMs and attention at 1024px and above.
- These tests do not give the end-to-end gain. That needs `scale_e2e.py`, and `nsys` for the share of GPU time in GEMM vs attention vs norm-quant.
- Fusion is what helps. Removing launch cost did not help in the direct CUDA test.

## 5. Still to run

1. `pixi run smoke`, then `scale_e2e.py` for bf16 / stock / fused at 256 and 512 (1024 if time allows).
2. `graph_step_bench.py` for eager vs CUDA Graph on the real model.
3. The `nsys` phase if time allows.
