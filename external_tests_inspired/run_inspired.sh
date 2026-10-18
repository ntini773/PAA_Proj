#!/usr/bin/env bash
# Run from anywhere:  bash rtx_5000_runs/external_tests_inspired/run_inspired.sh
# 1) CUDA C++ launch-vs-fusion microbench (needs nvcc only, ~1 min)
# 2) real-model CUDA-Graph step benchmark, eager vs graph (needs models downloaded)
set -u
cd "$(dirname "$0")"
mkdir -p ../results
nvcc -O3 -std=c++17 -arch=native launch_vs_fusion.cu -o launch_vs_fusion 2>&1 | tee ../results/cu_build.log
./launch_vs_fusion | tee ../results/launch_vs_fusion.log
PY=${PY:-../.pixi/envs/default/bin/python}
for S in 256 512 1024; do for M in bf16 stock fused; do
  $PY graph_step_bench.py --size $S --mode $M 2>&1 | grep '^{' | tee -a ../results/graph_step.log
done; done
