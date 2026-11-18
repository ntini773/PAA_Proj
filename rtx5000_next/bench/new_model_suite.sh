#!/usr/bin/env bash
# bash bench/new_model_suite.sh <model> <size>
# download (transformer only) -> smoke -> reduced config matrix -> precision ablation. Stops at the first failure.
set -e
cd "$(dirname "$0")/.."
PY=${PY:-python}
MODEL=$1; SIZE=$2
$PY hf/download_model.py $MODEL --size $SIZE
$PY model_smoke.py --model $MODEL --size $SIZE
source profiling/configs.sh
for cfg in ${CONFIGS_NEW:-bf16 stock stock_compile fused_triton fused_cuda_swizzled free_bound}; do
  $PY step_target.py --model $MODEL --size $SIZE $(cfg_args $cfg) --config ${cfg} --iters 20 --graph || echo "config $cfg failed"
done
$PY precision/precision_ablation.py --model $MODEL --size $SIZE
