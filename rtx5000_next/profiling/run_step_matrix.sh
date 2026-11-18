#!/usr/bin/env bash
# Step time, CPU launch time and CUDA Graph time for every config and size.
cd "$(dirname "$0")/.."
source profiling/configs.sh
ITERS=${ITERS:-30}
mkdir -p logs/matrix
for size in $SIZES; do
  for cfg in $CONFIGS; do
    echo "[$(date +%H:%M:%S)] $MODEL $size $cfg"
    $PY step_target.py --model $MODEL --size $size $(cfg_args $cfg) --config $cfg --iters $ITERS --graph \
      > logs/matrix/${MODEL}_${cfg}_${size}.log 2>&1 || echo "  FAILED, see logs/matrix/${MODEL}_${cfg}_${size}.log"
  done
done
