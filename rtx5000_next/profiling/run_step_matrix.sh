#!/usr/bin/env bash
# Step time, CPU launch time and CUDA Graph time for every config and size.
cd "$(dirname "$0")/.."
source profiling/configs.sh
ITERS=${ITERS:-30}
mkdir -p logs/matrix
nfail=0
for size in $SIZES; do
  for cfg in $CONFIGS; do
    echo "[$(date +%H:%M:%S)] $MODEL $size $cfg"
    $PY step_target.py --model $MODEL --size $size $(cfg_args $cfg) --config $cfg --iters $ITERS --graph \
      > logs/matrix/${MODEL}_${cfg}_${size}.log 2>&1 || { echo "  FAILED, see logs/matrix/${MODEL}_${cfg}_${size}.log"; nfail=$((nfail+1)); }
  done
done
# a failed config is reported in the exit code too (eval3: the stage must not look green when rows are missing)
[ $nfail -eq 0 ] || { echo "$nfail config(s) failed"; exit 1; }
