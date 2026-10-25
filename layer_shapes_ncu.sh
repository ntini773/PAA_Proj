#!/usr/bin/env bash
# Stage split, GPU time and DRAM for the layer shapes the first boundary test did not cover:
#   mlp_fc2  : width 4608, rows = 2 x tokens   (the layer our kernel does not handle yet)
#   crosskv  : width 1152, rows = 2 x 300      (text K/V layers, same at every image size)
# Only the quantize stages matter here (these layers have no LayerNorm in front). About 4 minutes.
#   bash layer_shapes_ncu.sh
set -u
cd "$(dirname "$0")"
PY=${PY:-.pixi/envs/default/bin/python}
NCU=${NCU:-ncu}
OUT=results/shapes; mkdir -p $OUT
run_shape() {  # label size hidden tokens
  local L=$1 S=$2 HID=$3 TOK=$4 TARG=""
  [ -n "$TOK" ] && TARG="--tokens $TOK"
  echo "[$(date +%H:%M:%S)] $L: timing + correctness"
  $PY boundary_stages.py --size $S --hidden $HID $TARG --mode time 2>&1 | grep '^{' | tee $OUT/time_${L}.json
  for ST in amax quant_unfused quant_fused; do
    $NCU --nvtx --nvtx-include "STAGE_${ST}/" \
      --metrics gpu__time_duration.sum,dram__bytes.sum --csv --page raw \
      $PY boundary_stages.py --size $S --hidden $HID $TARG --mode ncu --stage $ST \
      > $OUT/ncu_${L}_${ST}.csv 2> $OUT/ncu_${L}_${ST}.err
  done
}
for S in 256 512 1024; do run_shape fc2_$S $S 4608 ""; done
run_shape crosskv 512 1152 300
$PY summarize_shapes.py | tee results/shapes_table.md
