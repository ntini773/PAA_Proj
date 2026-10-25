#!/usr/bin/env bash
# Per-stage GPU time and DRAM traffic (ncu) + CUDA-event latency of the LN -> modulate -> quantize boundary.
# Run on the server from the repo folder:  bash boundary_ncu.sh        (~10-15 min)
#   SIZES="256 1024" bash boundary_ncu.sh   to run fewer sizes
set -u
cd "$(dirname "$0")"
PY=${PY:-.pixi/envs/default/bin/python}
NCU=${NCU:-ncu}
OUT=results/boundary; mkdir -p $OUT
for S in ${SIZES:-256 512 1024}; do
  echo "[$(date +%H:%M:%S)] size $S: event timing"
  $PY boundary_stages.py --size $S --mode time 2>&1 | grep '^{'
  for ST in ln mod amax quant_unfused quant_fused full_fused; do
    echo "[$(date +%H:%M:%S)] size $S: ncu $ST"
    $NCU --nvtx --nvtx-include "STAGE_${ST}/" \
      --metrics gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum \
      --csv --page raw \
      $PY boundary_stages.py --size $S --mode ncu --stage $ST \
      > $OUT/ncu_${S}_${ST}.csv 2> $OUT/ncu_${S}_${ST}.err
  done
done
$PY summarize_boundary.py | tee results/boundary_table.md
