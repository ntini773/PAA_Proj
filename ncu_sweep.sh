#!/usr/bin/env bash
# ncu: per-iteration GPU time + DRAM bytes of the norm->quant boundary, fused vs unfused.
# Usage: bash ncu_sweep.sh   (from repo root; writes results/ncu_*.csv)
set -u
cd "$(dirname "$0")"
PY=${PY:-.pixi/envs/default/bin/python}
NCU=${NCU:-ncu}
OUT=results; mkdir -p $OUT
ITERS=3
for TOK in ${TOKENS:-256 1024 4096 16384}; do
  for PATHN in fused unfused; do
    $NCU --nvtx --nvtx-include "${PATHN}_loop/" \
      --metrics gpu__time_duration.sum,dram__bytes.sum \
      --csv --page raw \
      $PY profile_kernel.py --path $PATHN --tokens $TOK --iters $ITERS --warmup 3 \
      > $OUT/ncu_${PATHN}_${TOK}.csv 2> $OUT/ncu_${PATHN}_${TOK}.err
  done
done
$PY summarize_ncu.py $OUT
