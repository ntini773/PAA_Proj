#!/usr/bin/env bash
# Re-run only the per-part breakdown (about 5 minutes). Run on the server from the repo folder.
set -u
cd "$(dirname "$0")"
PY=${PY:-.pixi/envs/default/bin/python}
mkdir -p results/logs
[ -f results/breakdown.jsonl ] && mv results/breakdown.jsonl results/breakdown_v1.jsonl
for S in 256 512 1024; do for M in stock fused fused_wide bf16; do
  echo "[$(date +%H:%M:%S)] breakdown $M $S"
  timeout 600 $PY breakdown_step.py --size $S --mode $M > results/logs/breakdown_${M}_${S}.log 2>&1 \
    || echo "  FAILED $M $S (see results/logs/breakdown_${M}_${S}.log)"
done; done
$PY breakdown_table.py | tee results/breakdown_table.md
