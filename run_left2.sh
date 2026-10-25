#!/usr/bin/env bash
# Last two open items: fused_wide breakdown (3 sizes) and the torch.compile test.
# Run on the server from the repo folder:  bash run_left2.sh   (~10 min)
cd "$(dirname "$0")"
PY=.pixi/envs/default/bin/python
F='^W10\|Unable'
mkdir -p results/logs

for S in 256 512 1024; do
  $PY breakdown_step.py --size $S --mode fused_wide 2>&1 | grep -v "$F" > results/logs/breakdown_fused_wide_$S.log
  echo "[$(date +%H:%M:%S)] breakdown fused_wide $S done"
done
echo "fused_wide rows in breakdown.jsonl:"
grep -c fused_wide results/breakdown.jsonl
$PY breakdown_table.py > results/breakdown_table.md

for M in bf16 stock fused_wide; do
  timeout 900 $PY scale_e2e.py --size 256 --mode $M --compile --steps 14 --reps 2 --tag _compile \
    > results/logs/compile_$M.log 2>&1
  echo "== $M: rc=$?"; grep -v "$F" results/logs/compile_$M.log | grep -E 'step_gpu_ms_median|Error|error' | cut -c1-300 | tail -5
done

tar czf results_$(date +%m%d)b.tgz results
echo "DONE: results_$(date +%m%d)b.tgz"
