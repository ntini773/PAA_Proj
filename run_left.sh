#!/usr/bin/env bash
# Remaining runs: CUDA Graph + fused_wide, e2e repeat, torch.compile test.
# Run on the server from the repo folder:  bash run_left.sh   (~12-15 min)
cd "$(dirname "$0")"
PY=.pixi/envs/default/bin/python
F='^W10\|Unable'
mkdir -p results/logs

for S in 256 512 1024; do
  $PY external_tests_inspired/graph_step_bench.py --size $S --mode fused_wide
done 2>&1 | grep -v "$F" | tee results/logs/graph_wide.log

for S in 256 512 1024; do
  $PY scale_e2e.py --size $S --mode fused_wide --steps 14 --reps 3 --tag _rep2
done 2>&1 | grep -v "$F" | tee results/logs/e2e_wide_rep2.log

for M in stock fused_wide; do
  timeout 600 $PY scale_e2e.py --size 256 --mode $M --compile --steps 14 --reps 2 --tag _compile
done 2>&1 | grep -v "$F" | tail -30 | tee results/logs/compile.log

echo "fused_wide rows in breakdown.jsonl:"
grep -c fused_wide results/breakdown.jsonl
tar czf results_$(date +%m%d).tgz results
echo "DONE: results_$(date +%m%d).tgz"
