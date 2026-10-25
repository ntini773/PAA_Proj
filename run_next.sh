#!/usr/bin/env bash
# Next test (about 6 minutes). Everything printed is also saved to results/next_<time>.log
#   bash run_next.sh
cd "$(dirname "$0")"
PY=.pixi/envs/default/bin/python
B=$(ls -d results_bak_* 2>/dev/null | tail -1)       # results of the earlier full run
mkdir -p results
[ -n "$B" ] && cp -n $B/*.npy results/ 2>/dev/null   # old images, needed for the comparisons
LOG=results/next_$(date +%H%M%S).log
{
echo "=== GPU check (look for other users' processes) ==="
nvidia-smi | tail -8

echo
echo "=== A. Make the fast kernel also handle the big MLP layer (width 4608) ==="
echo "Question: does the step get faster, and are the images still identical?"
for S in 1024 512 256; do
  echo "--- ${S}px"
  EXTRA_OK_K=4608,4096 $PY scale_e2e.py --size $S --mode fused --steps 14 --reps 3 --tag _k4608 2>&1 | grep '^{' | cut -c1-330
  $PY -c "import numpy as np; a=np.load('results/stock_${S}_b1.npy').astype('f4'); b=np.load('results/fused_${S}_b1_k4608.npy').astype('f4'); print('image difference vs stock: mean', abs(a-b).mean(), ' max', abs(a-b).max(), ' (0.0 = identical)')"
done

echo
echo "=== B. Earlier step times for comparison (from $B) ==="
$PY - <<PYEOF
import json
for l in open("$B/e2e.jsonl"):
    r = json.loads(l)
    print(f"{r['size']}px {r['mode']:6s} step median {r['step_gpu_ms_median']:7.1f} ms  min {r['step_gpu_ms_min']:7.1f} ms  fused calls {r['fused_calls']}  fallback calls {r['fallback_calls']}")
PYEOF

echo
echo "=== C. How far each precision drifts from bf16 ==="
$PY quality_compare.py

echo
echo "=== D. Old results we never read ==="
echo "--- images identical stock vs fused?"; cat $B/images.log
echo "--- CUDA Graph test"; cat $B/graph_step.jsonl | cut -c1-300
echo "--- torch.compile error (last lines)"; tail -8 $B/logs/compile_stock_256.log
} 2>&1 | tee $LOG
echo "saved: $LOG"
