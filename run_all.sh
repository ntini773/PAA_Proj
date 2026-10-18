#!/usr/bin/env bash
# Full experiment driver. Run from repo root: bash run_all.sh [phase...]
# Phases: tier1 ncu e2e nsys stretch   (default: tier1 ncu e2e nsys)
# Every run is a separate process (no patch contamination). Logs in results/.
set -u
cd "$(dirname "$0")"
PY=${PY:-.pixi/envs/default/bin/python}
R=results; mkdir -p $R
E2E="$PY scale_e2e.py"
PHASES=${@:-tier1 ncu e2e nsys}

for ph in $PHASES; do case $ph in
tier1)
  $PY sweep_tier1.py 2>&1 | grep -v "^W10\|Unable" | tee $R/tier1.log ;;
ncu)
  bash ncu_sweep.sh 2>&1 | tee $R/ncu_summary.log ;;
e2e)   # latency per step + wall + memory + image equality, 14 steps
  for S in 256 512 1024; do for M in bf16 stock fused; do
    $E2E --size $S --mode $M --steps 14 --reps 3 2>&1 | grep '^{' | tee -a $R/e2e.log
  done; done
  $PY compare_images.py | tee $R/images.log ;;
nsys)  # kernel-time split per size/mode (2 steps, 1 rep, only timed region captured)
  for S in 256 512 1024; do for M in stock fused; do
    nsys profile -o $R/nsys_${M}_$S --force-overwrite=true --trace=cuda,nvtx \
      --capture-range=cudaProfilerApi --capture-range-end=stop \
      $E2E --size $S --mode $M --steps 2 --reps 1 --profile --tag _nsys --out $R/nsys_tmp > $R/nsys_${M}_$S.log 2>&1
    nsys stats --report cuda_gpu_kern_sum --format csv --force-export=true \
      $R/nsys_${M}_$S.nsys-rep > $R/nsys_${M}_${S}_kern.csv 2>/dev/null
  done; done
  $PY classify_nsys.py $R/nsys_*_kern.csv | tee $R/nsys_summary.log ;;
stretch) # batch scaling + torch.compile (honest baseline); may be slow or fail
  for B in 1 4; do for M in stock fused; do
    $E2E --size 512 --mode $M --batch $B --steps 14 --reps 3 --tag _b 2>&1 | grep '^{' | tee -a $R/e2e.log
  done; done
  for M in stock fused; do
    timeout 900 $E2E --size 512 --mode $M --compile --steps 14 --reps 3 2>&1 | grep '^{' | tee -a $R/e2e.log
  done ;;
esac; done
