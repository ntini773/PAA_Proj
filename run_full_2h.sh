#!/usr/bin/env bash
# Full 2-hour benchmark. Run on the server from the repo folder:
#   nohup bash run_full_2h.sh > run.log 2>&1 &     then:  tail -f results/progress.log
# Stages run in priority order. Every run is its own process with a timeout, so one
# failure does not stop the rest. Progress and finish time are in results/progress.log.
# Stop early at any time:  pkill -u $USER -f run_full_2h.sh; pkill -u $USER -f scale_e2e.py
set -u
cd "$(dirname "$0")"
PY=${PY:-.pixi/envs/default/bin/python}

# one copy at a time (two copies on one GPU ruined timings before)
exec 9> .run.lock
flock -n 9 || { echo "another run_full_2h.sh is already running"; exit 1; }

# keep any earlier results
if [ -s results/e2e.jsonl ]; then mv results "results_prev_$(date +%Y%m%d_%H%M%S)"; fi
mkdir -p results
P=results/progress.log
stage() { echo "[$(date +%H:%M:%S)] $*" | tee -a $P; }
run() {  # run <timeout_s> <logfile> <cmd...>  (full output kept in the per-run log)
  local t=$1 lf=$2; shift 2
  timeout "$t" "$@" > "$lf" 2>&1; local rc=$?
  [ $rc -ne 0 ] && stage "  FAILED rc=$rc: $* (see $lf)"
  return 0
}

stage "START  host=$(hostname)  GPU snapshot:"
nvidia-smi >> $P 2>&1
mkdir -p results/logs

# ---------------- A. whole-model step time, 5 precisions x 3 sizes ----------------
stage "A. e2e: bf16 fp16 fp8 stock fused at 256 512 1024 (3 reps x 14 steps)"
for S in 256 512 1024; do for M in bf16 fp16 fp8 stock fused; do
  run 900 results/logs/e2e_${M}_${S}.log $PY scale_e2e.py --size $S --mode $M --steps 14 --reps 3
  stage "  done e2e $M $S"
done; done
$PY compare_images.py > results/images.log 2>&1

# ---------------- B. per-part breakdown (the pipeline contribution table) ----------------
stage "B. breakdown per part: bf16 stock fused at 256 512 1024"
for S in 256 512 1024; do for M in stock fused bf16; do
  run 600 results/logs/breakdown_${M}_${S}.log $PY breakdown_step.py --size $S --mode $M
  stage "  done breakdown $M $S"
done; done
$PY breakdown_table.py > results/breakdown_table.md 2>&1

# ---------------- C. batch scaling ----------------
stage "C. batch scaling (stock vs fused)"
for B in 4 8; do for M in stock fused; do
  run 900 results/logs/batch_${M}_256_b${B}.log $PY scale_e2e.py --size 256 --mode $M --batch $B --steps 14 --reps 2 --tag _b$B
done; done
for B in 4; do for M in stock fused; do
  run 900 results/logs/batch_${M}_512_b${B}.log $PY scale_e2e.py --size 512 --mode $M --batch $B --steps 14 --reps 2 --tag _b$B
done; done

# ---------------- D. CUDA Graph on the real model ----------------
stage "D. CUDA Graph replay on the real model"
for S in 256 512 1024; do for M in stock fused bf16; do
  run 600 results/logs/graph_${M}_${S}.log $PY external_tests_inspired/graph_step_bench.py --size $S --mode $M
done; done

# ---------------- E. nsys kernel split (whole run) ----------------
stage "E. nsys kernel split"
for S in 256 512 1024; do for M in stock fused; do
  timeout 600 nsys profile -o results/nsys_${M}_$S --force-overwrite=true --trace=cuda,nvtx \
    --capture-range=cudaProfilerApi --capture-range-end=stop \
    $PY scale_e2e.py --size $S --mode $M --steps 2 --reps 1 --profile --tag _nsys --out results/nsys_tmp \
    > results/logs/nsys_${M}_$S.log 2>&1
  nsys stats --report cuda_gpu_kern_sum --format csv --force-export=true \
    results/nsys_${M}_$S.nsys-rep > results/nsys_${M}_${S}_kern.csv 2>/dev/null
done; done
$PY classify_nsys.py results/nsys_*_kern.csv > results/nsys_summary.log 2>&1

# ---------------- G. kernel-level: Triton sweep + ncu ----------------
stage "G. tier1 sweep + ncu"
$PY sweep_tier1.py 2>&1 | grep -v '^W10\|Unable' > results/tier1.log
TOKENS="256 1024 4096" bash ncu_sweep.sh > results/ncu_summary.log 2>&1

# ---------------- H. CUDA C++ fusion vs launch cost ----------------
stage "H. CUDA C++ test"
( cd external_tests_inspired && nvcc -O3 -std=c++17 -arch=native launch_vs_fusion.cu -o launch_vs_fusion \
  && ./launch_vs_fusion ) > results/launch_vs_fusion.log 2>&1

# ---------------- I. torch.compile (slowest, runs last) baseline (the fair baseline) ----------------
stage "I. torch.compile (slowest, runs last) baseline at 256 and 512"
for S in 256 512; do for M in stock fused; do
  run 600 results/logs/compile_${M}_${S}.log $PY scale_e2e.py --size $S --mode $M --compile --steps 14 --reps 2
done; done

$PY make_report.py >> $P 2>&1
stage "ALL_DONE"
