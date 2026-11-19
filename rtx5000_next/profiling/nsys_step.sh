#!/usr/bin/env bash
# nsys timeline of a few forwards per config and size, plus CSV reports for analyze_nsys.py
cd "$(dirname "$0")/.."
source profiling/configs.sh
ITERS=${ITERS:-5}
EXTRA_ARGS=${EXTRA_ARGS:-}   # e.g. EXTRA_ARGS="--stage-ranges" for per-stage NVTX ranges (eager variants)
OUT=results/nsys
mkdir -p $OUT logs/nsys
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv
for size in $SIZES; do
  for cfg in $CONFIGS; do
    name=${cfg}_${size}
    echo "[$(date +%H:%M:%S)] nsys $MODEL $name"
    nsys profile -o $OUT/$name --force-overwrite=true --trace=cuda,nvtx --sample=none --cpuctxsw=none \
      --capture-range=cudaProfilerApi --capture-range-end=stop \
      $PY step_target.py --model $MODEL --size $size $(cfg_args $cfg) --config $cfg --iters $ITERS --warmup 5 \
      --scope step --nvtx $EXTRA_ARGS > logs/nsys/$name.log 2>&1 || { echo "  FAILED, see logs/nsys/$name.log"; continue; }
    # nvtx_gpu_proj_sum: GPU time projected onto each NVTX range (per stage with --stage-ranges); nvtx_kern_sum: kernels per range;
    # cuda_kern_exec_sum: launch API time vs queue time vs kernel time (launch overhead)
    for rep in cuda_gpu_kern_sum cuda_gpu_trace cuda_api_sum nvtx_sum nvtx_gpu_proj_sum nvtx_kern_sum cuda_kern_exec_sum; do
      nsys stats -r $rep --format csv --force-export=true --output $OUT/$name $OUT/$name.nsys-rep \
        > /dev/null 2>> logs/nsys/$name.log || true
    done
  done
done
