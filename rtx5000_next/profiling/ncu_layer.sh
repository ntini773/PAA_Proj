#!/usr/bin/env bash
# ncu on one layer (quantize + GEMM) of block 3, last forward only. Cold caches, so DRAM bytes are worst case.
cd "$(dirname "$0")/.."
source profiling/configs.sh
CONFIGS=${CONFIGS:-$NCU_CONFIGS}
SIZES=${SIZES:-"512 1024"}
NCU=${NCU:-ncu}
LAYERS=${LAYERS:-"qkv:transformer_blocks\.3\.attn1\.to_q$ fc1:transformer_blocks\.3\.ff\.net\.0\.proj$ fc2:transformer_blocks\.3\.ff\.net\.2$ crosskv:transformer_blocks\.3\.attn2\.to_k$"}
METRICS=gpu__time_duration.sum,dram__bytes_op_read.sum,dram__bytes_op_write.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,launch__grid_size,launch__block_size,launch__registers_per_thread,lts__t_sector_hit_rate.pct,l1tex__t_sector_hit_rate.pct,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,l1tex__t_requests_pipe_lsu_mem_global_op_st.sum,lts__t_bytes.sum
# sections go into the .ncu-rep for the Nsight Compute GUI: memory chart, roofline, access tables, source counters
SECTIONS="--section SpeedOfLight --section SpeedOfLight_RooflineChart --section MemoryWorkloadAnalysis --section MemoryWorkloadAnalysis_Chart --section MemoryWorkloadAnalysis_Tables --section LaunchStats --section Occupancy --section WarpStateStats --section SourceCounters"
OUT=results/ncu
mkdir -p $OUT logs/ncu
for size in $SIZES; do
  for cfg in $CONFIGS; do
    for entry in $LAYERS; do
      layer=${entry%%:*}; rx=${entry#*:}
      name=${cfg}_${layer}_${size}
      echo "[$(date +%H:%M:%S)] ncu $MODEL $name"
      $NCU --target-processes all --profile-from-start off --cache-control all --clock-control none \
        $SECTIONS --metrics $METRICS --export $OUT/$name.ncu-rep --force-overwrite \
        $PY step_target.py --model $MODEL --size $size $(cfg_args $cfg) --config ${cfg}_${layer} --iters 3 --warmup 3 \
        --scope layer --layer-regex "$rx" > logs/ncu/$name.log 2>&1 || { echo "  FAILED, see logs/ncu/$name.log"; continue; }
      $NCU -i $OUT/$name.ncu-rep --csv --page raw --metrics $METRICS --log-file $OUT/$name.csv > /dev/null 2>> logs/ncu/$name.log
    done
  done
done
