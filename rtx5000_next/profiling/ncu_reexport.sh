#!/usr/bin/env bash
# Rebuild results/ncu/*.csv from the saved .ncu-rep files. No GPU, no re-profiling. Needs only ncu.
# Use it after changing METRICS: the reports already hold every section metric, including dram__bytes_op_read/write.
cd "$(dirname "$0")/.."
NCU=${NCU:-ncu}
METRICS=$(sed -n 's/^METRICS=//p' profiling/ncu_layer.sh)
for rep in results/ncu/*.ncu-rep; do
  $NCU -i "$rep" --csv --page raw --metrics "$METRICS" --log-file "${rep%.ncu-rep}.csv" > /dev/null
  echo "wrote ${rep%.ncu-rep}.csv"
done
