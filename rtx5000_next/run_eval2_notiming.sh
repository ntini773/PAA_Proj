#!/usr/bin/env bash
# The eval2 stages that do not need clean timing, run side by side to save wall time (about 50 min instead of about 90).
#   bash run_eval2_notiming.sh          then run the timing stages alone:  for s in E02 E05 E06 E07 E08 E12; do bash run_eval2.sh --only $s; done
#
# Why these can overlap: they measure accuracy or produce files (embeddings, scales, plans, images, metrics), not speed, so
# sharing the GPU only slows them down a little. Do NOT run them while a timing stage (E02 E05-E09 E12) is running: the
# timing numbers would be wrong. The "ms per image" column of E11 is informational and is contaminated here; use E12 for time.
# Order and why:
#   1. E01  calibration prompt embeddings (the next stages need them)
#   2. E04 + E10 + E03 together   static scales, DiTFastAttn plans, real w4a16 vs w4a4 (all need only E01 / the eval prompts)
#   3. E11 per size together      real-prompt quality (needs the scales from E04 and the plans from E10)
# GPU memory: the 1024 px quality run is the largest (CFG batch of 12 prompts+uncond); each job here fits in 48 GB, two at 1024 px
# together should too. E14 (FLUX 24 GB) is left out on purpose: run it alone, afterwards.
cd "$(dirname "$0")"
source ./pyenv.sh
SIZES=${SIZES:-"512 1024"}

echo "[1/3] E01"
bash run_eval2.sh --only E01

echo "[2/3] E04, E10, E03 in parallel"
TAG=_e04 bash run_eval2.sh --only E04 &
TAG=_e10 bash run_eval2.sh --only E10 &
TAG=_e03 bash run_eval2.sh --only E03 &
wait

echo "[3/3] E11, one process per size, in parallel"
for s in $SIZES; do
  SIZES=$s TAG=_$s bash run_eval2.sh --only E11 &
done
wait

echo "done. Failed stages: $(ls results/.fail 2>/dev/null | tr '\n' ' ')   (logs: logs/eval2/)"
echo "Next, alone on the GPU: for s in E02 E05 E06 E07 E08 E12; do bash run_eval2.sh --only \$s; done"
