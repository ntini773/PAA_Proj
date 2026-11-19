#!/usr/bin/env bash
# The headline results in about 25 minutes, for all three sizes, with selective quantization.
#
#   bash run_final30.sh
#
#   1. step time (alone on the GPU)   6 configs x 256/512/1024                          (about 10 min)
#   2. real-prompt quality (parallel, one process per size)   bf16, stock, w4a16, module stack, + trim + cuDNN, + selective
#      each compared with bf16 AND with stock NVFP4                                       (about 8-12 min)
#   3. analysis/final_table.py -> results/final_table.md (step ms, speed-up, LPIPS, CLIP)
# Results are saved to the results-eval2 branch after each phase. Each phase continues if the previous one failed.
# Needs results/static_scales_<size>.json (made here if missing). Skips E11/E12 on purpose.
cd "$(dirname "$0")"
source ./pyenv.sh
SIZES=${SIZES:-"256 512 1024"}
STEPS=${STEPS:-14}
EMBEDS=${EMBEDS:-prompts/embeds_pixart.pt}
CALIB=${CALIB:-prompts/embeds_calib.pt}
mkdir -p logs/final30
say() { echo; echo "[$(date +%H:%M:%S)] ===== $* ====="; }
save() { [ -z "$NO_SAVE" ] && bash save_results.sh "final30-$1" >> logs/final30/save.log 2>&1; }

say "0/3 static scales for sizes that lack them"
for s in $SIZES; do
  [ -f results/static_scales_$s.json ] || $PY calibrate_static_scales.py --size $s --steps $STEPS --embeds $CALIB --n 8 > logs/final30/scales_$s.log 2>&1 \
    || echo "  scales for $s FAILED (logs/final30/scales_$s.log): +static variants will be skipped at that size"
done

say "1/3 step time, alone on the GPU"
source profiling/configs.sh
CONFIGS="bf16_compile fp8_compile stock_compile module_cuda_compile module_best_compile module_best_xcudnn module_best_sel_xcudnn" \
  SIZES="$SIZES" ITERS=20 bash profiling/run_step_matrix.sh > logs/final30/timing.log 2>&1
tail -n 3 logs/final30/timing.log
save timing

say "2/3 quality in parallel (one process per size)"
BEST="module+be=cuda_swizzled+static+concat"
for s in $SIZES; do
  $PY quality/run_quality.py --size $s --steps $STEPS --embeds $EMBEDS ${NO_CLIP:+--no-clip} --reference bf16 --extra-ref stock \
    --out results/quality_final/$s \
    --variants stock w4a16 module+be=cuda_swizzled $BEST $BEST+trim+xcudnn $BEST+selective+trim+xcudnn > logs/final30/quality_$s.log 2>&1 &
done
wait
for s in $SIZES; do grep -E "ms/img|FAILED" logs/final30/quality_$s.log | cut -c1-130; done
save quality

say "3/3 table"
$PY analysis/final_table.py 2>&1 | tee logs/final30/table.log
save table
echo; echo "done. Table: results/final_table.md   Images: results/quality_final/<size>/"
