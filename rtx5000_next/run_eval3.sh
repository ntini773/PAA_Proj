#!/usr/bin/env bash
# Eval 3 pipeline. Wave 1 = setup and measurements that need no new model code. Later waves add stages W2xx, W3xx.
#
#   bash run_eval3.sh --list                 stages, what they do, rough time
#   bash run_eval3.sh --setup                W100-W103: preflight, downloads, T5 embeddings (needs no idle GPU except T5 encode)
#   bash run_eval3.sh --wave1                W100-W106: setup + roofs + noise floor + baseline step times
#   bash run_eval3.sh --only W104            one stage        --from W104   this stage and the rest
#   bash run_eval3.sh --dry-run ...          print commands only
#   FORCE=1 bash run_eval3.sh --only W104    rerun a finished stage
# Env: PY, SIZE_Q (quality size, default 512), NPROMPTS (default 31), SEEDS (default 2), STEPS (14), WITH_MODELS=1 (also download
# SD3.5 Medium + FLUX schnell with text encoders), HF_TOKEN for gated repos.
# One stage at a time on purpose (bug 27: parallel jobs were killed by host memory). Failed stages are marked and the next one runs.
# Markers: results/eval3/.done/<ID>, .fail/<ID>; logs: logs/eval3/<ID>.log.  Results go to results/eval3/.
cd "$(dirname "$0")"
source ./pyenv.sh
SIZE_Q=${SIZE_Q:-512}; NPROMPTS=${NPROMPTS:-30}; SEEDS=${SEEDS:-2}; STEPS=${STEPS:-14}
# Knobs for the reviewer stages (W2xx-W3xx). The defaults are the real run; eval3/orchestrate.py --mode smoke sets tiny values
# and EVAL3_RESULTS=results_smoke so a smoke run exercises the same code without touching real results.
RES=${EVAL3_RESULTS:-results}
STEP_SIZES=${STEP_SIZES:-"512 1024"}; REPS=${REPS:-3}; STEP_ITERS=${STEP_ITERS:-30}; HEAD_REPS=${HEAD_REPS:-2}
ROOF_SIZES=${ROOF_SIZES:-"256 512 1024"}; ROOF_ITERS=${ROOF_ITERS:-3}; STATIC_SIZES=${STATIC_SIZES:-"512 1024"}; CALIB_N=${CALIB_N:-8}
SEEDS3=${SEEDS3:-3}; QHI=${QHI:-1024}; NQHI=${NQHI:-20}; XGPU_SIZES=${XGPU_SIZES:-"256 512"}; ATTN_SIZES=${ATTN_SIZES:-"1024"}
PEAKS_ARGS=${PEAKS_ARGS:-}; KISO_ARGS=${KISO_ARGS:-}; TAG_SFX=${TAG_SFX:-}
mkdir -p $RES/eval3/.done $RES/eval3/.fail logs/eval3
source profiling/configs.sh

ONLY=""; FROM=""; DRY=0; LIST=0; SET=""
while [ $# -gt 0 ]; do
  case $1 in
    --only) ONLY=$2; shift 2;; --from) FROM=$2; shift 2;; --dry-run) DRY=1; shift;; --list) LIST=1; shift;;
    --setup) SET="W100 W101 W102 W103"; shift;; --wave1) SET="W100 W101 W102 W103 W104 W105 W106"; shift;;
    --queue) SET="W106 W107 W108 W109"; shift;;
    --queue2) SET="W110 W112 W111"; shift;;
    --wave2) SET="W200 W201 W202 W203"; shift;;   # needs W104 (roofs) done; SD3.5 and FLUX weights for their rows (MODELS="pixart_sigma:512" for PixArt only)   # round 2: the cheap stages first, the 1024 px quality run last   # GPU-only queue that needs no new model downloads; run it in the background with nohup
    *) echo "unknown argument $1"; exit 2;;
  esac
done
STAGES="W100 W101 W102 W103 W104 W105 W106 W107 W108 W109 W110 W111 W112 W200 W201 W202 W203 W299 W300 W301 W302 W303 W304 W305 W306 W399"
desc() { case $1 in
  W100) echo "preflight: GPU, packages, tools, disk, HF cache (models, metric weights)  ~1 min";;
  W101) echo "downloads: metric weights (CLIP, PickScore, LPIPS, FID); PixArt sizes; with WITH_MODELS=1 SD3.5-M and FLUX schnell incl. text encoders  ~5-40 min, network";;
  W102) echo "variant-name validation for every name used in wave 1  ~5 s";;
  W103) echo "T5 embeddings for calibration and showcase prompts (needs the GPU for about 2 min)  ~5 min";;
  W104) echo "roofs: M/N/K sweep of FP4 and bf16 GEMM, flash attention per head dim, bandwidth (idle GPU)  ~5 min";;
  W105) echo "noise floor + controls: bf16 eager vs compiled vs other seed vs fp8 vs stock, $NPROMPTS prompts x $SEEDS seeds at ${SIZE_Q}px  ~25-40 min";;
  W106) echo "baseline step times 512 and 1024, 3 repeats: bf16c, fp8c, stockc (stock_tk c is skipped unless mslk works: STOCK_TK=1)  ~20 min";;
  W107) echo "microbenchmarks: FP4 GEMM at real shapes, attention backends (flash/cudnn/efficient/sage2), idle GPU  ~5 min";;
  W108) echo "lever step times 512 and 1024 (no static scale): base, concat, noshare, selective, trim, xcudnn, sage2, bf16 attention variants  ~25 min";;
  W299) echo "static activation scales per size (needed only for the +static ablation rows; skipped if the files exist)  ~6 min";;
  W300) echo "reviewer point 1: ladder (stock -> final stack) + leave-one-out step times, 512 and 1024, 3 repeats, CUDA Graph; +static row if scales exist  ~40 min";;
  W301) echo "reviewer point 3: ladder / leave-one-out / static / w4a16 quality at ${SIZE_Q}px, $NPROMPTS prompts x 3 seeds, paired CIs  ~35 min";;
  W302) echo "reviewer point 1: quantize kernel in isolation (torchao eager/compiled vs ours Triton/CUDA), bit-exactness, GB/s; ncu DRAM bytes  ~6 min";;
  W303) echo "reviewer point 4: attention backends: microbench + step time (bf16 and final stack with flash/cudnn/efficient/sage2) at ${ATTN_SIZES}px  ~25 min";;
  W304) echo "quality at ${QHI}px, $NQHI prompts x $SEEDS seeds: stock, w4a16, fp8, bf16+sage2, final stack with/without sage2, +static  ~35 min";;
  W305) echo "headline rows repeated $HEAD_REPS more times for tighter ranges  ~15 min";;
  W306) echo "reviewer point 5: reduced matrix at ${XGPU_SIZES}px, the same set the laptop runs  ~10 min";;
  W399) echo "verify the run (files, consistency checks, tests) and package results_eval3.tgz (CPU)  ~2 min";;
  W200) echo "T-conv/T-cover: NVFP4 conversion + per-layer bit-exactness vs torchao: PixArt 512, SD3.5-M, FLUX schnell (MODELS env limits)  ~10 min";;
  W201) echo "T-assume: PixArt premises (memory bound, FP4 beats bf16, attention share, mask, CFG, error vs fp8) on each model  ~15 min";;
  W202) echo "T-roof/T-attn: per-class kernel time vs speed of light, PixArt 512+1024 (incl. sage check), SD3.5-M, FLUX  ~20 min";;
  W203) echo "roofline report + figures + consistency checks (CPU)  ~1 min";;
  W110) echo "round 2 step times 512+1024 (3 repeats): key rows again + new combos (fp8+sage2, final stack, stock+sage2)  ~20 min";;
  W111) echo "round 2 quality at 1024px (NQ1024 prompts x 2 seeds): bf16/fp8/stock/final stack with and without sage2, selective  ~35 min";;
  W112) echo "round 2 quality at 512px for the new combos (fp8+sage2, final stack)  ~12 min";;
  W109) echo "lever quality at ${SIZE_Q}px, $NPROMPTS prompts x $SEEDS seeds, paired CI vs stock  ~30 min";;
esac; }
if [ $LIST -eq 1 ]; then for s in $STAGES; do echo "$s  $(desc $s)"; done; exit 0; fi

go=1; [ -n "$FROM" ] && go=0
run() {
  local id=$1; shift
  [ -n "$FROM" ] && [ "$id" = "$FROM" ] && go=1
  [ $go -eq 0 ] && return 0
  [ -n "$ONLY" ] && [ "$ONLY" != "$id" ] && return 0
  if [ -n "$SET" ] && ! echo " $SET " | grep -q " $id "; then return 0; fi
  if [ -f $RES/eval3/.done/$id ] && [ -z "$FORCE" ] && [ -z "$ONLY" ]; then echo "[$id] already done (FORCE=1 to rerun)"; return 0; fi
  echo "[$(date +%H:%M:%S)] $id: $(desc $id)"
  # GPU guard: a stage that needs the GPU must not start while another process uses it (it caused the wave 2 OOM and spoils timings).
  # Stages W100-W102 and W203 do not need the GPU. Waits up to 10 min for the other job, then marks the stage failed.
  case $id in W100|W101|W102|W203) ;; *)
    if [ $DRY -eq 0 ]; then
      for i in $(seq 1 60); do
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
        [ -z "$used" ] || [ "$used" -lt 2000 ] && break
        [ $i -eq 1 ] && echo "  GPU busy (${used} MiB in use, other jobs: $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ';')), waiting up to 10 min"
        sleep 10
      done
      if [ -n "$used" ] && [ "$used" -ge 2000 ]; then echo "  GPU still busy: $id not started"; touch $RES/eval3/.fail/$id; return 0; fi
    fi;;
  esac
  if [ $DRY -eq 1 ]; then declare -f "$1" | sed '1,2d;$d' | sed 's/^/    /'; return 0; fi
  local log=logs/eval3/$id${TAG_SFX}.log
  if "$@" > $log 2>&1; then touch $RES/eval3/.done/$id; rm -f $RES/eval3/.fail/$id
  else touch $RES/eval3/.fail/$id; echo "  FAILED, see $log (last lines:)"; tail -n 6 $log | sed 's/^/    /'; fi
}

BEST="module+be=cuda_swizzled+concat"   # eval3 method: no static scale (image quality)
w100() { $PY eval3/preflight.py --models pixart_sigma:256,512,1024 sd35_medium:1024 flux_schnell:1024; }
w101() { $PY eval3/prefetch.py --metrics --models pixart_sigma || return 1
         [ -n "$WITH_MODELS" ] && { $PY eval3/prefetch.py --models sd35_medium flux_schnell --with-text-encoder; }; return 0; }
w102() { $PY eval3/validate_variants.py bf16 bf16+compile bf16@seedshift fp8+compile stock stock+compile stock_tk+compile \
           module+be=cuda_swizzled+compile $BEST+compile $BEST+trim+xcudnn+compile; }
w103() { $PY prompts/make_prompt_embeds.py --prompts prompts/prompts_calib.json --out prompts/embeds_calib.pt &&
         $PY prompts/make_prompt_embeds.py --prompts prompts/prompts_showcase.json --out prompts/embeds_showcase.pt; }
w104() { $PY bench/peaks_sweep.py $PEAKS_ARGS; }
w105() { $PY eval3/quality_grid.py --size $SIZE_Q --embeds prompts/embeds_showcase.pt --n-prompts $NPROMPTS --seeds $SEEDS --steps $STEPS \
           --tag noise_floor_$SIZE_Q; }
w106() { local cfgs="bf16_compile fp8_compile stock_compile${STOCK_TK:+ stock_tk_compile}"
         for rep in 1 2 3; do CONFIGS="$cfgs" SIZES="512 1024" ITERS=30 bash profiling/run_step_matrix.sh || return 1; done; }
w107() { $PY bench/fp4_gemm_shapes.py && $PY bench/attention_backends.py; }
w108() { CONFIGS="$EVAL3_LEVERS" SIZES="512 1024" ITERS=30 bash profiling/run_step_matrix.sh; }
MODELS=${MODELS:-"pixart_sigma:512 sd35_medium:1024 flux_schnell:1024"}   # model:size pairs for wave 2
each_model() {  # each_model fn: run "fn model size" for every pair; one failing model does not stop the others; exit 1 if any failed
  local rc=0 p; for p in $MODELS; do "$@" ${p%%:*} ${p##*:} || rc=1; done; return $rc; }
w200() { each_model w200_one; }
w200_one() { $PY eval3/model_check.py --model $1 --size $2 --backend cuda_swizzled; }
w201() { each_model w201_one; }
w201_one() { $PY eval3/assume_check.py --model $1 --size $2; }
w202() { local P="module+be=cuda_swizzled" rc=0
         for s in $ROOF_SIZES; do   # 256 px is needed by eval3/amdahl.py (attention share vs size)
           $PY eval3/roofline_profile.py --model pixart_sigma --size $s --iters $ROOF_ITERS --variants bf16+compile fp8+compile stock+compile $P+compile \
             $P+concat+selective+trim+xcudnn+compile bf16+sage2+compile $P+concat+selective+trim+xcudnn+sage2+compile || rc=1
         done
         for p in $MODELS; do case $p in pixart_sigma:*) ;; *)
           $PY eval3/roofline_profile.py --model ${p%%:*} --size ${p##*:} --iters $ROOF_ITERS --variants bf16+compile fp8+compile stock+compile module+compile || rc=1;; esac; done
         return $rc; }
w203() { $PY eval3/roofline_report.py; }
# --- eval3 reviewer points 1 and 3 -----------------------------------------------------------------------------------------------
has_static() { [ -f $RES/static_scales_$1.json ]; }   # per-size file only: a scale calibrated at another size is not used here
w299() { local s rc=0
         for s in $STATIC_SIZES; do
           has_static $s && { echo "static scales for $s exist"; continue; }
           $PY calibrate_static_scales.py --size $s --steps $STEPS --embeds prompts/embeds_calib.pt --n $CALIB_N || rc=1
           has_static $s || { echo "calibrate_static_scales.py did not write $RES/static_scales_$s.json"; rc=1; }
         done; return $rc; }
w300() { local rc=0 rep s
         for rep in $(seq 1 $REPS); do
           CONFIGS="$EVAL3_LADDER $EVAL3_LOO" SIZES="$STEP_SIZES" ITERS=$STEP_ITERS bash profiling/run_step_matrix.sh || rc=1
           for s in $STEP_SIZES; do   # the static ablation row only where the per-size scale file exists
             has_static $s && { CONFIGS="$EVAL3_STATIC" SIZES="$s" ITERS=$STEP_ITERS bash profiling/run_step_matrix.sh || rc=1; }
           done
         done; return $rc; }
w301() { local M="module+be=cuda_swizzled" S=""
         has_static $SIZE_Q && S="$M+concat+selective+trim+xcudnn+sage2+static+compile"
         # quality-relevant steps of the ladder, then leave-one-out; trim and cuDNN are near-exact, kept so the table is complete
         $PY eval3/quality_grid.py --size $SIZE_Q --embeds prompts/embeds_showcase.pt --n-prompts $NPROMPTS --seeds $SEEDS3 --steps $STEPS \
           --controls none --tag ladder_$SIZE_Q$TAG_SFX --variants stock w4a16 fp8+compile \
           $M+noshare+compile $M+concat+compile $M+concat+selective+compile $M+concat+selective+trim+xcudnn+compile \
           $M+concat+selective+trim+xcudnn+sage2+compile $S \
           $M+selective+trim+xcudnn+sage2+compile $M+concat+trim+xcudnn+sage2+compile $M+concat+selective+xcudnn+sage2+compile \
           $M+concat+selective+trim+sage2+compile module+be=triton+concat+selective+trim+xcudnn+sage2+compile; }
w302() { $PY eval3/kernel_isolation.py $KISO_ARGS || return 1
         # DRAM bytes per quantize call from ncu (optional: a missing ncu or counter permission does not fail the stage)
         if command -v ncu > /dev/null; then
           ncu --metrics dram__bytes_op_read.sum,dram__bytes_op_write.sum,gpu__time_duration.sum --csv --page raw \
             $PY eval3/kernel_isolation.py --once > $RES/eval3/kernel_isolation_ncu.csv 2> logs/eval3/W302_ncu${TAG_SFX}.err \
             || echo "ncu failed (optional), see logs/eval3/W302_ncu${TAG_SFX}.err"
         else echo "ncu not found (optional)"; fi; return 0; }
w303() { local rc=0 rep
         $PY bench/attention_backends.py || rc=1
         for rep in $(seq 1 $REPS); do
           CONFIGS="$EVAL3_ATTN" SIZES="$ATTN_SIZES" ITERS=$STEP_ITERS bash profiling/run_step_matrix.sh || rc=1
         done; return $rc; }
w304() { local M="module+be=cuda_swizzled" S=""
         has_static $QHI && S="$M+concat+selective+trim+xcudnn+sage2+static+compile"
         $PY eval3/quality_grid.py --size $QHI --embeds prompts/embeds_showcase.pt --n-prompts $NQHI --seeds $SEEDS --steps $STEPS --batch 4 \
           --controls none --tag key_$QHI$TAG_SFX --variants stock w4a16 fp8+compile bf16+sage2+compile \
           $M+concat+selective+trim+xcudnn+compile $M+concat+selective+trim+xcudnn+sage2+compile $S; }
w305() { local rc=0 rep
         for rep in $(seq 1 $HEAD_REPS); do
           CONFIGS="$EVAL3_HEAD" SIZES="$STEP_SIZES" ITERS=$STEP_ITERS bash profiling/run_step_matrix.sh || rc=1
         done; return $rc; }
w306() { local rc=0 rep
         for rep in $(seq 1 $REPS); do
           CONFIGS="$EVAL3_XGPU" SIZES="$XGPU_SIZES" ITERS=$STEP_ITERS bash profiling/run_step_matrix.sh || rc=1
         done; return $rc; }
w399() { $PY eval3/verify_run.py --mode ${RUN_MODE:-slot1}; }
NQ1024=${NQ1024:-20}
w110() { for rep in 1 2 3; do CONFIGS="$EVAL3_KEY $EVAL3_COMBOS" SIZES="512 1024" ITERS=30 bash profiling/run_step_matrix.sh || return 1; done; }
w111() { local M="module+be=cuda_swizzled"
         $PY eval3/quality_grid.py --size 1024 --embeds prompts/embeds_showcase.pt --n-prompts $NQ1024 --seeds $SEEDS --steps $STEPS --batch 4 \
           --controls none --tag round2_1024 --variants bf16+compile stock fp8+compile bf16+trim+xcudnn+sage2+compile fp8+trim+xcudnn+sage2+compile \
           $M+concat+trim+xcudnn+compile $M+trim+xcudnn+sage2+compile $M+concat+selective+trim+xcudnn+sage2+compile; }
w112() { local M="module+be=cuda_swizzled"
         $PY eval3/quality_grid.py --size $SIZE_Q --embeds prompts/embeds_showcase.pt --n-prompts $NPROMPTS --seeds $SEEDS --steps $STEPS \
           --controls none --tag round2_$SIZE_Q --variants stock fp8+sage2+compile fp8+trim+xcudnn+sage2+compile stock+sage2+compile \
           $M+trim+xcudnn+sage2+compile $M+concat+selective+trim+xcudnn+sage2+compile; }
w109() { local M="module+be=cuda_swizzled"
         $PY eval3/quality_grid.py --size $SIZE_Q --embeds prompts/embeds_showcase.pt --n-prompts $NPROMPTS --seeds $SEEDS --steps $STEPS \
           --controls none --tag levers_$SIZE_Q --variants stock \
           $M+compile $M+concat+compile $M+noshare+compile $M+concat+selective+compile $M+concat+trim+compile \
           $M+concat+trim+xcudnn+compile $M+concat+trim+xcudnn+sage2+compile \
           bf16+sage2+compile bf16+trim+xcudnn+compile bf16+licfg0.2-0.8+compile $M+concat+trim+xcudnn+licfg0.2-0.8+compile; }

for s in $STAGES; do run $s $(echo $s | tr 'W' 'w'); done
echo "finished. Failed stages: $(ls results/eval3/.fail 2>/dev/null | tr '\n' ' ')"
