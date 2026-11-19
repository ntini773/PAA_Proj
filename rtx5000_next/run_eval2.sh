#!/usr/bin/env bash
# The eval2 experiment pipeline: baselines, the NVFP4 module stack, attention options, DiTFastAttn, limited-interval CFG, quality.
#
#   bash run_eval2.sh                run every stage in order (each runs to completion; no timeouts)
#   bash run_eval2.sh --list         stages with what they do and a rough time (guesses, RTX PRO 5000)
#   bash run_eval2.sh --only E06     one stage          --from E05   this stage and all after it
#   bash run_eval2.sh --dry-run      print the commands, run nothing
#   FORCE=1 bash run_eval2.sh --only E06    rerun a stage that is already marked done
# Needs an idle GPU for E02-E10. Stages are independent where possible: a failed stage is logged, marked failed, and the
# next stage still runs (a stage that needs its output reports exactly what to run). Markers: results/.done/<ID> and
# results/.fail/<ID>; logs: logs/eval2/<ID>.log. progress_report.py refreshes PROGRESS.md.
# Env: PY (python), SIZES="512 1024", STEPS=14, EMBEDS, CLIP (default openai/clip-vit-large-patch14), NO_CLIP=1.
cd "$(dirname "$0")"
source ./pyenv.sh   # finds a python with torch (with or without `pixi run`)
SIZES=${SIZES:-"512 1024"}
STEPS=${STEPS:-14}
EMBEDS=${EMBEDS:-prompts/embeds_pixart.pt}
CALIB=${CALIB:-prompts/embeds_calib.pt}
CLIPARG=${NO_CLIP:+--no-clip}
mkdir -p results/.done results/.fail logs/eval2
source profiling/configs.sh

ONLY=""; FROM=""; DRY=0; LIST=0
while [ $# -gt 0 ]; do
  case $1 in
    --only) ONLY=$2; shift 2;; --from) FROM=$2; shift 2;; --dry-run) DRY=1; shift;; --list) LIST=1; shift;;
    *) echo "unknown argument $1"; exit 2;;
  esac
done

STAGES="E01 E02 E03 E04 E05 E06 E07 E08 E09 E10 E11 E12 E13 E14 E15"
desc() { case $1 in
  E01) echo "calibration prompt embeddings (T5 encode of prompts_calib.json)  ~5 min";;
  E02) echo "microbenchmarks: FP4 GEMM shapes, attention backends  ~5 min";;
  E03) echo "real w4a16 vs w4a4 (weights-only NVFP4 vs full NVFP4), 6 real prompts  ~15 min";;
  E04) echo "static activation scale calibration per size  ~5 min";;
  E05) echo "fair baselines step time: bf16/fp8/stock compiled, stock_tk  ~25 min";;
  E06) echo "NVFP4 module stack step time (dynamic, static, concat, selective, CUDA kernel)  ~35 min";;
  E07) echo "attention options on the best stack: trim, cuDNN, Sage, K/V pooling  ~20 min";;
  E08) echo "per-stage timing tables (GPU busy, gaps, kernels, CPU launch)  ~10 min";;
  E09) echo "nsys: kernel launches and GPU busy per stage for the key variants  ~25 min";;
  E10) echo "DiTFastAttn plan search (3 error thresholds, per size)  ~15 min";;
  E11) echo "real-prompt quality of every variant (LPIPS, SSIM, PSNR, CLIP) with generation time  ~45 min";;
  E12) echo "end-to-end pipeline timing (T5, steps, VAE) for the headline variants  ~15 min";;
  E13) echo "traffic ledger and whole-forward roofline refresh (CPU)  ~1 min";;
  E14) echo "smoke of the new models (Sana 0.6B, SD3.5 Medium, FLUX)  ~15 min";;
  E15) echo "package results  ~1 min";;
esac; }

if [ $LIST -eq 1 ]; then for s in $STAGES; do echo "$s  $(desc $s)"; done; exit 0; fi

go=1; [ -n "$FROM" ] && go=0
run() {  # run ID command...   (honours --only / --from / --dry-run / done markers)
  local id=$1; shift
  [ -n "$FROM" ] && [ "$id" = "$FROM" ] && go=1
  [ $go -eq 0 ] && return 0
  [ -n "$ONLY" ] && [ "$ONLY" != "$id" ] && return 0
  if [ -f results/.done/$id ] && [ -z "$FORCE" ] && [ -z "$ONLY" ]; then echo "[$id] already done (FORCE=1 to rerun)"; return 0; fi
  echo "[$(date +%H:%M:%S)] $id: $(desc $id)"
  if [ $DRY -eq 1 ]; then declare -f "$1" | sed '1,2d;$d' | sed 's/^/    /'; return 0; fi
  local log=logs/eval2/$id${TAG}.log   # TAG lets two copies of one stage (e.g. E11 per size) run side by side
  if "$@" > $log 2>&1; then touch results/.done/$id; rm -f results/.fail/$id
  else touch results/.fail/$id; echo "  FAILED, see $log (last lines:)"; tail -n 5 $log | sed 's/^/    /'; fi
  $PY progress_report.py > /dev/null 2>&1
}

e01() { $PY prompts/make_prompt_embeds.py --prompts prompts/prompts_calib.json --out $CALIB; }
e02() { $PY bench/fp4_gemm_shapes.py && $PY bench/attention_backends.py; }
e03() { for s in $SIZES; do $PY precision/real_w4a16_vs_w4a4.py --size $s --embeds $EMBEDS --steps $STEPS $CLIPARG || return 1; done; }
e04() { for s in $SIZES; do $PY calibrate_static_scales.py --size $s --steps $STEPS --embeds $CALIB --n 8 || return 1; done; }
matrix() { CONFIGS="$1" SIZES="$SIZES" ITERS=30 bash profiling/run_step_matrix.sh; }
e05() { matrix "$EVAL2_BASELINES"; }
e06() { matrix "$EVAL2_MODULE"; }
e07() { matrix "$EVAL2_ATTENTION"; }
e08() { for s in $SIZES; do for v in bf16 stock module+be=triton module+be=cuda_swizzled+static+concat; do
          $PY timing/profile_stages.py --variant $v --size $s --iters 3 || return 1; done; done; $PY analysis/analyze_stages.py; }
e09() { CONFIGS="bf16_compile stock_compile module_best_compile" SIZES="$SIZES" bash profiling/nsys_step.sh &&
        EXTRA_ARGS="--stage-ranges" CONFIGS="module_triton" SIZES="$SIZES" bash profiling/nsys_step.sh && $PY analysis/analyze_nsys.py; }
e10() { for s in $SIZES; do $PY ditfast/make_plans.py --size $s --deltas 0.05 0.1 0.2 --steps $STEPS --embeds $CALIB || return 1; done; }
quality_variants() {
  local best="module+be=cuda_swizzled+static+concat"
  echo "stock w4a16 module+be=cuda_swizzled $best module+be=cuda_swizzled+selective bf16+trim bf16+xcudnn bf16+trim+xcudnn" \
       "bf16+kvds2 bf16+dfa0.05 bf16+dfa0.1 bf16+dfa0.2 bf16+licfg0.2-0.8 bf16+licfg0.3-0.7 $best+trim+xcudnn $best+trim+xcudnn+dfa0.1" \
       "$best+trim+xcudnn+licfg0.2-0.8 $best+trim+xcudnn+dfa0.1+licfg0.2-0.8"
  $PY -c "import sageattention" 2>/dev/null && echo "bf16+sage2 $best+trim+xcudnn+sage2"
  $PY -c "import sageattn3" 2>/dev/null && echo "bf16+sage3 $best+trim+xcudnn+sage3"
}
e11() { for s in $SIZES; do $PY quality/run_quality.py --size $s --steps $STEPS --embeds $EMBEDS ${CLIP:+--clip $CLIP} $CLIPARG \
          --variants $(quality_variants | tr '\n' ' ') || return 1; done; }
e12() { local best="module+be=cuda_swizzled+static+concat+trim+xcudnn"
        $PY bench/pipeline_e2e.py --size 1024 --images 4 --steps $STEPS --t5 --variants bf16 bf16+compile stock_compile $best+compile \
          $best+dfa0.1+compile $best+licfg0.2-0.8+compile $best+dfa0.1+licfg0.2-0.8+compile; }
e13() { for s in $SIZES; do $PY analysis/traffic_ledger.py --size $s || return 1; done; $PY analysis/roofline_whole_forward.py; }
e14() { for m in "sana_600m 1024" "sd35_medium 1024" "flux_schnell 1024"; do set -- $m; $PY model_smoke.py --model $1 --size $2 || return 1; done; }
e15() { tar czf results_eval2.tgz results logs prompts/embeds_calib.pt; }

for s in $STAGES; do
  fn=$(echo $s | tr 'E' 'e')
  run $s $fn
done
[ $DRY -eq 0 ] && $PY progress_report.py
echo "finished. Failed stages: $(ls results/.fail 2>/dev/null | tr '\n' ' ')"
