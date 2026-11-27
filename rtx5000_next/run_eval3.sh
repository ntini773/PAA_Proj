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
mkdir -p results/eval3/.done results/eval3/.fail logs/eval3
source profiling/configs.sh

ONLY=""; FROM=""; DRY=0; LIST=0; SET=""
while [ $# -gt 0 ]; do
  case $1 in
    --only) ONLY=$2; shift 2;; --from) FROM=$2; shift 2;; --dry-run) DRY=1; shift;; --list) LIST=1; shift;;
    --setup) SET="W100 W101 W102 W103"; shift;; --wave1) SET="W100 W101 W102 W103 W104 W105 W106"; shift;;
    *) echo "unknown argument $1"; exit 2;;
  esac
done
STAGES="W100 W101 W102 W103 W104 W105 W106"
desc() { case $1 in
  W100) echo "preflight: GPU, packages, tools, disk, HF cache (models, metric weights)  ~1 min";;
  W101) echo "downloads: metric weights (CLIP, PickScore, LPIPS, FID); PixArt sizes; with WITH_MODELS=1 SD3.5-M and FLUX schnell incl. text encoders  ~5-40 min, network";;
  W102) echo "variant-name validation for every name used in wave 1  ~5 s";;
  W103) echo "T5 embeddings for calibration and showcase prompts (needs the GPU for about 2 min)  ~5 min";;
  W104) echo "roofs: M/N/K sweep of FP4 and bf16 GEMM, flash attention per head dim, bandwidth (idle GPU)  ~5 min";;
  W105) echo "noise floor + controls: bf16 eager vs compiled vs other seed vs fp8 vs stock, $NPROMPTS prompts x $SEEDS seeds at ${SIZE_Q}px  ~25-40 min";;
  W106) echo "baseline step times 512 and 1024, 3 repeats: bf16c, fp8c, stockc, stock_tk c  ~25 min";;
esac; }
if [ $LIST -eq 1 ]; then for s in $STAGES; do echo "$s  $(desc $s)"; done; exit 0; fi

go=1; [ -n "$FROM" ] && go=0
run() {
  local id=$1; shift
  [ -n "$FROM" ] && [ "$id" = "$FROM" ] && go=1
  [ $go -eq 0 ] && return 0
  [ -n "$ONLY" ] && [ "$ONLY" != "$id" ] && return 0
  if [ -n "$SET" ] && ! echo " $SET " | grep -q " $id "; then return 0; fi
  if [ -f results/eval3/.done/$id ] && [ -z "$FORCE" ] && [ -z "$ONLY" ]; then echo "[$id] already done (FORCE=1 to rerun)"; return 0; fi
  echo "[$(date +%H:%M:%S)] $id: $(desc $id)"
  if [ $DRY -eq 1 ]; then declare -f "$1" | sed '1,2d;$d' | sed 's/^/    /'; return 0; fi
  local log=logs/eval3/$id.log
  if "$@" > $log 2>&1; then touch results/eval3/.done/$id; rm -f results/eval3/.fail/$id
  else touch results/eval3/.fail/$id; echo "  FAILED, see $log (last lines:)"; tail -n 6 $log | sed 's/^/    /'; fi
}

BEST="module+be=cuda_swizzled+concat"   # eval3 method: no static scale (image quality)
w100() { $PY eval3/preflight.py --models pixart_sigma:256,512,1024 sd35_medium:1024 flux_schnell:1024; }
w101() { $PY eval3/prefetch.py --metrics --models pixart_sigma || return 1
         [ -n "$WITH_MODELS" ] && { $PY eval3/prefetch.py --models sd35_medium flux_schnell --with-text-encoder; }; return 0; }
w102() { $PY eval3/validate_variants.py bf16 bf16+compile bf16@seedshift fp8+compile stock stock+compile stock_tk+compile \
           module+be=cuda_swizzled+compile $BEST+compile $BEST+trim+xcudnn+compile; }
w103() { $PY prompts/make_prompt_embeds.py --prompts prompts/prompts_calib.json --out prompts/embeds_calib.pt &&
         $PY prompts/make_prompt_embeds.py --prompts prompts/prompts_showcase.json --out prompts/embeds_showcase.pt; }
w104() { $PY bench/peaks_sweep.py; }
w105() { $PY eval3/quality_grid.py --size $SIZE_Q --embeds prompts/embeds_showcase.pt --n-prompts $NPROMPTS --seeds $SEEDS --steps $STEPS \
           --tag noise_floor_$SIZE_Q; }
w106() { local cfgs="bf16_compile fp8_compile stock_compile stock_tk_compile"
         for rep in 1 2 3; do CONFIGS="$cfgs" SIZES="512 1024" ITERS=30 bash profiling/run_step_matrix.sh || return 1; done; }

for s in $STAGES; do run $s $(echo $s | tr 'W' 'w'); done
echo "finished. Failed stages: $(ls results/eval3/.fail 2>/dev/null | tr '\n' ' ')"
