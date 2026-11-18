#!/usr/bin/env bash
# bash run_all.sh [stage ...]    stages: hf kernels time prof embeds precision plots models   (default: all except models, in this order)
# Each test writes results/.done/<ID> or results/.fail/<ID>; PROGRESS.md is refreshed after every test.
# Env: PY=python-to-use  MODEL=pixart_sigma  SIZES="256 512 1024"  CONFIGS="..."  ITERS=30
cd "$(dirname "$0")"
export PY=${PY:-python}
export MODEL=${MODEL:-pixart_sigma}
mkdir -p results/.done results/.fail logs

t() {  # t ID command...
  local id=$1; shift
  echo "[$(date +%H:%M:%S)] $id: $*"
  if "$@" > logs/$id.log 2>&1; then touch results/.done/$id; rm -f results/.fail/$id
  else touch results/.fail/$id; echo "  FAILED, see logs/$id.log"; fi
  $PY progress_report.py > /dev/null 2>&1
}

stage_hf() {
  t N01 $PY hf/check_model_repos.py
  t N02 $PY hf/probe_model_arch.py --size 512
}
stage_kernels() {
  t K01 $PY bench/cuda_vs_triton.py --only cuda_row
  t K02 $PY bench/cuda_vs_triton.py --only cuda_swizzled amax_swizzled
  t K03 $PY bench/cuda_vs_triton.py --only ln_mod_quant
}
stage_time() { t S01 bash profiling/run_step_matrix.sh; }
stage_prof() {
  t P01 bash profiling/nsys_step.sh
  t P02 bash profiling/ncu_layer.sh
  t P03 $PY analysis/analyze_nsys.py
  t P04 $PY analysis/analyze_ncu.py
}
stage_embeds() { t T01 $PY prompts/make_prompt_embeds.py; }
stage_precision() {
  local e=${EMBEDS:-prompts/embeds_pixart.pt}
  [ -f "$e" ] || { echo "no $e: run 'bash run_all.sh embeds' first, or the precision tests use synthetic embeddings"; e=; }
  local ea=${e:+--embeds $e}
  t Q01 $PY precision/precision_ablation.py --model $MODEL --size 512 --validate-stock $ea
  t Q02 $PY precision/precision_ablation.py --model $MODEL --size 1024 $ea
  t Q03 $PY precision/precision_images_pixart.py --size 1024 $ea
}
stage_models() {
  t M01 bash bench/new_model_suite.sh sana_600m 1024
  t M02 bash bench/new_model_suite.sh sd35_medium 1024
  t M03 bash bench/new_model_suite.sh flux_schnell 1024
}
stage_plots() { t V01 $PY viz/plot_all.py; }

stages=${*:-hf kernels time prof embeds precision plots}
for s in $stages; do "stage_$s"; done
$PY progress_report.py
