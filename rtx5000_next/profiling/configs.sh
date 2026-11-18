#!/usr/bin/env bash
# Config name -> step_target.py arguments. Sourced by the other scripts.
ALL_CONFIGS="bf16 fp8 stock stock_compile fused_triton_narrow fused_triton fused_triton_op_compile fused_cuda_row fused_cuda_swizzled fused_cuda_swizzled_compile free_bound"
NCU_CONFIGS="stock fused_triton fused_cuda_row fused_cuda_swizzled"

cfg_args() {
  case "$1" in
    bf16) echo "--mode bf16";;
    fp8) echo "--mode fp8";;
    stock) echo "--mode stock";;
    stock_compile) echo "--mode stock --compile";;
    fused_triton_narrow) echo "--mode fused --backend triton --width narrow";;
    fused_triton) echo "--mode fused --backend triton --width wide";;
    fused_triton_op_compile) echo "--mode fused --backend triton_op --width wide --compile";;
    fused_cuda_row) echo "--mode fused --backend cuda_row --width wide";;
    fused_cuda_swizzled) echo "--mode fused --backend cuda_swizzled --width wide";;
    fused_cuda_swizzled_compile) echo "--mode fused --backend cuda_swizzled --width wide --compile";;
    free_bound) echo "--mode fused --backend free --width wide";;
    *) echo "unknown config $1" >&2; return 1;;
  esac
}

PY=${PY:-python}
MODEL=${MODEL:-pixart_sigma}
SIZES=${SIZES:-"256 512 1024"}
CONFIGS=${CONFIGS:-$ALL_CONFIGS}
