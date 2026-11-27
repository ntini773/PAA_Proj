#!/usr/bin/env bash
# Config name -> step_target.py arguments. Sourced by the other scripts.
ALL_CONFIGS="bf16 fp8 stock stock_compile fused_triton_narrow fused_triton fused_triton_op_compile fused_cuda_row fused_cuda_swizzled fused_cuda_swizzled_compile free_bound"
NCU_CONFIGS="stock fused_triton fused_cuda_row fused_cuda_swizzled"
# eval2: fair baselines first, then the module stack step by step, then attention options on the best stack
EVAL2_BASELINES="bf16_compile fp8_compile stock_compile stock_tk stock_tk_compile"
EVAL2_MODULE="module_triton module_triton_compile module_noshare_compile module_static_compile module_concat_compile module_selective_compile module_cuda_compile module_best_compile"
EVAL2_ATTENTION="bf16_trim_compile module_best_trim module_best_xcudnn module_best_sel_xcudnn module_best_cudnn module_best_sage2 module_best_kvds2"

# eval3: module stack WITHOUT static scale (static is dropped from the method for image quality)
EVAL3_LEVERS="mod_base_compile mod_concat_compile mod_noshare_compile mod_sel_compile mod_trim_compile mod_trim_xcudnn_compile mod_trim_xcudnn_sage2_compile bf16_trim_xcudnn_compile bf16_sage2_compile bf16_trim_compile"

cfg_args() {
  case "$1" in
    mod_base_compile) echo "--variant module+be=cuda_swizzled+compile";;
    mod_concat_compile) echo "--variant module+be=cuda_swizzled+concat+compile";;
    mod_noshare_compile) echo "--variant module+be=cuda_swizzled+noshare+compile";;
    mod_sel_compile) echo "--variant module+be=cuda_swizzled+concat+selective+compile";;
    mod_trim_compile) echo "--variant module+be=cuda_swizzled+concat+trim+compile";;
    mod_trim_xcudnn_compile) echo "--variant module+be=cuda_swizzled+concat+trim+xcudnn+compile";;
    mod_trim_xcudnn_sage2_compile) echo "--variant module+be=cuda_swizzled+concat+trim+xcudnn+sage2+compile";;
    bf16_trim_xcudnn_compile) echo "--variant bf16+trim+xcudnn+compile";;
    bf16_sage2_compile) echo "--variant bf16+sage2+compile";;
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
    # eval2 configs: a variants.py string says everything (see variants.py for the grammar)
    bf16_compile) echo "--variant bf16+compile";;
    fp8_compile) echo "--variant fp8+compile";;
    stock_tk) echo "--variant stock_tk";;
    stock_tk_compile) echo "--variant stock_tk+compile";;
    module_triton) echo "--variant module+be=triton";;
    module_triton_compile) echo "--variant module+be=triton+compile";;
    module_noshare_compile) echo "--variant module+be=triton+noshare+compile";;
    module_static_compile) echo "--variant module+be=triton+static+compile";;
    module_concat_compile) echo "--variant module+be=triton+concat+compile";;
    module_selective_compile) echo "--variant module+be=triton+selective+compile";;
    module_cuda_compile) echo "--variant module+be=cuda_swizzled+compile";;
    module_best_compile) echo "--variant module+be=cuda_swizzled+static+concat+compile";;
    module_best_trim) echo "--variant module+be=cuda_swizzled+static+concat+trim+compile";;
    module_best_sel_xcudnn) echo "--variant module+be=cuda_swizzled+static+concat+selective+trim+xcudnn+compile";;
    module_best_xcudnn) echo "--variant module+be=cuda_swizzled+static+concat+trim+xcudnn+compile";;
    module_best_cudnn) echo "--variant module+be=cuda_swizzled+static+concat+trim+cudnn+xcudnn+compile";;
    module_best_sage2) echo "--variant module+be=cuda_swizzled+static+concat+trim+sage2+compile";;
    module_best_kvds2) echo "--variant module+be=cuda_swizzled+static+concat+trim+kvds2+compile";;
    bf16_trim_compile) echo "--variant bf16+trim+compile";;
    *) echo "unknown config $1" >&2; return 1;;
  esac
}

PY=${PY:-python}
MODEL=${MODEL:-pixart_sigma}
SIZES=${SIZES:-"256 512 1024"}
CONFIGS=${CONFIGS:-$ALL_CONFIGS}
