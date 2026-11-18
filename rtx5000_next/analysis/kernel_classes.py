import re

CATEGORIES = ["fp4_gemm", "bf16_gemm", "attention", "our_quant", "inductor", "layernorm", "copy", "elementwise", "reduce", "other"]
QUANT_PATH = {"our_quant", "inductor", "copy", "elementwise", "reduce", "layernorm", "other"}

_RULES = [
    ("our_quant", r"quant2lvl|ln_mod_quant|amax_quant|fused_nvfp4|_fused_.*nvfp4|fill_swizzle_padding"),
    ("fp4_gemm", r"block_scaled|ue4m3xe2m1|e2m1|nvfp4|mxf4"),
    ("attention", r"flash|fmha|attention|sdpa|mem_eff|softmax"),
    ("bf16_gemm", r"gemm|cutlass|xmma|cublas|nvjet|sm80_|sm90_|sm100_|sm120_|splitK|implicit_convolve"),
    ("inductor", r"^triton_"),
    ("layernorm", r"layer_norm|layernorm|welford|group_norm"),
    ("copy", r"copy_kernel|direct_copy|_copy|CatArray"),
    ("elementwise", r"elementwise|BinaryFunctor|UnaryFunctor|CUDAFunctor|index_kernel|clamp|where"),
    ("reduce", r"reduce_kernel|ReduceOp|reduce"),
]


def classify(name):
    for cat, rx in _RULES:
        if re.search(rx, name, re.I):
            return cat
    return "other"
