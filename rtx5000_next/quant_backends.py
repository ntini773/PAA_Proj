"""Activation-quantize backends spliced into torchao's NVFP4Tensor.to_nvfp4.

backend  what it is
triton         existing Triton kernel (fused_adaln_nvfp4.fused_nvfp4_quantize), torchao swizzle afterwards
triton_op      same kernel wrapped as a torch custom op (torch.compile safe)
cuda_row       CUDA kernel, logical scales, torchao swizzle afterwards
cuda_swizzled  CUDA kernel that also writes the swizzled scales
free           no quantize work at all (cached zero buffers): lower bound for any fusion
"""
import torch

import cuda_ext

_ORIG = {}
_OPS = {}


def _ceil(a, b):
    return -(-a // b)


def _triton_impl(x, pts):
    from fused_adaln_nvfp4 import fused_nvfp4_quantize
    scales, packed = fused_nvfp4_quantize(x, pts)
    return packed, scales


def _cuda_row_impl(x, pts):
    packed, scales = cuda_ext.load_ext("quant2lvl_row").quant2lvl_row(x, pts.reshape(()).float())
    return packed, scales


def _cuda_swizzled_impl(x, pts):
    packed, scales = cuda_ext.load_ext("quant2lvl_swizzled").quant2lvl_swizzled(x, pts.reshape(()).float())
    return packed, scales


IMPLS = {
    "triton": (_triton_impl, False),
    "triton_op": (_triton_impl, False),
    "cuda_row": (_cuda_row_impl, False),
    "cuda_swizzled": (_cuda_swizzled_impl, True),
}


def _as_op(name, impl, swizzled):
    if name in _OPS:
        return _OPS[name]

    @torch.library.custom_op(f"nvfp4fusion::{name}", mutates_args=())
    def op(x: torch.Tensor, pts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return impl(x, pts)

    @op.register_fake
    def _(x, pts):
        M, K = x.shape
        packed = x.new_empty((M, K // 2), dtype=torch.uint8)
        n = _ceil(M, 128) * 128 * _ceil(K // 16, 4) * 4 if swizzled else M * (K // 16)
        scales = x.new_empty((n,) if swizzled else (M, K // 16), dtype=torch.float8_e4m3fn)
        return packed, scales

    _OPS[name] = op
    return op


_FREE = {}


def _free_buffers(M, K, swizzled, dev):
    key = (M, K, swizzled, str(dev))
    if key not in _FREE:
        packed = torch.zeros((M, K // 2), dtype=torch.uint8, device=dev)
        shape = (_ceil(M, 128) * 128 * _ceil(K // 16, 4) * 4,) if swizzled else (M, K // 16)
        _FREE[key] = (packed, torch.ones(shape, dtype=torch.float32, device=dev).to(torch.float8_e4m3fn))
    return _FREE[key]


def install(backend, ok_k=(1152,), use_op=False, count=True):
    import torchao.prototype.mx_formats.nvfp4_tensor as m
    from torchao.prototype.mx_formats.nvfp4_tensor import hp_data_dims_to_swizzled_scale_dims_nvfp4
    from torchao.prototype.mx_formats.utils import to_blocked

    _ORIG.setdefault("to_nvfp4", m.NVFP4Tensor.to_nvfp4)
    orig = _ORIG["to_nvfp4"]
    ok_k = set(ok_k)
    counter = {"n": 0, "fallback": 0}

    if backend == "free":
        fn, native_swz = None, None
    else:
        impl, native_swz = IMPLS[backend]
        fn = _as_op(backend, impl, native_swz) if (use_op or backend == "triton_op") else impl

    def patched(data_hp, block_size=16, per_tensor_scale=None, act_per_tensor_scale=None,
                is_swizzled_scales=False, use_triton_kernel=False, act_quant_kwargs=None):
        ok = (data_hp.dtype == torch.bfloat16 and data_hp.dim() == 2 and data_hp.is_contiguous()
              and data_hp.shape[-1] in ok_k and data_hp.shape[-1] % 16 == 0
              and per_tensor_scale is not None and per_tensor_scale.numel() == 1
              and not (native_swz and not is_swizzled_scales))
        if not ok:
            if count:
                counter["fallback"] += 1
            return orig(data_hp, block_size, per_tensor_scale, act_per_tensor_scale,
                        is_swizzled_scales, use_triton_kernel, act_quant_kwargs)
        M, K = data_hp.shape
        if backend == "free":
            codes, scales = _free_buffers(M, K, is_swizzled_scales, data_hp.device)
            already_swizzled = is_swizzled_scales
        else:
            codes, scales = fn(data_hp, per_tensor_scale)
            already_swizzled = native_swz
        if is_swizzled_scales:
            if not already_swizzled:
                scales = to_blocked(scales.view(M, K // block_size)).flatten()
            scale_m, scale_k = hp_data_dims_to_swizzled_scale_dims_nvfp4(M, K)
        else:
            scale_m, scale_k = M, K // block_size
        scales = scales.view(scale_m, scale_k)
        if count:
            counter["n"] += 1
        return m.NVFP4Tensor(codes, scales, block_size, data_hp.dtype, per_tensor_scale,
                             act_per_tensor_scale, is_swizzled_scales, use_triton_kernel, act_quant_kwargs)

    m.NVFP4Tensor.to_nvfp4 = staticmethod(patched)
    return counter


def uninstall():
    import torchao.prototype.mx_formats.nvfp4_tensor as m
    if "to_nvfp4" in _ORIG:
        m.NVFP4Tensor.to_nvfp4 = staticmethod(_ORIG["to_nvfp4"])


def wrap_nvtx(label="quantize"):
    import torchao.prototype.mx_formats.nvfp4_tensor as m
    inner = m.NVFP4Tensor.to_nvfp4

    def wrapped(*args, **kwargs):
        torch.cuda.nvtx.range_push(label)
        try:
            return inner(*args, **kwargs)
        finally:
            torch.cuda.nvtx.range_pop()

    m.NVFP4Tensor.to_nvfp4 = staticmethod(wrapped)


def backend_fn(backend):
    impl, swizzled = IMPLS[backend]
    return impl, swizzled
