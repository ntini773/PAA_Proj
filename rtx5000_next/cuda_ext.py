import functools
import os

from common import ROOT

CUDA_FLAGS = ["-O3", "-std=c++17", "--fmad=false", "-lineinfo"]


@functools.lru_cache(maxsize=None)
def load_ext(name):
    from torch.utils.cpp_extension import load
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
    build = ROOT / "build" / name
    build.mkdir(parents=True, exist_ok=True)
    return load(name=name, sources=[str(ROOT / "cuda_kernels" / f"{name}.cu")],
                extra_cflags=["-O3", "-std=c++17"], extra_cuda_cflags=CUDA_FLAGS,
                build_directory=str(build), verbose=False)
