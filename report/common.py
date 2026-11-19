"""Shared helpers for the report scripts: load facts.json once, fixed labels, fixed colours.

Every number in figures and text is read through these functions so that one quantity has one value everywhere.
"""
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
BUILD = HERE / "build"
FIG = BUILD / "fig"
FIG.mkdir(parents=True, exist_ok=True)
SIZES = (256, 512, 1024)
TOKENS = {256: 256, 512: 1024, 1024: 4096}   # transformer tokens for one image: (size/16)^2
BEST = "module+be=cuda_swizzled+static+concat"

_F = json.loads((BUILD / "facts.json").read_text())


def facts():
    return _F


# (label shown in text and figures, key in step facts)
STEP_ROWS = [
    ("bf16, compiled", "bf16_compile"),
    ("fp8, compiled", "fp8_compile"),
    ("stock NVFP4, compiled", "stock_compile"),
    ("module stack (our CUDA quantize kernel)", "module_cuda"),
    ("+ static scale + concat Q/K/V", "module_best"),
    ("+ text trim + cuDNN cross-attention", "module_best_xcudnn"),
    ("+ selective quantization", "module_best_selective"),
]
SHORT = {"bf16_compile": "bf16 compiled", "fp8_compile": "fp8 compiled", "stock_compile": "stock NVFP4 compiled",
         "module_cuda": "module stack", "module_best": "+ static + concat", "module_best_xcudnn": "+ trim + cuDNN cross",
         "module_best_selective": "+ selective quant"}
COLORS = {"bf16_compile": "#8c8c8c", "fp8_compile": "#5b8def", "stock_compile": "#e8903a", "module_cuda": "#7bc67b",
          "module_best": "#4fae6a", "module_best_xcudnn": "#2e8b57", "module_best_selective": "#1b5e3a"}
CLASS_COLORS = {"attention": "#d1495b", "fp4_gemm": "#2e86ab", "bf16_gemm": "#7a9cc6", "our_quant": "#59a14f", "inductor": "#edae49",
                "elementwise": "#b07aa1", "copy": "#9c755f", "reduce": "#bab0ac", "layernorm": "#76b7b2", "other": "#dddddd"}


def chain(key):
    """nsys kernel time per forward that is neither attention nor GEMM: the quantize step plus the glue around it."""
    return sum(v for k, v in _F["nsys"][key]["class_ms"].items() if k not in ("attention", "fp4_gemm", "bf16_gemm"))


def step(label, size, field="graph"):
    k = f"{label}|{size}"
    return _F["step"][k][field] if k in _F["step"] else None


def step_runs(label, size):
    return _F["step"][f"{label}|{size}"]["runs"]


def speedup(label, size, base="bf16_compile"):
    a, b = step(base, size), step(label, size)
    return None if a is None or b is None else a / b


QV = {"bf16": "bf16", "stock": "stock", "w4a16": "w4a16", "module": "module+be=cuda_swizzled",
      "best": BEST, "trim": BEST + "+trim+xcudnn", "selective": BEST + "+selective+trim+xcudnn"}


def quality(size, variant):
    """Headline quality row; accepts the short names of QV or the full variant string."""
    return _F["quality_final"][str(size)][QV.get(variant, variant)]


def ext(size, variant):
    """Longer variant list (first E11 run), keyed by the variant string used in run_quality."""
    return _F["quality_ext"][str(size)][variant]
