"""Variant registry: one string describes a complete inference setup, and build() turns it into a ready pipeline.

Why: the speed benchmarks, the real-prompt quality runs, the end-to-end timing and the calibration scripts all need
to construct "PixArt with this quantization, this attention and this sampler". Writing it once keeps them consistent.

Name grammar: <base>[+modifier]...        e.g.  module+static+concat+dfa0.1+licfg0.2-0.8
  base        bf16 | fp8 | stock | stock_tk | w4a16 | module
                stock     torchao NVFP4 (use_triton_kernel=False), the baseline measured so far
                stock_tk  torchao NVFP4 with its own fused Triton quantize kernel (needs the mslk package)
                w4a16     real torchao NVFP4 weights, activations left in bf16 (act_quant_kwargs=None)
                module    our FusedNVFP4Linear stack (fused_linear.py)
  modifiers   static      static activation scales from results/static_scales_<size>.json (calibrate_static_scales.py --size <size>)
              noshare     do not share the Q/K/V quantize          concat   one GEMM for Q/K/V and for K/V
              selective   keep cross-attn K/V, caption, adaln and proj_out in bf16
              be=<name>   quantize backend: triton (default) | cuda_swizzled | cuda_row | torchref
              trim        cross-attention keeps only the real text tokens (exact)
              kvds<f>     self-attention K/V pooling factor        flash | cudnn | efficient | sage2 | sage3   self-attention backend
              xcudnn | xefficient   cross-attention backend (the masked cross-attention otherwise falls to the efficient kernel)
              dfa<delta>  DiTFastAttn plan from results/dfa_plans/pixart_<size>_d<delta>.json (ditfast/plan_search.py)
              licfg<lo>-<hi>  guidance only for denoise steps with lo <= i/steps < hi
              compile     torch.compile on every transformer block
"""
import dataclasses
import pathlib
import re

import torch

import attention_patch as ap
import cfg_interval
import fused_linear as fl
from common import RESULTS

REPO = {256: "PixArt-alpha/PixArt-Sigma-XL-2-256x256", 512: "PixArt-alpha/PixArt-Sigma-XL-2-512-MS",
        1024: "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS"}
STATIC_PATH = RESULTS / "static_scales.json"


def static_path(size):
    """results/static_scales_<size>.json written by calibrate_static_scales.py, else the generic file."""
    p = RESULTS / f"static_scales_{size}.json"
    return p if p.exists() else STATIC_PATH


def plan_path(size, delta):
    return RESULTS / "dfa_plans" / f"pixart_{size}_d{delta}.json"


@dataclasses.dataclass
class Variant:
    name: str
    base: str = "bf16"
    backend: str = "triton"
    static: bool = False
    share: bool = True
    concat: bool = False
    selective: bool = False
    trim: bool = False
    kv_ds: int = 1
    attn: str = "default"
    attn_cross: str = "default"
    dfa_delta: float = None
    interval: tuple = None
    compile: bool = False


def parse(name):
    toks = name.split("+")
    v = Variant(name=name, base=toks[0])
    assert v.base in ("bf16", "fp8", "stock", "stock_tk", "w4a16", "module"), f"unknown base in {name}"
    for t in toks[1:]:
        if t == "static":
            v.static = True
        elif t == "noshare":
            v.share = False
        elif t == "concat":
            v.concat = True
        elif t == "selective":
            v.selective = True
        elif t == "trim":
            v.trim = True
        elif t == "compile":
            v.compile = True
        elif t in ("flash", "cudnn", "efficient", "sage2", "sage3", "math"):
            v.attn = t
        elif t in ("xcudnn", "xefficient"):
            v.attn_cross = t[1:]
        elif t.startswith("be="):
            v.backend = t[3:]
        elif m := re.fullmatch(r"kvds(\d+)", t):
            v.kv_ds = int(m.group(1))
        elif m := re.fullmatch(r"dfa([0-9.]+)", t):
            v.dfa_delta = float(m.group(1))
        elif m := re.fullmatch(r"licfg([0-9.]+)-([0-9.]+)", t):
            v.interval = (float(m.group(1)), float(m.group(2)))
        else:
            raise ValueError(f"unknown modifier '{t}' in '{name}'")
    return v


def _load_pipe(size, base):
    """Pipeline with the transformer in the format `base` asks for (everything else in bf16 on the GPU)."""
    from diffusers import PixArtSigmaPipeline, PixArtTransformer2DModel, TorchAoConfig
    kw = {}
    if base in ("stock", "stock_tk", "w4a16"):
        from torchao.prototype.mx_formats.inference_workflow import NVFP4DynamicActivationNVFP4WeightConfig
        kw["quantization_config"] = TorchAoConfig(NVFP4DynamicActivationNVFP4WeightConfig(
            use_dynamic_per_tensor_scale=True, use_triton_kernel=(base == "stock_tk")))
    elif base == "fp8":
        from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, PerRow
        kw["quantization_config"] = TorchAoConfig(Float8DynamicActivationFloat8WeightConfig(granularity=PerRow()))
    tr = PixArtTransformer2DModel.from_pretrained(REPO[size], subfolder="transformer", torch_dtype=torch.bfloat16, **kw)
    return PixArtSigmaPipeline.from_pretrained(REPO[1024], transformer=tr, text_encoder=None, tokenizer=None,
                                               torch_dtype=torch.bfloat16).to("cuda")


def weights_only(transformer):
    """Turn a torchao NVFP4 transformer into weights-only: activations stay bf16 (nvfp4_linear's weight-only branch)."""
    n = 0
    for m in transformer.modules():
        if isinstance(m, torch.nn.Linear) and getattr(m.weight, "act_quant_kwargs", None) is not None:
            m.weight.act_quant_kwargs = None
            n += 1
    assert n > 0, "no NVFP4 linear found: the model is not torchao-quantized"
    return n


def load_static_amax(size, margin=1.0):
    import json
    p = static_path(size)
    if not p.exists():
        raise SystemExit(f"{p} missing: run calibrate_static_scales.py --size {size} first")
    return {k: float(a) * margin for k, a in json.loads(p.read_text())["amax"].items()}


def pad16(n):
    return -(-int(n) // 16) * 16


class Built:
    def __init__(self, v, pipe, cfg, size):
        self.v, self.pipe, self.cfg, self.size = v, pipe, cfg, size
        self.report = {}

    def prepare(self, items):
        """Per-run settings that depend on the prompts: the trimmed text length."""
        if self.v.trim:
            valid = max(int(it["pm"].sum()) for it in items)
            self.cfg.text_len = min(pad16(valid), items[0]["pm"].shape[-1])

    def run(self, items, steps=14, seeds=None, guidance=4.5, output="np", timing=False):
        self.prepare(items)
        return cfg_interval.sample(self.pipe, items, self.size, steps=steps, seeds=seeds, guidance=guidance,
                                   interval=self.v.interval, dfa_active=self.v.dfa_delta is not None, output=output, timing=timing)


def make_forward(built, items, size, batch=1, t=500):
    """One transformer forward (one denoise step with CFG: transformer batch 2*batch) on fixed real-prompt inputs.

    Used by step_target.py and timing/profile_stages.py so every config is timed on the same inputs. Returns a closure."""
    tr = built.pipe.transformer
    its = [items[i % len(items)] for i in range(batch)]
    built.prepare(its)
    g = torch.Generator("cuda").manual_seed(0)
    x = torch.randn(batch, 4, size // 8, size // 8, generator=g, device="cuda", dtype=torch.bfloat16).repeat(2, 1, 1, 1)
    enc = torch.cat([it["ne"] for it in its] + [it["pe"] for it in its]).to("cuda", torch.bfloat16)
    mask = torch.cat([it["nm"] for it in its] + [it["pm"] for it in its]).to("cuda")
    tt = torch.full((2 * batch,), t, device="cuda")
    added = {"resolution": None, "aspect_ratio": None}

    def fwd():
        return tr(x, encoder_hidden_states=enc, encoder_attention_mask=mask, timestep=tt, added_cond_kwargs=added, return_dict=False)[0]

    return fwd


def build(name, size):
    v = parse(name)
    pipe = _load_pipe(size, "bf16" if v.base == "module" else v.base)
    tr = pipe.transformer
    report = {}
    if v.base == "w4a16":
        report["weights_only_layers"] = weights_only(tr)
    if v.base == "module":
        static = load_static_amax(size) if v.static else None
        report.update(fl.convert_pixart(tr, backend=v.backend, static_amax=static, shared_qkv=v.share, concat=v.concat,
                                        selective=v.selective))
    cfg = ap.AttnConfig(backend=v.attn, backend_cross=v.attn_cross, kv_ds=v.kv_ds)
    use_dfa = v.dfa_delta is not None
    if v.base == "module" or v.trim or v.kv_ds > 1 or v.attn != "default" or v.attn_cross != "default" or use_dfa:
        ap.install_attention(tr, cfg, use_dfa=use_dfa)
    if use_dfa:
        import ditfast.attention as dfa
        import ditfast.plan_search as ps
        p = plan_path(size, v.dfa_delta)
        if not p.exists():
            raise SystemExit(f"{p} missing: run ditfast/make_plans.py for size {size} first")
        dfa.STATE.cfg = cfg
        report["dfa_plan"] = {k: ps.load_plan(p)[k] for k in ("counts", "modeled_attention_cost")}
    if v.compile:
        for b in tr.transformer_blocks:
            b.compile()
    built = Built(v, pipe, cfg, size)
    built.report = report
    return built
