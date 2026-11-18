"""Transformer-only runners (no text encoder, no VAE) so every model needs only its transformer download.
PixArt is the validated path; the others follow the diffusers forward signatures and need model_smoke.py first.
"""
import collections
import json

import torch
import torch.nn as nn

from common import ROOT

CATALOG = json.loads((ROOT / "hf" / "catalog.json").read_text())


class Runner:
    def __init__(self, name, transformer, blocks, inputs, make_t, tokens, rows):
        self.name, self.transformer, self.blocks = name, transformer, blocks
        self.inputs, self.make_t, self.tokens, self.rows = inputs, make_t, tokens, rows

    def forward(self, t=None):
        kw = dict(self.inputs)
        if t is not None:
            kw["timestep"] = self.make_t(t)
        return self.transformer(**kw, return_dict=False)[0]


def linear_widths(transformer):
    return collections.Counter(m.in_features for m in transformer.modules() if isinstance(m, nn.Linear))


def _load(model, size, quant):
    import diffusers
    from diffusers import TorchAoConfig
    spec = CATALOG[model]
    repo = spec["repos"][str(size)]
    cls = getattr(diffusers, next(c for c in spec["classes"] if hasattr(diffusers, c)))
    kw = {}
    if quant == "nvfp4":
        from torchao.prototype.mx_formats.inference_workflow import NVFP4DynamicActivationNVFP4WeightConfig
        kw["quantization_config"] = TorchAoConfig(NVFP4DynamicActivationNVFP4WeightConfig(
            use_dynamic_per_tensor_scale=True, use_triton_kernel=False))
    elif quant == "fp8":
        from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, PerRow
        kw["quantization_config"] = TorchAoConfig(Float8DynamicActivationFloat8WeightConfig(granularity=PerRow()))
    return cls.from_pretrained(repo, subfolder="transformer", torch_dtype=torch.bfloat16, **kw).to("cuda").eval()


def _inputs(model, tr, size, B, g):
    c = tr.config
    dev, bf = "cuda", torch.bfloat16
    rn = lambda *s: torch.randn(*s, device=dev, dtype=bf, generator=g)
    f = CATALOG[model]["vae_factor"]
    lat = size // f
    patch = getattr(c, "patch_size", 1) or 1
    full_t = lambda v: (lambda t: torch.full((B,), float(v if t is None else t), device=dev))

    if model == "pixart_sigma":
        inp = dict(hidden_states=rn(B, c.in_channels, lat, lat),
                   encoder_hidden_states=rn(B, 300, c.caption_channels) * 0.5,
                   timestep=full_t(500)(None),
                   encoder_attention_mask=torch.ones(B, 300, dtype=torch.long, device=dev),
                   added_cond_kwargs={"resolution": None, "aspect_ratio": None})
        return inp, full_t(500), (lat // patch) ** 2
    if model == "dit_xl2":
        labels = torch.tensor([207] * (B // 2) + [1000] * (B - B // 2), device=dev)
        mk = lambda t: torch.full((B,), int(500 if t is None else t), device=dev, dtype=torch.long)
        return dict(hidden_states=rn(B, c.in_channels, lat, lat), timestep=mk(None), class_labels=labels), mk, (lat // patch) ** 2
    if model.startswith("sd35"):
        inp = dict(hidden_states=rn(B, c.in_channels, lat, lat),
                   encoder_hidden_states=rn(B, 333, c.joint_attention_dim) * 0.5,
                   pooled_projections=rn(B, c.pooled_projection_dim) * 0.5,
                   timestep=full_t(500)(None))
        return inp, full_t(500), (lat // patch) ** 2
    if model.startswith("sana"):
        inp = dict(hidden_states=rn(B, c.in_channels, lat, lat),
                   encoder_hidden_states=rn(B, 300, c.caption_channels) * 0.5,
                   timestep=full_t(500)(None),
                   encoder_attention_mask=torch.ones(B, 300, dtype=torch.long, device=dev))
        return inp, full_t(500), (lat // patch) ** 2
    if model.startswith("flux"):
        h = w = size // 16
        ids = torch.zeros(h, w, 3)
        ids[..., 1] = torch.arange(h)[:, None]
        ids[..., 2] = torch.arange(w)[None, :]
        inp = dict(hidden_states=rn(B, h * w, c.in_channels),
                   encoder_hidden_states=rn(B, 512, c.joint_attention_dim) * 0.5,
                   pooled_projections=rn(B, c.pooled_projection_dim) * 0.5,
                   timestep=torch.full((B,), 0.5, device=dev),
                   img_ids=ids.reshape(-1, 3).to(dev, bf), txt_ids=torch.zeros(512, 3, device=dev, dtype=bf))
        if getattr(c, "guidance_embeds", False):
            inp["guidance"] = torch.full((B,), 3.5, device=dev)
        mk = lambda t: torch.full((B,), float(500 if t is None else t) / 1000, device=dev)
        return inp, mk, h * w
    raise KeyError(model)


def build(model, size, batch=1, quant="bf16"):
    tr = _load(model, size, quant)
    B = CATALOG[model]["cfg_batch"] * batch
    g = torch.Generator("cuda").manual_seed(0)
    inputs, make_t, tokens = _inputs(model, tr, size, B, g)
    blocks = list(getattr(tr, "transformer_blocks", [])) + list(getattr(tr, "single_transformer_blocks", []))
    return Runner(model, tr, blocks, inputs, make_t, tokens, B * tokens)
