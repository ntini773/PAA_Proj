"""Attention core options and the PixArt attention processor used by the eval2 experiments.

What it does
    sdpa()                  one place that runs softmax(QK^T)V with a chosen backend:
                              default / flash / cudnn / efficient / math   torch SDPA backends
                              sage2 / sage3                                SageAttention INT8-FP8 / FP4 kernels (optional installs)
                            plus two training-free speed options:
                              text_len     cross-attention: keep only the first text_len text tokens (padding is
                                           masked anyway, so this is exact; real prompts use 26-36 of 300 tokens)
                              kv_ds        self-attention: average-pool K and V over kv_ds x kv_ds token tiles
                                           (about kv_ds^2 fewer attention FLOPs, changes the output)
    Eval2AttnProcessor      replaces diffusers' AttnProcessor2_0 for PixArt. It uses fused NVFP4 projections when
                            fused_linear.convert_pixart() installed them, runs the core through sdpa(), and routes
                            self-attention through the DiTFastAttn op when a plan is active.

Why a custom processor: the Q/K/V projections must see the shared-quantize modules, and the attention core must
be swappable (backend, trimming, DiTFastAttn) without touching diffusers.
"""
import contextlib
import dataclasses
import math

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

_BACKENDS = {
    "flash": [SDPBackend.FLASH_ATTENTION],
    "efficient": [SDPBackend.EFFICIENT_ATTENTION],
    "cudnn": [SDPBackend.CUDNN_ATTENTION],
    "math": [SDPBackend.MATH],
}


@dataclasses.dataclass
class AttnConfig:
    backend: str = "default"      # self-attention: default | flash | efficient | cudnn | math | sage2 | sage3
    backend_cross: str = "default"  # cross-attention (masked; flash cannot take a mask): default | cudnn | efficient
    text_len: int = 0             # 0 = keep all text tokens
    kv_ds: int = 1                # K/V token-grid pooling factor for self-attention
    sage_on_cross: bool = False   # Sage kernels take no mask; cross-attention stays on torch SDPA unless True


def _sage(q, k, v, which):
    if which == "sage2":
        from sageattention import sageattn
        return sageattn(q, k, v, tensor_layout="HND", is_causal=False)
    from sageattn3 import sageattn3_blackwell
    return sageattn3_blackwell(q, k, v)


def pool_kv(k, v, f):
    """Average-pool K and V over f x f tiles of the square token grid. (B,H,N,D) -> (B,H,N/f^2,D)."""
    B, H, N, D = k.shape
    g = int(math.isqrt(N))
    if f <= 1 or g * g != N or g % f:
        return k, v

    def pool(t):
        t = t.reshape(B * H, g, g, D).permute(0, 3, 1, 2)
        t = F.avg_pool2d(t, f)
        return t.permute(0, 2, 3, 1).reshape(B, H, -1, D)

    return pool(k), pool(v)


def sdpa(q, k, v, mask, cfg, is_self):
    """q,k,v (B,H,N,D); mask None or an additive bias broadcastable to (B,H,Nq,Nk)."""
    if not is_self and cfg.text_len and k.shape[-2] > cfg.text_len:
        k, v = k[..., :cfg.text_len, :], v[..., :cfg.text_len, :]
        if mask is not None:
            mask = mask[..., :cfg.text_len]
    if is_self and cfg.kv_ds > 1:
        k, v = pool_kv(k, v, cfg.kv_ds)
    backend = cfg.backend if is_self else cfg.backend_cross
    if backend in ("sage2", "sage3") and (is_self or cfg.sage_on_cross) and mask is None:
        return _sage(q, k, v, backend)
    ctx = sdpa_kernel(_BACKENDS[backend]) if backend in _BACKENDS else contextlib.nullcontext()
    with ctx:
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)


class Eval2AttnProcessor:
    """Drop-in for AttnProcessor2_0 on PixArt blocks. One instance per Attention module."""

    def __init__(self, cfg, layer_id, is_self, use_dfa=False):
        self.cfg, self.layer_id, self.is_self, self.use_dfa = cfg, layer_id, is_self, use_dfa

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, *args, **kwargs):
        B = hidden_states.shape[0]
        if encoder_hidden_states is None:  # self-attention
            if getattr(attn, "fused_qkv", None) is not None:
                q, k, v = attn.fused_qkv(hidden_states)
            else:
                q, k, v = attn.to_q(hidden_states), attn.to_k(hidden_states), attn.to_v(hidden_states)
        else:  # cross-attention
            q = attn.to_q(hidden_states)
            if getattr(attn, "fused_kv", None) is not None:
                k, v = attn.fused_kv(encoder_hidden_states)
            else:
                k, v = attn.to_k(encoder_hidden_states), attn.to_v(encoder_hidden_states)

        mask = None
        if attention_mask is not None:  # (B,1,T) additive bias from PixArtTransformer2DModel
            mask = attention_mask.view(B, 1, -1, attention_mask.shape[-1]).to(q.dtype)
        heads = attn.heads
        d = q.shape[-1] // heads
        q = q.view(B, -1, heads, d).transpose(1, 2)
        k = k.view(B, -1, heads, d).transpose(1, 2)
        v = v.view(B, -1, heads, d).transpose(1, 2)

        if self.use_dfa and self.is_self:
            out = torch.ops.ditfast.attn(q, k, v, self.layer_id)
        else:
            out = sdpa(q, k, v, mask, self.cfg, self.is_self)

        out = out.transpose(1, 2).reshape(B, -1, heads * d).to(q.dtype)
        out = attn.to_out[0](out)
        out = attn.to_out[1](out)
        return out / attn.rescale_output_factor


def install_attention(transformer, cfg, use_dfa=False):
    """Set an Eval2AttnProcessor on every block's attn1 (self) and attn2 (cross)."""
    for i, block in enumerate(transformer.transformer_blocks):
        block.attn1.set_processor(Eval2AttnProcessor(cfg, i, True, use_dfa))
        block.attn2.set_processor(Eval2AttnProcessor(cfg, i, False, False))
