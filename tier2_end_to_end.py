"""
Tier-2 end-to-end: real PixArt-Sigma NVFP4 image generation, comparing the
stock pipeline against one where the fused NVFP4 quant kernel is spliced into
the transformer's quantized-Linear activation path.

Input requirement: just a TEXT PROMPT (+ a fixed seed). No dataset. Quality is
measured as LPIPS between the two generated images (same prompt, same seed), so
it isolates the kernel's effect. LPIPS < 0.01 == quality-neutral (per the doc).

No torch.compile (it stalled on the 5060); this measures correctness/quality
end-to-end, not peak compiled speed.

Run:
    HF_HUB_OFFLINE=1 pixi run python tier2_end_to_end.py --steps 20
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse
import time
import torch
import numpy as np

from diffusers import PixArtSigmaPipeline, PixArtTransformer2DModel, TorchAoConfig
from torchao.prototype.mx_formats.inference_workflow import (
    NVFP4DynamicActivationNVFP4WeightConfig,
)


def encode_prompt_then_free(prompt):
    """Encode the prompt with T5-XXL using accelerate DISK offload so the 19GB
    encoder never needs to fit in RAM/VRAM at once (8GB GPU / 14GB RAM here).
    Layers stream from disk->GPU one at a time. Slow but fits. Encoder is freed
    after, so the diffusion loop runs with only the transformer+VAE on GPU."""
    import gc, os as _os
    from transformers import T5EncoderModel, T5Tokenizer
    repo = "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS"
    off = _os.path.abspath("./t5_offload")
    _os.makedirs(off, exist_ok=True)
    tok = T5Tokenizer.from_pretrained(repo, subfolder="tokenizer")
    te = T5EncoderModel.from_pretrained(
        repo, subfolder="text_encoder", dtype=torch.bfloat16,
        device_map="auto", offload_folder=off, low_cpu_mem_usage=True,
        max_memory={0: "5GiB", "cpu": "6GiB"},
    )
    max_len = 300
    with torch.no_grad():
        def emb(text):
            ids = tok(text, padding="max_length", max_length=max_len,
                      truncation=True, return_tensors="pt")
            iid = ids.input_ids; am = ids.attention_mask
            e = te(input_ids=iid.to(te.device), attention_mask=am.to(te.device))[0]
            return e.detach().cpu(), am.detach().cpu()
        pe, pmask = emb(prompt)
        ne, nmask = emb("")
    del te, tok
    gc.collect(); torch.cuda.empty_cache()
    return pe, pmask, ne, nmask


def load_transformer_pipe():
    """Load the pipeline WITHOUT the text encoder (we feed cached embeds)."""
    # use_triton_kernel=False -> torchao's pure-PyTorch nvfp4_quantize path.
    # (use_triton_kernel=True needs MSLK installed, which isn't available here.)
    nvfp4_cfg = NVFP4DynamicActivationNVFP4WeightConfig(
        use_dynamic_per_tensor_scale=True, use_triton_kernel=False,
    )
    transformer = PixArtTransformer2DModel.from_pretrained(
        "PixArt-alpha/PixArt-Sigma-XL-2-256x256",
        subfolder="transformer", torch_dtype=torch.bfloat16,
        quantization_config=TorchAoConfig(nvfp4_cfg),
    )
    pipe = PixArtSigmaPipeline.from_pretrained(
        "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
        transformer=transformer, text_encoder=None, tokenizer=None,
        torch_dtype=torch.bfloat16,
    ).to("cuda")
    return pipe


def gen(pipe, embeds, steps, seed):
    pe, pmask, ne, nmask = embeds
    g = torch.Generator(device="cuda").manual_seed(seed)
    img = pipe(
        prompt=None,
        negative_prompt=None,
        prompt_embeds=pe.to("cuda"),
        prompt_attention_mask=pmask.to("cuda"),
        negative_prompt_embeds=ne.to("cuda"),
        negative_prompt_attention_mask=nmask.to("cuda"),
        height=256, width=256,
        num_inference_steps=steps, generator=g,
    ).images[0]
    return np.array(img).astype(np.float32) / 255.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", default="a photo of a red fox in snow")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    print(f"Encoding prompt (text encoder loaded then freed): '{a.prompt}'")
    embeds = encode_prompt_then_free(a.prompt)
    print("  prompt encoded, text encoder freed")

    print("Loading real PixArt-Sigma NVFP4 transformer pipeline (from cache)...")
    pipe = load_transformer_pipe()

    print(f"\nGenerating STOCK NVFP4 image (seed={a.seed}, steps={a.steps})...")
    t0 = time.time()
    img_stock = gen(pipe, embeds, a.steps, a.seed)
    t_stock = time.time() - t0
    print(f"  stock done in {t_stock:.1f}s")

    # ---- splice in the fused quant ----
    counter = install_fused_quant()

    print(f"Generating FUSED NVFP4 image (same seed)...")
    t0 = time.time()
    img_fused = gen(pipe, embeds, a.steps, a.seed)
    t_fused = time.time() - t0
    print(f"  fused done in {t_fused:.1f}s  "
          f"(fused quant calls: {counter['n']}, fallback: {counter['fallback']})")

    # ---- compare ----
    mae = float(np.abs(img_stock - img_fused).mean())
    maxd = float(np.abs(img_stock - img_fused).max())
    print("\n" + "=" * 60)
    print("END-TO-END COMPARISON (256x256, same prompt/seed)")
    print("=" * 60)
    print(f"  pixel mean|diff| : {mae:.6f}")
    print(f"  pixel max|diff|  : {maxd:.6f}")
    try:
        import lpips
        lp = lpips.LPIPS(net="alex").to("cuda")
        def to_t(x):
            return torch.from_numpy(x).permute(2, 0, 1)[None].to("cuda") * 2 - 1
        d = lp(to_t(img_stock), to_t(img_fused)).item()
        print(f"  LPIPS            : {d:.6f}   ({'PASS <0.01' if d < 0.01 else 'CHECK'})")
    except Exception as e:
        print(f"  LPIPS unavailable ({type(e).__name__}); pixel diff above.")
    print(f"  wall time stock={t_stock:.1f}s  fused={t_fused:.1f}s "
          f"(uncompiled; CPU-offload-free transformer path)")
    print("=" * 60)

    from PIL import Image
    Image.fromarray((img_stock*255).astype(np.uint8)).save("e2e_stock.png")
    Image.fromarray((img_fused*255).astype(np.uint8)).save("e2e_fused.png")
    print("  saved e2e_stock.png / e2e_fused.png")


def install_fused_quant():
    """Patch NVFP4Tensor.to_nvfp4 so the block-quant is produced by our fused
    kernel (fused_nvfp4_quantize), then reuse torchao's exact swizzle/assembly
    for GEMM-layout compatibility. Falls back to the original for any case we
    don't cover (transpose, 3D, non-contiguous, shapes we haven't validated).
    """
    import math
    import torchao.prototype.mx_formats.nvfp4_tensor as m
    from torchao.prototype.mx_formats.utils import to_blocked
    from torchao.prototype.mx_formats.nvfp4_tensor import (
        hp_data_dims_to_swizzled_scale_dims_nvfp4,
    )
    from fused_adaln_nvfp4 import fused_nvfp4_quantize

    orig = m.NVFP4Tensor.to_nvfp4
    counter = {"n": 0, "fallback": 0}

    # shapes our two-level kernel is validated bit-exact on (DiT hidden dim)
    # EXTRA_OK_K=4608,4096 widens the fused path to the MLP second layer / caption
    # projection (default: only the validated 1152). Verify images stay identical.
    OK_K = {1152} | {int(v) for v in os.environ.get("EXTRA_OK_K", "").split(",") if v.strip()}

    def patched(data_hp, block_size=16, per_tensor_scale=None,
                act_per_tensor_scale=None, is_swizzled_scales=False,
                use_triton_kernel=False, act_quant_kwargs=None):
        ok = (data_hp.dtype == torch.bfloat16 and data_hp.dim() == 2
              and data_hp.is_contiguous() and data_hp.shape[-1] in OK_K
              and data_hp.shape[-1] % 16 == 0
              and (per_tensor_scale is None or per_tensor_scale.numel() == 1))
        if not ok:
            counter["fallback"] += 1
            return orig(data_hp, block_size, per_tensor_scale,
                        act_per_tensor_scale, is_swizzled_scales,
                        use_triton_kernel, act_quant_kwargs)

        M, K = data_hp.shape
        # fused block-quant (matches nvfp4_quantize's (out_scales, data_lp))
        blockwise_scales, data_lp = fused_nvfp4_quantize(data_hp, per_tensor_scale)
        # reuse torchao's exact swizzle assembly
        if is_swizzled_scales:
            scale_shape = (M, K // block_size)
            blockwise_scales = to_blocked(blockwise_scales.view(scale_shape)).flatten()
            scale_M, scale_K = hp_data_dims_to_swizzled_scale_dims_nvfp4(M, K)
        else:
            scale_M, scale_K = M, K // block_size
        blockwise_scales = blockwise_scales.view(scale_M, scale_K)
        counter["n"] += 1
        return m.NVFP4Tensor(
            data_lp, blockwise_scales, block_size, data_hp.dtype,
            per_tensor_scale, act_per_tensor_scale, is_swizzled_scales,
            use_triton_kernel, act_quant_kwargs,
        )

    m.NVFP4Tensor.to_nvfp4 = staticmethod(patched)
    return counter


if __name__ == "__main__":
    main()
