"""Download ONLY what the experiments need (no T5 text encoder, ~8 GB total).

PixArt-Sigma has no pre-quantized NVFP4 checkpoint on the Hub. The bf16/fp32
transformer is downloaded once and quantized to NVFP4 at load time by torchao
(TorchAoConfig + NVFP4DynamicActivationNVFP4WeightConfig), exactly as in the
local Tier-2 runs. One transformer per resolution + shared VAE/scheduler.

    pixi run download
"""
import os
from huggingface_hub import snapshot_download

TRANSFORMERS = {  # resolution -> repo whose transformer/ we use
    256: "PixArt-alpha/PixArt-Sigma-XL-2-256x256",
    512: "PixArt-alpha/PixArt-Sigma-XL-2-512-MS",
    1024: "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
}
for size, repo in TRANSFORMERS.items():
    pats = ["transformer/*"]
    if size == 1024:  # shared pipeline parts: VAE, scheduler, index (no text_encoder/tokenizer)
        pats += ["vae/*", "scheduler/*", "model_index.json"]
    print(f"[{size}] {repo} {pats}")
    snapshot_download(repo, allow_patterns=pats)
print("done. HF cache:", os.environ.get("HF_HOME", "~/.cache/huggingface"))
