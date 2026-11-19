"""CPU tests for cfg_interval.sample: it must reproduce diffusers' PixArtSigmaPipeline, and limited-interval CFG must
change only the batch size inside/outside the guided band."""
import _boot
from _boot import check, torch

import cfg_interval


def diffusers_latents(pipe, item, steps, seed, guidance):
    g = torch.Generator().manual_seed(seed)
    out = pipe(prompt=None, negative_prompt=None, prompt_embeds=item["pe"], prompt_attention_mask=item["pm"],
               negative_prompt_embeds=item["ne"], negative_prompt_attention_mask=item["nm"], height=_boot.SIZE, width=_boot.SIZE,
               num_inference_steps=steps, guidance_scale=guidance, generator=g, output_type="latent", use_resolution_binning=False)
    return out.images


def test_matches_diffusers_pipeline():
    pipe = _boot.tiny_pipe()
    item = _boot.tiny_items(1)[0]
    ref = diffusers_latents(pipe, item, 4, seed=3, guidance=4.5)
    ours = cfg_interval.sample(pipe, [item], _boot.SIZE, steps=4, seeds=[3], guidance=4.5, output="latent")["latents"]
    check("sample() equals PixArtSigmaPipeline (interval=None)", torch.allclose(ref, ours, atol=1e-5), f"{(ref - ours).abs().max():.2e}")


def test_interval_batches_and_edges():
    pipe = _boot.tiny_pipe()
    items = _boot.tiny_items(2)
    full = cfg_interval.sample(pipe, items, _boot.SIZE, steps=6, output="latent")
    all_in = cfg_interval.sample(pipe, items, _boot.SIZE, steps=6, interval=(0.0, 1.0), output="latent")
    check("interval (0,1) equals no interval", torch.equal(full["latents"], all_in["latents"]))
    check("full CFG runs batch 2n every step", full["model_batches"] == [4] * 6, str(full["model_batches"]))
    mid = cfg_interval.sample(pipe, items, _boot.SIZE, steps=6, interval=(1 / 3, 2 / 3), output="latent")
    check("limited interval: 2n only inside the band", mid["model_batches"] == [2, 2, 4, 4, 2, 2], str(mid["model_batches"]))
    none = cfg_interval.sample(pipe, items, _boot.SIZE, steps=6, interval=(2.0, 3.0), output="latent")
    unguided = cfg_interval.sample(pipe, items, _boot.SIZE, steps=6, guidance=1.0, output="latent")
    check("empty interval equals guidance 1.0", torch.allclose(none["latents"], unguided["latents"], atol=1e-4),
          f"{(none['latents'] - unguided['latents']).abs().max():.2e}")


def test_decode_runs():
    pipe = _boot.tiny_pipe()
    res = cfg_interval.sample(pipe, _boot.tiny_items(2), _boot.SIZE, steps=2)
    imgs = res["images"]
    check("decode returns one HxWx3 image per prompt", len(imgs) == 2 and imgs[0].shape == (_boot.SIZE, _boot.SIZE, 3))


if __name__ == "__main__":
    _boot.run(test_matches_diffusers_pipeline, test_interval_batches_and_edges, test_decode_runs)
