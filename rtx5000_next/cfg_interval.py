"""PixArt-Sigma sampling loop with limited-interval classifier-free guidance and DiTFastAttn hooks.

Why a custom loop
    diffusers' PixArtSigmaPipeline always runs the transformer on a doubled (uncond, cond) batch. Limited-interval
    CFG (Kynkaanniemi et al., NeurIPS 2024, https://arxiv.org/abs/2404.07724) guides only a middle band of the
    denoising path; outside it the model runs on the conditional half alone (batch 2n -> n, about half the work).
    The loop below reproduces the pipeline step for step (same scheduler, timestep handling, learned-sigma split),
    so interval=None gives the pipeline's result, and adds:
      - interval=(lo, hi): guidance only for steps with lo <= i/steps < hi
      - batching of several prompts: batch layout [uncond x n, cond x n], which DiTFastAttn's CFG sharing needs
      - per-step and per-phase timing (CUDA events), used by bench/pipeline_e2e.py and the quality runs
      - DiTFastAttn: STATE.reset() before the image, STATE.begin_step(i) before every model call

Inputs  items: list of {"pe","pm","ne","nm"} prompt embeddings (prompts/embeds_pixart.pt entries)
Outputs dict(images=[HxWx3 float32 in 0..1], latents, denoise_ms, vae_ms, step_ms, model_batches)
"""
import torch

import ditfast.attention as dfa


def _latents(pipe, n, size, seeds, device, dtype):
    out = []
    for s in seeds[:n]:
        g = torch.Generator(device=device).manual_seed(int(s))
        out.append(pipe.prepare_latents(1, pipe.transformer.config.in_channels, size, size, dtype, device, g))
    return torch.cat(out)


@torch.no_grad()
def sample(pipe, items, size, steps=14, seeds=None, guidance=4.5, interval=None, dfa_active=False, output="np", timing=False,
           on_step=None):
    tr, sch, vae = pipe.transformer, pipe.scheduler, pipe.vae
    device = next(tr.parameters()).device if len(list(tr.parameters())) else next(tr.buffers()).device
    dtype = tr.dtype if hasattr(tr, "dtype") else torch.bfloat16
    n = len(items)
    seeds = list(seeds) if seeds is not None else list(range(n))
    cat = lambda key, dt=None: torch.cat([it[key] for it in items]).to(device, dt) if dt else torch.cat([it[key] for it in items]).to(device)
    pe, ne = cat("pe", dtype), cat("ne", dtype)
    pm, nm = cat("pm"), cat("nm")
    enc_cfg, mask_cfg = torch.cat([ne, pe]), torch.cat([nm, pm])
    latents = _latents(pipe, n, size, seeds, device, dtype)

    sch.set_timesteps(steps, device=device)
    extra = pipe.prepare_extra_step_kwargs(torch.Generator(device=device).manual_seed(0), 0.0)
    latent_channels = tr.config.in_channels
    added = {"resolution": None, "aspect_ratio": None}
    if dfa_active:
        dfa.STATE.reset()

    events = []
    cuda = timing and device.type == "cuda"
    if cuda:
        e0 = torch.cuda.Event(enable_timing=True)
        e0.record()
    batches = []
    for i, t in enumerate(sch.timesteps):
        use_cfg = interval is None or (interval[0] <= i / steps < interval[1])
        x = torch.cat([latents, latents]) if use_cfg else latents
        x = sch.scale_model_input(x, t)
        tt = t.to(device).reshape(1).expand(x.shape[0])
        if dfa_active:
            dfa.STATE.begin_step(i)
            dfa.STATE.cfg_pair = use_cfg
        if on_step is not None:
            on_step(i)
        if cuda:
            s_ev = torch.cuda.Event(enable_timing=True)
            s_ev.record()
        out = tr(x, encoder_hidden_states=enc_cfg if use_cfg else pe, encoder_attention_mask=mask_cfg if use_cfg else pm,
                 timestep=tt, added_cond_kwargs=added, return_dict=False)[0]
        if cuda:
            f_ev = torch.cuda.Event(enable_timing=True)
            f_ev.record()
            events.append((s_ev, f_ev))
        batches.append(int(x.shape[0]))
        if use_cfg:
            u, c = out.chunk(2)
            out = u + guidance * (c - u)
        if tr.config.out_channels // 2 == latent_channels:
            out = out.chunk(2, dim=1)[0]
        latents = sch.step(out, t, latents, **extra, return_dict=False)[0]
    if cuda:
        e1 = torch.cuda.Event(enable_timing=True)
        e1.record()

    res = dict(latents=latents, model_batches=batches)
    if output == "latent":
        res["images"] = None
    else:
        if cuda:
            v0 = torch.cuda.Event(enable_timing=True)
            v0.record()
        img = vae.decode(latents.to(vae.dtype) / vae.config.scaling_factor, return_dict=False)[0]
        res["images"] = list(pipe.image_processor.postprocess(img, output_type="np"))
        if cuda:
            v1 = torch.cuda.Event(enable_timing=True)
            v1.record()
    if cuda:
        torch.cuda.synchronize()
        res["denoise_ms"] = e0.elapsed_time(e1)
        res["step_ms"] = [a.elapsed_time(b) for a, b in events]
        if output != "latent":
            res["vae_ms"] = v0.elapsed_time(v1)
    return res
