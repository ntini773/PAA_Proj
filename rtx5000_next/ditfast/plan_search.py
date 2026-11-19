"""Greedy DiTFastAttn compression-plan search (the calibration step).

What it does
    Runs the real sampling loop on a few calibration prompts, in probe mode (ditfast.attention.DFAState). At every
    (denoise step, layer) the self-attention op computes full attention plus every technique that its caches allow,
    measures each technique's relative L2 error against full attention, and takes the cheapest technique whose error
    is below delta. The chosen output flows on, so later decisions see the compressed trajectory. The result is a
    plan: plan[step][layer] in {full, wars, ast, asc, wars_asc}.

Why greedy and per-layer: it is the paper's recipe in miniature and costs about one sampling run per calibration
batch (all prompts go through together, so the decision is shared by every prompt).

    plan = search_plan(pipe, items, size=512, steps=14, delta=0.1)
    save_plan(plan, "ditfast/plans/pixart_512_d010.json")
"""
import collections
import json
import pathlib

import cfg_interval
import ditfast.attention as dfa


def enable(pipe, cfg=None, use_dfa=True):
    """Install the Eval2 attention processor (with the DiTFastAttn op) on the transformer."""
    import attention_patch as ap
    cfg = cfg or ap.AttnConfig()
    dfa.STATE.cfg = cfg
    ap.install_attention(pipe.transformer, cfg, use_dfa=use_dfa)


def search_plan(pipe, items, size, steps=14, delta=0.1, window_div=8, interval=None, techniques=dfa.TECHNIQUES, guidance=4.5):
    S = dfa.STATE
    S.reset()
    S.mode, S.delta, S.window_div, S.techniques = "probe", delta, window_div, tuple(techniques)
    S.plan = {}
    cfg_interval.sample(pipe, items, size, steps=steps, guidance=guidance, interval=interval, dfa_active=True, output="latent")
    layers = len(pipe.transformer.transformer_blocks)
    rows = [[S.plan.get((s, l), "full") for l in range(layers)] for s in range(steps)]
    counts = collections.Counter(t for r in rows for t in r)
    n_tokens = (size // 8 // pipe.transformer.config.patch_size) ** 2
    W = n_tokens // window_div
    cost = dfa.plan_cost(S.plan, layers, steps, (W / n_tokens) if W else 1.0)
    mean_err = {t: sum(e[t] for e in S.record.values() if t in e) / max(1, sum(1 for e in S.record.values() if t in e))
                for t in techniques if t != "full"}
    S.mode = "run"
    return dict(size=size, steps=steps, layers=layers, delta=delta, window_div=window_div, interval=interval,
                counts=dict(counts), modeled_attention_cost=cost, mean_candidate_error=mean_err, plan=rows)


def save_plan(plan, path):
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(plan))


def load_plan(path):
    plan = json.loads(pathlib.Path(path).read_text())
    dfa.set_plan(plan)
    return plan
