"""CPU tests for attention_patch.py (text trimming, K/V pooling, processor equivalence) and ditfast/attention.py.

Each DiTFastAttn technique is checked against what it must mean: AST copies the previous output, ASC copies the
conditional half, WA-RS reproduces full attention on the step that cached its residual, an empty plan changes nothing,
and the state resets between images.
"""
import _boot
from _boot import check, torch

import attention_patch as ap
import cfg_interval
import ditfast.attention as dfa
import ditfast.plan_search as ps

F = torch.nn.functional


def qkv(B=4, H=2, N=16, D=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(B, H, N, D, generator=g) for _ in range(3)]


def test_trim_text_is_exact():
    q, k, v = qkv(B=3, N=16)
    T = 20
    kc, vc = torch.randn(3, 2, T, 8), torch.randn(3, 2, T, 8)
    valid = torch.tensor([6, 3, 5])
    bias = torch.zeros(3, 1, 1, T)
    for b in range(3):
        bias[b, ..., valid[b]:] = -10000.0
    full = ap.sdpa(q, kc, vc, bias, ap.AttnConfig(), False)
    trim = ap.sdpa(q, kc, vc, bias, ap.AttnConfig(text_len=int(valid.max())), False)
    check("text trim equals masked full cross-attention", torch.allclose(full, trim, atol=1e-5), f"{(full - trim).abs().max():.2e}")


def test_kv_pool_shapes():
    q, k, v = qkv(N=16)
    kp, vp = ap.pool_kv(k, v, 2)
    check("kv pool 2x2: 16 tokens -> 4", kp.shape[-2] == 4 and vp.shape == kp.shape)
    k1, v1 = ap.pool_kv(k, v, 1)
    check("kv pool factor 1 is identity", k1 is k and v1 is v)
    out = ap.sdpa(q, k, v, None, ap.AttnConfig(kv_ds=2), True)
    check("pooled attention keeps the query length", out.shape == q.shape)


def test_backend_names():
    q, k, v = qkv()
    a = ap.sdpa(q, k, v, None, ap.AttnConfig("math"), True)
    b = ap.sdpa(q, k, v, None, ap.AttnConfig("default"), True)
    check("math backend equals default", torch.allclose(a, b, atol=1e-5))


def test_ast_asc_wars_semantics():
    S = dfa.STATE
    S.reset()
    S.mode, S.window_div, S._need_resid, S.cfg_pair = "run", 4, True, True
    S.cfg = ap.AttnConfig()
    q, k, v = qkv(B=4, N=16)
    # step 0: full, caches last and residual
    S.plan = {(0, 0): "full", (1, 0): "ast", (2, 0): "asc", (3, 0): "wars", (4, 0): "wars_asc"}
    S.begin_step(0)
    o0 = S.run(q, k, v, 0)
    ref = ap.sdpa(q, k, v, None, ap.AttnConfig(), True)
    check("full step equals sdpa", torch.allclose(o0, ref, atol=1e-6))
    S.begin_step(1)
    q1, k1, v1 = qkv(B=4, N=16, seed=5)
    o1 = S.run(q1, k1, v1, 0)
    check("AST returns the previous step's output", torch.equal(o1, o0) and S.used[(1, 0)] == "ast")
    S.begin_step(2)
    o2 = S.run(q1, k1, v1, 0)
    cond = ap.sdpa(q1[2:], k1[2:], v1[2:], None, ap.AttnConfig(), True)
    check("ASC: both halves equal conditional attention", torch.allclose(o2[:2], cond, atol=1e-6) and torch.allclose(o2[2:], cond, atol=1e-6)
          and S.used[(2, 0)] == "asc")
    # wars on the same inputs as the cached full step reproduces full attention (residual = full - window)
    S.reset()
    S.plan = {(0, 0): "full", (1, 0): "wars", (2, 0): "wars_asc"}
    S.begin_step(0)
    of = S.run(q, k, v, 0)
    S.begin_step(1)
    ow = S.run(q, k, v, 0)
    check("WA-RS on the cached step reproduces full attention", torch.allclose(ow, of, atol=1e-5) and S.used[(1, 0)] == "wars",
          f"{(ow - of).abs().max():.2e}")
    S.begin_step(2)
    owa = S.run(q, k, v, 0)
    check("WA-RS+ASC copies the conditional half", torch.allclose(owa[:2], owa[2:], atol=1e-6) and S.used[(2, 0)] == "wars_asc")
    # missing cache falls back to full
    S.reset()
    S.plan = {(0, 0): "ast"}
    S.begin_step(0)
    S.run(q, k, v, 0)
    check("AST without a cache falls back to full", S.used[(0, 0)] == "full")
    # shape change (limited-interval CFG) invalidates caches
    S.reset()
    S.plan = {(0, 0): "full", (1, 0): "ast"}
    S.begin_step(0)
    S.run(q, k, v, 0)
    S.begin_step(1)
    S.run(q[:2], k[:2], v[:2], 0)
    check("batch-shape change invalidates the AST cache", S.used[(1, 0)] == "full")


def _setup(use_dfa):
    pipe = _boot.tiny_pipe()
    ps.enable(pipe, ap.AttnConfig(), use_dfa=use_dfa)
    return pipe


def test_empty_plan_equals_plain_attention():
    items = _boot.tiny_items(2)
    plain = cfg_interval.sample(_setup(False), items, _boot.SIZE, steps=4, output="latent")["latents"]
    pipe = _setup(True)
    dfa.STATE.plan, dfa.STATE.mode, dfa.STATE._need_resid = {}, "run", False
    with_dfa = cfg_interval.sample(pipe, items, _boot.SIZE, steps=4, dfa_active=True, output="latent")["latents"]
    check("DiTFastAttn with an empty plan equals plain attention", torch.allclose(plain, with_dfa, atol=1e-5),
          f"{(plain - with_dfa).abs().max():.2e}")


def test_plan_search_and_run_consistency():
    items = _boot.tiny_items(2)
    pipe = _setup(True)
    plan0 = ps.search_plan(pipe, items, _boot.SIZE, steps=5, delta=0.0)
    # block 0 sees identical uncond/cond inputs (same latent and timestep, text enters only at cross-attention),
    # so ASC is exact there (error 0) and is allowed even at delta=0; every other layer must stay full
    check("delta=0 keeps every layer full except exact ASC at block 0",
          set(t for r in plan0["plan"] for l, t in enumerate(r) if l > 0) == {"full"} and
          set(r[0] for r in plan0["plan"]) <= {"full", "asc"})
    plan_a = ps.search_plan(pipe, items, _boot.SIZE, steps=5, delta=10.0)
    plan_b = ps.search_plan(pipe, items, _boot.SIZE, steps=5, delta=10.0)
    check("plan search is deterministic", plan_a["plan"] == plan_b["plan"])
    check("huge delta uses AST from step 1 on and ASC where no cache exists",
          all(t == "asc" for t in plan_a["plan"][0]) and all(t == "ast" for t in plan_a["plan"][1]))
    check("modeled attention cost below 1 when compressing", plan_a["modeled_attention_cost"] < 1.0, str(plan_a["modeled_attention_cost"]))
    # run mode must follow the plan and be repeatable; state must reset between images
    dfa.set_plan(plan_a)
    r1 = cfg_interval.sample(pipe, items, _boot.SIZE, steps=5, dfa_active=True, output="latent")["latents"]
    used1 = dict(dfa.STATE.used)
    r2 = cfg_interval.sample(pipe, items, _boot.SIZE, steps=5, dfa_active=True, output="latent")["latents"]
    check("two runs with the same plan are identical (state resets)", torch.equal(r1, r2))
    check("run mode executes the planned techniques", all(used1.get(k, "full") == t or t != "full" and used1.get(k) == "full"
                                                         for k, t in dfa.STATE.plan.items()))


if __name__ == "__main__":
    _boot.run(test_trim_text_is_exact, test_kv_pool_shapes, test_backend_names, test_ast_asc_wars_semantics,
              test_empty_plan_equals_plain_attention, test_plan_search_and_run_consistency)
