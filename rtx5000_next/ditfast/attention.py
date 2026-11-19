"""DiTFastAttn-style attention compression for PixArt self-attention (training-free, inference time).

Idea (Yuan et al., NeurIPS 2024, https://arxiv.org/abs/2406.08552): self-attention in a diffusion transformer is
redundant across tokens, timesteps and the two halves of classifier-free guidance. For each (denoise step, layer)
one technique is chosen from a calibration run (plan_search.py):

    full      ordinary attention
    wars      Window Attention with Residual Sharing: attention inside non-overlapping token windows of size W
              plus the residual (full - window) cached from the last full step of this layer
    ast       Attention Sharing across Timesteps: reuse this layer's attention output from the previous step
    asc       Attention Sharing across CFG: attention only on the conditional half; the unconditional half copies it
    wars_asc  wars on the conditional half, copied to the unconditional half

Batch layout assumed (as in diffusers PixArt CFG): [unconditional x n, conditional x n].

How it plugs in
    The core is one custom op, torch.ops.ditfast.attn(q, k, v, layer_id). It is opaque to torch.compile, so the
    compiled graph is the same at every denoise step; the per-step decision is read from STATE at run time.
    The sampling loop calls STATE.begin_step(i) before each model call and STATE.reset() before each image.
    Caches hold tensors per layer; they are dropped whenever the batch shape changes (limited-interval CFG).

Modes
    STATE.mode == "run"    follow STATE.plan[(step, layer)]; unavailable caches fall back to full
    STATE.mode == "probe"  plan search: compute full attention and every candidate, pick the cheapest candidate
                           whose relative L2 error against full is below STATE.delta, use it, and record it
"""
import torch

import attention_patch as ap

# modeled attention cost of each technique relative to full (window share w = W / N; asc halves the batch)
TECHNIQUES = ("ast", "wars_asc", "asc", "wars", "full")


def modeled_cost(tech, window_share):
    return {"ast": 0.02, "wars_asc": 0.5 * window_share + 0.02, "asc": 0.5, "wars": window_share + 0.02, "full": 1.0}[tech]


def window_sdpa(q, k, v, W):
    """Attention inside non-overlapping windows of W consecutive tokens. (B,H,N,D), N % W == 0."""
    B, H, N, D = q.shape
    g = N // W

    def split(t):
        return t.reshape(B, H * g, W, D)

    out = torch.nn.functional.scaled_dot_product_attention(split(q), split(k), split(v))
    return out.reshape(B, H, N, D)


def rel_l2(a, ref):
    return float((a.float() - ref.float()).norm() / ref.float().norm().clamp(min=1e-12))


class DFAState:
    def __init__(self):
        self.cfg = ap.AttnConfig()
        self.mode = "run"
        self.plan = {}            # (step, layer) -> technique
        self.window_div = 8       # window = N // window_div (N=4096 -> 512, the paper's setting)
        self.delta = 0.1
        self.techniques = TECHNIQUES
        self.cfg_pair = True      # batch is [uncond, cond]
        self.reset()

    def reset(self):
        self.step = 0
        self.cache = {}           # layer -> {"last": tensor, "resid": tensor}
        self.record = {}          # probe: (step, layer) -> {tech: error}
        self.used = {}            # probe/run: (step, layer) -> technique actually executed
        self.calls = 0

    def begin_step(self, i):
        self.step = i

    # -- helpers -------------------------------------------------------------------------------------------
    def _window(self, N):
        W = N // self.window_div
        return W if W >= 1 and N % W == 0 and W < N else 0

    def _full(self, q, k, v):
        return ap.sdpa(q, k, v, None, self.cfg, True)

    def _valid(self, tensor, like):
        return tensor is not None and tensor.shape == like.shape

    def _asc_ok(self, B):
        return self.cfg_pair and B % 2 == 0 and B >= 2

    # -- one technique -------------------------------------------------------------------------------------
    def _candidate(self, tech, q, k, v, c):
        """Output of one technique, or None if its cache or shape prerequisite is missing."""
        B, H, N, D = q.shape
        W = self._window(N)
        h = B // 2
        if tech == "ast":
            return c["last"].clone() if self._valid(c.get("last"), q) else None
        if tech == "asc":
            if not self._asc_ok(B):
                return None
            cond = self._full(q[h:], k[h:], v[h:])
            return torch.cat([cond, cond])
        if tech == "wars":
            if not W or not self._valid(c.get("resid"), q):
                return None
            return window_sdpa(q, k, v, W) + c["resid"]
        if tech == "wars_asc":
            if not W or not self._asc_ok(B) or not self._valid(c.get("resid"), q):
                return None
            cond = window_sdpa(q[h:], k[h:], v[h:], W) + c["resid"][h:]
            return torch.cat([cond, cond])
        return self._full(q, k, v)

    def _update_full_cache(self, q, k, v, full, c):
        N = q.shape[2]
        W = self._window(N)
        c["last"] = full
        if W and (self.mode == "probe" or self.need_resid):
            c["resid"] = full - window_sdpa(q, k, v, W)

    # -- entry point ---------------------------------------------------------------------------------------
    @property
    def need_resid(self):
        return getattr(self, "_need_resid", True)

    def run(self, q, k, v, layer):
        self.calls += 1
        c = self.cache.setdefault(layer, {})
        key = (self.step, layer)
        if self.mode == "probe":
            return self._probe(q, k, v, layer, c, key)
        tech = self.plan.get(key, "full")
        out = self._candidate(tech, q, k, v, c) if tech != "full" else None
        if out is None:
            tech = "full"
            out = self._full(q, k, v)
            self._update_full_cache(q, k, v, out, c)
        else:
            c["last"] = out
        self.used[key] = tech
        return out

    def _probe(self, q, k, v, layer, c, key):
        full = self._full(q, k, v)
        errs, outs = {}, {}
        for tech in self.techniques:
            if tech == "full":
                continue
            o = self._candidate(tech, q, k, v, c)
            if o is not None:
                errs[tech] = rel_l2(o, full)
                outs[tech] = o
        self.record[key] = errs
        window_share = (self._window(q.shape[2]) or q.shape[2]) / q.shape[2]
        ok = [t for t in self.techniques if t in outs and errs[t] <= self.delta]
        tech = min(ok, key=lambda t: modeled_cost(t, window_share)) if ok else "full"
        self.plan[key] = tech
        self.used[key] = tech
        if tech == "full":
            self._update_full_cache(q, k, v, full, c)
            return full
        c["last"] = outs[tech]
        return outs[tech]


STATE = DFAState()


@torch.library.custom_op("ditfast::attn", mutates_args=())
def dfa_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int) -> torch.Tensor:
    return STATE.run(q, k, v, layer_id)


@dfa_attn.register_fake
def _(q, k, v, layer_id):
    return torch.empty_like(q)


def set_plan(plan_json):
    """Load a plan saved by plan_search.save_plan into STATE. Also tells run() whether residuals are needed."""
    STATE.plan = {(s, l): t for s, row in enumerate(plan_json["plan"]) for l, t in enumerate(row)}
    STATE.window_div = plan_json.get("window_div", STATE.window_div)
    STATE._need_resid = any(t in ("wars", "wars_asc") for t in STATE.plan.values())
    STATE.mode = "run"


def plan_cost(plan, n_layers, steps, window_share):
    """Modeled share of full self-attention cost under a plan (1.0 = no compression)."""
    total = sum(modeled_cost(plan.get((s, l), "full"), window_share) for s in range(steps) for l in range(n_layers))
    return total / (steps * n_layers)
