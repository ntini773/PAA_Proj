"""Per-stage timing for one transformer forward: GPU span, GPU busy time, kernel count and CPU launch time.

Used by timing/profile_stages.py (tables) and step_target.py --stage-ranges (NVTX ranges for nsys).

Stages (names are stable so tables from different configs line up)
    block level   norm1 attn1 norm2/glue attn2 ff        (the pieces of a PixArt block)
    inside a FusedNVFP4Linear   quant  gemm              (only for the module stack: quantize op vs _scaled_mm + epilogue)
    outside       patch_embed caption_proj adaln final

How each number is measured
    span_ms    CUDA events recorded at stage start/end (no sync inside the forward): time the GPU spent from the first to the
               last kernel of the stage, gaps included
    busy_ms    torch.profiler: sum of kernel durations launched inside the stage
    gap_ms     span - busy: GPU idle time inside the stage (launch-bound or dependency stalls)
    cpu_ms     host time between entering and leaving the stage (Python + launch cost)
    kernels    number of kernels launched inside the stage
A stage is launch-bound when gap_ms > busy_ms.
"""
import collections
import contextlib
import time

import torch


class StageRecorder:
    """Collects (stage, start event, end event, cpu ms). Re-entrant stages are recorded separately and summed."""

    def __init__(self, use_nvtx=False, profiler_labels=False):
        self.use_nvtx = use_nvtx                  # NVTX ranges for nsys (nvtx_gpu_proj_sum)
        self.profiler_labels = profiler_labels    # torch.profiler record_function labels "stage::<name>"
        self.rows = []

    @contextlib.contextmanager
    def stage(self, name):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        if self.use_nvtx:
            torch.cuda.nvtx.range_push(name)
        label = torch.profiler.record_function(f"stage::{name}") if self.profiler_labels else contextlib.nullcontext()
        t0 = time.perf_counter()
        s.record()
        try:
            with label:
                yield
        finally:
            e.record()
            if self.use_nvtx:
                torch.cuda.nvtx.range_pop()
            self.rows.append((name, s, e, (time.perf_counter() - t0) * 1e3))

    def summary(self):
        torch.cuda.synchronize()
        out = collections.defaultdict(lambda: dict(span_ms=0.0, cpu_ms=0.0, calls=0))
        for name, s, e, cpu in self.rows:
            d = out[name]
            d["span_ms"] += s.elapsed_time(e)
            d["cpu_ms"] += cpu
            d["calls"] += 1
        return dict(out)

    def clear(self):
        self.rows.clear()


def _wrap_attr(obj, attr, rec, name, undo_list):
    """Replace obj.<attr> (a bound method) by a version that runs inside a stage; undo restores the class method."""
    inner = getattr(obj, attr)

    def wrapped(*a, **k):
        with rec.stage(name):
            return inner(*a, **k)

    setattr(obj, attr, wrapped)
    undo_list.append(lambda: delattr(obj, attr))


def install(transformer, rec):
    """Add stage ranges to a (bf16, stock or module-stack) PixArt transformer. Returns a function that removes them."""
    undo_list = []

    def wrap(mod, name):
        _wrap_attr(mod, "forward", rec, name, undo_list)

    for b in transformer.transformer_blocks:
        wrap(b.norm1, "norm1")
        wrap(b.attn1, "attn1")
        wrap(b.attn2, "attn2")
        wrap(b.norm2, "norm2")
        wrap(b.ff, "ff")
    for attr, name in (("pos_embed", "patch_embed"), ("caption_projection", "caption_proj"), ("adaln_single", "adaln")):
        if getattr(transformer, attr, None) is not None:
            wrap(getattr(transformer, attr), name)
    for m in transformer.modules():
        if type(m).__name__ == "FusedNVFP4Linear":  # split it into quantize and GEMM stages
            _wrap_attr(m, "_quantize", rec, "quant", undo_list)
            _wrap_attr(m, "_mm", rec, "gemm", undo_list)

    def undo():
        for u in undo_list:
            u()

    return undo
