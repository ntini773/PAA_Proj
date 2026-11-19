"""One model forward (one denoise step) under a chosen quantization config.
Used for timing, nsys (--scope step) and ncu (--scope layer).

    python step_target.py --model pixart_sigma --size 1024 --mode fused --backend cuda_swizzled --graph
    python step_target.py --size 512 --mode stock --list-layers
"""
import argparse
import re
import statistics
import time

import torch

import quant_backends as qb
from common import RESULTS, append_jsonl, stats, write_json
from model_adapters import build, linear_widths


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="pixart_sigma")
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--mode", choices=["bf16", "fp8", "stock", "fused"], default="bf16")
    p.add_argument("--variant", default="", help="eval2 variant string (variants.py), e.g. module+static+concat+compile")
    p.add_argument("--embeds", default="prompts/embeds_pixart.pt")
    p.add_argument("--stage-ranges", action="store_true", help="NVTX ranges per stage (norm1, attn1, ..., quant, gemm) for nsys")
    p.add_argument("--backend", choices=["triton", "triton_op", "cuda_row", "cuda_swizzled", "free"], default="triton")
    p.add_argument("--width", choices=["narrow", "wide"], default="wide")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--graph", action="store_true")
    p.add_argument("--scope", choices=["none", "step", "layer"], default="none")
    p.add_argument("--layer-regex", default=r"transformer_blocks\.3\.ff\.net\.2$")
    p.add_argument("--nvtx", action="store_true")
    p.add_argument("--config", default="")
    p.add_argument("--list-layers", action="store_true")
    return p.parse_args()


class VariantRunner:
    """Adapter so a variants.py variant runs through the same timing / nsys / ncu code as the legacy modes."""

    def __init__(self, built, fwd, size, batch):
        self.transformer, self.blocks = built.pipe.transformer, list(built.pipe.transformer.transformer_blocks)
        self.tokens = (size // 16) ** 2
        self.rows = 2 * batch * self.tokens
        self._fwd = fwd

    def forward(self, t=None):
        return self._fwd()


def main():
    a = parse()
    if a.variant:  # eval2 path: the variant string says everything (quantization, attention, compile)
        import variants as vr
        from common import load_prompt_embeds
        built = vr.build(a.variant, a.size)
        r = VariantRunner(built, vr.make_forward(built, load_prompt_embeds(a.embeds), a.size, a.batch), a.size, a.batch)
        a.mode, a.compile = "variant", False  # compile is part of the variant string
        if a.stage_ranges:
            from timing.stages import StageRecorder, install
            install(r.transformer, StageRecorder(use_nvtx=True))
    else:
        r = build(a.model, a.size, a.batch, "nvfp4" if a.mode in ("stock", "fused") else a.mode)
    tr = r.transformer
    if a.list_layers:
        for n, m in tr.named_modules():
            if isinstance(m, torch.nn.Linear):
                print(f"{n}\t{m.in_features}\t{m.out_features}")
        return

    counter = None
    if a.mode == "fused":
        widths = linear_widths(tr)
        ok_k = {widths.most_common(1)[0][0]} if a.width == "narrow" else {k for k in widths if k % 16 == 0}
        use_op = a.backend == "triton_op" or (a.compile and a.backend.startswith("cuda"))
        counter = qb.install(a.backend, ok_k, use_op=use_op, count=not a.compile)
    if a.nvtx and not a.compile and a.mode in ("stock", "fused"):
        qb.wrap_nvtx("quantize")
        def block_push(i):  # hooks must return None: a returned value replaces the block output or its args
            def hook(m, args, kwargs):
                torch.cuda.nvtx.range_push(f"blk{i}")
            return hook

        def block_pop(m, args, kwargs, out):
            torch.cuda.nvtx.range_pop()

        for i, b in enumerate(r.blocks):
            b.register_forward_pre_hook(block_push(i), with_kwargs=True)
            b.register_forward_hook(block_pop, with_kwargs=True)
    if a.compile:
        for b in r.blocks:
            b.compile(fullgraph=False)

    layer = {}
    if a.scope == "layer":
        rx = re.compile(a.layer_regex)
        name, mod = next(((n, m) for n, m in tr.named_modules() if isinstance(m, torch.nn.Linear) and rx.search(n)), (None, None))
        if mod is None:
            raise SystemExit(f"no Linear matches {a.layer_regex}; use --list-layers")
        layer.update(name=name, in_features=mod.in_features, out_features=mod.out_features, armed=False)

        def pre(m, args):
            if layer["armed"]:
                layer["input_shape"] = list(args[0].shape)
                torch.cuda.synchronize()
                torch.cuda.profiler.start()

        def post(m, args, out):
            if layer["armed"]:
                torch.cuda.synchronize()
                torch.cuda.profiler.stop()
                layer["armed"] = False

        mod.register_forward_pre_hook(pre)
        mod.register_forward_hook(post)

    res = dict(model=a.model, size=a.size, batch=a.batch, mode=a.mode, backend=a.backend if a.mode == "fused" else None,
               width=a.width if a.mode == "fused" else None, compile=a.compile, tokens=r.tokens, rows=r.rows,
               config=a.config or (a.variant or a.mode), variant=a.variant)
    with torch.no_grad():
        for _ in range(a.warmup):
            r.forward()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        if counter:
            counter["n"] = counter["fallback"] = 0
        if a.scope == "step":
            torch.cuda.profiler.start()
        gpu, cpu = [], []
        for i in range(a.iters):
            if a.scope == "layer" and i == a.iters - 1:
                layer["armed"] = True
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            s.record()
            c0 = time.perf_counter()
            r.forward()
            cpu.append((time.perf_counter() - c0) * 1e3)
            e.record()
            torch.cuda.synchronize()
            gpu.append(s.elapsed_time(e))
        if a.scope == "step":
            torch.cuda.profiler.stop()
        res["gpu_ms"] = stats(gpu)
        res["cpu_launch_ms"] = statistics.median(cpu)
        res["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 2**30
        if counter and not a.compile:
            res["fused_calls"], res["fallback_calls"] = counter["n"], counter["fallback"]

        if a.graph:
            try:
                st = torch.cuda.Stream()
                st.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(st):
                    for _ in range(3):
                        r.forward()
                torch.cuda.current_stream().wait_stream(st)
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    r.forward()
                gg = []
                for _ in range(a.iters):
                    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    s.record()
                    g.replay()
                    e.record()
                    torch.cuda.synchronize()
                    gg.append(s.elapsed_time(e))
                res["graph_ms"] = stats(gg)
            except Exception as ex:
                res["graph_error"] = f"{type(ex).__name__}: {str(ex)[:200]}"

    layer.pop("armed", None)
    if layer:
        res["layer"] = layer
    print(res)
    append_jsonl("step_target.jsonl", res)
    if a.config:
        write_json(f"meta/{a.config}_{a.size}.json", {**res, "iters": a.iters})


if __name__ == "__main__":
    main()
