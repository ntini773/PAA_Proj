"""CUDA-Graph replay of the whole PixArt-Sigma transformer forward (1 denoise step).

Inspired by NVlabs/GraspGenX#7, whose big win is replaying the full forward as
one CUDA Graph with persistent buffers. Here we measure, for the REAL model:

  eager  : normal Python-launched forward           (GPU ms + CPU launch ms)
  graph  : the same forward replayed as one graph    (GPU ms)

for --mode bf16 | stock (torchao NVFP4) | fused (NVFP4 + our fused quant).
If cpu_launch_ms > gpu_ms in eager, that step is launch-bound and graphs help.
Input is a random latent / synthetic text embedding (speed only, no image).

    pixi run python external_tests_inspired/graph_step_bench.py --size 256 --mode fused
Results are appended to results/graph_step.jsonl
"""
import os, sys, json, time, argparse, statistics
os.environ.setdefault("HF_HUB_OFFLINE", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import torch
from diffusers import PixArtTransformer2DModel, TorchAoConfig
from torchao.prototype.mx_formats.inference_workflow import NVFP4DynamicActivationNVFP4WeightConfig
from scale_e2e import REPO
from tier2_end_to_end import install_fused_quant


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--mode", choices=["bf16", "stock", "fused", "fused_wide"], required=True)
    p.add_argument("--batch", type=int, default=1, help="images; CFG doubles it inside")
    p.add_argument("--iters", type=int, default=50)
    a = p.parse_args()

    kw = {}
    if a.mode != "bf16":
        kw["quantization_config"] = TorchAoConfig(NVFP4DynamicActivationNVFP4WeightConfig(
            use_dynamic_per_tensor_scale=True, use_triton_kernel=False))
    tr = PixArtTransformer2DModel.from_pretrained(
        REPO[a.size], subfolder="transformer", torch_dtype=torch.bfloat16, **kw).to("cuda").eval()
    if a.mode == "fused_wide":
        os.environ["EXTRA_OK_K"] = "4608,4096"
    counter = install_fused_quant() if a.mode in ("fused", "fused_wide") else None

    B2 = 2 * a.batch
    lat = a.size // 8
    g = torch.Generator("cuda").manual_seed(0)
    x = torch.randn(B2, 4, lat, lat, device="cuda", dtype=torch.bfloat16, generator=g)
    txt = torch.randn(B2, 300, 4096, device="cuda", dtype=torch.bfloat16, generator=g) * 0.5
    msk = torch.ones(B2, 300, dtype=torch.long, device="cuda")
    t = torch.full((B2,), 500.0, device="cuda")

    def fwd():
        return tr(x, encoder_hidden_states=txt, timestep=t, encoder_attention_mask=msk,
                  added_cond_kwargs={"resolution": None, "aspect_ratio": None},
                  return_dict=False)[0]

    res = dict(mode=a.mode, size=a.size, batch=a.batch, tokens=(a.size // 16) ** 2)
    with torch.no_grad():
        for _ in range(5): out_eager = fwd()          # warmup: Triton JIT, allocator
        torch.cuda.synchronize()

        # ---- eager: GPU time (events) and CPU launch time (no sync) ----
        gpu, cpu = [], []
        for _ in range(a.iters):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            s.record(); c0 = time.perf_counter()
            fwd()
            cpu.append((time.perf_counter() - c0) * 1e3)
            e.record(); torch.cuda.synchronize()
            gpu.append(s.elapsed_time(e))
        res["eager_gpu_ms"] = statistics.median(gpu)
        res["eager_cpu_launch_ms"] = statistics.median(cpu)

        # ---- graph: capture once, replay ----
        try:
            st = torch.cuda.Stream()
            st.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(st):
                for _ in range(3): fwd()
            torch.cuda.current_stream().wait_stream(st)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out_graph = fwd()
            gg = []
            for _ in range(a.iters):
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s.record(); graph.replay(); e.record(); torch.cuda.synchronize()
                gg.append(s.elapsed_time(e))
            res["graph_gpu_ms"] = statistics.median(gg)
            res["graph_speedup"] = res["eager_gpu_ms"] / res["graph_gpu_ms"]
            res["graph_matches_eager"] = bool(torch.equal(out_graph, out_eager))
        except Exception as ex:                       # report instead of crashing the sweep
            res["graph_error"] = f"{type(ex).__name__}: {str(ex)[:200]}"
    if counter: res["fused_calls_total"] = counter["n"]; res["fallback_calls_total"] = counter["fallback"]
    print(json.dumps(res))
    os.makedirs(os.path.join(os.path.dirname(HERE), "results"), exist_ok=True)
    with open(os.path.join(os.path.dirname(HERE), "results", "graph_step.jsonl"), "a") as f:
        f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
