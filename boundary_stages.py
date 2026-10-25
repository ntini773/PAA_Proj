"""The LayerNorm -> AdaLN modulate -> NVFP4 quantize boundary, stage by stage.

Same ops and shapes as the real PixArt block (x: [2*batch, N, 1152] bf16, CFG doubles the batch).
Unfused stages (what the stock model does):
  ln              F.layer_norm (one kernel)
  mod             norm*(1+scale)+shift in bf16 (3 elementwise kernels)
  amax            per-tensor scale: torch.max(torch.abs(x))       (torchao does this before every quantize)
  quant_unfused   NVFP4Tensor.to_nvfp4(per-tensor scale, swizzled) (bf16->fp32 copy, ~20 elementwise ops, scale swizzle)
Fused variants:
  quant_fused     our fused block-quant kernel + torchao's scale swizzle  (what the model run uses)
  full_fused      ONE kernel: LN + modulate + block-quant   (idealized: single-level scale, no swizzle)

  --mode time : CUDA-event median per stage and per chain, plus a byte-equality check
  --mode ncu --stage NAME : run one stage once inside NVTX range STAGE_NAME (for boundary_ncu.sh)
"""
import os, sys, json, argparse, statistics
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import torch
import torch.nn.functional as F
from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor, per_tensor_amax_to_scale
from torchao.prototype.mx_formats.utils import to_blocked
from fused_adaln_nvfp4 import fused_nvfp4_quantize, fused_ln_adaln_nvfp4

H, EPS = 1152, 1e-6
STAGES = ["ln", "mod", "amax", "quant_unfused", "quant_fused", "full_fused", "quant_mslk"]


def build(size, batch):
    B2, N = 2 * batch, (size // 16) ** 2
    g = torch.Generator("cuda").manual_seed(0)
    bf = torch.bfloat16
    x = torch.randn(B2, N, H, device="cuda", dtype=bf, generator=g)
    scale = (torch.randn(B2, 1, H, device="cuda", dtype=bf, generator=g) * 0.1)
    shift = (torch.randn(B2, 1, H, device="cuda", dtype=bf, generator=g) * 0.1)
    norm = F.layer_norm(x, (H,), eps=EPS)
    xm = norm * (1 + scale) + shift
    pts = per_tensor_amax_to_scale(torch.max(torch.abs(xm)))
    xm2 = xm.view(-1, H).contiguous()
    x2 = x.view(-1, H).contiguous()
    sc_rows = scale.expand(B2, N, H).reshape(-1, H).contiguous()
    sh_rows = shift.expand(B2, N, H).reshape(-1, H).contiguous()

    def quant_unfused(m2=xm2, p=pts):
        return NVFP4Tensor.to_nvfp4(m2, block_size=16, per_tensor_scale=p,
                                    is_swizzled_scales=True, use_triton_kernel=False)

    def quant_fused(m2=xm2, p=pts):
        bs, dl = fused_nvfp4_quantize(m2, p)
        return to_blocked(bs.view(-1, H // 16)).flatten(), dl

    fns = {
        "ln": lambda: F.layer_norm(x, (H,), eps=EPS),
        "mod": lambda: norm * (1 + scale) + shift,
        "amax": lambda: per_tensor_amax_to_scale(torch.max(torch.abs(xm))),
        "quant_unfused": quant_unfused,
        "quant_fused": quant_fused,
        "full_fused": lambda: fused_ln_adaln_nvfp4(x2, sc_rows, sh_rows, EPS),
    }

    try:                       # torchao's own optimized quantize kernel (needs the `mslk` package)
        NVFP4Tensor.to_nvfp4(xm2, block_size=16, per_tensor_scale=pts, is_swizzled_scales=True, use_triton_kernel=True)
        fns["quant_mslk"] = lambda: NVFP4Tensor.to_nvfp4(xm2, block_size=16, per_tensor_scale=pts,
                                                         is_swizzled_scales=True, use_triton_kernel=True)
    except Exception as ex:
        print(f"quant_mslk not available: {type(ex).__name__}: {str(ex)[:100]}", file=sys.stderr)

    def chain(q):
        n = F.layer_norm(x, (H,), eps=EPS)
        m = n * (1 + scale) + shift
        p = per_tensor_amax_to_scale(torch.max(torch.abs(m)))
        return q(m.view(-1, H), p)
    chains = {"chain_unfused": lambda: chain(quant_unfused),
              "chain_fused_quant": lambda: chain(quant_fused),
              "chain_full_fused": fns["full_fused"]}
    return fns, chains, dict(rows=B2 * N, B2=B2, N=N)


def gpu_ms(fn, iters=100, warm=20):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    s = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    e = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        s[i].record(); fn(); e[i].record()
    torch.cuda.synchronize()
    return statistics.median(a.elapsed_time(b) for a, b in zip(s, e))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--hidden", type=int, default=1152, help="other DiT widths: 1536 (SD3.5-M), 2432 (SD3.5-L), 3072 (Flux)")
    p.add_argument("--mode", choices=["time", "ncu"], default="time")
    p.add_argument("--stage", choices=STAGES)
    a = p.parse_args()
    global H
    H = a.hidden
    fns, chains, info = build(a.size, a.batch)

    if a.mode == "ncu":
        if a.stage not in fns:
            sys.exit(f"stage {a.stage} not available")
        fn = fns[a.stage]
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_push(f"STAGE_{a.stage}")
        fn()
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
        return

    res = dict(size=a.size, batch=a.batch, hidden=H, **info)
    res["stage_ms"] = {k: round(gpu_ms(f), 4) for k, f in fns.items()}
    res["chain_ms"] = {k: round(gpu_ms(f), 4) for k, f in chains.items()}
    try:                                   # fused quant must equal torchao's quantize bit for bit
        q = fns["quant_unfused"]()
        bs, dl = fns["quant_fused"]()
        res["diff_packed_bytes"] = int((q.qdata != dl).sum())
        res["diff_scale_bytes"] = int((q.scale.flatten().view(torch.uint8) != bs.view(torch.uint8)).sum())
    except Exception as ex:
        res["diff_error"] = f"{type(ex).__name__}: {str(ex)[:120]}"
    print(json.dumps(res))
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    name = "boundary_time.jsonl" if H == 1152 else "boundary_time_hidden.jsonl"
    with open(os.path.join(HERE, "results", name), "a") as f:
        f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
