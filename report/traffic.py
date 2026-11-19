"""Quantize-traffic numbers for one forward, from the repo's own ledger code (rtx5000_next/analysis/traffic_ledger.py), so the report
and the scripts cannot disagree. Bytes are per forward at batch 2 (classifier-free guidance with one image)."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "rtx5000_next" / "analysis"))
import traffic_ledger as tl  # noqa: E402


def sol_ms(size, mode, peaks, batch=2):
    """Model-based lower bound for one forward: sum over GEMM, attention and quantize of max(compute time, memory time) at the
    measured peaks. Optimistic: peaks come from large benchmark shapes, and LayerNorm / GELU / residual traffic is not modelled."""
    import roofline_pixart as rp
    m = rp.model(size, batch, 0)
    bw = peaks["copy_gbs"] * 1e9
    bf16 = mode == "bf16"
    gpeak = (peaks["bf16_tflops"] if bf16 else peaks["fp4_tflops"]) * 1e12
    gbytes = m["gemm_bytes_bf16"] if bf16 else m["gemm_bytes_fp4"]
    t = max(m["gemm_flops"] / gpeak, gbytes / bw)
    t += max(m["attn_flops"] / (peaks["attn_tflops"] * 1e12), m["attn_bytes"] / bw)
    t += 0 if bf16 else m["quant_bytes"] / bw
    return t * 1e3


def numbers(size, batch=2):
    cur, quant_now, epi_now, quant_floor, static, shared_saved, ln_fuse, fc1_epi, qc = tl.lever_rows(size, batch)
    g = 1e9
    return dict(now=quant_now / g, floor=quant_floor / g, static=static / g, shared=shared_saved / g, ln=ln_fuse / g, epi=epi_now / g,
                fc1_epi=fc1_epi / g, gemm=cur["gemm_traffic"] / g, calls_now=qc["current"], calls_shared=qc["shared"])
