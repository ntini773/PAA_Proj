"""DRAM-traffic ledger for one PixArt-Sigma forward: where the bytes go and what each lever removes. No GPU needed.

    python analysis/traffic_ledger.py [--size 1024] [--batch 2] [--bw 1113]

Why: the quantize path and the glue around the GEMMs are memory-bound, so their time is bytes / bandwidth. Listing the bytes per
site for the current path and for each lever shows which fusion is worth building before any kernel is written.
Counting rule: ideal traffic of one pass = each tensor read or written once at its true size (L2 reuse ignored). "Current" is
torchao's compiled path as read from its kernels: amax pass over x, quantize pass over x writing FP4 codes and scales, then the
GEMM, then a separate output-scale pass and bias pass over y. Levers change which of these exist.
Inputs  nothing is read from disk except an optional results/analysis/ncu_layers.csv to print measured DRAM next to the model.
Output  results/analysis/traffic_ledger.md and a table on stdout. Bytes are per forward; ms = bytes / --bw (GB/s, default = measured copy bandwidth).
"""
import argparse
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
D, FF, LAYERS, TEXT = 1152, 4608, 28, 300
BF, FP4 = 2.0, 0.5 + 1 / 16  # bytes per element: bf16, packed FP4 + one e4m3 scale per 16


def sites(rows, text_rows):
    """(name, M, K, N, group) of every linear in one block. group marks inputs shared by several linears."""
    return [("qkv", rows, D, 3 * D, "shared3"), ("attn1_out", rows, D, D, ""), ("attn2_q", rows, D, D, ""),
            ("attn2_kv", text_rows, D, 2 * D, "shared2"), ("attn2_out", rows, D, D, ""), ("fc1", rows, D, FF, ""), ("fc2", rows, FF, D, "")]


def ledger(size, batch):
    n = (size // 16) ** 2
    rows, trows = batch * n, batch * TEXT
    tot = dict(zip(("x_read_amax", "x_read_quant", "codes_scales_write", "y_scale_pass", "y_bias_pass", "gemm_traffic"), [0.0] * 6))
    quant_calls = {"current": 0, "shared": 0}
    for name, M, K, N, grp in sites(rows, trows):
        reps = {"shared3": 3, "shared2": 2}.get(grp, 1)  # current path quantizes the same input once per linear
        for _ in range(LAYERS):
            quant_calls["current"] += reps
            quant_calls["shared"] += 1
            tot["x_read_amax"] += reps * M * K * BF
            tot["x_read_quant"] += reps * M * K * BF
            tot["codes_scales_write"] += reps * M * K * FP4
            tot["y_scale_pass"] += 2 * M * N * BF
            tot["y_bias_pass"] += 2 * M * N * BF
            tot["gemm_traffic"] += (M * K + N * K) * FP4 + M * N * BF  # read FP4 x and w, write bf16 y
    return tot, quant_calls, rows, trows


def lever_rows(size, batch):
    n = (size // 16) ** 2
    rows, trows = batch * n, batch * TEXT
    cur, qc, _, _ = ledger(size, batch)
    quant_now = cur["x_read_amax"] + cur["x_read_quant"] + cur["codes_scales_write"]
    epi_now = cur["y_scale_pass"] + cur["y_bias_pass"]
    # one-pass quantize with a static scale: read x once, write codes and scales once
    quant_floor = sum(LAYERS * M * K * (BF + FP4) for _, M, K, _, _ in sites(rows, trows))  # shared inputs counted once
    static = cur["x_read_amax"] + 0
    shared_saved = 0.0
    for name, M, K, N, grp in sites(rows, trows):
        reps = {"shared3": 3, "shared2": 2}.get(grp, 1)
        shared_saved += LAYERS * (reps - 1) * M * K * (BF + BF + FP4)
    shared_epi = 0.0  # y passes folded by inductor into the next op: remove both extra passes
    ln_fuse = 0.0  # LN+modulate fused into the quantize kernel: the normalized bf16 tensor is not written and re-read
    for name, M, K, N, grp in sites(rows, trows):
        if name in ("qkv", "fc1", "attn2_q"):
            ln_fuse += LAYERS * 2 * M * K * BF
    fc1_epilogue = LAYERS * 2 * rows * FF * BF  # GELU+quantize in the fc1 GEMM epilogue: fc1 output never written as bf16
    return cur, quant_now, epi_now, quant_floor, static, shared_saved, ln_fuse, fc1_epilogue, qc


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--size", type=int, default=1024)
    a.add_argument("--batch", type=int, default=2, help="transformer batch (2 = CFG with one image)")
    a.add_argument("--bw", type=float, default=None, help="GB/s; default: results/peaks.json copy_gbs or 1113")
    a = a.parse_args()
    bw = a.bw
    peaks = ROOT / "results" / "peaks.json"
    if bw is None:
        bw = json.loads(peaks.read_text())["copy_gbs"] if peaks.exists() else 1113.0
    cur, quant_now, epi_now, quant_floor, static, shared_saved, ln_fuse, fc1_epi, qc = lever_rows(a.size, a.batch)
    gb = lambda x: x / 1e9
    ms = lambda x: x / (bw * 1e9) * 1e3
    lines = [f"Ledger for PixArt-Sigma {a.size}px, transformer batch {a.batch}, bandwidth {bw:.0f} GB/s.",
             f"Quantize calls per forward: current {qc['current']}, shared Q/K/V and K/V quantize {qc['shared']}.", "",
             "| item | GB | ms at bandwidth |", "|---|---|---|",
             f"| quantize path now (amax read + quantize read + codes/scales write) | {gb(quant_now):.2f} | {ms(quant_now):.2f} |",
             f"| output scale + bias passes, eager torchao only (inductor already folds them into the next op when compiled) | {gb(epi_now):.2f} | {ms(epi_now):.2f} |",
             f"| GEMM operand and result traffic (not removable) | {gb(cur['gemm_traffic']):.2f} | {ms(cur['gemm_traffic']):.2f} |",
             f"| floor: one-pass quantize, each input read once | {gb(quant_floor):.2f} | {ms(quant_floor):.2f} |", "",
             "| lever | traffic removed GB | ms saved |", "|---|---|---|",
             f"| static activation scale (no amax pass) | {gb(static):.2f} | {ms(static):.2f} |",
             f"| share the Q/K/V and K/V quantize | {gb(shared_saved):.2f} | {ms(shared_saved):.2f} |",
             f"| fold output scale and bias into the next op (eager path; already true when compiled) | {gb(epi_now):.2f} | {ms(epi_now):.2f} |",
             f"| LN+modulate fused into quantize (qkv, attn2 q, fc1) | {gb(ln_fuse):.2f} | {ms(ln_fuse):.2f} |",
             f"| GELU+quantize in the fc1 GEMM epilogue (draft, needs a CUTLASS epilogue) | {gb(fc1_epi):.2f} | {ms(fc1_epi):.2f} |"]
    ncu = ROOT / "results" / "analysis" / "ncu_layers.csv"
    if ncu.exists():
        lines += ["", f"Measured quantize-path DRAM per layer from ncu is in {ncu.name} (dram_vs_ideal column)."]
    text = "\n".join(lines)
    out = ROOT / "results" / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    (out / "traffic_ledger.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
