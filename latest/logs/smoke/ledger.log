Ledger for PixArt-Sigma 512px, transformer batch 2, bandwidth 1113 GB/s.
Quantize calls per forward: current 280, shared Q/K/V and K/V quantize 196.

| item | GB | ms at bandwidth |
|---|---|---|
| quantize path now (amax read + quantize read + codes/scales write) | 3.49 | 3.14 |
| output scale + bias passes, eager torchao only (inductor already folds them into the next op when compiled) | 6.12 | 5.50 |
| GEMM operand and result traffic (not removable) | 2.21 | 1.99 |
| floor: one-pass quantize, each input read once | 1.57 | 1.41 |

| lever | traffic removed GB | ms saved |
|---|---|---|
| static activation scale (no amax pass) | 1.53 | 1.38 |
| share the Q/K/V and K/V quantize | 0.69 | 0.62 |
| fold output scale and bias into the next op (eager path; already true when compiled) | 6.12 | 5.50 |
| LN+modulate fused into quantize (qkv, attn2 q, fc1) | 0.79 | 0.71 |
| GELU+quantize in the fc1 GEMM epilogue (draft, needs a CUTLASS epilogue) | 1.06 | 0.95 |

Measured quantize-path DRAM per layer from ncu is in ncu_layers.csv (dram_vs_ideal column).
