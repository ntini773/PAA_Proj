Ledger for PixArt-Sigma 1024px, transformer batch 2, bandwidth 1113 GB/s.
Quantize calls per forward: current 280, shared Q/K/V and K/V quantize 196.

| item | GB | ms at bandwidth |
|---|---|---|
| quantize path now (amax read + quantize read + codes/scales write) | 13.44 | 12.08 |
| output scale + bias passes, eager torchao only (inductor already folds them into the next op when compiled) | 23.56 | 21.18 |
| GEMM operand and result traffic (not removable) | 7.57 | 6.81 |
| floor: one-pass quantize, each input read once | 6.14 | 5.52 |

| lever | traffic removed GB | ms saved |
|---|---|---|
| static activation scale (no amax pass) | 5.89 | 5.29 |
| share the Q/K/V and K/V quantize | 2.50 | 2.25 |
| fold output scale and bias into the next op (eager path; already true when compiled) | 23.56 | 21.18 |
| LN+modulate fused into quantize (qkv, attn2 q, fc1) | 3.17 | 2.85 |
| GELU+quantize in the fc1 GEMM epilogue (draft, needs a CUTLASS epilogue) | 4.23 | 3.80 |

Measured quantize-path DRAM per layer from ncu is in ncu_layers.csv (dram_vs_ideal column).
