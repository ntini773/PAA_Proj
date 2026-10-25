# Where the time goes in one PixArt-Sigma denoise step (measured)

All numbers: RTX PRO 5000, clean GPU, 2026-10-25, one denoise step (the transformer forward, batch 1 with guidance),
from `breakdown_step.py` (`results/breakdown.jsonl`). Percentages are shares of the whole step.

- **Quantize** = converting the bf16 input of a layer to 4-bit FP4 (the part we fuse).
- **Overhead** = the rest of the quantization work around it (per-tensor max and similar).
- **GEMM** = the FP4 matrix multiply itself.

Colors: red = above 20% of the step, orange = 5% to 20%, green = below 5%.

## 1. NVFP4 stock, 1024px (step 373 ms)

```mermaid
flowchart TB
  IN["Latent image + timestep + text<br/>patch embed and caption layers: 0.4%"] --> BLK
  subgraph BLK ["One of 28 identical blocks (shares are summed over all 28)"]
    direction TB
    N1["LayerNorm + modulate (spot 1)<br/>LN 0.2% · modulate in 'rest'"]
    QKV["Self-attention Q, K, V layers<br/>16.3% of step<br/>quantize 13.7% · overhead 1.7% · GEMM 0.9%"]
    SDPA["Attention core (scores, softmax)<br/>10.3%"]
    SO["Self-attention output layer<br/>5.3%<br/>quantize 4.5%"]
    CQ["Cross-attention Q layer<br/>5.4%<br/>quantize 4.6%"]
    CKV["Cross-attention K, V layers (text)<br/>1.8%"]
    CO["Cross-attention output layer<br/>5.4%<br/>quantize 4.6%"]
    N2["LayerNorm + modulate (spot 2)<br/>LN 0.2%"]
    F1["MLP first layer, 1152 to 4608<br/>7.7%<br/>quantize 4.5% · overhead 2.2% · GEMM 1.0%"]
    F2["MLP SECOND layer, 4608 wide<br/>44.3%<br/>quantize 42.0% · overhead 1.4% · GEMM 0.8%"]
    N1 --> QKV --> SDPA --> SO --> CQ --> CKV --> CO --> N2 --> F1 --> F2
  end
  BLK --> OUT["final LayerNorm + output layer<br/>0.1%"]
  classDef red fill:#f8b4b4,stroke:#b91c1c,color:#000
  classDef orange fill:#fde1b0,stroke:#c2410c,color:#000
  classDef green fill:#cdeccd,stroke:#166534,color:#000
  class F2 red
  class QKV,SDPA,SO,CQ,CO,F1 orange
  class N1,N2,CKV,IN,OUT green
```

Where the 373 ms goes, by kind of work:

```mermaid
pie showData title Stock NVFP4, 1024px, step 373 ms
  "Quantize kernel (all layers)" : 281.5
  "Attention core" : 38.4
  "Quantize overhead" : 27.3
  "FP4 matmul (GEMM)" : 14.0
  "Modulate, GELU, copies" : 10.2
  "LayerNorm" : 1.6
```

## 2. NVFP4 fused (1152-wide layers only), 1024px (step 254 ms)

The fused quantize kernel took the first eight layer types from 16 to 51 ms down to under 3 ms each.
The MLP second layer was not covered, so it is now the whole story.

```mermaid
flowchart TB
  subgraph BLK2 ["One of 28 identical blocks (shares of the 254 ms step)"]
    direction TB
    A1["Self-attention Q, K, V<br/>4.5%<br/>quantize 1.0%"] --> A2["Attention core<br/>15.0%"]
    A2 --> A3["Self-attention output<br/>1.5%"] --> A4["Cross-attention Q<br/>1.6%"] --> A5["Cross-attention K, V<br/>1.0%"] --> A6["Cross-attention output<br/>1.5%"]
    A6 --> A7["MLP first layer<br/>5.1%<br/>overhead 3.2%"]
    A7 --> A8["MLP SECOND layer, 4608 wide, NOT fused<br/>65.2%<br/>quantize 61.8%"]
  end
  classDef red fill:#f8b4b4,stroke:#b91c1c,color:#000
  classDef orange fill:#fde1b0,stroke:#c2410c,color:#000
  classDef green fill:#cdeccd,stroke:#166534,color:#000
  class A8 red
  class A2,A7 orange
  class A1,A3,A4,A5,A6 green
```

```mermaid
pie showData title Fused (1152 wide), 1024px, step 254 ms
  "Quantize kernel (MLP second layer, unfused)" : 156.8
  "Attention core" : 38.1
  "Quantize overhead" : 26.5
  "FP4 matmul (GEMM)" : 13.8
  "Modulate, GELU, copies" : 10.1
  "Quantize kernel (all fused layers)" : 6.9
  "LayerNorm" : 1.5
```

## 3. After also fusing the MLP second layer (`fused_wide`), 1024px: step 98.7 ms

Only the step time was measured (bf16 is 103.1 ms). The split below is an **estimate**, made by subtracting the measured
parts from the earlier run. It has not been measured. Run `bash run_breakdown_only.sh` (now includes `fused_wide`) to measure it.

```mermaid
pie showData title ESTIMATE: fused + MLP second layer, 1024px, step 98.7 ms
  "Attention core" : 38
  "Quantize overhead (per-tensor max etc.)" : 27
  "FP4 matmul (GEMM)" : 14
  "Modulate, GELU, copies, LayerNorm" : 12
  "Quantize kernel (all layers, fused)" : 8
```

## 4. The same split at all image sizes (measured)

| Image | Mode | Step ms | Quantize | Overhead | GEMM | Attention core | LayerNorm | Rest |
|---|---|---|---|---|---|---|---|---|
| 256px | bf16 | 15.2 | 0% | 0% | 50% | 17% | 5% | 29% |
| 256px | stock | 118.8 | **57%** | 23% | 8% | 2% | 1% | 8% |
| 256px | fused | 72.0 | 33% | 36% | 12% | 4% | 2% | 14% |
| 512px | bf16 | 24.5 | 0% | 0% | 64% | 22% | 2% | 12% |
| 512px | stock | 119.5 | **61%** | 18% | 9% | 5% | 1% | 6% |
| 512px | fused | 72.0 | 40% | 27% | 13% | 9% | 1% | 10% |
| 1024px | bf16 | 104.4 | 0% | 0% | 48% | 40% | 2% | 10% |
| 1024px | stock | 373.0 | **76%** | 7% | 4% | 10% | 0% | 3% |
| 1024px | fused | 253.7 | 64% | 10% | 5% | 15% | 1% | 4% |

("Rest" includes the modulate step, GELU, residual adds and copies.)

## 5. Takeaways

1. Quantization is 57% to 76% of the stock step, plus 7% to 23% overhead. The FP4 matmul is only 4% to 9%.
2. LayerNorm plus modulate cost 1% to 3% of the step. They are not where the unfused cost sits.
3. At 1024px the MLP second layer alone is 44% of the stock step and 65% of the 1152-only fused step. Fusing it took the step from 250.8 ms to 98.7 ms.
4. At 256px and 512px the step is limited by the CPU launching kernels (CUDA Graph gave 3.2x and 1.3x), so percentages there include idle time.
