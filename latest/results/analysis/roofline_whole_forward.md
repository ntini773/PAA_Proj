Roofs: DRAM 1113 GB/s, bf16 206, fp4 707.5073130127788, attention 155 TFLOP/s

### 256 px
| config | eager ms | graph ms | SoL ms | eff eager | eff graph | TFLOP/s eager |
|---|---|---|---|---|---|---|
| bf16 | 9.02 | 7.72 | 3.27 | 0.363 | 0.424 | 73.01 |
| free_bound | 29.86 | 9.95 | 1.13 | 0.038 | 0.113 | 22.06 |
| fused_cuda_row | 36.33 | 11.84 | 1.63 | 0.045 | 0.138 | 18.13 |
| fused_cuda_swizzled | 32.99 | 10.55 | 1.63 | 0.05 | 0.155 | 19.97 |
| fused_cuda_swizzled_compile | 13.13 | 6.55 | 1.63 | 0.124 | 0.249 | 50.16 |
| fused_triton | 45.1 | 11.89 | 1.63 | 0.036 | 0.137 | 14.6 |
| fused_triton_narrow | 57.09 | 15.68 | 1.63 | 0.029 | 0.104 | 11.54 |
| fused_triton_op_compile | 14.8 | 6.58 | 1.63 | 0.11 | 0.248 | 44.5 |
| stock | 98.33 | 30.73 | 1.63 | 0.017 | 0.053 | 6.7 |
| stock_compile | 16.03 | 6.96 | 1.63 | 0.102 | 0.235 | 41.08 |

### 512 px
| config | eager ms | graph ms | SoL ms | eff eager | eff graph | TFLOP/s eager |
|---|---|---|---|---|---|---|
| bf16 | 23.73 | 23.52 | 13.06 | 0.55 | 0.555 | 108.31 |
| free_bound | 29.5 | 19.49 | 5.4 | 0.183 | 0.277 | 87.13 |
| fused_cuda_row | 36.44 | 22.52 | 7.17 | 0.197 | 0.318 | 70.52 |
| fused_cuda_swizzled | 30.04 | 20.86 | 7.17 | 0.239 | 0.344 | 85.54 |
| fused_cuda_swizzled_compile | 15.85 | 14.72 | 7.17 | 0.452 | 0.487 | 162.18 |
| fused_triton | 47.94 | 23.04 | 7.17 | 0.149 | 0.311 | 53.6 |
| fused_triton_narrow | 54.27 | 39.0 | 7.17 | 0.132 | 0.184 | 47.36 |
| fused_triton_op_compile | 16.18 | 14.74 | 7.17 | 0.443 | 0.486 | 158.85 |
| stock | 98.79 | 67.65 | 7.17 | 0.073 | 0.106 | 26.01 |
| stock_compile | 17.18 | 15.24 | 7.17 | 0.417 | 0.47 | 149.59 |

### 1024 px
| config | eager ms | graph ms | SoL ms | eff eager | eff graph | TFLOP/s eager |
|---|---|---|---|---|---|---|
| bf16 | 103.68 | 103.69 | 71.94 | 0.694 | 0.694 | 127.88 |
| free_bound | 87.05 | 86.36 | 42.25 | 0.485 | 0.489 | 152.31 |
| fused_cuda_row | 99.32 | 99.29 | 49.03 | 0.494 | 0.494 | 133.5 |
| fused_cuda_swizzled | 96.77 | 97.02 | 49.03 | 0.507 | 0.505 | 137.01 |
| fused_cuda_swizzled_compile | 75.37 | 74.81 | 49.03 | 0.651 | 0.655 | 175.92 |
| fused_triton | 102.23 | 102.09 | 49.03 | 0.48 | 0.48 | 129.69 |
| fused_triton_narrow | 251.2 | 249.55 | 49.03 | 0.195 | 0.196 | 52.78 |
| fused_triton_op_compile | 75.57 | 74.83 | 49.03 | 0.649 | 0.655 | 175.46 |
| stock | 370.48 | 369.73 | 49.03 | 0.132 | 0.133 | 35.79 |
| stock_compile | 77.24 | 76.4 | 49.03 | 0.635 | 0.642 | 171.66 |
