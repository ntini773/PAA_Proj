
### 256 px

| setup | step ms | speed-up vs bf16+compile | LPIPS to bf16 | LPIPS to stock NVFP4 | CLIP score |
|---|---|---|---|---|---|
| bf16 compiled | 7.1 | 1.00x | 0.000 | 0.377 | 28.9 |
| fp8 compiled | 5.4 | 1.32x | - | - | - |
| stock NVFP4 compiled | 7.0 | 1.02x | 0.377 | 0.000 | 27.9 |
| NVFP4 weights only (w4a16) | - | - | 0.283 | 0.328 | 29.1 |
| module stack (CUDA kernel, compiled) | 5.7 | 1.24x | 0.353 | 0.326 | 28.4 |
| + static scale + concat | 4.8 | 1.49x | 0.414 | 0.408 | 27.7 |
| + trim + cuDNN cross-attn | 3.7 | 1.91x | 0.414 | 0.420 | 27.4 |
| + selective quantization | 4.0 | 1.77x | 0.326 | 0.327 | 29.1 |

### 512 px

| setup | step ms | speed-up vs bf16+compile | LPIPS to bf16 | LPIPS to stock NVFP4 | CLIP score |
|---|---|---|---|---|---|
| bf16 compiled | 22.0 | 1.00x | 0.000 | 0.369 | 30.0 |
| fp8 compiled | 15.0 | 1.47x | - | - | - |
| stock NVFP4 compiled | 15.2 | 1.45x | 0.369 | 0.000 | 29.3 |
| NVFP4 weights only (w4a16) | - | - | 0.283 | 0.316 | 29.6 |
| module stack (CUDA kernel, compiled) | 12.8 | 1.72x | 0.383 | 0.302 | 29.4 |
| + static scale + concat | 11.6 | 1.90x | 0.419 | 0.392 | 28.0 |
| + trim + cuDNN cross-attn | 9.2 | 2.38x | 0.402 | 0.385 | 28.2 |
| + selective quantization | 9.5 | 2.31x | 0.332 | 0.350 | 29.4 |

### 1024 px

| setup | step ms | speed-up vs bf16+compile | LPIPS to bf16 | LPIPS to stock NVFP4 | CLIP score |
|---|---|---|---|---|---|
| bf16 compiled | 100.1 | 1.00x | 0.000 | 0.564 | 28.9 |
| fp8 compiled | 77.2 | 1.30x | - | - | - |
| stock NVFP4 compiled | 75.7 | 1.32x | 0.564 | 0.000 | 30.1 |
| NVFP4 weights only (w4a16) | - | - | 0.419 | 0.472 | 29.7 |
| module stack (CUDA kernel, compiled) | 66.1 | 1.51x | 0.551 | 0.420 | 30.2 |
| + static scale + concat | 65.3 | 1.53x | 0.553 | 0.497 | 30.3 |
| + trim + cuDNN cross-attn | 57.7 | 1.73x | 0.547 | 0.493 | 30.0 |
| + selective quantization | 58.4 | 1.71x | 0.519 | 0.566 | 28.5 |
