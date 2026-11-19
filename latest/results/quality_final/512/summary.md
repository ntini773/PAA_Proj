Real prompts (6), 512 px, 14 steps, reference `bf16`. Mean over prompts.

| variant | ms/img | LPIPS | PSNR | SSIM | CLIP score | CLIP sim to ref |
|---|---|---|---|---|---|---|
| bf16 | 564 | 0.000±0.000 | 99.0 | 1.000±0.000 | 30.0 | 1.000±0.000 |
| stock | 1447 | 0.369±0.028 | 15.7 | 0.532±0.053 | 29.3 | 0.931±0.017 |
| w4a16 | 361 | 0.283±0.022 | 17.0 | 0.619±0.047 | 29.6 | 0.963±0.010 |
| module+be=cuda_swizzled | 256 | 0.383±0.024 | 15.4 | 0.519±0.059 | 29.4 | 0.921±0.035 |
| module+be=cuda_swizzled+static+concat | 253 | 0.419±0.044 | 15.4 | 0.503±0.034 | 28.0 | 0.895±0.061 |
| module+be=cuda_swizzled+static+concat+trim+xcudnn | 297 | 0.402±0.043 | 15.4 | 0.508±0.046 | 28.2 | 0.903±0.046 |
| module+be=cuda_swizzled+static+concat+selective+trim+xcudnn | 451 | 0.332±0.020 | 17.1 | 0.588±0.057 | 29.4 | 0.928±0.021 |
