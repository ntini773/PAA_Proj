Real prompts (6), 256 px, 14 steps, reference `bf16`. Mean over prompts.

| variant | ms/img | LPIPS | PSNR | SSIM | CLIP score | CLIP sim to ref |
|---|---|---|---|---|---|---|
| bf16 | 90 | 0.000±0.000 | 99.0 | 1.000±0.000 | 28.9 | 1.000±0.000 |
| stock | 298 | 0.377±0.043 | 15.1 | 0.453±0.091 | 27.9 | 0.865±0.051 |
| w4a16 | 352 | 0.283±0.045 | 16.5 | 0.551±0.056 | 29.1 | 0.961±0.009 |
| module+be=cuda_swizzled | 94 | 0.353±0.040 | 15.5 | 0.465±0.101 | 28.4 | 0.875±0.031 |
| module+be=cuda_swizzled+static+concat | 60 | 0.414±0.057 | 13.6 | 0.414±0.109 | 27.7 | 0.857±0.057 |
| module+be=cuda_swizzled+static+concat+trim+xcudnn | 102 | 0.414±0.077 | 13.7 | 0.412±0.111 | 27.4 | 0.847±0.060 |
| module+be=cuda_swizzled+static+concat+selective+trim+xcudnn | 58 | 0.326±0.063 | 16.6 | 0.521±0.079 | 29.1 | 0.891±0.031 |
