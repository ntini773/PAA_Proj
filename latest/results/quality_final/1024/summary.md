Real prompts (6), 1024 px, 14 steps, reference `bf16`. Mean over prompts.

| variant | ms/img | LPIPS | PSNR | SSIM | CLIP score | CLIP sim to ref |
|---|---|---|---|---|---|---|
| bf16 | 1752 | 0.000±0.000 | 99.0 | 1.000±0.000 | 28.9 | 1.000±0.000 |
| stock | 7858 | 0.564±0.045 | 14.2 | 0.426±0.077 | 30.1 | 0.918±0.017 |
| w4a16 | 1811 | 0.419±0.060 | 15.6 | 0.528±0.070 | 29.7 | 0.942±0.035 |
| module+be=cuda_swizzled | 1563 | 0.551±0.064 | 14.2 | 0.439±0.085 | 30.2 | 0.910±0.029 |
| module+be=cuda_swizzled+static+concat | 1409 | 0.553±0.062 | 14.1 | 0.436±0.064 | 30.3 | 0.902±0.058 |
| module+be=cuda_swizzled+static+concat+trim+xcudnn | 1307 | 0.547±0.057 | 14.3 | 0.436±0.075 | 30.0 | 0.895±0.056 |
| module+be=cuda_swizzled+static+concat+selective+trim+xcudnn | 1316 | 0.519±0.070 | 15.7 | 0.490±0.090 | 28.5 | 0.907±0.057 |
