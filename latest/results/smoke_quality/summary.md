Real prompts (2), 256 px, 4 steps, reference `bf16`. Mean over prompts.

| variant | ms/img | LPIPS | PSNR | SSIM | CLIP score | CLIP sim to ref |
|---|---|---|---|---|---|---|
| bf16 | 34 | 0.000±0.000 | 99.0 | 1.000±0.000 | - | 1.000±0.000 |
| stock | 293 | 0.753±0.050 | 7.8 | 0.140±0.000 | - | 0.973±0.006 |
| module+be=triton | 108 | 0.754±0.038 | 7.4 | 0.086±0.009 | - | 0.972±0.009 |
