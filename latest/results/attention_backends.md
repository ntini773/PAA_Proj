| shape | backend | ms | TFLOP/s | rel L2 vs fp32 | note |
|---|---|---|---|---|---|
| pixart_self_512 | default | 0.099 | 98 | 0.0022 |  |
| pixart_self_512 | flash | 0.104 | 93 | 0.0022 |  |
| pixart_self_512 | cudnn | 0.124 | 78 | 0.0022 |  |
| pixart_self_512 | efficient | 0.232 | 42 | 0.0022 |  |
| pixart_self_512 | sage2 | nan | nan | - | AssertionError: headdim should be in [64, 96, 128]. |
| pixart_self_512 | sage3 | nan | nan | - | ModuleNotFoundError: No module named 'sageattn3' |
| pixart_self_1024 | default | 1.046 | 148 | 0.0023 |  |
| pixart_self_1024 | flash | 1.060 | 146 | 0.0023 |  |
| pixart_self_1024 | cudnn | 1.352 | 114 | 0.0023 |  |
| pixart_self_1024 | efficient | 3.050 | 51 | 0.0023 |  |
| pixart_self_1024 | sage2 | nan | nan | - | AssertionError: headdim should be in [64, 96, 128]. |
| pixart_self_1024 | sage3 | nan | nan | - | ModuleNotFoundError: No module named 'sageattn3' |
| pixart_cross_T300 | default | 0.371 | 31 | 0.0018 |  |
| pixart_cross_T300 | flash | 0.370 | 31 | 0.0018 |  |
| pixart_cross_T300 | cudnn | 0.371 | 31 | 0.0018 |  |
| pixart_cross_T300 | efficient | 0.370 | 31 | 0.0018 |  |
| pixart_cross_T300 | sage2 | 0.370 | 31 | 0.0018 | mask: torch fallback |
| pixart_cross_T300 | sage3 | 0.370 | 31 | 0.0018 | mask: torch fallback |
| pixart_cross_T64 | default | 0.134 | 18 | 0.0018 |  |
| pixart_cross_T64 | flash | 0.134 | 18 | 0.0018 |  |
| pixart_cross_T64 | cudnn | 0.134 | 18 | 0.0018 |  |
| pixart_cross_T64 | efficient | 0.134 | 18 | 0.0018 |  |
| pixart_cross_T64 | sage2 | 0.134 | 18 | 0.0018 | mask: torch fallback |
| pixart_cross_T64 | sage3 | 0.134 | 18 | 0.0018 | mask: torch fallback |
| sd35_medium_joint | default | 1.204 | 200 | 0.0023 |  |
| sd35_medium_joint | flash | 1.231 | 196 | 0.0023 |  |
| sd35_medium_joint | cudnn | 1.289 | 187 | 0.0023 |  |
| sd35_medium_joint | efficient | 2.649 | 91 | 0.0023 |  |
| sd35_medium_joint | sage2 | 1.165 | 207 | 0.0146 |  |
| sd35_medium_joint | sage3 | nan | nan | - | ModuleNotFoundError: No module named 'sageattn3' |
| flux_joint | default | 2.783 | 188 | 0.0023 |  |
| flux_joint | flash | 2.849 | 183 | 0.0023 |  |
| flux_joint | cudnn | 2.540 | 205 | 0.0023 |  |
| flux_joint | efficient | 6.448 | 81 | 0.0023 |  |
| flux_joint | sage2 | 2.504 | 208 | 0.0134 |  |
| flux_joint | sage3 | nan | nan | - | ModuleNotFoundError: No module named 'sageattn3' |
