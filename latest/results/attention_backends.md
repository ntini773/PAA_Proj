| shape | backend | ms | TFLOP/s | rel L2 vs fp32 | note |
|---|---|---|---|---|---|
| pixart_self_512 | default | 0.092 | 104 | 0.0022 |  |
| pixart_self_512 | flash | 0.095 | 101 | 0.0022 |  |
| pixart_self_512 | cudnn | 0.114 | 85 | 0.0022 |  |
| pixart_self_512 | efficient | 0.219 | 44 | 0.0022 |  |
| pixart_self_512 | sage2 | 0.182 | 53 | 0.0376 |  |
| pixart_self_512 | sage3 | nan | nan | - | ModuleNotFoundError: No module named 'sageattn3' |
| pixart_self_1024 | default | 1.008 | 153 | 0.0023 |  |
| pixart_self_1024 | flash | 1.010 | 153 | 0.0023 |  |
| pixart_self_1024 | cudnn | 1.298 | 119 | 0.0023 |  |
| pixart_self_1024 | efficient | 3.006 | 51 | 0.0023 |  |
| pixart_self_1024 | sage2 | 1.256 | 123 | 0.0385 |  |
| pixart_self_1024 | sage3 | nan | nan | - | ModuleNotFoundError: No module named 'sageattn3' |
| pixart_cross_T300 | default | 0.355 | 32 | 0.0018 |  |
| pixart_cross_T300 | flash | 0.354 | 32 | 0.0018 |  |
| pixart_cross_T300 | cudnn | 0.353 | 32 | 0.0018 |  |
| pixart_cross_T300 | efficient | 0.353 | 32 | 0.0018 |  |
| pixart_cross_T300 | sage2 | 0.354 | 32 | 0.0018 | mask: torch fallback |
| pixart_cross_T300 | sage3 | 0.353 | 32 | 0.0018 | mask: torch fallback |
| pixart_cross_T64 | default | 0.124 | 19 | 0.0018 |  |
| pixart_cross_T64 | flash | 0.124 | 19 | 0.0018 |  |
| pixart_cross_T64 | cudnn | 0.124 | 19 | 0.0018 |  |
| pixart_cross_T64 | efficient | 0.124 | 19 | 0.0018 |  |
| pixart_cross_T64 | sage2 | 0.124 | 19 | 0.0018 | mask: torch fallback |
| pixart_cross_T64 | sage3 | 0.124 | 19 | 0.0018 | mask: torch fallback |
| sd35_medium_joint | default | 1.156 | 208 | 0.0023 |  |
| sd35_medium_joint | flash | 1.162 | 207 | 0.0023 |  |
| sd35_medium_joint | cudnn | 1.249 | 193 | 0.0023 |  |
| sd35_medium_joint | efficient | 2.525 | 95 | 0.0023 |  |
| sd35_medium_joint | sage2 | 0.938 | 257 | 0.0386 |  |
| sd35_medium_joint | sage3 | nan | nan | - | ModuleNotFoundError: No module named 'sageattn3' |
| flux_joint | default | 2.686 | 194 | 0.0023 |  |
| flux_joint | flash | 2.735 | 191 | 0.0023 |  |
| flux_joint | cudnn | 2.476 | 211 | 0.0023 |  |
| flux_joint | efficient | 6.203 | 84 | 0.0023 |  |
| flux_joint | sage2 | 1.805 | 289 | 0.0385 |  |
| flux_joint | sage3 | nan | nan | - | ModuleNotFoundError: No module named 'sageattn3' |
