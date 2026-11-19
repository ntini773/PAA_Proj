
### bf16_1024
| ms/step | calls/step | class | kernel |
|---|---|---|---|
| 32.06 | 28 | attention | `void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<(int)96, (int)128, (int)64, (` |
| 22.96 | 169 | bf16_gemm | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_tn_align8>(` |
| 13.55 | 28 | bf16_gemm | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x128_64x3_tn_align8>` |
| 12.16 | 28 | bf16_gemm | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_32x3_tn_align8` |
| 10.25 | 28 | attention | `fmha_cutlassF_bf16_aligned_64x128_rf_sm80(PyTorchMemEffAttention::AttentionKernel<cutlass:` |
| 3.25 | 113 | elementwise | `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_noc` |
| 2.53 | 29 | elementwise | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::GeluCUDAKernelImpl(at::` |
| 1.80 | 88 | elementwise | `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_noc` |

### bf16_512
| ms/step | calls/step | class | kernel |
|---|---|---|---|
| 7.79 | 169 | bf16_gemm | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_tn_align8>(` |
| 3.96 | 85 | bf16_gemm | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x64_64x3_tn_align8>` |
| 2.93 | 28 | bf16_gemm | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x128_32x3_tn_align8` |
| 2.65 | 28 | attention | `fmha_cutlassF_bf16_aligned_64x128_rf_sm80(PyTorchMemEffAttention::AttentionKernel<cutlass:` |
| 2.41 | 28 | attention | `void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<(int)96, (int)128, (int)64, (` |
| 0.79 | 113 | elementwise | `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_noc` |
| 0.49 | 88 | elementwise | `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_noc` |
| 0.39 | 57 | layernorm | `void at::native::<unnamed>::vectorized_layer_norm_kernel<c10::BFloat16, float, (bool)0>(in` |

### bf16_compile_1024
| ms/step | calls/step | class | kernel |
|---|---|---|---|
| 32.65 | 28 | attention | `void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<(int)96, (int)128, (int)64, (` |
| 23.31 | 169 | bf16_gemm | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_tn_align8>(` |
| 13.89 | 28 | bf16_gemm | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x128_64x3_tn_align8>` |
| 12.40 | 28 | bf16_gemm | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_32x3_tn_align8` |
| 10.84 | 28 | attention | `fmha_cutlassF_bf16_aligned_64x128_rf_sm80(PyTorchMemEffAttention::AttentionKernel<cutlass:` |
| 1.99 | 28 | attention | `triton_poi_fused__scaled_dot_product_efficient_attention_clone_constant_pad_nd_expand_slic` |
| 1.85 | 28 | inductor | `triton_poi_fused_gelu_view_4` |
| 1.03 | 28 | inductor | `triton_red_fused_add_div_mul_native_layer_norm_split_unsqueeze_view_3` |

### bf16_compile_512
| ms/step | calls/step | class | kernel |
|---|---|---|---|
| 7.86 | 169 | bf16_gemm | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_tn_align8>(` |
| 3.85 | 85 | bf16_gemm | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x64_64x3_tn_align8>` |
| 2.95 | 28 | bf16_gemm | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x128_32x3_tn_align8` |
| 2.69 | 28 | attention | `fmha_cutlassF_bf16_aligned_64x128_rf_sm80(PyTorchMemEffAttention::AttentionKernel<cutlass:` |
| 2.42 | 28 | attention | `void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<(int)96, (int)128, (int)64, (` |
| 0.30 | 27 | inductor | `triton_poi_fused_gelu_view_4` |
| 0.23 | 27 | inductor | `triton_red_fused_add_div_mul_native_layer_norm_split_unsqueeze_view_3` |
| 0.21 | 27 | inductor | `triton_red_fused_add_mul_native_layer_norm_split_unsqueeze_view_0` |

### fp8_1024
| ms/step | calls/step | class | kernel |
|---|---|---|---|
| 31.82 | 28 | attention | `void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<(int)96, (int)128, (int)64, (` |
| 23.96 | 225 | bf16_gemm | `void cutlass::device_kernel<at::cuda::detail::enable_3x_kernel_for_sm10_or_later<cutlass::` |
| 13.40 | 286 | elementwise | `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_noc` |
| 11.31 | 286 | elementwise | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::<unnamed>::launch_clamp` |
| 11.14 | 572 | copy | `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::Tenso` |
| 9.88 | 28 | attention | `fmha_cutlassF_bf16_aligned_64x128_rf_sm80(PyTorchMemEffAttention::AttentionKernel<cutlass:` |
| 6.16 | 286 | elementwise | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::ConvertToFloat8E4M3fnOp` |
| 5.56 | 286 | elementwise | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::AbsFunctor<c10::BFloat1` |

### fp8_512
| ms/step | calls/step | class | kernel |
|---|---|---|---|
| 7.07 | 224 | bf16_gemm | `void cutlass::device_kernel<at::cuda::detail::enable_3x_kernel_for_sm10_or_later<cutlass::` |
| 2.73 | 572 | copy | `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::Tenso` |
| 2.61 | 28 | attention | `fmha_cutlassF_bf16_aligned_64x128_rf_sm80(PyTorchMemEffAttention::AttentionKernel<cutlass:` |
| 2.35 | 28 | attention | `void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<(int)96, (int)128, (int)64, (` |
| 2.15 | 286 | elementwise | `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_noc` |
| 1.66 | 286 | elementwise | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::<unnamed>::launch_clamp` |
| 1.07 | 286 | reduce | `void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<c10::BFloat16, at::n` |
| 0.81 | 286 | elementwise | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::ConvertToFloat8E4M3fnOp` |

### free_bound_1024
| ms/step | calls/step | class | kernel |
|---|---|---|---|
| 27.33 | 28 | attention | `void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<(int)96, (int)128, (int)64, (` |
| 9.60 | 28 | attention | `fmha_cutlassF_bf16_aligned_64x128_rf_sm80(PyTorchMemEffAttention::AttentionKernel<cutlass:` |
| 9.45 | 374 | elementwise | `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_noc` |
| 9.28 | 399 | elementwise | `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_noc` |
| 9.24 | 258 | fp4_gemm | `cutlass3x_sm120_bstensorop_s16864gemm_block_scaled_ue4m3xe2m1_ue4m3xe2m1_f32_bf16_bf16_128` |
| 4.85 | 286 | elementwise | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::AbsFunctor<c10::BFloat1` |
| 3.58 | 28 | fp4_gemm | `cutlass3x_sm120_bstensorop_s16864gemm_block_scaled_ue4m3xe2m1_ue4m3xe2m1_f32_bf16_bf16_128` |
| 2.81 | 29 | elementwise | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::GeluCUDAKernelImpl(at::` |

### free_bound_512
| ms/step | calls/step | class | kernel |
|---|---|---|---|
| 4.75 | 286 | fp4_gemm | `cutlass3x_sm120_bstensorop_s16864gemm_block_scaled_ue4m3xe2m1_ue4m3xe2m1_f32_bf16_bf16_128` |
| 2.58 | 28 | attention | `fmha_cutlassF_bf16_aligned_64x128_rf_sm80(PyTorchMemEffAttention::AttentionKernel<cutlass:` |
| 2.38 | 374 | elementwise | `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_noc` |
| 2.30 | 28 | attention | `void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<(int)96, (int)128, (int)64, (` |
| 2.29 | 399 | elementwise | `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_noc` |
| 1.37 | 286 | reduce | `void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<c10::BFloat16, at::n` |
| 0.75 | 286 | elementwise | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::AbsFunctor<c10::BFloat1` |
| 0.37 | 57 | layernorm | `void at::native::<unnamed>::vectorized_layer_norm_kernel<c10::BFloat16, float, (bool)0>(in` |

### fused_cuda_row_1024
| ms/step | calls/step | class | kernel |
|---|---|---|---|
| 29.80 | 28 | attention | `void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<(int)96, (int)128, (int)64, (` |
| 9.75 | 374 | elementwise | `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_noc` |
| 9.73 | 28 | attention | `fmha_cutlassF_bf16_aligned_64x128_rf_sm80(PyTorchMemEffAttention::AttentionKernel<cutlass:` |
| 9.66 | 399 | elementwise | `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_noc` |
| 9.66 | 258 | fp4_gemm | `cutlass3x_sm120_bstensorop_s16864gemm_block_scaled_ue4m3xe2m1_ue4m3xe2m1_f32_bf16_bf16_128` |
| 5.16 | 286 | our_quant | `<unnamed>::quant2lvl_kernel(const uint4 *, const float *, unsigned long *, unsigned char *` |
| 4.88 | 286 | elementwise | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::AbsFunctor<c10::BFloat1` |
| 3.88 | 28 | fp4_gemm | `cutlass3x_sm120_bstensorop_s16864gemm_block_scaled_ue4m3xe2m1_ue4m3xe2m1_f32_bf16_bf16_128` |

### fused_cuda_row_512
| ms/step | calls/step | class | kernel |
|---|---|---|---|
| 4.79 | 286 | fp4_gemm | `cutlass3x_sm120_bstensorop_s16864gemm_block_scaled_ue4m3xe2m1_ue4m3xe2m1_f32_bf16_bf16_128` |
| 2.61 | 28 | attention | `fmha_cutlassF_bf16_aligned_64x128_rf_sm80(PyTorchMemEffAttention::AttentionKernel<cutlass:` |
| 2.41 | 374 | elementwise | `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_noc` |
| 2.35 | 28 | attention | `void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<(int)96, (int)128, (int)64, (` |
| 2.33 | 399 | elementwise | `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_noc` |
| 1.52 | 286 | our_quant | `<unnamed>::quant2lvl_kernel(const uint4 *, const float *, unsigned long *, unsigned char *` |
| 1.36 | 286 | reduce | `void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<c10::BFloat16, at::n` |
| 1.05 | 569 | copy | `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_noc` |

### fused_cuda_swizzled_1024
| ms/step | calls/step | class | kernel |
|---|---|---|---|
| 29.86 | 28 | attention | `void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<(int)96, (int)128, (int)64, (` |
| 9.76 | 374 | elementwise | `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_noc` |
| 9.73 | 258 | fp4_gemm | `cutlass3x_sm120_bstensorop_s16864gemm_block_scaled_ue4m3xe2m1_ue4m3xe2m1_f32_bf16_bf16_128` |
| 9.73 | 28 | attention | `fmha_cutlassF_bf16_aligned_64x128_rf_sm80(PyTorchMemEffAttention::AttentionKernel<cutlass:` |
| 9.69 | 399 | elementwise | `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_noc` |
| 4.89 | 286 | elementwise | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::AbsFunctor<c10::BFloat1` |
| 4.36 | 286 | other | `quant_kernel(const __nv_bfloat16 *, const float *, unsigned char *, unsigned char *, float` |
| 3.92 | 28 | fp4_gemm | `cutlass3x_sm120_bstensorop_s16864gemm_block_scaled_ue4m3xe2m1_ue4m3xe2m1_f32_bf16_bf16_128` |

### fused_cuda_swizzled_512
| ms/step | calls/step | class | kernel |
|---|---|---|---|
| 4.80 | 286 | fp4_gemm | `cutlass3x_sm120_bstensorop_s16864gemm_block_scaled_ue4m3xe2m1_ue4m3xe2m1_f32_bf16_bf16_128` |
| 2.61 | 28 | attention | `fmha_cutlassF_bf16_aligned_64x128_rf_sm80(PyTorchMemEffAttention::AttentionKernel<cutlass:` |
| 2.40 | 374 | elementwise | `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_noc` |
| 2.34 | 28 | attention | `void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<(int)96, (int)128, (int)64, (` |
| 2.32 | 399 | elementwise | `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_noc` |
| 1.37 | 286 | reduce | `void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<c10::BFloat16, at::n` |
| 1.08 | 286 | other | `quant_kernel(const __nv_bfloat16 *, const float *, unsigned char *, unsigned char *, float` |
| 0.75 | 286 | elementwise | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::AbsFunctor<c10::BFloat1` |

### fused_cuda_swizzled_compile_1024
| ms/step | calls/step | class | kernel |
|---|---|---|---|
| 30.32 | 28 | attention | `void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<(int)96, (int)128, (int)64, (` |
| 10.32 | 28 | attention | `fmha_cutlassF_bf16_aligned_64x128_rf_sm80(PyTorchMemEffAttention::AttentionKernel<cutlass:` |
| 10.12 | 258 | fp4_gemm | `cutlass3x_sm120_bstensorop_s16864gemm_block_scaled_ue4m3xe2m1_ue4m3xe2m1_f32_bf16_bf16_128` |
| 3.88 | 28 | fp4_gemm | `cutlass3x_sm120_bstensorop_s16864gemm_block_scaled_ue4m3xe2m1_ue4m3xe2m1_f32_bf16_bf16_128` |
| 2.87 | 28 | inductor | `triton_red_fused__to_copy_abs_add_div_gelu_max_mul_view_19` |
| 2.26 | 28 | inductor | `triton_per_fused___rshift____to_copy_abs_add_amax_bitwise_and_bitwise_or_clamp_div_full_li` |
| 1.95 | 28 | attention | `triton_poi_fused__scaled_dot_product_efficient_attention__to_copy_add_clone_constant_pad_n` |
| 1.16 | 196 | inductor | `triton_poi_fused___lshift_____rshift____scaled_mm__to_copy__unsafe_view_bitwise_or_clamp_c` |

### fused_cuda_swizzled_compile_512
| ms/step | calls/step | class | kernel |
|---|---|---|---|
| 4.79 | 286 | fp4_gemm | `cutlass3x_sm120_bstensorop_s16864gemm_block_scaled_ue4m3xe2m1_ue4m3xe2m1_f32_bf16_bf16_128` |
| 2.69 | 28 | attention | `fmha_cutlassF_bf16_aligned_64x128_rf_sm80(PyTorchMemEffAttention::AttentionKernel<cutlass:` |
| 2.33 | 28 | attention | `void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<(int)96, (int)128, (int)64, (` |
| 0.54 | 27 | inductor | `triton_red_fused__to_copy_abs_add_div_gelu_max_mul_view_19` |
| 0.52 | 27 | inductor | `triton_per_fused___rshift____to_copy_abs_add_amax_bitwise_and_bitwise_or_clamp_div_full_li` |
| 0.35 | 189 | inductor | `triton_poi_fused___lshift_____rshift____scaled_mm__to_copy__unsafe_view_bitwise_or_clamp_c` |
| 0.31 | 189 | inductor | `triton_poi_fused___lshift_____rshift____scaled_mm__to_copy__unsafe_view_bitwise_or_clamp_c` |
| 0.31 | 108 | attention | `triton_poi_fused__scaled_dot_product_flash_attention__to_copy_add_div_mul_transpose_view_5` |

### fused_triton_1024
| ms/step | calls/step | class | kernel |
|---|---|---|---|
| 29.75 | 28 | attention | `void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<(int)96, (int)128, (int)64, (` |
| 9.73 | 374 | elementwise | `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_noc` |
| 9.72 | 28 | attention | `fmha_cutlassF_bf16_aligned_64x128_rf_sm80(PyTorchMemEffAttention::AttentionKernel<cutlass:` |
| 9.66 | 258 | fp4_gemm | `cutlass3x_sm120_bstensorop_s16864gemm_block_scaled_ue4m3xe2m1_ue4m3xe2m1_f32_bf16_bf16_128` |
| 9.65 | 399 | elementwise | `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_noc` |
| 7.45 | 286 | our_quant | `_fused_quant2lvl_kernel` |
| 4.89 | 286 | elementwise | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::AbsFunctor<c10::BFloat1` |
| 3.88 | 28 | fp4_gemm | `cutlass3x_sm120_bstensorop_s16864gemm_block_scaled_ue4m3xe2m1_ue4m3xe2m1_f32_bf16_bf16_128` |

### fused_triton_512
| ms/step | calls/step | class | kernel |
|---|---|---|---|
| 4.79 | 286 | fp4_gemm | `cutlass3x_sm120_bstensorop_s16864gemm_block_scaled_ue4m3xe2m1_ue4m3xe2m1_f32_bf16_bf16_128` |
| 2.61 | 28 | attention | `fmha_cutlassF_bf16_aligned_64x128_rf_sm80(PyTorchMemEffAttention::AttentionKernel<cutlass:` |
| 2.41 | 374 | elementwise | `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_noc` |
| 2.35 | 28 | attention | `void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<(int)96, (int)128, (int)64, (` |
| 2.33 | 399 | elementwise | `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_noc` |
| 2.08 | 286 | our_quant | `_fused_quant2lvl_kernel` |
| 1.35 | 286 | reduce | `void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<c10::BFloat16, at::n` |
| 1.05 | 569 | copy | `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_noc` |

### fused_triton_narrow_1024
| ms/step | calls/step | class | kernel |
|---|---|---|---|
| 28.39 | 28 | attention | `void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<(int)96, (int)128, (int)64, (` |
| 21.58 | 90 | elementwise | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::BUnaryFunctor<int, int,` |
| 15.26 | 60 | elementwise | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::AUnaryFunctor<int, int,` |
| 15.21 | 60 | elementwise | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctorOnSelf_add<i` |
| 12.58 | 90 | copy | `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::Tenso` |
| 11.20 | 30 | elementwise | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<int, int,` |
| 11.18 | 30 | elementwise | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add<int>, s` |
| 9.74 | 28 | attention | `fmha_cutlassF_bf16_aligned_64x128_rf_sm80(PyTorchMemEffAttention::AttentionKernel<cutlass:` |

### fused_triton_narrow_512
| ms/step | calls/step | class | kernel |
|---|---|---|---|
| 4.82 | 286 | fp4_gemm | `cutlass3x_sm120_bstensorop_s16864gemm_block_scaled_ue4m3xe2m1_ue4m3xe2m1_f32_bf16_bf16_128` |
| 2.62 | 28 | attention | `fmha_cutlassF_bf16_aligned_64x128_rf_sm80(PyTorchMemEffAttention::AttentionKernel<cutlass:` |
| 2.42 | 374 | elementwise | `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_noc` |
| 2.37 | 90 | elementwise | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::BUnaryFunctor<int, int,` |
| 2.36 | 28 | attention | `void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<(int)96, (int)128, (int)64, (` |
| 2.33 | 399 | elementwise | `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_noc` |
| 2.12 | 60 | elementwise | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctorOnSelf_add<i` |
| 1.81 | 30 | reduce | `void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<float, at::native::f` |

### fused_triton_op_compile_1024
| ms/step | calls/step | class | kernel |
|---|---|---|---|
| 30.41 | 28 | attention | `void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<(int)96, (int)128, (int)64, (` |
| 10.33 | 28 | attention | `fmha_cutlassF_bf16_aligned_64x128_rf_sm80(PyTorchMemEffAttention::AttentionKernel<cutlass:` |
| 10.14 | 258 | fp4_gemm | `cutlass3x_sm120_bstensorop_s16864gemm_block_scaled_ue4m3xe2m1_ue4m3xe2m1_f32_bf16_bf16_128` |
| 3.89 | 28 | fp4_gemm | `cutlass3x_sm120_bstensorop_s16864gemm_block_scaled_ue4m3xe2m1_ue4m3xe2m1_f32_bf16_bf16_128` |
| 2.88 | 28 | inductor | `triton_red_fused__to_copy_abs_add_div_gelu_max_mul_view_19` |
| 2.26 | 28 | inductor | `triton_per_fused___rshift____to_copy_abs_add_amax_bitwise_and_bitwise_or_clamp_div_full_li` |
| 1.95 | 28 | attention | `triton_poi_fused__scaled_dot_product_efficient_attention__to_copy_add_clone_constant_pad_n` |
| 1.17 | 196 | inductor | `triton_poi_fused___lshift_____rshift____scaled_mm__to_copy__unsafe_view_bitwise_or_clamp_c` |

### fused_triton_op_compile_512
| ms/step | calls/step | class | kernel |
|---|---|---|---|
| 4.78 | 286 | fp4_gemm | `cutlass3x_sm120_bstensorop_s16864gemm_block_scaled_ue4m3xe2m1_ue4m3xe2m1_f32_bf16_bf16_128` |
| 2.69 | 28 | attention | `fmha_cutlassF_bf16_aligned_64x128_rf_sm80(PyTorchMemEffAttention::AttentionKernel<cutlass:` |
| 2.33 | 28 | attention | `void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<(int)96, (int)128, (int)64, (` |
| 0.53 | 27 | inductor | `triton_red_fused__to_copy_abs_add_div_gelu_max_mul_view_19` |
| 0.52 | 27 | inductor | `triton_per_fused___rshift____to_copy_abs_add_amax_bitwise_and_bitwise_or_clamp_div_full_li` |
| 0.35 | 189 | inductor | `triton_poi_fused___lshift_____rshift____scaled_mm__to_copy__unsafe_view_bitwise_or_clamp_c` |
| 0.31 | 189 | inductor | `triton_poi_fused___lshift_____rshift____scaled_mm__to_copy__unsafe_view_bitwise_or_clamp_c` |
| 0.30 | 108 | attention | `triton_poi_fused__scaled_dot_product_flash_attention__to_copy_add_div_mul_transpose_view_5` |

### module_best_compile_1024
| ms/step | calls/step | class | kernel |
|---|---|---|---|
| 31.74 | 28 | attention | `void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<(int)96, (int)128, (int)64, (` |
| 9.99 | 28 | attention | `fmha_cutlassF_bf16_aligned_64x128_rf_sm80(PyTorchMemEffAttention::AttentionKernel<cutlass:` |
| 9.98 | 174 | fp4_gemm | `cutlass3x_sm120_bstensorop_s16864gemm_block_scaled_ue4m3xe2m1_ue4m3xe2m1_f32_bf16_bf16_128` |
| 3.89 | 28 | fp4_gemm | `cutlass3x_sm120_bstensorop_s16864gemm_block_scaled_ue4m3xe2m1_ue4m3xe2m1_f32_bf16_bf16_128` |
| 3.04 | 202 | other | `quant_kernel(const __nv_bfloat16 *, const float *, unsigned char *, unsigned char *, float` |
| 1.72 | 28 | inductor | `triton_poi_fused__to_copy_add_cuda_swizzled_gelu_mul_view_10` |
| 0.87 | 28 | inductor | `triton_red_fused__to_copy_add_cuda_swizzled_div_mul_native_layer_norm_split_unsqueeze_view` |
| 0.75 | 28 | inductor | `triton_red_fused_add_cuda_swizzled_mul_native_layer_norm_split_unsqueeze_view_0` |

### module_best_compile_512
| ms/step | calls/step | class | kernel |
|---|---|---|---|
| 4.14 | 202 | fp4_gemm | `cutlass3x_sm120_bstensorop_s16864gemm_block_scaled_ue4m3xe2m1_ue4m3xe2m1_f32_bf16_bf16_128` |
| 2.65 | 28 | attention | `fmha_cutlassF_bf16_aligned_64x128_rf_sm80(PyTorchMemEffAttention::AttentionKernel<cutlass:` |
| 2.37 | 28 | attention | `void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<(int)96, (int)128, (int)64, (` |
| 0.86 | 202 | other | `quant_kernel(const __nv_bfloat16 *, const float *, unsigned char *, unsigned char *, float` |
| 0.37 | 27 | inductor | `triton_poi_fused__to_copy_add_cuda_swizzled_gelu_mul_view_10` |
| 0.24 | 27 | inductor | `triton_red_fused__to_copy_add_cuda_swizzled_div_mul_native_layer_norm_split_unsqueeze_view` |
| 0.19 | 27 | inductor | `triton_red_fused_add_cuda_swizzled_mul_native_layer_norm_split_unsqueeze_view_0` |
| 0.09 | 27 | inductor | `triton_poi_fused__to_copy_add_mul_split_unsqueeze_view_11` |

### module_triton_1024
| ms/step | calls/step | class | kernel |
|---|---|---|---|
| 30.74 | 28 | attention | `void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<(int)96, (int)128, (int)64, (` |
| 10.07 | 258 | fp4_gemm | `cutlass3x_sm120_bstensorop_s16864gemm_block_scaled_ue4m3xe2m1_ue4m3xe2m1_f32_bf16_bf16_128` |
| 9.91 | 374 | elementwise | `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_noc` |
| 9.89 | 399 | elementwise | `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_noc` |
| 9.48 | 28 | attention | `fmha_cutlassF_bf16_aligned_64x128_rf_sm80(PyTorchMemEffAttention::AttentionKernel<cutlass:` |
| 6.26 | 202 | our_quant | `_fused_quant2lvl_kernel` |
| 4.39 | 202 | elementwise | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::AbsFunctor<c10::BFloat1` |
| 3.97 | 28 | fp4_gemm | `cutlass3x_sm120_bstensorop_s16864gemm_block_scaled_ue4m3xe2m1_ue4m3xe2m1_f32_bf16_bf16_128` |

### module_triton_512
| ms/step | calls/step | class | kernel |
|---|---|---|---|
| 4.81 | 286 | fp4_gemm | `cutlass3x_sm120_bstensorop_s16864gemm_block_scaled_ue4m3xe2m1_ue4m3xe2m1_f32_bf16_bf16_128` |
| 2.55 | 28 | attention | `fmha_cutlassF_bf16_aligned_64x128_rf_sm80(PyTorchMemEffAttention::AttentionKernel<cutlass:` |
| 2.42 | 374 | elementwise | `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_noc` |
| 2.38 | 28 | attention | `void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<(int)96, (int)128, (int)64, (` |
| 2.36 | 399 | elementwise | `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_noc` |
| 1.66 | 202 | our_quant | `_fused_quant2lvl_kernel` |
| 1.00 | 202 | reduce | `void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<c10::BFloat16, at::n` |
| 0.75 | 401 | copy | `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_noc` |

### stock_1024
| ms/step | calls/step | class | kernel |
|---|---|---|---|
| 38.68 | 858 | elementwise | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::BUnaryFunctor<int, int,` |
| 30.01 | 572 | elementwise | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctorOnSelf_add<i` |
| 28.62 | 28 | attention | `void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<(int)96, (int)128, (int)64, (` |
| 22.42 | 858 | copy | `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::Tenso` |
| 21.51 | 572 | elementwise | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::AUnaryFunctor<int, int,` |
| 21.30 | 286 | reduce | `void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<float, at::native::f` |
| 18.61 | 286 | elementwise | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<int, int,` |
| 15.47 | 286 | elementwise | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add<int>, s` |

### stock_512
| ms/step | calls/step | class | kernel |
|---|---|---|---|
| 5.36 | 286 | reduce | `void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<float, at::native::f` |
| 4.91 | 858 | elementwise | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::BUnaryFunctor<int, int,` |
| 4.83 | 286 | fp4_gemm | `cutlass3x_sm120_bstensorop_s16864gemm_block_scaled_ue4m3xe2m1_ue4m3xe2m1_f32_bf16_bf16_128` |
| 4.47 | 858 | copy | `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::Tenso` |
| 3.88 | 572 | elementwise | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctorOnSelf_add<i` |
| 3.10 | 858 | copy | `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::Tenso` |
| 2.61 | 572 | elementwise | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::AUnaryFunctor<int, int,` |
| 2.61 | 28 | attention | `fmha_cutlassF_bf16_aligned_64x128_rf_sm80(PyTorchMemEffAttention::AttentionKernel<cutlass:` |

### stock_compile_1024
| ms/step | calls/step | class | kernel |
|---|---|---|---|
| 30.26 | 28 | attention | `void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<(int)96, (int)128, (int)64, (` |
| 10.30 | 28 | attention | `fmha_cutlassF_bf16_aligned_64x128_rf_sm80(PyTorchMemEffAttention::AttentionKernel<cutlass:` |
| 10.11 | 258 | fp4_gemm | `cutlass3x_sm120_bstensorop_s16864gemm_block_scaled_ue4m3xe2m1_ue4m3xe2m1_f32_bf16_bf16_128` |
| 3.87 | 28 | fp4_gemm | `cutlass3x_sm120_bstensorop_s16864gemm_block_scaled_ue4m3xe2m1_ue4m3xe2m1_f32_bf16_bf16_128` |
| 2.87 | 28 | inductor | `triton_red_fused__to_copy_abs_add_div_gelu_max_mul_view_19` |
| 2.25 | 28 | inductor | `triton_per_fused___rshift____to_copy_abs_add_amax_bitwise_and_bitwise_or_clamp_div_full_li` |
| 1.95 | 28 | attention | `triton_poi_fused__scaled_dot_product_efficient_attention__to_copy_add_clone_constant_pad_n` |
| 1.15 | 196 | inductor | `triton_poi_fused___lshift_____rshift____scaled_mm__to_copy__unsafe_view_bitwise_or_clamp_c` |

### stock_compile_512
| ms/step | calls/step | class | kernel |
|---|---|---|---|
| 4.79 | 286 | fp4_gemm | `cutlass3x_sm120_bstensorop_s16864gemm_block_scaled_ue4m3xe2m1_ue4m3xe2m1_f32_bf16_bf16_128` |
| 2.69 | 28 | attention | `fmha_cutlassF_bf16_aligned_64x128_rf_sm80(PyTorchMemEffAttention::AttentionKernel<cutlass:` |
| 2.33 | 28 | attention | `void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<(int)96, (int)128, (int)64, (` |
| 0.53 | 27 | inductor | `triton_red_fused__to_copy_abs_add_div_gelu_max_mul_view_19` |
| 0.52 | 27 | inductor | `triton_per_fused___rshift____to_copy_abs_add_amax_bitwise_and_bitwise_or_clamp_div_full_li` |
| 0.35 | 189 | inductor | `triton_poi_fused___lshift_____rshift____scaled_mm__to_copy__unsafe_view_bitwise_or_clamp_c` |
| 0.31 | 189 | inductor | `triton_poi_fused___lshift_____rshift____scaled_mm__to_copy__unsafe_view_bitwise_or_clamp_c` |
| 0.30 | 108 | attention | `triton_poi_fused__scaled_dot_product_flash_attention__to_copy_add_div_mul_transpose_view_5` |
