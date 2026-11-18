// Two-level NVFP4 quantize (per-tensor scale + per-16 e4m3 block scale), logical layout.
// Byte-exact replacement for _fused_quant2lvl_kernel in fused_adaln_nvfp4.py.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <vector>

namespace {

constexpr int kThreads = 256;

__device__ __forceinline__ uint32_t e2m1(float v) {
  uint32_t sign = v < 0.f;
  float a = fabsf(v);
  uint32_t c = 0;
  c += a > 0.25f; c += a >= 0.75f; c += a > 1.25f; c += a >= 1.75f;
  c += a > 2.5f;  c += a >= 3.5f;  c += a > 5.0f;
  return c | (sign << 3);
}

__device__ __forceinline__ void unpack_bf16x8(const uint4& u, float* f) {
  const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
  for (int i = 0; i < 4; i++) {
    f[2 * i]     = __uint_as_float(w[i] << 16);
    f[2 * i + 1] = __uint_as_float(w[i] & 0xFFFF0000u);
  }
}

// One thread = one 16-element block. Blocks are flattened over [M * nblk].
__global__ void quant2lvl_kernel(const uint4* __restrict__ x, const float* __restrict__ pts_ptr,
                                 uint64_t* __restrict__ packed, uint8_t* __restrict__ scales,
                                 long long total_blocks) {
  long long g = (long long)blockIdx.x * kThreads + threadIdx.x;
  if (g >= total_blocks) return;

  float v[16];
  uint4 lo = __ldg(x + 2 * g);
  uint4 hi = __ldg(x + 2 * g + 1);
  unpack_bf16x8(lo, v);
  unpack_bf16x8(hi, v + 8);

  const float pts = __ldg(pts_ptr);

  float amax = 0.f;
#pragma unroll
  for (int i = 0; i < 16; i++) amax = fmaxf(amax, fabsf(v[i]));

  float bs = __fdiv_rn(amax, 6.f);
  float sbs = __fdiv_rn(bs, pts);
  sbs = fminf(fmaxf(sbs, 0.015625f), 448.f);  // E4M3 tiny = 2^-6
  uint8_t s8 = (uint8_t)__nv_cvt_float_to_fp8(sbs, __NV_SATFINITE, __NV_E4M3);
  __half_raw hr = __nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)s8, __NV_E4M3);
  float sf = __half2float(*reinterpret_cast<__half*>(&hr));
  float recip = __fdiv_rn(__fdiv_rn(1.f, pts), sf);

  uint64_t out = 0;
#pragma unroll
  for (int j = 0; j < 8; j++) {
    float a = fminf(fmaxf(__fmul_rn(v[2 * j], recip), -6.f), 6.f);
    float b = fminf(fmaxf(__fmul_rn(v[2 * j + 1], recip), -6.f), 6.f);
    out |= (uint64_t)(e2m1(a) | (e2m1(b) << 4)) << (8 * j);
  }
  packed[g] = out;
  scales[g] = s8;
}

}  // namespace

std::vector<at::Tensor> quant2lvl_row(at::Tensor x, at::Tensor pts) {
  TORCH_CHECK(x.is_cuda() && pts.is_cuda(), "x and pts must be CUDA tensors");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16, "x must be bfloat16");
  TORCH_CHECK(x.dim() == 2, "x must be 2-D [M, K]");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(pts.scalar_type() == at::kFloat && pts.dim() == 0 && pts.is_contiguous(),
              "pts must be a 0-dim float32 tensor");
  TORCH_CHECK(x.device() == pts.device(), "x and pts must be on the same device");
  const int64_t M = x.size(0), K = x.size(1);
  TORCH_CHECK(K % 16 == 0 && K <= 8192, "K must be a multiple of 16 and <= 8192");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0, "x must be 16-byte aligned");

  c10::cuda::CUDAGuard guard(x.device());
  const int64_t nblk = K / 16;
  auto packed = at::empty({M, K / 2}, x.options().dtype(at::kByte));
  auto scales = at::empty({M, nblk}, x.options().dtype(at::kFloat8_e4m3fn));
  const long long total = M * nblk;
  if (total == 0) return {packed, scales};

  const long long grid = (total + kThreads - 1) / kThreads;
  quant2lvl_kernel<<<(unsigned)grid, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const uint4*>(x.data_ptr()), pts.data_ptr<float>(),
      reinterpret_cast<uint64_t*>(packed.data_ptr()),
      reinterpret_cast<uint8_t*>(scales.data_ptr()), total);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {packed, scales};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("quant2lvl_row", &quant2lvl_row, "NVFP4 two-level quantize (logical layout)");
}
