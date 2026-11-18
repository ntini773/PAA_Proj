// Fused LayerNorm + AdaLN modulate + NVFP4 quantize (single-level), one CTA per row.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <vector>

namespace {

__device__ __forceinline__ float bf_lo(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float bf_hi(uint32_t w) { return __uint_as_float(w & 0xffff0000u); }

// 16 contiguous bf16 (32 bytes, 16B aligned) -> fp32
__device__ __forceinline__ void load16(const __nv_bfloat16* p, float v[16]) {
  const uint4* q = reinterpret_cast<const uint4*>(p);
  uint4 a = q[0], b = q[1];
  v[0] = bf_lo(a.x);  v[1] = bf_hi(a.x);  v[2] = bf_lo(a.y);  v[3] = bf_hi(a.y);
  v[4] = bf_lo(a.z);  v[5] = bf_hi(a.z);  v[6] = bf_lo(a.w);  v[7] = bf_hi(a.w);
  v[8] = bf_lo(b.x);  v[9] = bf_hi(b.x);  v[10] = bf_lo(b.y); v[11] = bf_hi(b.y);
  v[12] = bf_lo(b.z); v[13] = bf_hi(b.z); v[14] = bf_lo(b.w); v[15] = bf_hi(b.w);
}

// Fixed order: xor-butterfly inside each warp, then warps summed sequentially 0..nw-1.
__device__ __forceinline__ float block_sum(float v, float* sm) {
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, o));
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  if (lane == 0) sm[warp] = v;
  __syncthreads();
  const int nw = blockDim.x >> 5;
  float t = sm[0];
  for (int w = 1; w < nw; w++) t = __fadd_rn(t, sm[w]);
  __syncthreads();
  return t;
}

__device__ __forceinline__ uint32_t e2m1(float v) {
  uint32_t sign = v < 0.f;
  float a = fabsf(v);
  uint32_t c = 0;
  c += a > 0.25f; c += a >= 0.75f; c += a > 1.25f; c += a >= 1.75f;
  c += a > 2.5f;  c += a >= 3.5f;  c += a > 5.0f;
  return c | (sign << 3);
}

// e4m3fn byte -> float (exact)
__device__ __forceinline__ float e4m3_to_f32(uint32_t s) {
  uint32_t e = (s >> 3) & 0xF, m = s & 7;
  if (e == 0) return __fmul_rn((float)m, 0.001953125f);  // 2^-9
  return __uint_as_float(((e + 120u) << 23) | (m << 20));
}

__global__ void ln_mod_quant_kernel(const __nv_bfloat16* __restrict__ x,
                                    const __nv_bfloat16* __restrict__ scale,
                                    const __nv_bfloat16* __restrict__ shift,
                                    uint2* __restrict__ packed, uint8_t* __restrict__ bscale,
                                    int64_t x_stride, int64_t s_stride, int H, float eps) {
  __shared__ float sm[32];
  const int64_t row = blockIdx.x;
  const int t = threadIdx.x;
  const int nblk = H >> 4;
  const bool active = t < nblk;

  float v[16];
  float s = 0.f;
  if (active) {
    load16(x + row * x_stride + (int64_t)t * 16, v);
    #pragma unroll
    for (int i = 0; i < 16; i++) s = __fadd_rn(s, v[i]);
  }
  const float mean = __fdiv_rn(block_sum(s, sm), (float)H);

  float q = 0.f;
  if (active) {
    #pragma unroll
    for (int i = 0; i < 16; i++) {
      float d = __fsub_rn(v[i], mean);
      q = __fadd_rn(q, __fmul_rn(d, d));
    }
  }
  const float var = __fdiv_rn(block_sum(q, sm), (float)H);
  const float rstd = rsqrtf(__fadd_rn(var, eps));
  if (!active) return;

  float sc[16], sh[16];
  load16(scale + row * s_stride + (int64_t)t * 16, sc);
  load16(shift + row * s_stride + (int64_t)t * 16, sh);

  float amax = 0.f;
  #pragma unroll
  for (int i = 0; i < 16; i++) {
    float n = __fmul_rn(__fsub_rn(v[i], mean), rstd);
    float y = __fadd_rn(__fmul_rn(n, __fadd_rn(1.f, sc[i])), sh[i]);
    v[i] = __bfloat162float(__float2bfloat16_rn(y));
    amax = fmaxf(amax, fabsf(v[i]));
  }

  float bs = __fdiv_rn(amax, 6.f);
  bs = fminf(fmaxf(bs, 0.015625f), 448.f);
  const uint32_t s8 = (uint32_t)__nv_cvt_float_to_fp8(bs, __NV_SATFINITE, __NV_E4M3);
  const float inv = __fdiv_rn(1.f, e4m3_to_f32(s8));

  uint32_t w[2] = {0u, 0u};
  #pragma unroll
  for (int j = 0; j < 8; j++) {
    float a = fminf(fmaxf(__fmul_rn(v[2 * j], inv), -6.f), 6.f);
    float b = fminf(fmaxf(__fmul_rn(v[2 * j + 1], inv), -6.f), 6.f);
    w[j >> 2] |= (e2m1(a) | (e2m1(b) << 4)) << (8 * (j & 3));
  }
  packed[row * nblk + t] = make_uint2(w[0], w[1]);
  bscale[row * nblk + t] = (uint8_t)s8;
}

at::Tensor prep(const at::Tensor& t, int64_t H) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == at::kBFloat16 && t.dim() == 2 && t.size(1) == H,
              "expected CUDA bf16 (M,H) tensor");
  bool ok = t.stride(1) == 1 && t.stride(0) % 8 == 0 && (reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0);
  return ok ? t : t.contiguous();
}

}  // namespace

std::vector<at::Tensor> ln_mod_quant(at::Tensor x, at::Tensor scale_rows, at::Tensor shift_rows, double eps) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.dim() == 2, "x must be CUDA bf16 (M,H)");
  const int64_t M = x.size(0), H = x.size(1);
  TORCH_CHECK(H % 16 == 0 && H >= 16 && H <= 16384, "H must be a multiple of 16, <= 16384");
  TORCH_CHECK(scale_rows.size(0) == M && shift_rows.size(0) == M, "scale/shift rows must have M rows (expand if broadcast)");
  const c10::cuda::CUDAGuard guard(x.device());
  x = prep(x, H);
  scale_rows = prep(scale_rows, H);
  shift_rows = prep(shift_rows, H);

  auto packed = at::empty({M, H / 2}, x.options().dtype(at::kByte));
  auto bscale = at::empty({M, H / 16}, x.options().dtype(at::kFloat8_e4m3fn));
  if (M == 0) return {packed, bscale};

  const int nblk = (int)(H / 16);
  const int threads = ((nblk + 31) / 32) * 32;
  TORCH_CHECK(scale_rows.stride(0) == shift_rows.stride(0), "scale/shift row strides must match");
  ln_mod_quant_kernel<<<(unsigned)M, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(scale_rows.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(shift_rows.data_ptr()),
      reinterpret_cast<uint2*>(packed.data_ptr()), reinterpret_cast<uint8_t*>(bscale.data_ptr()),
      x.stride(0), scale_rows.stride(0), (int)H, (float)eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {packed, bscale};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ln_mod_quant", &ln_mod_quant,
        "Fused LayerNorm + AdaLN modulate + NVFP4 quantize; returns [packed_uint8 (M,H/2), scale_e4m3 (M,H/16)]");
}
