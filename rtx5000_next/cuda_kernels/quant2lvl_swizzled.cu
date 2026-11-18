// NVFP4 two-level quantize with block scales written directly in the cuBLAS
// swizzled layout (fuses torchao to_blocked). Reference: Triton _fused_quant2lvl_kernel.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>

static __device__ __forceinline__ float bf16_lo(uint32_t w) { return __uint_as_float(w << 16); }
static __device__ __forceinline__ float bf16_hi(uint32_t w) { return __uint_as_float(w & 0xFFFF0000u); }

// same threshold ladder as _e2m1_encode (input already clamped to [-6,6])
static __device__ __forceinline__ uint32_t e2m1_encode(float v) {
  uint32_t sign = v < 0.0f;
  float a = fabsf(v);
  uint32_t code = 0;
  code = a > 0.25f ? 1u : code;
  code = a >= 0.75f ? 2u : code;
  code = a > 1.25f ? 3u : code;
  code = a >= 1.75f ? 4u : code;
  code = a > 2.5f ? 5u : code;
  code = a >= 3.5f ? 6u : code;
  code = a > 5.0f ? 7u : code;
  return code | (sign << 3);
}

static __device__ __forceinline__ float fp8e4m3_to_float(uint8_t b) {
  __half_raw h = __nv_cvt_fp8_to_halfraw(b, __NV_E4M3);
  return __half2float(__half(h));
}

static constexpr float kE4M3Eps = 0.001953125f;  // 2^-9, torchao E4M3_EPS

// One thread per 16-element block. pts_mode 0: *pts_or_amax is pts; 1: it is amax.
__global__ void quant_kernel(const __nv_bfloat16* __restrict__ x,
                             const float* __restrict__ pts_or_amax,
                             uint8_t* __restrict__ packed,
                             uint8_t* __restrict__ scales_sw,
                             float* __restrict__ pts_out,
                             int M, int K, int pts_mode) {
  float pts = *pts_or_amax;
  if (pts_mode) pts = __fdiv_rn(pts, 448.0f * 6.0f);
  if (pts_mode && blockIdx.x == 0 && threadIdx.x == 0) *pts_out = pts;

  const int nblk = K >> 4;
  const long long total = (long long)M * nblk;
  long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= total) return;
  const int r = (int)(t / nblk);
  const int c = (int)(t - (long long)r * nblk);

  const uint4* src = reinterpret_cast<const uint4*>(x + (size_t)r * K + (size_t)c * 16);
  uint4 q0 = src[0], q1 = src[1];
  uint32_t w[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
  float v[16];
  float amax = 0.0f;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    v[2 * i] = bf16_lo(w[i]);
    v[2 * i + 1] = bf16_hi(w[i]);
    amax = fmaxf(amax, fmaxf(fabsf(v[2 * i]), fabsf(v[2 * i + 1])));
  }

  float block_scale = __fdiv_rn(amax, 6.0f);
  float sbs = __fdiv_rn(block_scale, pts);
  sbs = fminf(fmaxf(sbs, kE4M3Eps), 448.0f);
  uint8_t sbs_e4m3 = (uint8_t)__nv_cvt_float_to_fp8(sbs, __NV_SATFINITE, __NV_E4M3);
  float sbs_f32 = fp8e4m3_to_float(sbs_e4m3);
  float recip = __fdiv_rn(__fdiv_rn(1.0f, pts), sbs_f32);

  uint32_t out[2] = {0u, 0u};
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    float s0 = fminf(fmaxf(__fmul_rn(v[2 * i], recip), -6.0f), 6.0f);
    float s1 = fminf(fmaxf(__fmul_rn(v[2 * i + 1], recip), -6.0f), 6.0f);
    uint32_t b = (e2m1_encode(s0) & 0xFu) | ((e2m1_encode(s1) & 0xFu) << 4);
    out[i >> 2] |= b << (8 * (i & 3));
  }
  *reinterpret_cast<uint2*>(packed + (size_t)r * (K >> 1) + (size_t)c * 8) = make_uint2(out[0], out[1]);

  const int ncb = (nblk + 3) >> 2;
  const long long tile = (long long)(r >> 7) * ncb + (c >> 2);
  const long long off = tile * 512 + (r & 31) * 16 + ((r & 127) >> 5) * 4 + (c & 3);
  scales_sw[off] = sbs_e4m3;
}

// Zero the swizzle padding: logical positions with r >= M or c >= nblk.
__global__ void pad_zero_kernel(uint8_t* __restrict__ scales_sw, int M, int nblk,
                                int rows_p, int cols_p) {
  long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (long long)rows_p * cols_p) return;
  const int r = (int)(t / cols_p);
  const int c = (int)(t - (long long)r * cols_p);
  if (r < M && c < nblk) return;
  const int ncb = cols_p >> 2;
  const long long tile = (long long)(r >> 7) * ncb + (c >> 2);
  scales_sw[tile * 512 + (r & 31) * 16 + ((r & 127) >> 5) * 4 + (c & 3)] = 0;
}

// x is read as 8 bf16 per uint4; numel % 8 == 0 is guaranteed by K % 16 == 0.
__global__ void amax_kernel(const __nv_bfloat16* __restrict__ x, long long n_vec,
                            float* __restrict__ amax_out) {
  float m = 0.0f;
  const uint4* p = reinterpret_cast<const uint4*>(x);
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n_vec;
       i += (long long)gridDim.x * blockDim.x) {
    uint4 q = p[i];
    uint32_t w[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
    for (int j = 0; j < 4; ++j)
      m = fmaxf(m, fmaxf(fabsf(bf16_lo(w[j])), fabsf(bf16_hi(w[j]))));
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFFu, m, o));
  if ((threadIdx.x & 31) == 0) atomicMax(reinterpret_cast<int*>(amax_out), __float_as_int(m));
}

static void check_x(const at::Tensor& x) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16, "x must be bf16 CUDA");
  TORCH_CHECK(x.dim() == 2 && x.is_contiguous(), "x must be contiguous 2D");
  TORCH_CHECK(x.size(1) % 16 == 0, "K must be divisible by 16");
  TORCH_CHECK(x.size(0) > 0, "M must be > 0");
}

static void launch_quant(const at::Tensor& x, const float* p, at::Tensor& packed,
                         at::Tensor& scales, float* pts_out, int mode) {
  const int M = (int)x.size(0), K = (int)x.size(1), nblk = K / 16;
  const int rows_p = (M + 127) / 128 * 128, cols_p = (nblk + 3) / 4 * 4;
  auto stream = at::cuda::getCurrentCUDAStream();
  auto* sc = reinterpret_cast<uint8_t*>(scales.data_ptr());
  if (rows_p * cols_p != M * nblk) {
    long long n = (long long)rows_p * cols_p;
    pad_zero_kernel<<<(unsigned)((n + 255) / 256), 256, 0, stream>>>(sc, M, nblk, rows_p, cols_p);
  }
  long long total = (long long)M * nblk;
  quant_kernel<<<(unsigned)((total + 127) / 128), 128, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), p,
      packed.data_ptr<uint8_t>(), sc, pts_out, M, K, mode);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

static std::pair<at::Tensor, at::Tensor> alloc_outputs(const at::Tensor& x) {
  const int64_t M = x.size(0), K = x.size(1);
  const int64_t rows_p = (M + 127) / 128 * 128, cols_p = (K / 16 + 3) / 4 * 4;
  auto packed = at::empty({M, K / 2}, x.options().dtype(at::kByte));
  auto scales = at::empty({rows_p * cols_p}, x.options().dtype(at::kFloat8_e4m3fn));
  return {packed, scales};
}

std::vector<at::Tensor> quant2lvl_swizzled(at::Tensor x, at::Tensor pts) {
  check_x(x);
  TORCH_CHECK(pts.is_cuda() && pts.scalar_type() == at::kFloat && pts.numel() == 1,
              "pts must be fp32 CUDA scalar");
  c10::cuda::CUDAGuard guard(x.device());
  pts = pts.contiguous();
  auto [packed, scales] = alloc_outputs(x);
  launch_quant(x, pts.data_ptr<float>(), packed, scales, nullptr, 0);
  return {packed, scales};
}

std::vector<at::Tensor> amax_quant_swizzled(at::Tensor x) {
  check_x(x);
  c10::cuda::CUDAGuard guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  auto amax = torch::zeros({1}, x.options().dtype(at::kFloat));
  auto pts = torch::empty({}, x.options().dtype(at::kFloat));
  const long long n_vec = x.numel() / 8;
  const int threads = 256;
  int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  long long want = (n_vec + threads - 1) / threads;
  unsigned blocks = (unsigned)std::max<long long>(1, std::min<long long>(want, (long long)sms * 8));
  amax_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), n_vec, amax.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  auto [packed, scales] = alloc_outputs(x);
  launch_quant(x, amax.data_ptr<float>(), packed, scales, pts.data_ptr<float>(), 1);
  return {packed, scales, pts};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("quant2lvl_swizzled", &quant2lvl_swizzled, "NVFP4 2-level quantize, swizzled scales");
  m.def("amax_quant_swizzled", &amax_quant_swizzled, "amax + NVFP4 quantize, swizzled scales");
}
