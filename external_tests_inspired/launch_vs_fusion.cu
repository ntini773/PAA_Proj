// launch_vs_fusion.cu
//
// Question: when the norm->NVFP4-quant boundary gets faster, is it because of
//   (a) FUSION      : fewer kernels, no big DRAM round trip, or
//   (b) LAUNCH COST : the CPU needs ~5 us to start each tiny kernel?
// The GraspGenX PR (NVlabs/GraspGenX#7) gets its 30-40x mostly from removing
// (b) (CUDA Graph replay) plus fusing. This program separates the two effects.
//
// Four variants, each repeated R times per "step" (R=56: 28 blocks x 2 spots):
//   unfused/eager : 3 kernels per spot, launched one by one from the CPU
//   unfused/graph : same 3 kernels, but the whole step is ONE CUDA Graph replay
//   fused/eager   : 1 kernel per spot, launched one by one
//   fused/graph   : 1 kernel per spot, whole step is ONE CUDA Graph replay
//
// The math is LayerNorm -> AdaLN modulate -> bf16 round -> NVFP4 quantize
// (16-element blocks, e4m3 block scale, E2M1 codes, 2 codes per byte).
// All float ops use explicit _rn intrinsics so fused == unfused bit for bit.
//
// Build: nvcc -O3 -std=c++17 -arch=native launch_vs_fusion.cu -o launch_vs_fusion
// Run  : ./launch_vs_fusion            (optional: ./launch_vs_fusion 56 200)
//         args: R (spots per step, default 56)  iters (default 200)
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1);} } while (0)

constexpr int H = 1152;            // PixArt-Sigma hidden size
constexpr int NB = H / 16;         // 72 quant blocks per row
constexpr int THREADS = 96;        // 3 warps; threads 0..71 own one 16-element block
constexpr float EPS = 1e-6f;

// ---------------- shared device helpers ----------------
__device__ __forceinline__ float bf2f(__nv_bfloat16 v) { return __bfloat162float(v); }

__device__ __forceinline__ float block_sum(float v, float* sm) {
  for (int o = 16; o > 0; o >>= 1) v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, o));
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  if (lane == 0) sm[warp] = v;
  __syncthreads();
  float t = __fadd_rn(__fadd_rn(sm[0], sm[1]), sm[2]);
  __syncthreads();
  return t;
}

// Load this thread's 16 bf16 values (zeros for idle threads), compute row mean / rstd.
__device__ __forceinline__ void load_and_stats(const __nv_bfloat16* xrow, float v[16],
                                               float& mean, float& rstd, float* sm) {
  int t = threadIdx.x;
  float s = 0.f;
  if (t < NB) {
    const __nv_bfloat16* p = xrow + t * 16;
    #pragma unroll
    for (int i = 0; i < 16; i++) { v[i] = bf2f(p[i]); s = __fadd_rn(s, v[i]); }
  } else {
    #pragma unroll
    for (int i = 0; i < 16; i++) v[i] = 0.f;
  }
  mean = __fdiv_rn(block_sum(s, sm), (float)H);
  float q = 0.f;
  if (t < NB) {
    #pragma unroll
    for (int i = 0; i < 16; i++) { float d = __fsub_rn(v[i], mean); q = __fadd_rn(q, __fmul_rn(d, d)); }
  }
  float var = __fdiv_rn(block_sum(q, sm), (float)H);
  rstd = rsqrtf(__fadd_rn(var, EPS));
}

__device__ __forceinline__ __nv_bfloat16 modulate(float x, float mean, float rstd, float sc, float sh) {
  float n = __fmul_rn(__fsub_rn(x, mean), rstd);
  return __float2bfloat16_rn(__fadd_rn(__fmul_rn(n, __fadd_rn(1.f, sc)), sh));
}

// E2M1 round-to-nearest-even ladder (same thresholds as the Triton kernel / torchao).
__device__ __forceinline__ uint32_t e2m1(float v) {
  uint32_t sign = v < 0.f; float a = fabsf(v); uint32_t c = 0;
  c += a > 0.25f; c += a >= 0.75f; c += a > 1.25f; c += a >= 1.75f;
  c += a > 2.5f;  c += a >= 3.5f;  c += a > 5.0f;
  return c | (sign << 3);
}

// Quantize one 16-element block already rounded to bf16. Returns packed 8 bytes + e4m3 scale.
__device__ __forceinline__ void quant16(const float xm[16], uint64_t& packed, uint8_t& s8) {
  float amax = 0.f;
  #pragma unroll
  for (int i = 0; i < 16; i++) amax = fmaxf(amax, fabsf(xm[i]));
  float bs = __fdiv_rn(amax, 6.f);
  bs = fminf(fmaxf(bs, 0.015625f), 448.f);            // e4m3 tiny = 2^-6
  s8 = (uint8_t)__nv_cvt_float_to_fp8(bs, __NV_SATFINITE, __NV_E4M3);
  __half_raw hr = __nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)s8, __NV_E4M3);
  float sf = __half2float(*reinterpret_cast<__half*>(&hr));
  float inv = __fdiv_rn(1.f, sf);
  packed = 0;
  #pragma unroll
  for (int j = 0; j < 8; j++) {
    float a = fminf(fmaxf(__fmul_rn(xm[2 * j], inv), -6.f), 6.f);
    float b = fminf(fmaxf(__fmul_rn(xm[2 * j + 1], inv), -6.f), 6.f);
    uint64_t byte = (uint64_t)(e2m1(a) | (e2m1(b) << 4));
    packed |= byte << (8 * j);
  }
}

// ---------------- FUSED: LN + modulate + quant in registers, 1 kernel ----------------
__global__ void fused_kernel(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ scale,
                             const __nv_bfloat16* __restrict__ shift, uint64_t* __restrict__ packed,
                             uint8_t* __restrict__ bscale) {
  __shared__ float sm[3];
  int row = blockIdx.x, t = threadIdx.x;
  float v[16], mean, rstd;
  load_and_stats(x + (size_t)row * H, v, mean, rstd, sm);
  if (t < NB) {
    float xm[16];
    #pragma unroll
    for (int i = 0; i < 16; i++)
      xm[i] = bf2f(modulate(v[i], mean, rstd, bf2f(scale[t * 16 + i]), bf2f(shift[t * 16 + i])));
    uint64_t pk; uint8_t s8; quant16(xm, pk, s8);
    packed[(size_t)row * NB + t] = pk;
    bscale[(size_t)row * NB + t] = s8;
  }
}

// ---------------- UNFUSED: 3 kernels, intermediates go through DRAM ----------------
// K1: LayerNorm -> fp32 [M,H]      (like eager PyTorch's fp32 temporaries)
__global__ void k1_layernorm(const __nv_bfloat16* __restrict__ x, float* __restrict__ norm) {
  __shared__ float sm[3];
  int row = blockIdx.x, t = threadIdx.x;
  float v[16], mean, rstd;
  load_and_stats(x + (size_t)row * H, v, mean, rstd, sm);
  if (t < NB) {
    #pragma unroll
    for (int i = 0; i < 16; i++) norm[(size_t)row * H + t * 16 + i] = __fmul_rn(__fsub_rn(v[i], mean), rstd);
  }
}
// K2: modulate -> bf16 [M,H]       (the "x_mod" copy that fusion removes)
__global__ void k2_modulate(const float* __restrict__ norm, const __nv_bfloat16* __restrict__ scale,
                            const __nv_bfloat16* __restrict__ shift, __nv_bfloat16* __restrict__ xmod, size_t total) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < total; i += (size_t)gridDim.x * blockDim.x) {
    int c = (int)(i % H);
    float n = norm[i];
    xmod[i] = __float2bfloat16_rn(__fadd_rn(__fmul_rn(n, __fadd_rn(1.f, bf2f(scale[c]))), bf2f(shift[c])));
  }
}
// K3: quantize x_mod -> packed + scales
__global__ void k3_quant(const __nv_bfloat16* __restrict__ xmod, uint64_t* __restrict__ packed,
                         uint8_t* __restrict__ bscale) {
  int row = blockIdx.x, t = threadIdx.x;
  if (t < NB) {
    float xm[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) xm[i] = bf2f(xmod[(size_t)row * H + t * 16 + i]);
    uint64_t pk; uint8_t s8; quant16(xm, pk, s8);
    packed[(size_t)row * NB + t] = pk;
    bscale[(size_t)row * NB + t] = s8;
  }
}

// ---------------- harness ----------------
struct Bufs {
  __nv_bfloat16 *x, *scale, *shift, *xmod; float* norm; uint64_t* packed; uint8_t* bscale; int M;
};

static void spot_unfused(const Bufs& b, cudaStream_t s) {
  k1_layernorm<<<b.M, THREADS, 0, s>>>(b.x, b.norm);
  size_t total = (size_t)b.M * H;
  int grid = (int)std::min<size_t>((total + 255) / 256, 65535 * 4);
  k2_modulate<<<grid, 256, 0, s>>>(b.norm, b.scale, b.shift, b.xmod, total);
  k3_quant<<<b.M, THREADS, 0, s>>>(b.xmod, b.packed, b.bscale);
}
static void spot_fused(const Bufs& b, cudaStream_t s) {
  fused_kernel<<<b.M, THREADS, 0, s>>>(b.x, b.scale, b.shift, b.packed, b.bscale);
}

typedef void (*SpotFn)(const Bufs&, cudaStream_t);

// time one step (= R spots). Returns median ms over iters. graph=true replays a captured graph.
static double time_step(SpotFn fn, const Bufs& b, int R, int iters, bool graph, cudaStream_t s) {
  cudaGraph_t g = nullptr; cudaGraphExec_t ge = nullptr;
  if (graph) {
    CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
    for (int r = 0; r < R; r++) fn(b, s);
    CK(cudaStreamEndCapture(s, &g));
    CK(cudaGraphInstantiate(&ge, g, 0));
  }
  auto once = [&]() { if (graph) CK(cudaGraphLaunch(ge, s)); else for (int r = 0; r < R; r++) fn(b, s); };
  for (int i = 0; i < 10; i++) once();                 // warmup
  CK(cudaStreamSynchronize(s));
  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  std::vector<float> ms(iters);
  for (int i = 0; i < iters; i++) {
    CK(cudaEventRecord(e0, s)); once(); CK(cudaEventRecord(e1, s));
    CK(cudaEventSynchronize(e1)); CK(cudaEventElapsedTime(&ms[i], e0, e1));
  }
  std::sort(ms.begin(), ms.end());
  CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
  if (graph) { CK(cudaGraphExecDestroy(ge)); CK(cudaGraphDestroy(g)); }
  return ms[iters / 2];
}

int main(int argc, char** argv) {
  int R = argc > 1 ? atoi(argv[1]) : 56;
  int iters = argc > 2 ? atoi(argv[2]) : 200;
  cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
  printf("GPU: %s (sm_%d%d)   R=%d spots/step   iters=%d\n", prop.name, prop.major, prop.minor, R, iters);
  cudaStream_t s; CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));

  printf("\n%-6s %-7s | %-9s | %12s %12s %12s %12s | %9s %9s\n", "image", "rows M", "fused==unf",
         "unf eager", "unf graph", "fus eager", "fus graph", "fusion x", "graph x");
  printf("%-6s %-7s | %-9s | %12s %12s %12s %12s | %9s %9s\n", "", "(B=2)", "", "ms/step", "ms/step",
         "ms/step", "ms/step", "(eager)", "(unfused)");
  const int sizes[] = {256, 512, 1024, 2048};
  for (int size : sizes) {
    int N = (size / 16) * (size / 16), M = 2 * N;       // CFG doubles the batch
    Bufs b; b.M = M;
    size_t nH = (size_t)M * H;
    CK(cudaMalloc(&b.x, nH * 2)); CK(cudaMalloc(&b.xmod, nH * 2)); CK(cudaMalloc(&b.norm, nH * 4));
    CK(cudaMalloc(&b.scale, H * 2)); CK(cudaMalloc(&b.shift, H * 2));
    CK(cudaMalloc(&b.packed, (size_t)M * NB * 8)); CK(cudaMalloc(&b.bscale, (size_t)M * NB));
    uint64_t* packed2; uint8_t* bscale2;
    CK(cudaMalloc(&packed2, (size_t)M * NB * 8)); CK(cudaMalloc(&bscale2, (size_t)M * NB));

    std::vector<__nv_bfloat16> hx(nH), hs(H), hh(H);
    srand(1);
    auto rnd = []() { float u = 0; for (int i = 0; i < 12; i++) u += rand() / (float)RAND_MAX; return u - 6.f; };  // ~N(0,1)
    for (size_t i = 0; i < nH; i++) hx[i] = __float2bfloat16(rnd());
    for (int i = 0; i < H; i++) { hs[i] = __float2bfloat16(0.1f * rnd()); hh[i] = __float2bfloat16(0.1f * rnd()); }
    CK(cudaMemcpy(b.x, hx.data(), nH * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(b.scale, hs.data(), H * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(b.shift, hh.data(), H * 2, cudaMemcpyHostToDevice));

    // correctness: unfused output vs fused output must be byte-identical
    spot_unfused(b, s);
    fused_kernel<<<M, THREADS, 0, s>>>(b.x, b.scale, b.shift, packed2, bscale2);
    CK(cudaStreamSynchronize(s));
    std::vector<uint64_t> p1((size_t)M * NB), p2((size_t)M * NB);
    std::vector<uint8_t> s1((size_t)M * NB), s2((size_t)M * NB);
    CK(cudaMemcpy(p1.data(), b.packed, p1.size() * 8, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(p2.data(), packed2, p2.size() * 8, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(s1.data(), b.bscale, s1.size(), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(s2.data(), bscale2, s2.size(), cudaMemcpyDeviceToHost));
    size_t diff = 0;
    for (size_t i = 0; i < p1.size(); i++) diff += (p1[i] != p2[i]) + (s1[i] != s2[i]);

    double ue = time_step(spot_unfused, b, R, iters, false, s);
    double ug = time_step(spot_unfused, b, R, iters, true, s);
    double fe = time_step(spot_fused, b, R, iters, false, s);
    double fg = time_step(spot_fused, b, R, iters, true, s);
    char eq[32]; snprintf(eq, sizeof eq, diff ? "%zu DIFF" : "identical", diff);
    printf("%-6d %-7d | %-9s | %12.3f %12.3f %12.3f %12.3f | %8.1fx %8.1fx\n", size, M, eq, ue, ug, fe, fg,
           ue / fe, ue / ug);
    cudaFree(b.x); cudaFree(b.xmod); cudaFree(b.norm); cudaFree(b.scale); cudaFree(b.shift);
    cudaFree(b.packed); cudaFree(b.bscale); cudaFree(packed2); cudaFree(bscale2);
  }
  printf("\nRead it: 'fusion x' = benefit of fewer kernels + no DRAM round trip. 'graph x' = benefit of\n"
         "removing CPU launch cost alone. If graph x is big at small images and shrinks at large ones,\n"
         "the small-image win is launch overhead, not memory traffic.\n");
  return 0;
}
