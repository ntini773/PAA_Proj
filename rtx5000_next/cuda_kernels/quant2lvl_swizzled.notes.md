# quant2lvl_swizzled.cu notes

- quant kernel: 128 threads/block, one thread per 16-element block, grid = ceil(M*K/16 / 128). Two 16B loads in, one 8B store out.
- Swizzle offset: tile=(r/128)*ncb+(c/4); off=tile*512+(r%32)*16+((r%128)/32)*4+(c%4). Checked against torchao `to_blocked` source (view/permute/transpose).
- Padding bytes zeroed by a separate `pad_zero_kernel`, launched only when rows_p*cols_p != M*K/16. Scales buffer is `empty`, so every byte is written exactly once.
- amax kernel: 256 threads, grid = min(ceil(n/8/256), 8*SMs), uint4 loads, warp shuffle, atomicMax on int bits (valid for non-negative floats) into a `torch::zeros` scratch float. No host sync; graph-capture safe.
- amax mode: quant kernel computes pts = __fdiv_rn(amax, 2688) itself; block 0 thread 0 writes the pts output tensor.
- Math uses __fdiv_rn/__fmul_rn; no fast-math flags allowed (do not build with --use_fast_math).
- Risk: Triton `amax / 6.0` may lower to approximate division; here it is IEEE (matches torch). If Triton differs, a few block scales may differ. Compare on real data.
- Risk: NaN/Inf inputs and pts == 0 are not handled specially (atomicMax int trick ignores NaN ordering).
- Risk: e4m3 encode uses cvt.rn.satfinite; denormal scales (eps clamp 2^-9 is the min normal-ish) should match Triton's cast.
- Not compile-tested with torch (nvcc only checked if available); no python was run.
