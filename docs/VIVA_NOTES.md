# Viva notes (skeleton)

Purpose: for each component, a short explanation, where to open the code, and likely questions. The "my contribution" section is for the student to write. Do not copy claims from here without checking the code and result files.

## 0. My contribution (STUDENT TO FILL)
- Parts I designed or wrote: ______
- Parts written together or with AI assistance, and what I checked myself: ______
- Experiments I ran and what I learned from each: ______
- Bugs I found and fixed (see `rtx5000_next/BUGS.md`): ______

## 1. NVFP4 format and why it needs a quantize step
- Explanation: TODO (5 lines). Open: `rtx5000_next/fused_linear.py` (`quantize_weight`), torchao `NVFP4Tensor`.
- Likely questions: why a per-tensor scale as well as block scales? what is swizzled scale layout? what is e2m1?

## 2. The quantize kernel
- Explanation: TODO. Open: `cuda_kernels/quant2lvl_swizzled.cu`, `quant_backends.py`, `bench/cuda_vs_triton.py`.
- Likely questions: why memory bound? how many bytes per call? what does the swizzle do?

## 3. FusedNVFP4Linear and torch.compile engagement
- Explanation: TODO. Open: `fused_linear.py` (`FusedNVFP4Linear`, `COUNTS`), `tests_cpu/test_fused_linear.py`.
- Likely questions: why the old monkeypatch fusion measured nothing (bug 17)? how does the counter prove engagement?

## 4. Attention levers
- Explanation: TODO. Open: `attention_patch.py`, `bench/attention_backends.py`.
- Likely questions: why trim is exact? why head dim 72 is a problem? why Sage2 is inside the noise floor and what could still go wrong at 1024 px?

## 5. Quality metrics and the noise floor
- Explanation: TODO. Open: `eval3/quality_grid.py`, `eval3/stats.py`, `quality/eval_images.py`.
- Likely questions: why paired differences? what is a bootstrap CI? why LPIPS alone is not enough?

## 6. Roofline method
- Explanation: TODO. Open: `bench/peaks_sweep.py`, `eval3/roofline_profile.py`, `eval3/roofline_report.py`.
- Likely questions: how roofs are measured; why efficiency above 100% means a wrong model (bug in W202: roof timed through `F.linear`).

## 7. Experiment design and honesty
- Explanation: TODO. Open: `eval3/tests.json`, `eval3/check_tests.py`, `rtx5000_next/BUGS.md`.
- Likely questions: what did not work? what are the limits? what would you do with more time?

## 8. Repository history
- Plan: small focused commits; `git log --oneline` read-through; tag `eval3` at the end.
