#!/usr/bin/env bash
# GPU checks on the real PixArt-Sigma weights at 256 px, one test per process (keeps host memory low on small machines).
#   bash tests_gpu/run_gpu_checks.sh [SIZE]       needs the PixArt weights in the HF cache and prompts/embeds_pixart.pt
cd "$(dirname "$0")/.."
source ./pyenv.sh
SIZE=${1:-256}
fail=0
for t in test_layer_exact_vs_torchao test_model_error_matches_stock_and_compile_engages test_w4a16_real test_dfa_licfg_metrics; do
  echo "== $t"
  out=$($PY tests_gpu/test_gpu_smoke.py --size $SIZE --only $t 2>&1); rc=$?
  echo "$out" | grep -E "^(PASS|FAIL|SKIP)|checks passed|Traceback|Error|mean rel L2|rel L2 vs|plan counts|DFA vs"
  if [ $rc -ne 0 ] || ! echo "$out" | grep -q "checks passed"; then fail=1; echo "  -> $t FAILED (exit $rc)"; fi
done
[ $fail -eq 0 ] && echo "GPU CHECKS: ALL PASSED" || echo "GPU CHECKS: FAILURES"
exit $fail
