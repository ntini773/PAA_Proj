#!/usr/bin/env bash
# All CPU tests (no GPU needed). One process per file. Exit status is 1 if any file fails or does not report "checks passed".
#   bash tests_cpu/run_cpu_checks.sh            (python is found automatically; see pyenv.sh)
cd "$(dirname "$0")"
source ../pyenv.sh
export CUDA_VISIBLE_DEVICES=""
fail=0
for t in test_fused_linear.py test_attention_dfa.py test_sampler.py test_ledger_metrics.py test_variants.py; do
  echo "== $t"
  out=$($PY $t 2>&1); rc=$?
  echo "$out" | grep -E "^(PASS|FAIL|SKIP)|checks passed|Traceback|Error"
  # a file passes only if python exited 0 AND it printed its summary line (a crash before the tests would print neither)
  if [ $rc -ne 0 ] || ! echo "$out" | grep -q "checks passed"; then fail=1; echo "  -> $t FAILED (exit $rc)"; fi
done
[ $fail -eq 0 ] && echo "CPU CHECKS: ALL PASSED" || echo "CPU CHECKS: FAILURES (run the failing file directly for the traceback)"
exit $fail
