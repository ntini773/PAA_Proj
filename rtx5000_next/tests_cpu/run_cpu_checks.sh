#!/usr/bin/env bash
# All CPU tests (no GPU needed). One process per file. Exit status is 1 if any file fails.
#   bash tests_cpu/run_cpu_checks.sh            (PY=python by default; use PY=.pixi/envs/default/bin/python if needed)
cd "$(dirname "$0")"
PY=${PY:-python}
export CUDA_VISIBLE_DEVICES=""
fail=0
for t in test_fused_linear.py test_attention_dfa.py test_sampler.py test_ledger_metrics.py; do
  echo "== $t"
  $PY $t 2>&1 | grep -E "^(PASS|FAIL|SKIP)|checks passed|Traceback|Error" || true
  [ "${PIPESTATUS[0]}" -eq 0 ] || fail=1
done
[ $fail -eq 0 ] && echo "CPU CHECKS: ALL PASSED" || echo "CPU CHECKS: FAILURES (run the failing file directly for the traceback)"
exit $fail
