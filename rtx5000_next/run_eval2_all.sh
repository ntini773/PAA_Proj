#!/usr/bin/env bash
# One command for the whole eval2 run, in the order that protects your results if the GPU session ends early:
#
#   bash run_eval2_all.sh
#
#   1. no-timing stages in parallel   E01 -> (E04 | E10 | E03) -> E11 per size       (about 50 min)
#   2. timing stages, one at a time   E02 E05 E06 E07 E08 E12                        (about 95 min)
#   3. E13 (CPU) and E15 (package the big files into results_eval2.tgz)               (about 2 min)
#   4. extras, only if time is left   E09 (nsys, about 25 min), E14 (new-model smoke, about 15 min), then E15 again
# After every stage run_eval2.sh saves the small results to the git branch results-eval2 (see save_results.sh).
# A failed stage does not stop the rest. Rerun or resume any stage later with run_eval2.sh (--only / --from).
# Env: SIZES="512 1024", STEPS=14, NO_CLIP=1 (skip the CLIP metric), NO_SAVE=1 (no automatic saving), SKIP_EXTRAS=1 (stop after step 3).
cd "$(dirname "$0")"
source ./pyenv.sh

say() { echo; echo "[$(date +%H:%M:%S)] ===== $* ====="; }

say "1/4 no-timing stages in parallel (E01, E04+E10+E03, E11 per size)"
bash run_eval2_notiming.sh

say "2/4 timing stages, one at a time on the idle GPU (E02 E05 E06 E07 E08 E12)"
for s in E02 E05 E06 E07 E08 E12; do bash run_eval2.sh --only $s; done

say "3/4 E13 (ledger, roofline) and E15 (package)"
for s in E13 E15; do bash run_eval2.sh --only $s; done

if [ -z "$SKIP_EXTRAS" ]; then
  # E14 (new-model smoke: Sana, SD3.5, FLUX) is left out on purpose for this window; add it with WITH_MODELS=1
  say "4/4 extras: E09 (nsys), ${WITH_MODELS:+E14 (new-model smoke), }E15 again"
  for s in E09 ${WITH_MODELS:+E14} E15; do bash run_eval2.sh --only $s; done
fi

say "finished"
echo "Failed stages: $(ls results/.fail 2>/dev/null | tr '\n' ' ')"
echo "Done stages:   $(ls results/.done 2>/dev/null | grep '^E' | tr '\n' ' ')"
echo "Package: results_eval2.tgz. Small results are on the results-eval2 branch (if pushing worked)."
