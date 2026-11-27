# Commands (eval3)

Run from `rtx5000_next/` unless stated. `PY` is found by `pyenv.sh` (pixi env). Older eval1/eval2 commands: `../commands.md`.
Plan and rules: `docs/EVAL3_HANDOFF_PLAN.md`. Laptop specifics: `docs/LAPTOP_5060.md`.

## Gates before an unattended server run (in this order)
- `git pull && git log --oneline -1`: Update the server checkout; the commit must match the one the smoke run used.
- `bash run_eval3.sh --only W100`: Preflight (GPU, packages, tools, disk, models and metric weights in the HF cache).
- `pixi run python eval3/orchestrate.py --mode smoke --hours 1`: Smoke gate: every slot-1 stage on tiny inputs, results in `results_smoke/`; needs verdict GREEN in `results_smoke/eval3/STATUS.md`.
- `pixi run python eval3/orchestrate.py --mode slot1 --hours 4.8 --dry-run`: Print the slot-1 schedule and what a shorter budget would skip. `--hours` can be changed.
- `setsid nohup pixi run python eval3/orchestrate.py --mode slot1 --hours 4.8 > logs/eval3/slot1.log 2>&1 < /dev/null &`: Start the unattended slot-1 run (survives SSH loss; refuses to start without a GREEN smoke on the same commit).

## While and after a run
- `cat results/eval3/STATUS.md`: Stage-by-stage status, minutes left, verdict at the end.
- `cat results/eval3/VERIFY.md`: End-of-run verification; archive path `results_eval3.tgz` at the bottom.
- `python eval3/verify_run.py --stage W300`: Re-run one stage's checker. Any stage id `W1xx`-`W3xx` can be given.
- `python eval3/check_tests.py`: Render the test registry to `results/eval3/EVAL3_TESTS.md` (boxes ticked from result files only).

## Single stages (same code the orchestrator calls)
- `bash run_eval3.sh --list`: All stages with a one-line description and rough time.
- `bash run_eval3.sh --only W300`: Run one stage. Stage id can be changed; knobs via env (`STEP_SIZES`, `REPS`, `SIZE_Q`, `QHI`, `MODELS`, `EVAL3_RESULTS`, ...).
- `bash run_eval3.sh --dry-run --only W301`: Print the commands of one stage without running them.

## Reports (CPU only; run on the laptop after copying `results_eval3.tgz`)
- `python eval3/ablation_report.py --sizes "512 1024" --min-reps 3 --quality ladder_512`: Ladder waterfall, leave-one-out, paired quality vs the final stack.
- `python eval3/amdahl.py --model pixart_sigma`: Per-class time vs image size and the Amdahl explanation of the 1024 px speed-up.
- `python eval3/roofline_report.py`: Roofline tables and figures with consistency checks.
- `python eval3/cross_gpu.py --runs server=results_server laptop=results`: Compare roofs and speed-ups across machines. Names and folders can be changed.
- `python eval3/report_levers.py --since "$(cat results/eval3/CLEAN_SINCE)"`: One-paste step-time and quality summary.
- `python eval3/kernel_isolation.py --quick`: Quantize kernel alone (torchao vs ours), bit-exactness and GB/s; drop `--quick` for all shapes.
