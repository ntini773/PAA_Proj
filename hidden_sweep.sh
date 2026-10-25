#!/usr/bin/env bash
# Does the unfused boundary cost show up for other DiT widths (SD3.5, Flux)? Event timing only, ~2 min.
set -u
cd "$(dirname "$0")"
PY=${PY:-.pixi/envs/default/bin/python}
for H in 768 1152 1536 2048 3072; do
  $PY boundary_stages.py --size 512 --mode time --hidden $H 2>&1 | grep '^{'
done
