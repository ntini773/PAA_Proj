# Sourced by the eval2 scripts. Sets PY to a python that can import torch, or exits with a clear message.
# Order: $PY if set, python, python3, the pixi env next to the repo (../.pixi or ../../.pixi). So `bash smoke_eval2.sh` works
# with or without `pixi run` in front.
_pyenv_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for _c in "$PY" python python3 "$_pyenv_root/../.pixi/envs/default/bin/python" "$_pyenv_root/../../.pixi/envs/default/bin/python"; do
  if [ -n "$_c" ] && command -v "$_c" > /dev/null 2>&1 && "$_c" -c "import torch" > /dev/null 2>&1; then
    PY=$(command -v "$_c"); export PY; break
  fi
done
# Put the python's own bin first on PATH (what `pixi run` does): torch's CUDA-extension build needs the `ninja` executable
# that sits next to this python, and nsys/ncu children inherit the same PATH.
export PATH="$(dirname "$PY"):$PATH"
if ! "$PY" -c "import torch" > /dev/null 2>&1; then
  echo "No python with torch found. Run through pixi (pixi run bash $0) or set PY=/path/to/python." >&2
  exit 2
fi
