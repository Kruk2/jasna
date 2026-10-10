#!/usr/bin/env bash

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/.." && pwd)

if [[ -n ${JASNA_PYTHON:-} ]]; then
  python_bin=$JASNA_PYTHON
elif [[ -x $repo_root/.venv/bin/python ]]; then
  python_bin=$repo_root/.venv/bin/python
else
  python_bin=$(command -v python3 || true)
fi
if [[ -z $python_bin || ! -x $python_bin ]]; then
  printf 'Jasna Python environment not found; set JASNA_PYTHON explicitly.\n' >&2
  exit 1
fi

exec "$python_bin" "$repo_root/scripts/run_jasna_unified.py" "$@"
