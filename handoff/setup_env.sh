#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 3 ]]; then
  echo "Usage: bash handoff/setup_env.sh cpu|mps NEW_VENV_PATH PYTHON_EXECUTABLE" >&2
  exit 2
fi
mode="$1"; task_venv="$2"; task_python="$3"
case "$mode" in cpu|mps) ;; *) echo "mode must be cpu or mps" >&2; exit 2;; esac
if [[ -e "$task_venv" ]]; then echo "Refusing to change an existing environment: $task_venv" >&2; exit 1; fi
"$task_python" -m venv "$task_venv"
"$task_venv/bin/python" -m pip install --upgrade pip
"$task_venv/bin/python" -m pip install -r "$(dirname "$0")/requirements-${mode}-observed.txt"
"$task_venv/bin/python" -c 'import platform, numpy, torch; print(platform.python_version(), numpy.__version__, torch.__version__); print("MPS",torch.backends.mps.is_available())'
echo "Environment only. Training was NOT started; restore private datasets and verify paths first."
