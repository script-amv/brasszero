#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Manual bootstrap inside an already allocated interactive Linux runtime.
set -euo pipefail
usage() { echo 'Usage: cloud_bootstrap.sh [--archive NEW_TAR_GZ] | [--check] [--python PYTHON] [--venv PATH] [--core-venv PATH]' >&2; }
python_bin=python3.10
venv_path=training/.venv-linux
core_venv_path=engine/bindings/.venv
check_only=false
archive_path=
while (($#)); do
  case "$1" in
    --check) check_only=true; shift ;;
    --archive)
      if (($# < 2)); then usage; exit 2; fi
      archive_path="$2"; shift 2 ;;
    --python|--venv|--core-venv)
      if (($# < 2)); then usage; exit 2; fi
      case "$1" in
        --python) python_bin="$2" ;;
        --venv) venv_path="$2" ;;
        --core-venv) core_venv_path="$2" ;;
      esac
      shift 2 ;;
    *) usage; exit 2 ;;
  esac
done
cd "$(dirname "$0")/../.."
if [[ -n "$archive_path" ]]; then
  # Explicit tracked whitelist: HEAD only, no worktree/ignored/private run data.
  # No uploads. Preserve Cargo.lock and every engine source for fingerprints.
  ( set -o noclobber
    git archive --format=tar.gz HEAD Cargo.toml Cargo.lock engine/core engine/bindings engine/tools players/baselines training > "$archive_path"
  )
  echo 'Created tracked training-source archive from HEAD; inspect before manual transfer.'
  exit 0
fi
if [[ "$(uname -s)" != Linux ]]; then echo 'Linux runtime required; local syntax/argument checks are not Linux proof.' >&2; exit 2; fi
command -v "$python_bin" >/dev/null
command -v git >/dev/null || { echo 'Install Git for the complete source-provenance test gate, then rerun.' >&2; exit 2; }
command -v cargo >/dev/null || { echo 'Install Rust with the official rustup installer, inspect it before execution, then rerun.' >&2; exit 2; }
"$python_bin" -c 'import sys; assert sys.version_info[:2] == (3,10), "Proposed dependency profile requires Python 3.10; record/test a separate lock for other versions"'
if [[ "$check_only" == true ]]; then
  rustc --version
  "$python_bin" --version
  echo 'Preflight only: dependencies/build/runtime are still untested.'
  exit 0
fi
# Validate both destinations before creating either; never reuse/replace an env.
for destination in "$core_venv_path" "$venv_path"; do
  if [[ -e "$destination" || -L "$destination" ]]; then
    echo 'Choose new separate --venv/--core-venv paths; refusing to replace an existing environment.' >&2
    exit 2
  fi
done
"$python_bin" -c 'from pathlib import Path; import sys; a,b = [Path(p).resolve() for p in sys.argv[1:]]; assert a != b and a not in b.parents and b not in a.parents, "Core and RL environments must have distinct non-nested paths"' "$core_venv_path" "$venv_path"
"$python_bin" -m venv "$core_venv_path"
"$core_venv_path/bin/python" -m pip install maturin==1.9.6 pytest==8.4.2
VIRTUAL_ENV="$("$core_venv_path/bin/python" -c 'import sys; print(sys.prefix)')" "$core_venv_path/bin/maturin" develop --release -m engine/bindings/Cargo.toml
"$core_venv_path/bin/python" -c 'import importlib.util; import brass; assert importlib.util.find_spec("torch") is None, "Core environment must remain torch-free"'

"$python_bin" -m venv "$venv_path"
"$venv_path/bin/python" -m pip install -r training/requirements/linux.txt
VIRTUAL_ENV="$("$venv_path/bin/python" -c 'import sys; print(sys.prefix)')" "$venv_path/bin/maturin" develop --release -m engine/bindings/Cargo.toml
PYTHONPATH=engine/tools:training OMP_NUM_THREADS=1 "$venv_path/bin/python" -m brass_rl runtime --device cpu
printf 'Core CLI test interpreter: export BRASS_CORE_PYTHON="%s/bin/python"\n' "$(cd "$core_venv_path" && pwd)"
printf '%s\n' 'CPU runtime probe finished. Record pip freeze/hardware/source and test the allocated GPU separately before claiming Linux/GPU portability.'
