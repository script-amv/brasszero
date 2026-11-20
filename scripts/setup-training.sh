#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$root"
# CLI regressions check a physically separate interpreter without torch.
if test ! -x .venv/bin/python; then sh scripts/setup-core.sh; fi
case "$(uname -s)-$(uname -m)" in
 Darwin-x86_64) profile=macos-py39; interpreter=${TRAIN_PYTHON:-/usr/bin/python3} ;;
 Linux-x86_64) profile=linux; interpreter=${TRAIN_PYTHON:-python3.10} ;;
 *) echo "The verified training profiles are Linux x86_64 Python 3.10 and Intel macOS Python 3.9. Choose a verified profile." >&2; exit 2 ;;
esac
uv venv --allow-existing --python "$interpreter" training/.venv
if [ "$profile" = linux ]; then
  uv pip install --python training/.venv/bin/python --index-url https://download.pytorch.org/whl/cpu 'torch==2.2.2+cpu'
fi
uv pip install --python training/.venv/bin/python -r "training/requirements/$profile.txt" ./engine/bindings 
