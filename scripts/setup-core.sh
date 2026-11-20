#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$root"
export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1
uv venv --allow-existing --python "${PYTHON:-python3}" .venv
uv pip install --python .venv/bin/python ./engine/bindings pytest trueskill mpmath six ruff pre-commit pip-licenses
