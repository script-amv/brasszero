#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$root"
command -v uv >/dev/null
export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1
rustup component add rustfmt clippy
sh scripts/setup-core.sh
npm --prefix site ci --no-audit --no-fund
.venv/bin/pre-commit install
if [ -d gui/.git ]; then
  echo "Installing the private GUI and training/inference runtime"
  sh scripts/setup-training.sh
  uv sync --project gui/server --locked
  npm --prefix gui/web ci --no-audit --no-fund
  npm --prefix gui/web run build
fi
