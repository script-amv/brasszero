#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
set -eu
kind=${1:-ts}
case "$kind" in
  ts) vendor=engine/oracles/vendor/BrassBirmingham; adapter=engine/oracles/ts-adapter/adapter.mjs ;;
  yuiffy) vendor=engine/oracles/vendor/brass-birmingham-daiyu-ai; adapter=lab/oracles/yuiffy-adapter/target/release/yuiffy-adapter ;;
  *) echo "Expected ts or yuiffy" >&2; exit 2 ;;
esac
if test ! -d "$vendor"; then echo "SKIPPED optional $kind oracle, private reference absent"; exit 0; fi
if test "$kind" = yuiffy; then cargo build --locked --release --manifest-path lab/oracles/yuiffy-adapter/Cargo.toml; fi
cargo run --locked --release -p brass_engine --bin brass-diff -- --oracle "$kind" --games "${ORACLE_GAMES:-1}" --seed 1 --adapter "$adapter" --cases "${ORACLE_CASES:-engine/oracles/cases}"
