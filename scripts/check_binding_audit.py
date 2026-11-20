# SPDX-License-Identifier: Apache-2.0
"""Invalidate the two narrowly scoped dependency exceptions when the reviewed binding changes."""

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def audited_bytes():
    # Include every Rust source, the entire binding crate manifest, and the locked graph.
    paths = [
        ROOT / "Cargo.lock",
        ROOT / "Cargo.toml",
        ROOT / "engine/bindings/pyproject.toml",
        *sorted((ROOT / ".cargo").glob("config*")),
        *sorted((ROOT / "engine/bindings").glob("build.rs")),
        ROOT / "engine/bindings/Cargo.toml",
        *sorted((ROOT / "engine/bindings/src").rglob("*.rs")),
    ]
    h = hashlib.sha256()
    for p in paths:
        for part in [p.relative_to(ROOT).as_posix().encode(), p.read_bytes()]:
            h.update(len(part).to_bytes(8, "big"))
            h.update(part)
    return h.hexdigest()


if __name__ == "__main__":
    if audited_bytes() != "f8d42dc4a6020ff728ae849f1b964f2378ce4fb4c7e6eb2d3556f3eebe6f9dda":
        raise SystemExit(
            "Binding sources or dependency graph changed. Reassess PyO3 advisories in SECURITY.md and deny.toml."
        )
    print("Dependency exceptions match the reviewed native binding and locked dependencies.")
