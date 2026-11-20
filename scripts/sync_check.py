# SPDX-License-Identifier: Apache-2.0
"""Check the public repository boundary around private repositories and data."""

from pathlib import Path
import subprocess

for name in (
    "gui",
    "lab",
    "players/models",
    "training/runs",
    "replays/games",
    "replays/index.jsonl",
    "replays/locks",
    "engine/oracles/vendor",
    "docs/rules-sources",
):
    p = Path(name)
    if not p.exists():
        raise SystemExit(f"Missing local workspace member: {name}")
    if name in ("gui", "lab") and not (p / ".git").exists():
        raise SystemExit(f"Not a private Git repository: {name}")
    subprocess.run(["git", "check-ignore", "-q", name], check=True)
    if subprocess.check_output(["git", "ls-files", "--", name]) or subprocess.check_output(
        ["git", "status", "--porcelain", "--", name]
    ):
        raise SystemExit(f"Public Git can see {name}")
    print("present and ignored:", name)
