# SPDX-License-Identifier: Apache-2.0
"""Serve the local site and poll the durable indexes for new results."""

import argparse
import os
import signal
import subprocess
import sys
import time
from site_progress import ROOT, refresh


def stamp():
    paths = [
        ROOT / "replays/index.jsonl",
        ROOT / "players/models/catalog.json",
        ROOT / "lab/outreach/CLAIMS.md",
    ]
    runs = ROOT / "training/runs"
    if runs.exists():
        # Run controllers atomically replace a small pointer named latest.
        # A new scientific game also changes the archive index.
        paths.extend(runs.glob("*/latest"))
        paths.extend(runs.glob("*/*/latest"))
    return {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in paths if p.is_file()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=4321)
    args = parser.parse_args()
    refresh()
    before = stamp()
    proc = subprocess.Popen(
        # The watcher owns this process. Astro otherwise detaches under automated clients.
        [
            "npm",
            "--prefix",
            "site",
            "run",
            "dev",
            "--",
            "--ignore-lock",
            "--host",
            "127.0.0.1",
            "--port",
            str(args.port),
        ],
        cwd=ROOT,
        start_new_session=True,
    )

    def halt(*unused):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, halt)
    try:
        while proc.poll() is None:
            time.sleep(4)
            after = stamp()
            if before != after:
                try:
                    refresh()
                except Exception as exc:
                    print(f"Site refresh failed: {exc}", file=sys.stderr)
                else:
                    before = after
        return proc.returncode
    except KeyboardInterrupt:
        return 0
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
        proc.wait()


if __name__ == "__main__":
    raise SystemExit(main())
