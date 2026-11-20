# SPDX-License-Identifier: Apache-2.0
"""Check and create a local release commit and annotated tag. No networking."""

import argparse
import datetime as dt
from pathlib import Path
import re
import subprocess


def git(*args):
    return subprocess.check_output(["git", *args], text=True).strip()


def changelog(version):
    text = Path("CHANGELOG.md").read_text()
    match = re.search(r"^## \[" + re.escape(version) + r"\](?:[^\n]*)\n(.*?)(?=^## |\Z)", text, re.M | re.S)
    if not match or not match[1].strip():
        raise ValueError("version requires a nonempty CHANGELOG entry")
    return match[1].strip()


def run(version, confirm):
    if not re.fullmatch(r"(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)", version):
        raise ValueError("expected a semantic version x.y.z")
    if git("status", "--porcelain"):
        raise ValueError("release requires a clean working tree and index")
    changelog(version)
    tag = "v" + version
    if git("tag", "--list", tag):
        raise ValueError("tag already exists")
    p = Path("CITATION.cff")
    old = p.read_text()
    if not all(re.search("^" + k + ": .+", old, re.M) for k in ("version", "date-released")):
        raise ValueError("incomplete citation")
    new = re.sub(r"^version: .+$", "version: " + version, old, flags=re.M)
    new = re.sub(r"^date-released: .+$", "date-released: " + dt.date.today().isoformat(), new, flags=re.M)
    print(
        f"Release {version}: citation date {dt.date.today()}, changelog checked, local annotated tag {tag}."
    )
    if confirm:
        if old != new:
            p.write_text(new)
            git("add", "CITATION.cff")
            git("commit", "-m", "Prepare " + tag + " release citation")
        git("tag", "-a", tag, "-m", "BrassZero " + tag)
    else:
        print("Dry run. Use CONFIRM=1 to create the local tag.")
    print("git push origin HEAD")
    print("git push origin " + tag)


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("version")
    cli.add_argument("--confirm", action="store_true")
    cli.add_argument("--notes", action="store_true")
    args = cli.parse_args()
    if args.notes:
        print(changelog(args.version))
    else:
        run(args.version, args.confirm)
