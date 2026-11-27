# SPDX-License-Identifier: Apache-2.0
"""Check release metadata and create a local annotated tag. No networking."""

import argparse
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
    citation = Path("CITATION.cff").read_text()
    if not re.search("^version: " + re.escape(version) + "$", citation, re.M):
        raise ValueError("CITATION.cff must name this version; change it in a pull request first")
    date = re.search(r"^date-released: (\d{4}-\d{2}-\d{2})$", citation, re.M)
    if not date:
        raise ValueError("CITATION.cff needs date-released as YYYY-MM-DD")
    print(f"Release {version}: citation date {date[1]}, changelog checked, local annotated tag {tag}.")
    if confirm:
        git("tag", "-a", tag, "-m", "BrassZero " + tag)
    else:
        print("Dry run. Use CONFIRM=1 to create the local tag.")
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
