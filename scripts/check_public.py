# SPDX-License-Identifier: Apache-2.0
"""Audit tracked and candidate files; the pre-commit mode reads exact index bytes."""

import argparse
import ast
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DENIED_PARTS = {
    ".git",
    ".superpowers",
    ".worktrees",
    "vendor",
    "rules-sources",
    "outreach",
    "yuiffy-adapter",
    "node_modules",
    ".venv",
    "__pycache__",
}
DENIED_ROOTS = (
    "gui/",
    "lab/",
    "players/models/",
    "training/runs/",
    "replays/games/",
    "replays/locks/",
    "replays/index.jsonl",
    "docs/archive/",
    "docs/superpowers/",
    "docs/plans/",
    "dist/",
)
IMAGES = {".svg", ".jpg", ".jpeg", ".png", ".webp", ".gif", ".ico", ".avif"}
BANNED = (
    (re.compile(r"/(?:Us" r"ers|ho" r"me)/|(?<![\w/])~[/\\]"), "home directory"),
    (re.compile(r"https?://(?:www\.)?kaggle\.com/co" r"de/"), "private notebook link"),
    (
        re.compile(
            r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|(?:gh[pousr]_|github_pat_|sk"
            r"-live-|AKIA)[A-Za-z0-9_]{16,}"
        ),
        "credential",
    ),
    (re.compile(r"(?<![\w@])[\w.+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?![\w.])"), "email address"),
)
# The one public project address (DECISIONS.md D13). Every other address is rejected.
PROJECT_ADDRESS = "hello@brasszero.org"


def git(*args, check=True):
    return subprocess.run(["git", *args], cwd=ROOT, check=check, stdout=subprocess.PIPE).stdout


def scan_path(name):
    p = Path(name)
    if p.is_absolute() or ".." in p.parts or set(p.parts) & DENIED_PARTS or name.startswith(DENIED_ROOTS):
        return "private path"
    if any(piece.startswith(".env") for piece in p.parts):
        return "environment file"
    return None


def scan_text(name, content, private):
    for pattern, message in BANNED:
        if any(m.group(0) != PROJECT_ADDRESS for m in pattern.finditer(content)):
            yield message
    if any(value.casefold() in content.casefold() for value in private):
        yield "private identifier"
    if name.endswith(".py"):
        try:
            tree = ast.parse(content)
        except SyntaxError as exc:
            yield "invalid Python: " + str(exc)
            return
        for node in ast.walk(tree):
            modules = (
                [a.name for a in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
                if isinstance(node, ast.ImportFrom)
                else []
            )
            if any(m.split(".")[0] in {"gui", "brass_gui"} for m in modules):
                yield "private GUI import"
    if name.endswith((".mjs", ".js", ".tsx", ".ts", ".astro", ".py")):
        if re.search(
            r"""(?:from\s*|import\s*\(?\s*|require\s*\(\s*)["'](?:\.\.?/)*(?:gui/|brass_gui\b)""", content
        ):
            yield "private GUI import"


def media_text(data):
    """Inspect JPEG header metadata and any trailing document, excluding entropy-coded pixels."""
    if data.startswith(b"\xff\xd8"):
        offset = 2
        segments = []
        while offset < len(data) and data[offset] == 255:
            while offset < len(data) and data[offset] == 255:
                offset += 1
            if offset >= len(data):
                break
            marker = data[offset]
            offset += 1
            if marker in (0xDA, 0xD9):
                break
            size = int.from_bytes(data[offset : offset + 2], "big")
            if size < 2 or offset + size > len(data):
                raise ValueError("invalid JPEG metadata block")
            if 0xE0 <= marker <= 0xEF or marker == 0xFE:
                segments.append(data[offset + 2 : offset + size])
            offset += size
        end = data.rfind(b"\xff\xd9")
        if end < 0:
            raise ValueError("missing JPEG terminator")
        segments.append(data[end + 2 :])
        data = b"\n".join(segments)
    return b"\n".join(re.findall(rb"[ -~]{6,}", data)).decode("ascii")


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--staged", action="store_true")
    args = cli.parse_args()
    opts = ["--cached"] if args.staged else ["--cached", "--others", "--exclude-standard"]
    names = sorted(set(os.fsdecode(n) for n in git("ls-files", "-z", *opts).split(b"\0") if n))
    owner_email = git("config", "--get", "user.email", check=False).decode().strip()
    private = [owner_email] if owner_email else []
    local = ROOT / "lab/private-strings.txt"
    if local.is_file():
        private += [s.strip() for s in local.read_text().splitlines() if s.strip() and not s.startswith("#")]

    def read(name):
        return git("show", ":" + name) if args.staged else (ROOT / name).read_bytes()

    allow = {n for n in read(".public-files").decode().splitlines() if n and not n.startswith("#")}
    credits = "site/src/assets/art/CREDITS"
    allowed_art = {s.split(" | ", 1)[0] for s in read(credits).decode().splitlines() if " | " in s}
    problems = []
    modes = {
        os.fsdecode(r.split(b"\t", 1)[1]): r.split(b" ", 1)[0]
        for r in git("ls-files", "--stage", "-z").split(b"\0")
        if r
    }
    for name in names:
        p = ROOT / name
        if not args.staged and not p.exists() and not p.is_symlink():
            continue
        problem = scan_path(name) or (None if name in allow else "path absent from reviewed .public-files")
        if problem or modes.get(name) in {b"120000", b"160000"} or (not args.staged and p.is_symlink()):
            problems.append((name, problem or "symbolic link or nested repository"))
            continue
        data = read(name)
        if len(data) > 5_000_000:
            problems.append((name, "larger than 5 MB"))
        if p.suffix.lower() in IMAGES:
            if name not in allowed_art:
                problems.append((name, "uncredited image"))
            if p.suffix.lower() != ".svg":
                # Credit permits the image, not arbitrary identifying EXIF/XMP/IPTC metadata.
                # Compressed pixel bytes are not text. Search embedded printable strings and
                # the common UTF-16 representations of locally configured identifiers.
                strings = media_text(data)
                problems.extend((name, msg) for msg in scan_text(name, strings, private))
                if any(
                    value.encode(codec) in data
                    for value in private
                    for codec in ("utf-8", "utf-16le", "utf-16be")
                ):
                    problems.append((name, "embedded private identifier"))
                continue
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            problems.append((name, "unreviewed binary"))
            continue
        problems.extend((name, m) for m in scan_text(name, text, private))
    for name, msg in problems:
        print(f"{name}: {msg}", file=sys.stderr)
    print(f"Audited {len(names)} public paths; {len(problems)} violations.")
    return bool(problems)


if __name__ == "__main__":
    raise SystemExit(main())
