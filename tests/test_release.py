# SPDX-License-Identifier: Apache-2.0
"""The release helper tags the reviewed commit and never adds one."""

from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from release import run  # noqa: E402


def git(*args):
    return subprocess.check_output(["git", *args], text=True).strip()


@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("CHANGELOG.md").write_text("## [0.1.0] - 2026-01-02\n\n- First.\n")
    Path("CITATION.cff").write_text("version: 0.1.0\ndate-released: 2026-01-02\n")
    git("init", "-q")
    git("config", "user.name", "Test")
    git("config", "user.email", "test")
    git("config", "commit.gpgsign", "false")
    git("config", "tag.gpgsign", "false")
    git("add", ".")
    git("commit", "-q", "--no-verify", "-m", "Initial")


def test_confirmed_release_tags_head_without_committing(repo):
    head = git("rev-parse", "HEAD")
    run("0.1.0", True)
    assert git("rev-parse", "HEAD") == head
    assert git("rev-parse", "v0.1.0^{commit}") == head


def test_dry_run_creates_no_tag(repo):
    run("0.1.0", False)
    assert git("tag", "--list") == ""


def test_citation_must_already_name_the_version(repo):
    Path("CHANGELOG.md").write_text("## [0.2.0]\n\n- Next.\n")
    git("commit", "-q", "--no-verify", "-am", "Notes")
    with pytest.raises(ValueError, match="CITATION"):
        run("0.2.0", True)
    assert git("tag", "--list") == ""
