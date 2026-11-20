# SPDX-License-Identifier: Apache-2.0
"""Regressions for the public boundary, training chart denominators and QA exclusion."""

import copy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from check_public import media_text, scan_path, scan_text  # noqa: E402
from distributions import public_record  # noqa: E402
from site_progress import CLAIMS, RECORD, validate  # noqa: E402


def test_private_paths_and_imports():
    assert scan_path("gui/private.py") == "private path"
    assert scan_path("engine/oracles/vendor/README") == "private path"
    assert scan_path(".env.credentials") == "environment file"
    path = "/" + "Us" + "ers/"
    assert "home directory" in scan_text("note.md", path, [])
    module = "gu" + "i"
    assert "private GUI import" in scan_text("x.py", "from " + module + " import helper", [])
    assert "private GUI import" in scan_text("x.ts", 'import x from "../' + module + '/helper"', [])
    assert "private identifier" in scan_text("note.md", "Example Identifier", ["example identifier"])


def test_private_source_alias_marks_qa_for_export():
    record = {"category": "evaluation", "provenance": {"sources": [{"path": "old/qa/run"}]}}
    assert public_record(record) == ("qa/validation", None)


@pytest.fixture(scope="module")
def publication():
    return json.loads(RECORD.read_text()), json.loads(CLAIMS.read_text())


def test_unknown_update_cannot_become_zero(publication):
    record, claims = publication
    candidate = copy.deepcopy(record)
    next(row for row in candidate["scores"] if row[0] != 0)[0] = 0
    with pytest.raises(ValueError, match="missing update"):
        validate(candidate, claims)


def test_every_plotted_metric_has_samples(publication):
    record, claims = publication
    candidate = copy.deepcopy(record)
    del candidate["windows"][0]["samples"]["network"]
    with pytest.raises(ValueError, match="denominator"):
        validate(candidate, claims)


def test_rejects_unapproved_numeric_page_headline(publication, tmp_path, monkeypatch):
    import site_progress

    page = tmp_path / "site/src/pages/report.astro"
    page.parent.mkdir(parents=True)
    page.write_text("<h1>Our score is 999999</h1>")
    monkeypatch.setattr(site_progress, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="headline"):
        validate(*publication)


def test_reviewed_art_keeps_its_metadata_inside_the_public_boundary():
    image = ROOT / "site/src/assets/art/coalbrookdale.jpg"
    encoded = image.read_bytes()
    data = encoded + b"private-test-identifier"
    assert "private identifier" in scan_text(image.name, media_text(data), ["private-test-identifier"])
    assert list(scan_text(image.name, media_text(encoded), [])) == []
