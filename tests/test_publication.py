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
    assert scan_path("docs/superpowers/plans/2026-01-01-task.md") == "private path"
    assert scan_path("docs/plans/task.md") == "private path"
    path = "/" + "Us" + "ers/"
    assert "home directory" in scan_text("note.md", path, [])
    module = "gu" + "i"
    assert "private GUI import" in scan_text("x.py", "from " + module + " import helper", [])
    assert "private GUI import" in scan_text("x.ts", 'import x from "../' + module + '/helper"', [])
    assert "private identifier" in scan_text("note.md", "Example Identifier", ["example identifier"])


def test_only_the_project_address_is_public():
    assert "email address" not in list(scan_text("SECURITY.md", "Contact hello@brasszero.org.", []))
    at = "@"
    for other in (
        "someone" + at + "example.com",
        "hello" + at + "brasszero.org.evil.com",
        "xhello" + at + "brasszero.org",
    ):
        assert "email address" in list(scan_text("note.md", other, [])), other


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


@pytest.mark.parametrize(
    "fact",
    [
        '{ value: "987654", label: "self-play training games so far", claim: C-04 }',
        "{ value: '586', label: 'self-play training games so far', claim: C-04 }",
        "claim: C-04\n    value: '586'\n    label: self-play training games so far",
    ],
)
def test_home_facts_use_exact_parsed_claim_values(publication, tmp_path, monkeypatch, fact):
    import site_progress

    source = site_progress.HOME.read_text()
    original = "{ value: '1,586', label: 'self-play training games so far', claim: C-04 }"
    assert original in source
    page = tmp_path / "home.md"
    page.write_text(source.replace(original, fact))
    monkeypatch.setattr(site_progress, "HOME", page)
    with pytest.raises(ValueError, match="homepage fact.*C-04"):
        validate(*publication)


def test_committed_headlines_must_match_private_projection(publication, tmp_path, monkeypatch):
    import site_progress

    original = publication[1]
    stale = copy.deepcopy(original)
    stale["headlines"]["record"] = "987654 victory points"
    committed, private = tmp_path / "claims.json", tmp_path / "ledger.md"
    committed.write_text(json.dumps(stale))
    private.touch()
    monkeypatch.setattr(site_progress, "CLAIMS", committed)
    monkeypatch.setattr(site_progress, "LEDGER", private)
    monkeypatch.setattr(site_progress, "approved", lambda: original)
    with pytest.raises(ValueError, match="rendered claims differ"):
        site_progress.validate_current()


def test_dataset_omits_cursor_branches_and_private_aliases():
    from brass_ai.replay_archive import engine_id, labeled_seats, seat_identities

    seats = labeled_seats(["operator-laptop.local", "greedy", "greedy", "random"])
    record = {
        "format": "brass-replay/1",
        "game_id": "1" * 64,
        "engine_id": engine_id(),
        "category": "evaluation",
        "provenance": {},
        "status": "completed",
        "initial": {"seed": 12},
        "seats": seats,
        "seat_identities": seat_identities(seats),
        "moves": [],
        "result": {"ranking": [0, 1, 2, 3], "vp": [4, 3, 2, 1], "winner": 0},
    }
    reason, projected = public_record(record)
    assert reason == "included"
    assert projected["seats"][0] == {"kind": "ai", "version_id": None}
    assert "operator-laptop" not in json.dumps(projected)
    assert projected["seats"][1]["version_id"] == record["seat_identities"][1]["version_id"]
    record["provenance"]["cursor"] = 0
    assert public_record(record) == ("changed lineup or branch", None)
