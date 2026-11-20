# SPDX-License-Identifier: Apache-2.0
import pytest

from brass_ai.__main__ import format_report, main
from brass_ai.results import read_results, write_results
from brass_ai.arena import GameResult


def test_cli_match_and_rate(tmp_path, capsys):
    out = tmp_path / "m.jsonl"
    main(["match", "--agents", "greedy", "random", "random", "random", "--deals", "1", "--seed", "9", "--out", str(out)])
    assert len(read_results([str(out)])) == 4
    main(["rate", str(out)])
    assert "greedy" in capsys.readouterr().out


def test_write_results_creates_parent_dir(tmp_path):
    p = tmp_path / "a" / "b" / "r.jsonl"
    r = GameResult(seed=1, seats=("a", "b", "c", "d"), ranking=(0, 1, 2, 3), vp=(4, 3, 2, 1), moves=5)
    write_results(str(p), [r])
    assert read_results([str(p)]) == [r]


def _match(out, *extra):
    main(["match", "--agents", "random", "random", "random", "random", "--deals", "1", "--seed", "9", "--out", str(out)] + list(extra))


def test_match_refuses_existing_out_without_append(tmp_path, capsys):
    out = tmp_path / "m.jsonl"
    _match(out)
    with pytest.raises(SystemExit) as ei:
        _match(out)
    assert ei.value.code == 2
    assert "--append" in capsys.readouterr().err
    assert len(read_results([str(out)])) == 4


def test_match_append_adds_to_existing_out(tmp_path):
    out = tmp_path / "m.jsonl"
    _match(out)
    _match(out, "--append")
    assert len(read_results([str(out)])) == 8


def test_match_unknown_spec_errors_before_running(tmp_path, capsys):
    out = tmp_path / "m.jsonl"
    with pytest.raises(SystemExit) as ei:
        main(["match", "--agents", "random", "nope", "random", "random", "--deals", "1", "--out", str(out)])
    assert ei.value.code == 2
    cap = capsys.readouterr()
    assert "nope" in cap.err and "seed:" not in cap.out
    assert not out.exists()


def test_play_unknown_spec_errors(capsys):
    with pytest.raises(SystemExit) as ei:
        main(["play", "--seat", "0", "--opponents", "random", "nope", "random"])
    assert ei.value.code == 2 and "nope" in capsys.readouterr().err


def test_report_names_rating_anchor():
    rs = [GameResult(seed=1, seats=("a", "b", "random", "random"), ranking=(0, 1, 2, 3), vp=(4, 3, 2, 1), moves=5)]
    assert "(anchor: random)" in format_report(rs)
    assert "(anchor: a)" in format_report(rs, anchor="a")
