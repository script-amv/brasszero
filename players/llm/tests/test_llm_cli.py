# SPDX-License-Identifier: Apache-2.0
import json

import brass
from brass_ai.replay_archive import Archive, game_file


def test_profile_file_cli_with_injected_provider_records_and_replays_one_native_game(
    tmp_path, monkeypatch, capsys
):
    from brass_llm import ProviderRegistry, ProviderReply
    from brass_llm.cli import main

    registry = ProviderRegistry()

    class Fixture:
        def complete(self, profile, req):
            assert "18446744073709551609" not in req.turn_json
            actions = json.loads(req.turn_json)["actions"]
            index = next(
                (i for i, a in enumerate(actions) if a.startswith("develop ")), 0
            )
            return ProviderReply(
                json.dumps({"turn_id": req.turn_id, "action_id": index}),
                reported_model="fixture/model-v1",
                usage={"total_tokens": 2},
            )

    registry.register("fixture", "v1", lambda lease: Fixture())
    profile = tmp_path / "profile.json"
    profile.write_text(
        json.dumps({"name": "Fixture", "model": "fixture/alias", "provider": "fixture"})
    )
    monkeypatch.setenv("BRASS_FIXTURE_TOKEN", "fixture-key-for-tests")
    root = tmp_path / "archive"
    seed = 18446744073709551609
    assert (
        main(
            [
                "--seats",
                str(profile),
                "greedy",
                "random",
                "greedy",
                "--seed",
                str(seed),
                "--archive",
                str(root),
                "--key-env",
                "BRASS_FIXTURE_TOKEN",
            ],
            registry=registry,
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["seed"] == seed
    (record,) = Archive(root).records()
    assert record["status"] == "completed"
    assert record["seats"][0]["llm"]["profile"]["model"] == "fixture/alias"
    assert record["seat_identities"][0]["kind"] == "llm"
    status = record["llm_status"]["0"]
    assert status["reported_model"] == "fixture/model-v1"
    assert (
        0 < status["calls"] <= 100
        and status["usage"]["total_tokens"] == 2 * status["calls"]
    )
    text = json.dumps(record)
    for forbidden in ["fixture-key-for-tests", "my_hand", "turn_json", "legal_moves"]:
        assert forbidden not in text
    monkeypatch.delenv("BRASS_FIXTURE_TOKEN")
    file = game_file(record)
    assert (
        file["seed"] == seed
        and file["seats"][0]["bot"] == record["seat_identities"][0]["version_id"]
    )
    replay = brass.Game(file["seed"])
    for item in file["moves"]:
        assert item["player"] == replay.current_player()
        replay.apply(json.dumps(item["move"]))
    assert (
        replay.is_over()
        and replay.vp() == record["result"]["vp"]
        and list(replay.ranking()) == record["result"]["ranking"]
    )
