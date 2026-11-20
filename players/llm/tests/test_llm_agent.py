# SPDX-License-Identifier: Apache-2.0
import json
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import brass
import pytest
from brass_ai.arena import RefereeGame
from brass_ai.events import PublicEvent

SEED = 18446744073709551603


def setup_agent(callback, **limits):
    from brass_llm import Profile, ProviderRegistry

    class Adapter:
        def complete(self, profile, request):
            return callback(request)

    registry = ProviderRegistry()
    registry.register("fixture", "v1", lambda key: Adapter())
    profile = Profile("Fixture", "test/model", provider="fixture", **limits)
    return registry.make_agent(profile, "fixture-secret")


def answer(request, index=0, **metadata):
    from brass_llm import ProviderReply

    return ProviderReply(
        json.dumps({"turn_id": request.turn_id, "action_id": index}), **metadata
    )


def position(agent):
    game = brass.Game(SEED)
    obs = game.observation(game.current_player())
    agent.reset(obs.me, obs.public_json())
    return obs


def test_reservation_is_one_shot_and_stale_handles_cannot_dispatch():
    from brass_llm import LLMError

    calls = []
    agent = setup_agent(
        lambda request: calls.append(request.turn_id) or answer(request)
    )
    obs = position(agent)
    stale = agent.reserve_turn()
    with pytest.raises(LLMError, match="already pending"):
        agent.act(obs)
    agent.cancel()
    current = agent.reserve_turn()
    with pytest.raises(LLMError, match="cancelled"):
        agent.act(obs, reservation=stale)
    assert agent.status()["thinking"] and agent.status()["calls"] == 0
    assert agent.act(obs, reservation=current) == 0
    with pytest.raises(LLMError, match="cancelled"):
        agent.act(obs, reservation=current)
    assert len(calls) == agent.status()["calls"] == 1


def test_controller_receipt_filter_is_applied_before_accounting():
    agent = setup_agent(
        lambda req: answer(
            req,
            reported_model="private-label",
            reported_provider="fixture",
            usage={"total_tokens": 8},
        )
    )
    obs = position(agent)
    seen = []

    def sensitive(labels):
        seen.append(labels)
        return {"private-label"}

    assert agent.act(obs, sensitive_labels=sensitive) == 0
    status = agent.status()
    assert seen == [("private-label", "fixture")]
    assert status["reported_model"] is None
    assert status["routes"] == [{"model": None, "provider": "fixture", "replies": 1}]
    assert status["usage"]["total_tokens"] == 8


def test_native_seat_visibility_and_full_public_history():
    requests = []
    agents = [
        setup_agent(lambda req: requests.append(req) or answer(req))
        for seat in range(4)
    ]
    referee = RefereeGame(SEED, agents)
    initial_public = json.loads(referee.pending()[1].public_json())
    snapshots = []
    for ply in range(8):
        seat, obs = referee.pending()
        assert agents[seat].act(obs) == 0
        request = requests[-1]
        payload = json.loads(request.turn_json)
        assert (
            payload["me"] == seat
            and payload["my_hand"] == json.loads(obs.to_json())["my_hand"]
        )
        assert str(SEED) not in request.turn_json
        assert (
            "rng_seed" not in request.turn_json
            and 'face_down_discard"'
            not in request.turn_json.replace('has_face_down_discard"', "")
        )
        assert payload["public"] == json.loads(obs.public_json())
        assert payload["actions"] == [
            obs.describe_move(i) for i in range(obs.num_legal())
        ]
        state = payload["initial_public"]
        assert state == initial_public
        assert len(payload["history"]) == ply
        for item, (actor, move, public) in zip(payload["history"], snapshots):
            assert (item["actor"], item["move"]) == (actor, move)
            for path, value in item["changes"]:
                node = state
                for key in path[:-1]:
                    node = node[key]
                node[path[-1]] = value
            assert state == public
        event = referee.apply_index(0)
        snapshots.append(
            (event.actor, json.loads(event.move_json), json.loads(event.public_json))
        )


def test_hidden_determinizations_have_same_payload_except_turn_nonce():
    requests = []
    agent = setup_agent(lambda req: requests.append(req) or answer(req))
    obs = position(agent)
    for n in (700, 801):
        equivalent = obs.determinize(n).observation(obs.me)
        agent.reset(obs.me, equivalent.public_json())
        agent.act(equivalent)
    first, second = [json.loads(r.turn_json) for r in requests]
    assert first.pop("turn_id") != second.pop("turn_id")
    assert first == second


@pytest.mark.parametrize(
    "mode",
    [
        "stale",
        "boolean",
        "negative",
        "large",
        "extra",
        "missing",
        "duplicate",
        "prose",
        "float",
    ],
)
def test_rejects_unbound_or_malformed_selection(mode):
    from brass_llm import LLMError, ProviderReply

    def callback(req):
        data = {"turn_id": req.turn_id, "action_id": 0}
        if mode == "stale":
            data["turn_id"] = "stale"
        if mode in ("boolean", "negative", "large", "float"):
            data["action_id"] = {
                "boolean": True,
                "negative": -1,
                "large": 999999,
                "float": 0.0,
            }[mode]
        if mode == "extra":
            data["reasoning"] = "unrequested"
        if mode == "missing":
            del data["turn_id"]
        content = json.dumps(data)
        if mode == "duplicate":
            content = content.replace('"action_id":', '"action_id":2,"action_id":')
        return ProviderReply("hello" if mode == "prose" else content)

    agent = setup_agent(callback)
    with pytest.raises(LLMError, match="reply"):
        agent.act(position(agent))
    assert agent.status()["calls"] == 1


def test_call_budget_and_accounting_include_paid_invalid_replies():
    from brass_llm import LLMError

    agent = setup_agent(
        lambda r: answer(
            r,
            9999999,
            reported_model="vendor/version",
            reported_provider="Example A",
            usage={"total_tokens": 4, "cost": 0.0},
        ),
        max_calls=1,
    )
    obs = position(agent)
    with pytest.raises(LLMError):
        agent.act(obs)
    with pytest.raises(LLMError) as error:
        agent.act(obs)
    assert error.value.code == "call_limit"
    status = agent.status()
    assert status["calls"] == 1 and status["usage"]["total_tokens"] == 4
    assert status["usage"]["cost"] == 0 and status["usage"]["prompt_tokens"] is None
    assert (
        status["reported_model"] == "vendor/version"
        and status["reported_provider"] == "Example A"
    )


def test_request_size_error_preserves_all_choices_and_never_calls_provider():
    from brass_llm import LLMError

    agent = setup_agent(lambda r: pytest.fail("oversize request sent"), max_bytes=1024)
    with pytest.raises(LLMError) as error:
        agent.act(position(agent))
    assert error.value.code == "request_size" and agent.status()["calls"] == 0


@pytest.mark.parametrize("method", ["reset", "close", "cancel", "observe_public"])
def test_pending_completion_invalidated_without_blocking(method):
    from brass_llm import LLMError

    entered, resume = Event(), Event()

    def callback(req):
        entered.set()
        assert resume.wait(3)
        return answer(req, usage={"total_tokens": 10})

    agent = setup_agent(callback)
    obs = position(agent)
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(agent.act, obs)
        try:
            assert entered.wait(2)
            assert agent.status()["thinking"]
            with pytest.raises(LLMError) as error:
                agent.act(obs)
            assert error.value.code == "busy"
            if method == "reset":
                agent.reset(obs.me, obs.public_json())
            elif method == "observe_public":
                game = brass.Game(SEED)
                move = obs.legal_moves()[0]
                game.apply(move)
                agent.observe_public(
                    PublicEvent(1, obs.me, move, game.observation(obs.me).public_json())
                )
            else:
                getattr(agent, method)()
        finally:
            resume.set()
        with pytest.raises(LLMError) as failure:
            future.result()
        assert failure.value.code == "cancelled"
        expected_tokens = 10 if method in ("cancel", "observe_public") else None
        assert agent.status()["usage"]["total_tokens"] == expected_tokens


@pytest.mark.parametrize(
    "reset,older_first", [(False, True), (False, False), (True, False)]
)
def test_cancelled_receipt_and_newer_decision_complete_independently(
    reset, older_first
):
    from brass_llm import LLMError

    entered = [Event(), Event()]
    finish = [Event(), Event()]
    calls = iter(range(2))

    def callback(req):
        i = next(calls)
        entered[i].set()
        assert finish[i].wait(3)
        return answer(
            req,
            0 if i == 0 else -1,
            reported_model=f"model-{i}",
            reported_provider=f"route-{i}",
            usage={"total_tokens": 10 + i, "cost": 1 + i},
        )

    agent = setup_agent(callback)
    obs = position(agent)
    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(agent.act, obs)
        try:
            assert entered[0].wait(2)
            agent.cancel()
            if reset:
                agent.reset(obs.me, obs.public_json())
            second = pool.submit(agent.act, obs)
            assert entered[1].wait(2)
            futures = [first, second]
            order = (0, 1) if older_first else (1, 0)
            for i in order:
                finish[i].set()
                with pytest.raises(LLMError) as failure:
                    futures[i].result(timeout=2)
                assert failure.value.code == ("cancelled" if i == 0 else "reply")
                if older_first and i == 0:
                    pending = agent.status()
                    assert pending["thinking"] and pending["error"] is None
                    assert pending["usage"]["total_tokens"] == 10
        finally:
            for event in finish:
                event.set()
    status = agent.status()
    assert not status["thinking"] and status["error"] == "reply"
    assert (
        status["calls"]
        == status["usage_reports"]["total_tokens"]
        == (1 if reset else 2)
    )
    assert status["usage"]["total_tokens"] == (11 if reset else 21)
    assert status["usage"]["cost"] == (2 if reset else 3)
    assert (status["reported_model"], status["reported_provider"]) == (
        "model-1",
        "route-1",
    )
    routes = {(r["model"], r["provider"]): r["replies"] for r in status["routes"]}
    assert routes == (
        {("model-1", "route-1"): 1}
        if reset
        else {("model-0", "route-0"): 1, ("model-1", "route-1"): 1}
    )


def test_wrong_seat_and_out_of_order_events_never_dispatch():
    from brass_llm import LLMError

    agent = setup_agent(lambda r: pytest.fail("wrong observation"))
    obs = position(agent)
    with pytest.raises(LLMError):
        agent.act(brass.Game(SEED).observation((obs.me + 1) % 4))
    with pytest.raises(LLMError):
        agent.observe_public(
            PublicEvent(2, obs.me, obs.legal_moves()[0], obs.public_json())
        )
    assert agent.status()["calls"] == 0


@pytest.mark.parametrize(
    "case",
    [
        "move_extra",
        "nested_source",
        "nested_card",
        "public_tile",
        "public_hand",
        "wrong_phase",
    ],
)
def test_event_hooks_reject_nested_non_public_fields(case):
    from brass_llm import LLMError

    agent = setup_agent(lambda r: pytest.fail("invalid hook forwarded"))
    obs = position(agent)
    game = brass.Game(SEED)
    move = json.loads(obs.legal_moves()[0])
    game.apply(json.dumps(move))
    public = json.loads(game.observation(obs.me).public_json())
    if case == "move_extra":
        move["extra"] = {"hands": []}
    elif case == "nested_source":
        move["Build"]["coal"]["from_tiles"].append({"seed": SEED})
    elif case == "nested_card":
        move["Build"]["card"]["hidden"] = 9
    elif case == "public_tile":
        next(tile for tile in public["slots"] if tile)["seed"] = SEED
    elif case == "public_hand":
        public["players"][0]["hand"] = []
    else:
        move = "EndAction"
    with pytest.raises(LLMError) as error:
        agent.observe_public(
            PublicEvent(1, obs.me, json.dumps(move), json.dumps(public))
        )
    assert error.value.code == "history"


def test_native_index_round_trip_for_all_action_classes_and_resource_selections():
    """Exercise every phase using a bounded prefix of the existing native replay fixture."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    fixture = json.loads(
        (root / "engine/tools/tests/fixtures/strategy/record-0.json").read_text()
    )
    game = brass.Game(fixture["initial"]["seed"])
    seen = set()

    def exercise(position):
        obs = position.observation(position.current_player())
        moves = [json.loads(s) for s in obs.legal_moves()]
        for i, move in enumerate(moves):
            kind = next(iter(move)) if isinstance(move, dict) else move
            if kind in seen:
                continue

            def select(req, i=i):
                actions = json.loads(req.turn_json)["actions"]
                assert len(actions) == obs.num_legal() and actions[
                    i
                ] == obs.describe_move(i)
                assert len(actions) == len(set(actions))
                return answer(req, i)

            agent = setup_agent(select)
            agent.reset(obs.me, obs.public_json())
            selected = agent.act(obs)
            assert selected == i
            branch = position.clone()
            branch.apply_index(selected)
            agent.observe_public(
                PublicEvent(
                    1,
                    obs.me,
                    json.dumps(move),
                    branch.observation(obs.me).public_json(),
                )
            )
            seen.add(kind)

    for ply in range(146):
        exercise(game)
        if ply == 145:
            # This board already permits a native sale. Exercise the repeated sale
            # and merchant development branches without playing another game.
            for phase in ("SellMore", "BonusDevelop"):
                state = json.loads(game.to_json())
                state["phase"] = phase
                exercise(brass.Game.from_json(json.dumps(state)))
            break
        game.apply(json.dumps(fixture["moves"][ply]["move"]))
    assert seen == {
        "Build",
        "Canal",
        "Rail",
        "Develop",
        "Sell",
        "Loan",
        "Scout",
        "Pass",
        "SecondRail",
        "DevelopSecond",
        "SellAnother",
        "BonusDevelop",
        "ShortfallRemove",
        "EndAction",
    }


def test_pre_dispatch_error_visible_in_status():
    from brass_llm import LLMError

    agent = setup_agent(lambda r: answer(r), max_calls=1)
    obs = position(agent)
    agent.act(obs)
    with pytest.raises(LLMError):
        agent.act(obs)
    assert agent.status()["error"] == "call_limit" and agent.status()["calls"] == 1


def test_read_only_usage_mapping_is_accounted():
    from types import MappingProxyType

    agent = setup_agent(
        lambda req: answer(req, usage=MappingProxyType({"prompt_tokens": 0}))
    )
    assert agent.act(position(agent)) == 0
    assert agent.status()["usage"]["prompt_tokens"] == 0
    assert agent.status()["usage_reports"]["prompt_tokens"] == 1


def test_unreadable_usage_mapping_settles_turn_without_partial_usage():
    from collections import UserDict

    class UnreadableUsage(UserDict):
        def __getitem__(self, key):
            if key == "completion_tokens":
                raise RuntimeError("fixture-secret")
            return super().__getitem__(key)

    receipts = iter([UnreadableUsage(prompt_tokens=8, completion_tokens=1), {"completion_tokens": 3}])
    agent = setup_agent(lambda req: answer(req, usage=next(receipts)))
    obs = position(agent)
    assert agent.act(obs) == 0
    status = agent.status()
    assert not status["thinking"] and status["error"] is None
    assert "fixture-secret" not in repr(status)
    assert all(value is None for value in status["usage"].values())
    assert all(count == 0 for count in status["usage_reports"].values())
    assert agent.act(obs) == 0
    assert agent.status()["calls"] == 2
    assert agent.status()["usage"]["completion_tokens"] == 3
