"""One native seat. The network call never holds the lifecycle lock."""

import json
import threading
import uuid

import brass
from brass_ai.events import PublicEvent

from .history import public_move, public_snapshot
from .prompt import PROTOCOL_VERSION, changes, instructions
from .protocol import (
    USAGE_FIELDS,
    LLMError,
    ProviderReply,
    TurnRequest,
    compact,
    metadata_label,
    usage_fields,
)


def _object(pairs):
    result = dict(pairs)
    if len(result) != len(pairs):
        raise LLMError("reply")
    return result


def _select(reply, request, count, limit):
    if reply.error_code:
        raise LLMError(reply.error_code)
    if type(reply.content) is not str:
        raise LLMError("reply")
    if len(reply.content.encode("utf-8")) > limit:
        raise LLMError("response_size")
    try:
        obj = json.loads(reply.content, object_pairs_hook=_object)
    except (ValueError, RecursionError):
        raise LLMError("reply") from None
    if (
        type(obj) is not dict
        or obj.keys() != {"turn_id", "action_id"}
        or obj["turn_id"] != request.turn_id
        or type(obj["action_id"]) is not int
        or not 0 <= obj["action_id"] < count
    ):
        raise LLMError("reply")
    return obj["action_id"]


def _actor(public):
    if public["game_over"]:
        return None
    phase = public["phase"]
    if isinstance(phase, dict):
        return phase["Shortfall"]["player"]
    return public["turn_order"][public["turn_idx"]]


class LLMAgent:
    def __init__(self, profile, identity, provenance, credential, adapter):
        self._profile = profile
        self._identity = identity
        self._provenance = provenance
        self._adapter = adapter
        self._credential = credential
        self._lock = threading.Lock()
        self._closed = False
        self._pending = None
        self._seat = None
        self._initial = None
        self._public = None
        self._history = []
        self._reset_usage()

    @property
    def name(self):
        return self._profile.name

    @property
    def replay_identity(self):
        return self._identity

    def seat_record(self):
        return {
            "kind": "bot",
            "name": self.name,
            "bot": self.replay_identity,
            "bot_family": "llm",
            "llm": json.loads(self._provenance),
        }

    def _reset_usage(self):
        self._game_generation = object()
        self._request_sequence = 0
        self._reported_sequence = 0
        self._calls = 0
        self._totals = {k: None for k in USAGE_FIELDS}
        self._reports = {k: 0 for k in USAGE_FIELDS}
        self._model = self._provider = None
        self._routes = {}
        self._error = None

    def reset(self, seat, initial_public_json):
        public = public_snapshot(initial_public_json)
        with self._lock:
            if self._closed:
                raise LLMError("closed")
            if type(seat) is not int or not 0 <= seat < 4:
                raise LLMError("history")
            self._seat = seat
            self._initial = self._public = public
            self._history = []
            self._pending = None
            self._reset_usage()

    def observe_public(self, event):
        with self._lock:
            if self._closed:
                raise LLMError("closed")
            if (
                not isinstance(event, PublicEvent)
                or self._public is None
                or type(event.ply) is not int
                or event.ply != len(self._history) + 1
                or type(event.actor) is not int
                or event.actor != _actor(self._public)
            ):
                raise LLMError("history")
            public = public_snapshot(event.public_json)
            move = public_move(event.move_json, self._public["phase"])
            self._history.append(
                {
                    "ply": event.ply,
                    "actor": event.actor,
                    "move": move,
                    "changes": list(changes(self._public, public)),
                }
            )
            self._public = public
            self._pending = None

    def cancel(self):
        with self._lock:
            self._pending = None

    def close(self):
        with self._lock:
            self._closed = True
            self._pending = self._adapter = self._credential = None
            self._history = []
            self._initial = self._public = None

    def status(self):
        with self._lock:
            return {
                "thinking": self._pending is not None,
                "closed": self._closed,
                "calls": self._calls,
                "max_calls": self._profile.max_calls,
                "error": self._error,
                "requested_model": self._profile.model,
                "provider": self._profile.provider,
                "reported_model": self._model,
                "reported_provider": self._provider,
                "routes": [
                    {"model": m, "provider": p, "replies": n}
                    for (m, p), n in self._routes.items()
                ],
                "usage": self._totals.copy(),
                "usage_reports": self._reports.copy(),
            }

    def _prepare(self, obs):
        if self._closed:
            raise LLMError("closed")
        if self._pending is not None:
            raise LLMError("busy")
        if (
            not isinstance(obs, brass.Observation)
            or self._seat != obs.me
            or obs.num_legal() == 0
            or self._public != json.loads(obs.public_json())
        ):
            raise LLMError("history")
        if self._calls >= self._profile.max_calls:
            raise LLMError("call_limit")
        turn_id = uuid.uuid4().hex
        # Native descriptions distinguish every source, discard and continuation. Array index is the ID.
        payload = {
            "protocol": PROTOCOL_VERSION,
            "turn_id": turn_id,
            "me": self._seat,
            "my_hand": json.loads(obs.to_json())["my_hand"],
            "public": self._public,
            "initial_public": self._initial,
            "history": self._history,
            "actions": [obs.describe_move(i) for i in range(obs.num_legal())],
        }
        request = TurnRequest(turn_id, instructions(), compact(payload))
        if (
            len(request.turn_json.encode()) + len(request.instructions.encode())
            > self._profile.max_bytes
        ):
            raise LLMError("request_size")
        self._calls += 1
        self._pending = turn_id
        self._error = None
        return request

    def _account(self, reply, sequence, sensitive_labels):
        model = metadata_label(reply.reported_model, self._credential)
        provider = metadata_label(reply.reported_provider, self._credential)
        if sensitive_labels is not None:
            # A trusted controller can suppress additional labels, never add them.
            # Treat a failed filter as private metadata and keep the numeric receipt.
            try:
                blocked = set(sensitive_labels((model, provider)))
                model = None if model in blocked else model
                provider = None if provider in blocked else provider
            except Exception:  # noqa: BLE001 -- fail closed on controller filter errors
                model = provider = None
        if sequence > self._reported_sequence:
            self._reported_sequence = sequence
            self._model, self._provider = model, provider
        if model is not None or provider is not None:
            route = model, provider
            self._routes[route] = self._routes.get(route, 0) + 1
        for key, value in usage_fields(reply.usage).items():
            self._totals[key] = (self._totals[key] or 0) + value
            self._reports[key] += 1

    def reserve_turn(self):
        """Reserve a single decision before handing it to another thread. No call yet."""
        with self._lock:
            if self._closed:
                raise LLMError("closed")
            if self._pending is not None:
                raise LLMError("busy")
            self._pending = object()
            return self._pending

    def act(self, obs, *, reservation=None, sensitive_labels=None):
        """Use a reservation once. Optional receipt filtering precedes all retention."""
        with self._lock:
            if reservation is not None:
                if self._pending is not reservation:
                    raise LLMError("cancelled")
                self._pending = None
            try:
                request = self._prepare(obs)
            except LLMError as error:
                if error.code != "busy":
                    self._error = error.code
                raise
            adapter = self._adapter
            generation = self._game_generation
            self._request_sequence += 1
            sequence = self._request_sequence
        reply = None
        error = None
        try:
            reply = adapter.complete(self._profile, request)
            if not isinstance(reply, ProviderReply):
                raise LLMError("reply")
            index = _select(reply, request, obs.num_legal(), self._profile.max_bytes)
        except LLMError as exc:
            error = LLMError(exc.code)
        except Exception:  # noqa: BLE001 -- adapters can reflect credentials in any exception
            error = LLMError("provider")
        with self._lock:
            if self._closed or self._game_generation is not generation:
                raise LLMError("cancelled")
            # A cancelled move can still return a billable receipt for this game.
            if isinstance(reply, ProviderReply):
                self._account(reply, sequence, sensitive_labels)
            elif error and error.code == "request_size":
                self._calls -= 1  # wire encoder rejected before dispatch
            if self._pending != request.turn_id:
                raise LLMError("cancelled")
            self._pending = None
            if error:
                self._error = error.code
                raise error
            return index
