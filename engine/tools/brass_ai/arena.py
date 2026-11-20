from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import brass

from .agents import Agent, make_agent
from .events import PublicEvent
from .seeds import game_seeds, match_agent_seed
from .replay_archive import Recorder, recording_metadata, recording_context, digest, native_id

MAX_MOVES = 2000


class AgentError(RuntimeError):
    """An agent raised or returned an invalid move index."""


@dataclass(frozen=True)
class GameResult:
    seed: int
    seats: Tuple[str, ...]
    ranking: Tuple[int, ...]
    vp: Tuple[int, ...]
    moves: int


class RefereeGame:
    """Own the real game; agents receive only their observation and public events."""

    def __init__(self, seed: int, agents: Sequence[Agent],
                 labels: Optional[Sequence[str]] = None, *, recorder=None):
        if len(agents) != 4:
            raise ValueError("RefereeGame needs exactly 4 agents")
        self._recorder = recorder
        self._agents = tuple(agents)
        self._seats = tuple(labels) if labels is not None else tuple(a.name for a in agents)
        if len(self._seats) != 4:
            raise ValueError("need one label per agent")
        self._stateful = []
        missing = object()
        for seat, agent in enumerate(self._agents):
            reset = getattr(agent, "reset", missing)
            observe = getattr(agent, "observe_public", missing)
            if reset is missing and observe is missing:
                self._stateful.append(False)
            elif callable(reset) and callable(observe):
                self._stateful.append(True)
            else:
                raise ValueError("seat %d needs both callable reset and observe_public hooks" % seat)
        self._seed = seed
        self._game = brass.Game(seed)
        self._ply = 0
        initial = self._game.observation(0).public_json()
        for seat, stateful in enumerate(self._stateful):
            if stateful:
                self._hook(seat, "reset", seat, initial)

    def _label(self, seat: int) -> str:
        return "seat %d (%s), ply %d" % (seat, self._seats[seat], self._ply)

    def _hook(self, seat: int, name: str, *args) -> None:
        try:
            getattr(self._agents[seat], name)(*args)
        except Exception as e:
            raise AgentError("%s: %s raised %s: %s" %
                             (self._label(seat), name, type(e).__name__, e)) from e

    def is_over(self) -> bool:
        return self._game.is_over()

    def pending(self) -> Tuple[int, "brass.Observation"]:
        if self.is_over():
            raise RuntimeError("game is over")
        if self._ply >= MAX_MOVES:
            raise RuntimeError("game %d exceeded %d moves" % (self._seed, MAX_MOVES))
        seat = self._game.current_player()
        return seat, self._game.observation(seat)

    def _check_index(self, seat: int, obs: "brass.Observation", index: int) -> None:
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < obs.num_legal():
            raise AgentError("%s: invalid move index %r (num_legal=%d)" %
                             (self._label(seat), index, obs.num_legal()))

    def apply_index(self, index: int) -> PublicEvent:
        seat, obs = self.pending()
        self._check_index(seat, obs, index)
        move_json = obs.legal_moves()[index]
        if self._recorder is not None:
            import json
            self._recorder.append(seat, json.loads(move_json))
        self._game.apply_index(index)  # The engine's existing validated mutation path.
        self._ply += 1
        if self._recorder is not None and self.is_over():
            self._recorder.finish(self._game)
        event = PublicEvent(self._ply, seat, move_json, self._game.observation(seat).public_json())
        # Even a failing hook must not prevent other seats seeing an already-applied move.
        first_error = None
        for recipient, stateful in enumerate(self._stateful):
            if stateful:
                try:
                    self._hook(recipient, "observe_public", event)
                except AgentError as e:
                    if first_error is None:
                        first_error = e
        if first_error is not None:
            raise first_error
        return event

    def result(self) -> GameResult:
        if not self.is_over():
            raise RuntimeError("game is not over")
        return GameResult(seed=self._seed, seats=self._seats,
                          ranking=tuple(self._game.ranking()), vp=tuple(self._game.vp()),
                          moves=self._ply)


def play_game(
    agents: Sequence[Agent],
    seed: int,
    on_move: Optional[Callable[[int, "brass.Observation", int], None]] = None,
    labels: Optional[Sequence[str]] = None,
    *, archive=None, game_id=None,
) -> GameResult:
    """Play one game; on_move observes the validated index before its apply."""
    import hashlib
    import inspect
    import uuid
    referee = RefereeGame(seed, agents, labels)
    seats = []
    for seat, agent in enumerate(agents):
        label = referee._seats[seat]
        if isinstance(agent, brass.RandomBot):
            identity, family = native_id('random'), 'random'
        elif isinstance(agent, brass.GreedyBot):
            identity, family = native_id('greedy'), 'greedy'
        elif callable(getattr(agent, 'seat_record', None)):
            seats.append(dict(agent.seat_record(), name=label))
            continue
        elif getattr(agent, 'replay_identity', None):
            identity, family = agent.replay_identity, 'ai'
        elif type(agent).__name__ == 'HumanAgent':
            seats.append({'kind': 'human', 'name': label, 'bot': None})
            continue
        else:
            # Extension algorithms should expose their immutable replay_identity.
            # Source identity is an honest fallback for simple custom agents.
            try:
                source = inspect.getsource(type(agent)).encode()
                identity = 'custom-source-sha256:' + hashlib.sha256(source).hexdigest()
            except (OSError, TypeError):
                identity = 'unresolved-extension:' + type(agent).__qualname__
            family = 'custom'
        seats.append({'kind': 'bot', 'name': label, 'bot': identity, 'bot_family': family})
    metadata = recording_metadata()
    if not metadata:
        metadata = {'category': 'terminal', 'session': uuid.uuid4().hex}
    recorder = Recorder(archive, seed=seed, seats=seats, provenance=metadata, game_id=game_id)
    referee._recorder = recorder
    try:
        while not referee.is_over():
            seat, obs = referee.pending()
            try:
                index = agents[seat].act(obs)
            except Exception as e:
                raise AgentError("%s: act raised %s: %s" %
                                 (referee._label(seat), type(e).__name__, e)) from e
            finally:
                if seats[seat].get('bot_family') == 'llm':
                    recorder.set_llm_status(seat, agents[seat].status())
            referee._check_index(seat, obs, index)
            if on_move is not None:
                on_move(seat, obs, index)
            referee.apply_index(index)
        return referee.result()
    except BaseException:
        recorder.interrupt()
        raise


def run_match(specs: Sequence[str], deals: int, master_seed: int) -> List[GameResult]:
    if len(specs) != 4:
        raise ValueError("run_match needs exactly 4 agent specs, got %d" % len(specs))
    results = []
    for k, d in enumerate(game_seeds(master_seed, deals)):
        for r in range(4):
            lineup = [specs[(s + r) % 4] for s in range(4)]
            # Agent seeds are one-way hashes of (master, deal, rotation, seat): an agent cannot
            # turn its seed into the deal seed and replay the deal to see hidden information.
            agents = [make_agent(lineup[s], match_agent_seed(master_seed, k, r, s)) for s in range(4)]
            with recording_context(category='evaluation', experiment=digest({'master_seed': master_seed, 'specs': list(specs)}),
                                   game=k*4+r, deal=k, rotation=r, interface='arena'):
                results.append(play_game(agents, d, labels=lineup))
    return results
