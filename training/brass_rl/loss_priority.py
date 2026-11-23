# SPDX-License-Identifier: Apache-2.0
"""Native rank exposure and with-replacement sampling over the inherited BC pool."""
from .league import League


def initial_exposure(game_id):
    return {'first_game_id': game_id, 'next_game_id': game_id, 'counts': {}}


class LossPriorityLeague(League):
    @classmethod
    def from_checkpoint(cls, streams, snapshots, config, holdouts, history, exposure, next_game_id):
        if (not isinstance(exposure, dict) or exposure.keys() != initial_exposure(0).keys()
                or type(exposure['first_game_id']) is not int or not 0 <= exposure['first_game_id'] <= next_game_id
                or type(exposure['next_game_id']) is not int or exposure['next_game_id'] != next_game_id):
            raise ValueError('loss-priority cursor differs from training boundary')
        league = cls.restore(streams, snapshots, config, holdouts, history)
        counts = exposure['counts']
        if not isinstance(counts, dict):
            raise ValueError('invalid loss-priority counts')
        for sid, pair in counts.items():
            if (sid not in history or history[sid]['role'] == 'holdout' or not isinstance(pair, list)
                    or len(pair) != 2 or any(type(n) is not int for n in pair)
                    or pair[1] <= 0 or not 0 <= pair[0] <= 2*pair[1]):
                raise ValueError('invalid frozen opponent exposure')
        if sum(pair[1] for pair in counts.values()) > 3*(next_game_id-exposure['first_game_id']):
            raise ValueError('too many frozen opponent encounters')
        league._first = exposure['first_game_id']
        league._next = next_game_id
        league._counts = {sid: pair.copy() for sid, pair in counts.items()}
        return league

    @property
    def state(self):
        return {'first_game_id': self._first, 'next_game_id': self._next,
                'counts': {sid: pair.copy() for sid, pair in self._counts.items()}}

    def _opponents(self, rng, eligible):
        size = len(self.snapshots)
        weights = []
        for entry in eligible[:size]:
            twice_wins, encounters = self._counts.get(entry.snapshot_id, (0, 0))
            probability = (2. + .5*twice_wins)/(4 + encounters)
            weights.append(1. + 4.*(1. - probability)**2)
        scale = size / sum(weights)
        normalized = [scale*w for w in weights] + [1.]*(len(eligible)-size)
        return rng.choices(eligible, weights=normalized, k=3)

    def record_training(self, game_id, lineup, ranking):
        if game_id != self._next:
            raise ValueError('loss-priority cursor differs from training game')
        learners = [i for i, entry in enumerate(lineup.entries) if entry.trainable]
        if len(learners) == 1:
            if len(ranking) != 4 or any(type(rank) is not int or not 0 <= rank < 4 for rank in ranking):
                raise ValueError('invalid native ranking')
            focal = ranking[learners[0]]
            frozen = [(entry.snapshot_id, ranking[i]) for i, entry in enumerate(lineup.entries)
                      if entry.kind == 'frozen']
            active = {s.id for s in self.snapshots}
            if any(sid not in active for sid, rank in frozen):
                raise ValueError('frozen opponent outside active pool')
            for sid, rank in frozen:
                tally = self._counts.setdefault(sid, [0, 0])
                tally[0] += int(focal <= rank) + int(focal < rank)
                tally[1] += 1
        elif len(learners) != 4:
            raise ValueError('loss-priority requires one or four learner seats')
        self._next += 1
