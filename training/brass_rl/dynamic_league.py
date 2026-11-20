# SPDX-License-Identifier: Apache-2.0
"""Frozen main-history opponents with versioned, resumable TrueSkill state."""
from copy import deepcopy
import math
import random

from .league import Lineup, LineupEntry, Snapshot, _digest


def mainline_rows(rows, tip):
    """Select the canonical producing branch and the original export per update."""
    eligible = {}
    for row in rows:
        update = row['update']
        branch = row['branch']
        if row['lineage'] != 'mainline' or update > tip:
            continue
        allowed = ('school', 'school-original') if update <= 688 else (
            ('control',) if update <= 738 else ('BC',))
        if branch in allowed:
            eligible.setdefault(update, []).append(row)
    selected = []
    for update, group in sorted(eligible.items()):
        originals = [r for r in group if r['branch'] == 'school-original']
        candidates = originals or group
        if len({r['policy_id'] for r in candidates}) != 1:
            raise ValueError('ambiguous mainline identities at update %d' % update)
        selected.append(candidates[0])
    return selected


def catalog_members(project, tip, stride=20):
    from .model_inventory import SECTIONS
    from .policy_io import inspect_policy, read_json
    catalog = read_json(project/'players/models/catalog.json')
    rows = mainline_rows([row for k in SECTIONS for row in catalog.get(k, [])], tip)
    by_update = {r['update']: r for r in rows}
    members = []
    for update in range(0, tip+1, stride):
        if update not in by_update:
            raise ValueError('missing canonical checkpoint %d' % update)
        row = by_update[update]
        policy = inspect_policy(project/row['path'])
        if (policy.update != update or policy.id != row['policy_id'] or
                dict(policy.content_hashes) != row['files']):
            raise ValueError('catalog row differs from its immutable policy')
        members.append(Snapshot(policy.id, policy.portable_path/'policy.pt', update,
                                'initial' if update == 0 else 'training'))
    return members


def _record(snapshot):
    return dict(update=snapshot.update, role=snapshot.role, sha256=_digest(snapshot.weights_path))


def _prior():
    return dict(mu=25., sigma=25./3)


class DynamicLeague:
    def __init__(self, streams, snapshots, holdouts, history, state):
        self.streams = streams
        self.snapshots = tuple(snapshots)
        self.holdouts = tuple(holdouts)
        self._history = deepcopy(history)
        self._state = deepcopy(state)

    @classmethod
    def start(cls, streams, snapshots, holdouts, *, history_extent, next_game_id):
        state = dict(format='brass-dynamic-history/1', anchor_extent=history_extent,
                     history_extent=history_extent, stride=20, next_thinning_extent=2*history_extent,
                     first_game_id=next_game_id, next_game_id=next_game_id, rated_games=[],
                     ratings={s.id: _prior() for s in snapshots})
        history = {s.id: _record(s) for s in tuple(snapshots)+tuple(holdouts)}
        return cls.restore(streams,snapshots,holdouts,history,state)

    @classmethod
    def restore(cls, streams, snapshots, holdouts, history, state):
        self = cls(streams, snapshots, holdouts, history, state)
        anchor, extent, stride = state['anchor_extent'], state['history_extent'], state['stride']
        if (state['format'] != 'brass-dynamic-history/1' or
                any(type(n) is not int or n < 1 for n in (anchor,extent,stride)) or extent < anchor):
            raise ValueError('invalid dynamic history extent')
        generation = (extent//anchor).bit_length()-1
        if stride != 20*(2**generation) or state['next_thinning_extent'] != 2*anchor*(2**generation):
            raise ValueError('inconsistent dynamic thinning schedule')
        if ([s.update for s in snapshots] != list(range(0,extent,stride)) or len(snapshots)<3 or
                any(s.role != ('initial' if s.update==0 else 'training') for s in snapshots) or
                any(s.role!='holdout' for s in holdouts) or
                len({s.id for s in tuple(snapshots)+tuple(holdouts)}) != len(snapshots)+len(holdouts)):
            raise ValueError('invalid dynamic membership')
        if (set(state['ratings']) != {s.id for s in snapshots} or
                any(not math.isfinite(r['mu']) or not math.isfinite(r['sigma']) or r['sigma'] <= 0
                    for r in state['ratings'].values())):
            raise ValueError('invalid dynamic ratings')
        first, following = state['first_game_id'], state['next_game_id']
        counted = state['rated_games']
        if (type(first) is not int or type(following) is not int or not 0<=first<=following or
                counted != sorted(set(counted)) or any(type(g) is not int or not first<=g<following for g in counted)):
            raise ValueError('invalid dynamic counted results')
        for s in tuple(snapshots)+tuple(holdouts):
            self.verify(s)
        return self

    @property
    def state(self):
        return deepcopy(self._state)

    @property
    def history(self):
        return deepcopy(self._history)

    @property
    def stride(self):
        return self._state['stride']

    def verify(self, snapshot):
        if self._history.get(snapshot.id) != _record(snapshot):
            raise ValueError('dynamic snapshot differs from immutable history')

    def sample(self, game_id):
        pair = game_id//2
        current = LineupEntry('current', trainable=True)
        if game_id%2 == self.streams.seed('dynamic.self_play', pair)%2:
            return Lineup((current,)*4)
        rng = random.Random(self.streams.seed('opponents',game_id))
        focal = rng.randrange(4)
        remaining = [s.id for s in self.snapshots]
        entries = []
        for seat in range(4):
            if seat == focal:
                entries.append(current)
                continue
            ratings = [self._state['ratings'][sid] for sid in remaining]
            peak = max(r['mu'] for r in ratings)
            strength = [math.exp((r['mu']-peak)/(25/6)) for r in ratings]
            sigma = [r['sigma'] for r in ratings]
            a, b = sum(strength), sum(sigma)
            weights = [.6*s/a+.2*u/b+.2/len(remaining) for s,u in zip(strength,sigma)]
            selected, = rng.choices(remaining, weights)
            remaining.remove(selected)
            entries.append(LineupEntry('frozen',selected))
        return Lineup(tuple(entries))

    def record_training(self, game_id, lineup, ranking):
        if game_id != self._state['next_game_id']:
            raise ValueError('dynamic rating needs next unprocessed training game')
        if len(ranking)!=4 or any(type(r) is not int or not 0<=r<=3 for r in ranking):
            raise ValueError('invalid native finishing ranks')
        frozen = [(s.snapshot_id,r) for s,r in zip(lineup.entries,ranking) if s.kind=='frozen']
        if frozen:
            import trueskill
            if (len(frozen)!=3 or len({sid for sid,_ in frozen})!=3 or
                    sum(s.trainable for s in lineup.entries)!=1):
                raise ValueError('dynamic rating needs three distinct frozen opponents')
            env = trueskill.TrueSkill(mu=25., sigma=25/3, beta=25/6, tau=0., draw_probability=.1,
                                      backend='mpmath')
            ratings = self._state['ratings']
            groups = [(env.create_rating(**ratings[sid]),) for sid,_ in frozen]
            updated = env.rate(groups, ranks=[r for _,r in frozen])
            for (sid,_),(rating,) in zip(frozen,updated):
                ratings[sid] = dict(mu=float(rating.mu),sigma=float(rating.sigma))
            self._state['rated_games'].append(game_id)
        elif not all(s.trainable for s in lineup.entries):
            raise ValueError('dynamic self-play requires four current seats')
        self._state['next_game_id'] += 1

    def next_stride(self, update):
        return self.stride * (2 if update+1==self._state['next_thinning_extent'] else 1)

    def advance_main(self, update, snapshot=None):
        """Apply a confirmed canonical boundary. Experimental runs never call this."""
        state = self._state
        if update != state['history_extent']:
            raise ValueError('main-history update must advance by one')
        extent = update+1
        stride = self.next_stride(update)
        if (snapshot is not None) != (update%stride==0):
            raise ValueError('main-history admission must follow checkpoint stride')
        if snapshot is not None and (snapshot.update!=update or snapshot.role!='training' or snapshot.id in self._history):
            raise ValueError('invalid new main-history identity')
        if stride != self.stride:
            self.snapshots = tuple(s for s in self.snapshots if s.update%stride==0)
            state['ratings'] = {s.id:state['ratings'][s.id] for s in self.snapshots}
            state.update(stride=stride,next_thinning_extent=extent*2)
        if snapshot is not None:
            self._history[snapshot.id] = _record(snapshot)
            self.snapshots += (snapshot,)
            state['ratings'][snapshot.id] = _prior()
        state['history_extent'] = extent
