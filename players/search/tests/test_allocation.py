# SPDX-License-Identifier: Apache-2.0
"""Hand-calculated root comparisons catch unequal/partial allocation and wrong scale."""
from dataclasses import replace
from pathlib import Path
import math
import pytest

from brass_search.config import SearchConfig
from brass_search.allocation import allocate, root_candidates, relative_to_absolute, terminal_values

CONFIGS = Path(__file__).resolve().parents[1] / 'configs'


def test_profiles_and_budget():
    smoke = SearchConfig.from_json(CONFIGS / 'smoke.json')
    pilot = SearchConfig.from_json(CONFIGS / 'pilot.json')
    assert (smoke.root_candidates, smoke.worlds_per_round, smoke.rollout_depth, smoke.max_rollouts) == (2, 2, 2, 4)
    assert (pilot.root_candidates, pilot.worlds_per_round, pilot.rollout_depth, pilot.max_rollouts) == (4, 2, 8, 12)
    calls = []
    candidates = root_candidates(['d', 'c', 'b', 'a'], [-1]*4, [0]*4, 4)
    def evaluate(key, world):
        calls.append((key, world))
        return {'a': .1, 'b': .8, 'c': .3, 'd': .2}[key]
    result = allocate(candidates, pilot, evaluate, 'd')
    assert result.selected == 'b'
    assert result.completed_rounds == 2 and result.rollouts == 12
    assert result.rollouts * pilot.rollout_depth == 96
    assert calls == [(k, w) for w in (0, 1) for k in ('a', 'b', 'c', 'd')] + [(k, w) for w in (2, 3) for k in ('b', 'c')]


@pytest.mark.parametrize('field,value', [('root_candidates', True), ('root_candidates', 0), ('worlds_per_round', 0), ('rollout_depth', -1), ('max_rollouts', -1), ('max_rollouts', False), ('candidate_chunk', 0), ('value_scale', float('nan')), ('value_scale', float('inf')), ('value_scale', True), ('wall_time_seconds', 0), ('wall_time_seconds', True)])
def test_invalid_config(field, value):
    with pytest.raises(ValueError):
        SearchConfig(**{field: value})


def test_zero_and_partial_rounds_do_not_commit():
    config = SearchConfig(root_candidates=4, worlds_per_round=2, max_rollouts=9)
    candidates = root_candidates(['a', 'b', 'c', 'd'], [-1]*4, [0]*4, 4)
    def evaluate(key, world):
        return {'a': .5, 'b': .4, 'c': .2, 'd': .1}[key] if world < 2 else 1.0
    result = allocate(candidates, config, evaluate, 'd')
    assert result.selected == 'a' and result.completed_rounds == 1
    assert result.rollouts == 9 and result.discarded_rollouts == 1
    assert result.means == {'a': .5, 'b': .4, 'c': .2, 'd': .1}
    assert allocate(candidates, replace(config, max_rollouts=1), evaluate, 'd').selected == 'd'
    assert allocate(candidates, replace(config, max_rollouts=0), evaluate, 'd').rollouts == 0


def test_score_gumbel_scale_ties_and_order():
    # b's 0.2 return advantage outweighs a's 1.5 prior/Gumbel advantage at scale 10.
    candidates = root_candidates(['b', 'a', 'c'], [-2, -.5, -9], [0, 0, 1], 2)
    cfg = SearchConfig(root_candidates=2, worlds_per_round=1, max_rollouts=2)
    assert allocate(candidates, cfg, lambda key, w: {'a': .1, 'b': .3}[key], 'c').selected == 'b'
    assert root_candidates(['b', 'a'], [-1, -1], [0, 0], 1)[0].key == 'a'
    assert root_candidates(['a', 'b'], [-1, -2], [0, 3], 1)[0].key == 'b'
    with pytest.raises(ValueError, match='finite'):
        allocate(candidates, cfg, lambda key, w: math.nan, 'c')


def test_four_seat_mapping_and_ties():
    assert relative_to_absolute([.11, .22, .33, .44], 2) == (.33, .44, .11, .22)
    assert terminal_values([2, 1, 0, 3]) == (0., 0., 1., 0.)
    assert terminal_values([0, 0, 2, 3]) == (0., 0., 0., 0.)


def test_completed_search_invariant_under_candidate_permutation():
    from itertools import permutations
    from brass_search.inference import standard_gumbel
    scores = {'a': -.2, 'b': -2., 'c': -3., 'd': -5.}
    values = {'a': .0, 'b': .9, 'c': .3, 'd': .5}
    reference = None
    for keys in permutations(scores):
        candidates = root_candidates(keys, [scores[k] for k in keys],
                                     [standard_gumbel(81, k) for k in keys], 4)
        result = allocate(candidates, SearchConfig(), lambda k, w: values[k], 'a')
        if reference is None:
            reference = result
        assert result == reference
    assert reference.selected == 'b'
