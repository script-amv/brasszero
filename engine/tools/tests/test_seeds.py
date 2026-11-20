# SPDX-License-Identifier: Apache-2.0
from brass_ai.seeds import fresh_seed, game_seeds
from splitmix_ref import MASK, splitmix64, unsplitmix64


def test_game_seeds_are_64bit_distinct_and_reproducible():
    s = game_seeds(1, 1000)
    assert s == game_seeds(1, 1000) and len(set(s)) == 1000
    assert all(0 <= x < 2**64 for x in s) and min(s) > 2**32
    assert game_seeds(2, 3) != s[:3]


def test_game_seeds_prefix_stable():
    assert game_seeds(5, 3) == game_seeds(5, 10)[:3]


def test_game_seeds_not_invertible_to_a_shared_base():
    # splitmix64(splitmix64(master) ^ i) let one deal seed reveal the base for all deals.
    s = game_seeds(1, 8)
    bases = {unsplitmix64(x) ^ i for i, x in enumerate(s)}
    assert len(bases) == 8


def test_fresh_seed_in_range():
    assert 0 <= fresh_seed() < 2**64


def test_reference_splitmix_inverse_round_trips():
    for x in (0, 1, 7, MASK, 0x1234_5678_9ABC_DEF0):
        assert unsplitmix64(splitmix64(x)) == x
    assert splitmix64(0) == 0xE220A8397B1DCDAF  # engine/src/state.rs value
