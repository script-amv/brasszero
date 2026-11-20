# SPDX-License-Identifier: Apache-2.0
"""Opponent/seed contracts: retention, sampling boundaries and immutable artifacts."""
from dataclasses import FrozenInstanceError

import pytest

from brass_rl.config import LeagueConfig
from brass_rl.league import League, Snapshot
from brass_rl.seeding import SeedStreams


def snapshot(tmp_path, update, role="training", identifier=None):
    path = tmp_path / (identifier or str(update))
    path.write_bytes(("weights-%d-%s" % (update, role)).encode())
    return Snapshot(identifier or "s%d" % update, path, update, role)


def test_seed_domains_separate_deals_and_agents():
    streams = SeedStreams(987654321)
    domains = ("train.deal", "dev.deal", "final.deal", "policy", "opponents", "bootstrap")
    seeds = [streams.seed(d, 7, 2) for d in domains]
    assert len(set(seeds + [987654321])) == 7
    assert seeds == [SeedStreams(987654321).seed(d, 7, 2) for d in domains]
    assert streams.seed("policy", 7, 2) != streams.seed("policy", 2, 7)
    for args in (("", 1), ("policy|7", 2), ("policy", True), ("policy", -1)):
        with pytest.raises(ValueError):
            streams.seed(*args)
    with pytest.raises(ValueError):
        SeedStreams(-1)


class Draws:
    def __init__(self, draw, seat=2):
        self.draw, self.seat, self.calls = draw, seat, 0

    def random(self):
        return self.draw

    def randrange(self, n):
        self.calls += 1
        return self.seat % n

    def choice(self, population):
        result = population[(self.calls - 1) % len(population)]
        self.calls += 1
        return result


@pytest.mark.parametrize("draw,mode", [(.25,"self"),(.499999,"self"),(.5,"frozen"),
                                        (.849999,"frozen"),(.85,"mixed"),(.999,"mixed")])
def test_lineup_modes_and_random_focal_seat(tmp_path, monkeypatch, draw, mode):
    initial = snapshot(tmp_path, 0, "initial")
    holdout = snapshot(tmp_path, 5, "holdout")
    league = League(SeedStreams(42), initial, holdouts=(holdout,))
    monkeypatch.setattr("brass_rl.league.random.Random", lambda seed: Draws(draw))
    lineup = league.sample(8)
    assert len(lineup.entries) == 4
    assert all(e.snapshot_id != holdout.id for e in lineup.entries)
    if mode == "self":
        assert [e.kind for e in lineup.entries] == ["current"] * 4
        assert all(e.trainable for e in lineup.entries)
    else:
        assert lineup.entries[2].kind == "current"
        assert [i for i,e in enumerate(lineup.entries) if e.trainable] == [2]
        others = [e.kind for i,e in enumerate(lineup.entries) if i != 2]
        assert others == (["frozen"] * 3 if mode == "frozen" else ["frozen", "random", "greedy"])
    with pytest.raises(FrozenInstanceError):
        lineup.entries[0].kind = "random"


def test_sample_is_pure_uniform_and_randomizes_focal_seat(tmp_path):
    league = League(SeedStreams(123), snapshot(tmp_path, 0, "initial"),
                    LeagueConfig(lineup_probabilities=(0,0,1)))
    league.add(snapshot(tmp_path, 1))
    lineups = [league.sample(g) for g in range(800)]
    assert lineups == [league.sample(g) for g in range(800)]
    counts = {k:0 for k in ("s0", "s1", "random", "greedy")}
    seats = [0]*4
    for lineup in lineups:
        for s, entry in enumerate(lineup.entries):
            if entry.trainable:
                seats[s] += 1
            else:
                counts[entry.snapshot_id or entry.kind] += 1
    assert all(140 < n < 260 for n in seats)
    assert all(470 < n < 730 for n in counts.values())


def test_pool_preserves_initial_anchors_recent_and_excludes_holdout(tmp_path):
    league = League(SeedStreams(9), snapshot(tmp_path,0,"initial"))
    holdout = snapshot(tmp_path,5,"holdout")
    league.add(holdout)
    for update in range(1,21):
        if update != 5:
            league.add(snapshot(tmp_path, update))
    updates = {s.update for s in league.snapshots}
    assert len(updates) == 8
    assert {0,1,18,19,20} <= updates
    older = updates - {0,1,18,19,20}
    assert len(older) == 3 and min(older) < 8 and max(older) > 11
    assert league.holdouts == (holdout,)
    assert all(e.snapshot_id != holdout.id for g in range(50) for e in league.sample(g).entries)
    with pytest.raises(ValueError, match="ID|id|holdout"):
        league.add(snapshot(tmp_path,5,"training", identifier=holdout.id))


@pytest.mark.parametrize("capacity,expected", [(1,{0}),(2,{0,1}),(3,{0,1,10}),(4,{0,1,9,10})])
def test_small_pool_priority(tmp_path, capacity, expected):
    league = League(SeedStreams(1),snapshot(tmp_path,0,"initial"), LeagueConfig(max_snapshots=capacity))
    for u in range(1,11):
        league.add(snapshot(tmp_path,u))
    assert {s.update for s in league.snapshots} == expected


def test_snapshot_collision_and_modified_file_fail(tmp_path):
    initial = snapshot(tmp_path,0,"initial")
    league = League(SeedStreams(1), initial)
    league.add(initial)  # identical metadata/file is idempotent
    alias = Snapshot("alias", initial.weights_path, 1, "training")
    league.add(alias)  # distinct immutable IDs may share content-addressed files
    with pytest.raises(ValueError, match="ID|id"):
        league.add(Snapshot(initial.id, initial.weights_path, 2, "training"))
    initial.weights_path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="immut|chang"):
        league.sample(0)


def test_predeclared_lineups_exact_exposure_and_resume(tmp_path):
    cfg = LeagueConfig(lineup_schedule='four_of_five', lineup_probabilities=(.8, .2, 0))
    league = League(SeedStreams(42), snapshot(tmp_path, 0, "initial"), cfg)
    league.add(snapshot(tmp_path, 10))
    ids = range(1376, 1476)
    draws = {g: tuple(e.kind for e in league.sample(g).entries) for g in ids}
    assert sum(row.count('current') == 4 for row in draws.values()) == 80
    assert sum(row.count('current') == 1 and row.count('frozen') == 3 for row in draws.values()) == 20
    assert all(draws[g] == tuple(e.kind for e in league.sample(g).entries) for g in reversed(ids))
