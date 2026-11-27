# SPDX-License-Identifier: Apache-2.0
"""Owner-selected line uses exact policies and the inherited BC training profile."""
import json
from pathlib import Path

import pytest

from test_train import coordinator  # noqa: F401
from brass_rl.config import TrainConfig

ROOT = Path(__file__).resolve().parents[2]
BC = 'policy-sha256-v1:3466c7765c2d9fb8abb6ae249469aec8b1cefc1c3593279fa47f0ad14f74d4a1'
CONTROL = 'policy-sha256-v1:223cf6014af1a1e800b914b373f7ed555be3a870525a66879487c9a2f25c4201'


def test_normal_profile_matches_score_terminal_and_inherited_league():
    cfg = TrainConfig.from_json(ROOT/'training/configs/main-bc.json')
    assert (cfg.ppo.reward, cfg.ppo.critic_target, cfg.ppo.gae_lambda) == ('score', 'terminal', .95)
    assert (cfg.games_per_update, cfg.concurrent_games, cfg.runtime.torch_threads) == (2, 2, 1)
    assert cfg.eval.interval_updates == 0
    assert (cfg.league.lineup_probabilities, cfg.league.lineup_schedule) == ((.5, .35, .15), 'sample')
    assert (cfg.league.max_snapshots, cfg.league.snapshot_interval_updates, cfg.league.holdout_update) == (8, 10, 5)


def test_inventory_uses_exact_identity_for_branches_and_recommendation():
    path = ROOT / 'players/models/catalog.json'
    if not path.exists():
        pytest.skip('local scientific catalog is absent from a public checkout')
    cat = json.loads(path.read_text())
    rows = [r for section in ('models','other_preserved_snapshots','unavailable_preserved_snapshots')
            for r in cat[section]]
    assert len({r['policy_id'] for r in rows}) == len(rows)
    assert {r['lineage'] for r in rows} == {'mainline', 'experimental'}
    selected = [r for r in rows if r['name'] == cat['recommended']]
    assert len(selected) == 1 and selected[0]['lineage'] == 'mainline'
    assert cat['development_selected_best'] == 'ppo-600'
    for update, identity in ((738, CONTROL), (788, BC)):
        peers = [r for r in rows if r['update'] == update]
        assert {r['policy_id'] for r in peers if r['lineage'] == 'mainline'} == {identity}
        assert len([r for r in peers if r['lineage'] == 'experimental']) >= 2
    for update in range(689,789):
        assert sum(r['update'] == update and r['lineage'] == 'mainline' for r in rows) == 1


def test_bc_preparation_creates_once_and_refuses_unrelated_existing_run(tmp_path, monkeypatch):
    from brass_rl import main_bc
    from brass_rl import experiments
    import pytest

    checkpoint = tmp_path / main_bc.SOURCE
    checkpoint.mkdir(parents=True)
    (checkpoint/'manifest.json').write_bytes(b'original')
    import hashlib
    monkeypatch.setattr(main_bc, 'SOURCE_HASH', hashlib.sha256(b'original').hexdigest())
    calls = []
    def fake_branch(source, run):
        assert source == checkpoint
        boundary = run/'updates/000788'
        boundary.mkdir(parents=True)
        (boundary/'manifest.json').write_bytes(b'original')
        (run/'experiment.json').write_text(json.dumps({
            'format': experiments.FORMAT, 'id': 'exp-child',
            'parent': main_bc.source_parent()}))
        calls.append(run)
    monkeypatch.setattr(experiments, 'branch_experiment', fake_branch)
    run = main_bc.prepare_main_run(tmp_path)
    assert main_bc.prepare_main_run(tmp_path) == run and calls == [run]
    metadata = json.loads((run/'experiment.json').read_text())
    metadata['parent']['model_id'] = 'unrelated'
    (run/'experiment.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='parent'):
        main_bc.prepare_main_run(tmp_path)


def test_legacy_catalog_seeds_declared_tip_and_keeps_selection(tmp_path, coordinator):
    from brass_rl.library import reconcile_library
    from test_train import tiny_config
    run = tmp_path/'training/runs/trial'
    coordinator.train(tiny_config(), run, max_updates=0)
    directory = tmp_path/'players/models'
    catalog = json.loads((directory/'catalog.json').read_bytes())
    row = catalog['other_preserved_snapshots'].pop()
    row.update(lineage='mainline', branch='BC')
    catalog.update(models=[row], latest=row['name'], recommended='pinned', development_selected_best='selected')
    (directory/'catalog.json').write_text(json.dumps(catalog))
    (directory/'lineage.json').unlink()
    reconcile_library(tmp_path)
    registry = json.loads((directory/'lineage.json').read_bytes())
    assert registry['mainline_tip']['policy_id'] == row['policy_id']
    assert registry['mainline_tip']['checkpoint_sha256'] is not None
    updated = json.loads((directory/'catalog.json').read_bytes())
    assert (updated['latest'], updated['recommended'], updated['development_selected_best']) == (row['name'], 'pinned', 'selected')
