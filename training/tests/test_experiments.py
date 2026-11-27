# SPDX-License-Identifier: Apache-2.0
"""Portable experiment lifecycle; uses real tiny optimizer fixtures separately from games."""
import json

import pytest

from test_train import tiny_config, coordinator  # noqa: F401


def test_new_identity_resume_preflight_and_full_state_branch(coordinator, tmp_path, capsys):
    from brass_rl.experiments import branch_experiment
    config = tiny_config()
    run = tmp_path/'training/runs/first'
    coordinator.train(config, run, max_updates=1)
    metadata = json.loads((run/'experiment.json').read_text())
    assert metadata['mode'] == 'new'
    assert (tmp_path/'players/models').is_dir()
    resumed = coordinator.train(config, run, resume=run/'latest', max_updates=1)
    assert resumed.preflight['experiment_id'] == metadata['id']
    assert resumed.preflight['optimizer'] == 'restored'
    assert resumed.preflight['rng'] == 'restored'
    assert resumed.preflight['completed_training_games'] == 2
    assert resumed.preflight['next_update'] == 2
    branch = tmp_path/'training/runs/branch'
    branch_experiment(run/'updates/000000', branch)
    fork = coordinator.train(config, branch, resume=branch/'latest', max_updates=1)
    fork_meta = json.loads((branch/'experiment.json').read_text())
    assert fork_meta['id'] != metadata['id']
    assert fork_meta['parent']['experiment_id'] == metadata['id']
    assert fork_meta['parent']['update'] == 0
    assert fork_meta['mode'] == 'full-state-branch'
    assert fork.preflight['completed_training_games'] == 0
    assert fork.preflight['restoration'] == 'full-state'
    assert 'preflight' in capsys.readouterr().out


def test_branch_rejects_existing_destination_and_resume_historical(coordinator, tmp_path):
    from brass_rl.experiments import branch_experiment
    run = tmp_path/'run'
    coordinator.train(tiny_config(), run, max_updates=2)
    with pytest.raises(ValueError, match='latest committed'):
        coordinator.train(tiny_config(), run, resume=run/'updates/000001', max_updates=3)
    with pytest.raises(ValueError, match='empty|exists'):
        branch_experiment(run/'updates/000001', run)


def test_training_reuses_curated_public_model_name(coordinator, tmp_path):
    run = tmp_path/'training/runs/local'
    coordinator.train(tiny_config(), run, max_updates=0)
    models = tmp_path/'players/models'
    existing = next(p for p in models.iterdir() if p.is_dir())
    existing.rename(models/'ppo-curated')
    coordinator.train(tiny_config(), run, resume=run/'latest', max_updates=0)
    assert [p.name for p in models.iterdir() if p.is_dir()] == ['ppo-curated']
