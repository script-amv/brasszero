# SPDX-License-Identifier: Apache-2.0
"""Explicit compact forks, strict legacy boundaries and full-state experiment resume."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import shutil

import pytest
import torch

from brass_rl.artifacts import capture_rng_states, committed_records, load_update
from brass_rl.policy_io import ArtifactError, inspect_policy
from test_train import assert_nested_equal, coordinator, tiny_config  # noqa: F401


def source_checkpoint(coordinator, tmp_path):
    cfg = tiny_config()
    cfg = replace(cfg, ppo=replace(cfg.ppo, reward='score', critic_target='terminal', epochs=4))
    return coordinator.train(cfg, tmp_path/'source', max_updates=1).latest_checkpoint


def tree_digest(root):
    return {p.relative_to(root): hashlib.sha256(p.read_bytes()).digest() for p in root.rglob('*') if p.is_file()}


def test_compact_fork_preserves_exact_actor_adam_rng_and_old_league(coordinator, tmp_path):
    from brass_rl.stability_run import fork_stability, train_stability
    source = source_checkpoint(coordinator, tmp_path)
    parent_lock = tmp_path/'.source.writer.lock'
    parent_lock.unlink()
    original = tree_digest(source.parent.parent)
    loaded = load_update(source)
    for arm, candidate, games in (('control', False, 2), ('candidate', True, 8)):
        dest = tmp_path/arm
        base = fork_stability(source, dest, candidate)
        assert base == dest/'updates'/source.name
        own = load_update(base)
        assert [r['update'] for r in committed_records(dest)] == [loaded.state.completed_updates]
        assert (base/'policy.pt').read_bytes() == (source/'policy.pt').read_bytes()
        assert (base/'metadata.json').read_bytes() == (source/'metadata.json').read_bytes()
        assert inspect_policy(dest/'policies'/source.name).id == json.loads((source/'metadata.json').read_bytes())['id']
        assert own.state.config == replace(loaded.state.config, games_per_update=games)
        assert own.state.master_seed == loaded.state.master_seed
        assert own.state.next_game_id == loaded.state.next_game_id
        assert_nested_equal(own.state.controller_rng_states, loaded.state.controller_rng_states)
        assert_nested_equal(own.state.rng_states, loaded.state.rng_states)
        assert_nested_equal(own.optimizer_state, loaded.optimizer_state)
        assert own.state.league_history == loaded.state.league_history
        assert own.state.league_snapshot_ids == loaded.state.league_snapshot_ids
        assert own.state.manifests['stability_arm'] == arm
        exp = json.loads((dest/'experiment.json').read_bytes())
        assert exp['parent']['checkpoint_sha256'] == hashlib.sha256((source/'manifest.json').read_bytes()).hexdigest()
        assert exp['stability_arm'] == arm
        if candidate:
            assert_nested_equal(own.critic_state, loaded.model.state_dict())
            assert own.critic_optimizer_state['state'] == {}
        else:
            assert own.critic_state is None and own.critic_optimizer_state is None
        with pytest.raises(ValueError, match='stability'):
            coordinator.train(own.state.config, dest, resume=base, max_updates=2)
        with pytest.raises(ValueError, match='multiple|whole'):
            train_stability(dest, 1)
    assert tree_digest(source.parent.parent) == original
    assert not parent_lock.exists()


def test_control_matches_ordinary_ppo_exactly_and_measures_full_final_policy(coordinator, tmp_path):
    from brass_rl.stability_run import fork_stability, train_stability
    source = source_checkpoint(coordinator, tmp_path)
    control = tmp_path/'control'
    fork_stability(source, control, False)
    legacy = load_update(source)
    original = coordinator.train(legacy.state.config, source.parent.parent, resume=source, max_updates=3)
    arm = train_stability(control, 4)
    assert arm.choice_trace == original.choice_trace[-4:]
    a, b = load_update(arm.latest_checkpoint), load_update(original.latest_checkpoint)
    assert_nested_equal(a.model.state_dict(), b.model.state_dict())
    assert_nested_equal(a.optimizer_state, b.optimizer_state)
    assert_nested_equal(a.state.controller_rng_states, b.state.controller_rng_states)
    assert_nested_equal(a.state.rng_states, b.state.rng_states)
    metrics = a.state.metrics[0]['ppo']
    assert metrics['actor_steps'] == metrics['critic_steps'] == metrics['optimizer_steps']
    assert metrics['final_policy_decisions'] == metrics['policy_decisions'] == 2
    assert metrics['final_kl'] >= 0 and 0 <= metrics['final_clip_fraction'] <= 1


def test_split_resume_and_interrupted_candidate_reproduce_both_adams_and_rng(coordinator, tmp_path, monkeypatch):
    from brass_rl.stability_run import fork_stability, train_stability
    source = source_checkpoint(coordinator, tmp_path)
    direct = tmp_path/'direct'
    split = tmp_path/'split'
    fork_stability(source, direct, True)
    shutil.copytree(direct, split)
    run = train_stability(direct, 16)
    partial = train_stability(split, 8)
    before = tree_digest(partial.latest_checkpoint)
    assert train_stability(split, 8, stop_requested=lambda: True).latest_checkpoint == partial.latest_checkpoint
    def fail(*args, **kwargs):
        raise OSError('injected before commit')
    with monkeypatch.context() as patch:
        patch.setattr(coordinator, 'save_update', fail)
        with pytest.raises(OSError, match='before commit'):
            train_stability(split, 8)
    assert tree_digest(partial.latest_checkpoint) == before
    assert len(committed_records(split)) == 2
    resumed = train_stability(split, 8)
    assert resumed.choice_trace == run.choice_trace
    assert resumed.next_game_id == run.next_game_id == 18
    a, b = load_update(resumed.latest_checkpoint), load_update(run.latest_checkpoint)
    assert_nested_equal(a.model.state_dict(), b.model.state_dict())
    for key in ('optimizer_state', 'critic_state', 'critic_optimizer_state'):
        assert_nested_equal(getattr(a, key), getattr(b, key))
    assert_nested_equal(a.state.controller_rng_states, b.state.controller_rng_states)
    assert_nested_equal(a.state.rng_states, b.state.rng_states)
    assert_nested_equal(a.state.metrics[0]['ppo'], b.state.metrics[0]['ppo'])
    assert not (partial.latest_checkpoint/'optimizer.pt').exists()
    assert (split/'updates'/source.name/'optimizer.pt').is_file()
    assert {p.name for p in resumed.latest_policy.iterdir()} == {'manifest.json', 'metadata.json', 'policy.pt'}


@pytest.mark.parametrize('damage', ['critic_shape', 'critic_nonfinite', 'missing_tag', 'raw_adam', 'negative_second_moment'])
def test_malformed_candidate_never_resumes_or_alters_rng(coordinator, tmp_path, damage):
    from brass_rl.stability_run import fork_stability, train_stability
    source = source_checkpoint(coordinator, tmp_path)
    run = tmp_path/'candidate'
    path = fork_stability(source, run, True)
    if damage == 'negative_second_moment':
        path = train_stability(run, 8).latest_checkpoint
    if damage == 'missing_tag':
        changed = 'manifests.json'
        payload = json.loads((path/changed).read_bytes())
        payload.pop('stability_arm')
        (path/changed).write_text(json.dumps(payload))
    else:
        changed = 'optimizer.pt'
        payload = torch.load(path/changed, weights_only=False)
        if damage == 'negative_second_moment':
            next(iter(payload['critic_adam']['state'].values()))['exp_avg_sq'].flatten()[0] = -1.
        elif damage == 'raw_adam':
            payload = payload['actor_adam']
        elif damage == 'critic_shape':
            payload['critic_weights']['value_head.weight'] = torch.ones(3, 1)
        else:
            payload['critic_weights']['value_head.weight'].fill_(float('nan'))
        torch.save(payload, path/changed)
    manifest = json.loads((path/'manifest.json').read_bytes())
    manifest['content_hashes'][changed] = hashlib.sha256((path/changed).read_bytes()).hexdigest()
    (path/'manifest.json').write_text(json.dumps(manifest))
    before = deepcopy(capture_rng_states())
    with pytest.raises(ArtifactError):
        load_update(path)
    assert_nested_equal(capture_rng_states(), before)
