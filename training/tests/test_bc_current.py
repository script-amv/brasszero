# SPDX-License-Identifier: Apache-2.0
"""Full-source migration, resumed learner dispatch, and loss-priority accounting."""
from dataclasses import replace
import json
from pathlib import Path
import shutil

import pytest

from brass_rl.artifacts import load_update, restore_league
from brass_rl.config import LeagueConfig, TrainConfig
from brass_rl.league import Lineup, LineupEntry
from test_train import assert_nested_equal, coordinator, tiny_config, tiny_collection  # noqa: F401

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def seat_zero_collection(coordinator, monkeypatch):
    # The short native observation fixture has exactly one deciding seat (zero).
    def collect(*args):
        result = []
        for episode in tiny_collection(*args):
            entries = episode.lineup.entries
            first = next(i for i, entry in enumerate(entries) if entry.trainable)
            result.append(replace(episode, lineup=Lineup(entries[first:]+entries[:first])))
        return result
    monkeypatch.setattr(coordinator, 'collect_batch', collect)


def source_checkpoint(coordinator, root, max_updates=1):
    cfg = tiny_config()
    cfg = replace(cfg, league=LeagueConfig(),
                  ppo=replace(cfg.ppo, reward='score', critic_target='terminal', epochs=4))
    return coordinator.train(cfg, root/'source', max_updates=max_updates).latest_checkpoint


def test_profile_is_explicit_and_cannot_start_without_a_source(coordinator, tmp_path):
    path = ROOT/'training/configs/bc-current.json'
    assert path.is_file()
    cfg = TrainConfig.from_json(path)
    assert cfg.learner == 'brass-bc/1'
    assert (cfg.games_per_update, cfg.ppo.target_kl, cfg.ppo.epochs) == (8, .05, 4)
    with pytest.raises(ValueError, match='source'):
        coordinator.train(cfg, tmp_path/'new', max_updates=0)
    with pytest.raises(ValueError):
        replace(cfg, games_per_update=2)


@pytest.mark.parametrize('trained_critic', [False, True])
def test_fork_preserves_full_source_and_declares_critic(coordinator, tmp_path, trained_critic):
    from brass_rl import bc_current
    from brass_rl.stability_run import fork_stability, train_stability
    source = source_checkpoint(coordinator, tmp_path)
    if trained_critic:
        fork_stability(source, tmp_path/'study', True)
        source = train_stability(tmp_path/'study', 8).latest_checkpoint
    parent = load_update(source)
    if trained_critic:
        assert parent.critic_optimizer_state['state']
    dest = tmp_path/'bc'
    child = load_update(bc_current.fork_bc(source, dest))
    for filename in ('metadata.json', 'policy.pt'):
        assert (child.path/filename).read_bytes() == (source/filename).read_bytes()
    assert_nested_equal(child.optimizer_state, parent.optimizer_state)
    assert_nested_equal(child.state.rng_states, parent.state.rng_states)
    assert_nested_equal(child.state.controller_rng_states, parent.state.controller_rng_states)
    assert (child.state.master_seed, child.state.next_game_id, child.state.completed_updates) == (
        parent.state.master_seed, parent.state.next_game_id, parent.state.completed_updates)
    assert child.state.league_history == parent.state.league_history
    assert child.state.league_snapshot_ids == parent.state.league_snapshot_ids
    assert 'stability_arm' not in child.state.manifests
    assert child.state.config.learner == child.state.manifests['learner'] == 'brass-bc/1'
    assert child.state.loss_priority['counts'] == {}
    assert child.state.loss_priority['next_game_id'] == parent.state.next_game_id
    exp = json.loads((dest/'experiment.json').read_bytes())
    assert exp['learner'] == 'brass-bc/1'
    if trained_critic:
        assert_nested_equal(child.critic_state, parent.critic_state)
        assert_nested_equal(child.critic_optimizer_state, parent.critic_optimizer_state)
        assert exp['critic_initialization'] == 'restored'
    else:
        assert_nested_equal(child.critic_state, parent.model.state_dict())
        assert child.critic_optimizer_state['state'] == {}
        assert exp['critic_initialization'] == 'cloned'


def test_native_loss_priority_and_duplicate_encounters(coordinator, tmp_path):
    from brass_rl.bc_current import fork_bc
    from test_league import snapshot
    source = source_checkpoint(coordinator, tmp_path)
    base = load_update(fork_bc(source, tmp_path/'bc'))
    league = restore_league(base.state)
    initial = league.snapshots[0].id
    league.add(snapshot(tmp_path, 1))
    current = LineupEntry('current', trainable=True)
    frozen = LineupEntry('frozen', initial)
    other = LineupEntry('frozen', 's1')
    mixed = Lineup((current, frozen, frozen, other))
    g = base.state.next_game_id
    league.record_training(g, mixed, (1, 1, 0, 3))
    league.record_training(g+1, Lineup((current,)*4), (0, 1, 2, 3))
    assert league.state['counts'] == {initial: [1, 2], 's1': [2, 1]}
    assert league.state['next_game_id'] == g+2
    with pytest.raises(ValueError, match='cursor'):
        league.record_training(g+1, mixed, (1, 1, 0, 3))
    class Draw:
        def choices(self, entries, *, weights, k):
            assert k == 3
            w1, w2 = 1+4*(1-2.5/6)**2, 1+4*(1-3/5)**2
            assert weights == pytest.approx([2*w1/(w1+w2), 2*w2/(w1+w2), 1, 1])
            return [entries[0]]*3
    assert league._opponents(Draw(), (frozen, other, LineupEntry('random'), LineupEntry('greedy'))) == [frozen]*3


def test_normal_resume_after_failed_commit_matches_direct(coordinator, tmp_path, monkeypatch):
    from brass_rl.bc_current import fork_bc
    source = source_checkpoint(coordinator, tmp_path)
    run, split = tmp_path/'direct', tmp_path/'split'
    base = load_update(fork_bc(source, run))
    cfg = base.state.config
    shutil.copytree(run, split)
    direct = coordinator.train(cfg, run, run/'latest', 3)
    partial = coordinator.train(cfg, split, split/'latest', 3, stop_requested=iter([False, True]).__next__)
    assert partial.completed_updates == 2 and partial.next_game_id == 10
    with monkeypatch.context() as patch:
        def fail(*args, **kwargs):
            raise OSError('uncommitted')
        patch.setattr(coordinator, 'save_update', fail)
        with pytest.raises(OSError, match='uncommitted'):
            coordinator.train(cfg, split, split/'latest', 3)
    assert load_update(split/'latest').state.loss_priority == partial.state.loss_priority
    resumed = coordinator.train(cfg, split, split/'latest', 3)
    a, b = load_update(resumed.latest_checkpoint), load_update(direct.latest_checkpoint)
    assert a.state.loss_priority == b.state.loss_priority
    assert a.state.loss_priority['next_game_id'] == 18 and a.state.loss_priority['counts']
    assert a.state.metrics[0]['ppo']['critic_steps'] > 0
    assert a.state.metrics[0]['ppo']['full_epoch_kl_checks'] > 0
    assert_nested_equal(a.model.state_dict(), b.model.state_dict())
    for key in ('optimizer_state', 'critic_state', 'critic_optimizer_state'):
        assert_nested_equal(getattr(a, key), getattr(b, key))
    assert_nested_equal(a.state.rng_states, b.state.rng_states)
    assert_nested_equal(a.state.controller_rng_states, b.state.controller_rng_states)
    assert not (partial.latest_checkpoint/'optimizer.pt').exists()
    assert {p.name for p in resumed.latest_policy.iterdir()} == {'metadata.json', 'manifest.json', 'policy.pt'}
    child = load_update(fork_bc(a.path, tmp_path/'refork'))
    assert child.state.loss_priority == a.state.loss_priority


@pytest.mark.parametrize('trained_critic', [False, True])
def test_cloud_seed_and_restore_preserve_profile_and_critic(coordinator, tmp_path, monkeypatch, trained_critic):
    monkeypatch.syspath_prepend(str(ROOT/'training/scripts/kaggle_control'))
    from common import extract
    import prepare
    from train_run import verify
    assert hasattr(prepare, 'seed_bc')
    source = source_checkpoint(coordinator, tmp_path)
    if trained_critic:
        from brass_rl.stability_run import fork_stability, train_stability
        fork_stability(source, tmp_path/'study', True)
        source = train_stability(tmp_path/'study', 8).latest_checkpoint
    original = load_update(source)
    with pytest.raises(ValueError, match='archive'):
        prepare.seed_bc(source, tmp_path/'collision.json')
    archive = tmp_path/'seed.tar.gz'
    prepare.seed_bc(source, archive)
    extract(archive, tmp_path/'return')
    state = tmp_path/'return/state'
    summary = verify(state)
    assert summary['learner'] == 'brass-bc/1' and summary['update'] == original.state.completed_updates
    assert summary['next_game_id'] == summary['loss_priority_next_game_id'] == original.state.next_game_id
    assert summary['replay_count'] == 0
    loaded = load_update(state/'run/latest')
    assert loaded.state.loss_priority['counts'] == {}
    if trained_critic:
        assert summary['critic_adam_slots'] > 0
        assert_nested_equal(original.critic_state, loaded.critic_state)
        assert_nested_equal(original.critic_optimizer_state, loaded.critic_optimizer_state)
    else:
        assert summary['critic_adam_slots'] == 0


def test_cli_and_generic_package_restore_current_boundary(coordinator, tmp_path):
    from brass_rl.__main__ import main
    from brass_rl.experiments import branch_experiment
    from brass_rl.transfer import pack, verify, import_package

    runs = tmp_path/'project/training/runs'
    cfg = tiny_config()
    cfg = replace(cfg, ppo=replace(cfg.ppo, epochs=4, reward='score', critic_target='terminal'), league=LeagueConfig())
    source = coordinator.train(cfg, runs/'parent', max_updates=0).latest_checkpoint
    run = runs/'current'
    main(['train-bc', '--out', str(run), '--source', str(source), '--max-updates', '0'])
    main(['train-bc', '--out', str(run), '--resume', 'latest', '--max-updates', '0'])
    with pytest.raises(SystemExit):
        main(['train-bc', '--out', str(run), '--max-updates', '0'])
    with pytest.raises(SystemExit):
        main(['train-bc', '--out', str(source.parent.parent), '--resume', 'latest', '--max-updates', '0'])
    archive = tmp_path/'state.tar.gz'
    pack(run, archive)
    assert verify(archive)['experiments'] == 2
    moved = Path(import_package(archive, tmp_path/'relocated')['experiment_path'])
    base = load_update(moved/'latest')
    assert base.critic_state is not None
    assert_nested_equal(base.optimizer_state, load_update(run/'latest').optimizer_state)
    main(['train', '--config', str(moved/'config.json'), '--out', str(moved), '--resume', 'latest', '--max-updates', '0'])
    branch = tmp_path/'branch'
    branch_experiment(base.path, branch)
    result = coordinator.train(base.state.config, branch, branch/'latest', 0)
    assert result.preflight['learner'] == 'brass-bc/1'


@pytest.mark.parametrize('damage', ['raw_adam', 'missing_learner', 'missing_counts', 'invalid_cursor'])
def test_corrupt_supported_state_cannot_resume_as_shared(coordinator, tmp_path, damage):
    import hashlib
    import torch
    from brass_rl.bc_current import fork_bc
    from brass_rl.policy_io import ArtifactError
    path = fork_bc(source_checkpoint(coordinator, tmp_path), tmp_path/'bc')
    if damage == 'missing_learner':
        name = 'manifests.json'
        value = json.loads((path/name).read_bytes())
        del value['learner']
        (path/name).write_text(json.dumps(value))
    else:
        name = 'optimizer.pt' if damage == 'raw_adam' else 'controller.pt'
        value = torch.load(path/name, weights_only=False)
        if damage == 'raw_adam':
            value = value['actor_adam']
        elif damage == 'missing_counts':
            del value['loss_priority']
        else:
            value['loss_priority']['next_game_id'] += 1
        torch.save(value, path/name)
    manifest = json.loads((path/'manifest.json').read_bytes())
    manifest['content_hashes'][name] = hashlib.sha256((path/name).read_bytes()).hexdigest()
    (path/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ArtifactError):
        load_update(path)


def edit_journal(checkpoint, change):
    import hashlib
    path = checkpoint/'manifests.json'
    value = json.loads(path.read_bytes())
    change(value)
    path.write_text(json.dumps(value))
    envelope = json.loads((checkpoint/'manifest.json').read_bytes())
    envelope['content_hashes'][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (checkpoint/'manifest.json').write_text(json.dumps(envelope))


@pytest.mark.parametrize('location', ['experiment', 'journal'])
def test_research_marker_rejects_training_and_forks_but_allows_transport(coordinator, tmp_path, location):
    from brass_rl.bc_current import fork_bc
    from brass_rl.experiments import branch_experiment
    from brass_rl.learning_study import fork_arm
    from brass_rl.stability_run import fork_stability
    from brass_rl.transfer import pack, verify
    from test_stability_run import tree_digest

    base = source_checkpoint(coordinator, tmp_path, max_updates=0)
    run = base.parent.parent
    mark = {'format': 'brass-research-league/1', 'mode': 'loss-priority', 'replicate': 1,
            'first_game_id': 0, 'next_game_id': 0, 'counts': {}}
    if location == 'experiment':
        path = run/'experiment.json'
        data = json.loads(path.read_bytes())
        path.write_text(json.dumps(dict(data, league_research=mark)))
    else:
        edit_journal(base, lambda data: data.update(league_research=mark))
    before = tree_digest(run)
    config = load_update(base).state.config
    archive = tmp_path/'research.tar.gz'
    pack(run, archive)
    assert verify(archive)['updates'] == 1
    child = tmp_path/'child'
    attempts = (
        lambda: coordinator.train(config, run, run/'latest', 1),
        lambda: fork_bc(base, child),
        lambda: fork_stability(base, child, True),
        lambda: branch_experiment(base, child),
        lambda: fork_arm(base, child, .95),
    )
    for attempt in attempts:
        with pytest.raises(ValueError, match='pinned research runtime'):
            attempt()
        assert tree_digest(run) == before
        assert not child.exists()


@pytest.mark.parametrize('damage', ['no_config_learner', 'no_journal_learner', 'stability_arm', 'unsupported_settings'])
def test_pack_checks_public_bc_declarations(coordinator, tmp_path, damage):
    from brass_rl.bc_current import fork_bc
    from brass_rl.transfer import pack

    base = fork_bc(source_checkpoint(coordinator, tmp_path, max_updates=0), tmp_path/'bc')
    run = base.parent.parent
    if damage == 'no_journal_learner':
        edit_journal(base, lambda data: data.pop('learner'))
    elif damage == 'stability_arm':
        path = run/'experiment.json'
        data = json.loads(path.read_bytes())
        path.write_text(json.dumps(dict(data, stability_arm='candidate')))
    else:
        path = run/'config.json'
        data = json.loads(path.read_bytes())
        if damage == 'no_config_learner':
            del data['learner']
        else:
            data['ppo']['target_kl'] = .03
        path.write_text(json.dumps(data))
    archive = tmp_path/'bad.tar.gz'
    with pytest.raises(ValueError, match='learner|brass-bc/1'):
        pack(run, archive)
    assert not archive.exists()


def test_rehashed_package_with_conflicting_bc_config_rejected_before_import(coordinator, tmp_path):
    import io
    import tarfile
    from brass_rl.bc_current import fork_bc
    from brass_rl.transfer import pack, verify, import_package, digest, encoded

    base = fork_bc(source_checkpoint(coordinator, tmp_path, max_updates=0), tmp_path/'bc')
    run = base.parent.parent
    good, bad = tmp_path/'good.tar.gz', tmp_path/'bad.tar.gz'
    pack(run, good)
    # Model and checkpoint blobs remain authentic. Replace only this run's top-level config
    # and rehash its complete transport envelope to exercise semantic verification.
    config = json.loads((run/'config.json').read_bytes())
    config['learner'] = 'different-learner'
    replacement = encoded(config)
    with tarfile.open(good) as src, tarfile.open(bad, 'w:gz') as out:
        manifest = json.load(src.extractfile('manifest.json'))
        relative = manifest['experiments'][manifest['experiment_id']]+'/config.json'
        old = manifest['files'][relative]['sha256']
        assert sum(row['sha256'] == old for row in manifest['files'].values()) == 1
        manifest['files'][relative] = dict(sha256=digest(replacement), size=len(replacement))
        del manifest['package_id']
        manifest['package_id'] = digest(encoded(manifest))
        for item in src.getmembers():
            if item.name == 'manifest.json':
                content = encoded(manifest)
            elif item.name == 'blobs/'+old:
                item.name = 'blobs/'+digest(replacement)
                content = replacement
            else:
                out.addfile(item, src.extractfile(item))
                continue
            item.size = len(content)
            out.addfile(item, io.BytesIO(content))
    with pytest.raises(ValueError, match='learner'):
        verify(bad)
    destination = tmp_path/'destination'
    with pytest.raises(ValueError, match='learner'):
        import_package(bad, destination)
    assert not (destination/'training/runs').exists()
