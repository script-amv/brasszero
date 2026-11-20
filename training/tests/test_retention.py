# SPDX-License-Identifier: Apache-2.0
from dataclasses import replace
from pathlib import Path

import pytest

from test_artifacts import training  # noqa: F401
from test_train import coordinator  # noqa: F401


def test_retire_only_intermediate_and_preserve_journal_and_play(tmp_path, training):
    from brass_rl import artifacts
    from brass_rl import retention
    from brass_rl.experiments import branch_experiment, ensure_experiment
    model, optimizer, state = training
    run = tmp_path/'run'
    run.mkdir()
    for number in range(8, 12):
        checkpoint = artifacts.save_update(run, model, optimizer, replace(state, completed_updates=number))
        artifacts.export_policy(checkpoint, run/'policies'/checkpoint.name)
    ensure_experiment(run, state.config, state=replace(state, completed_updates=8))
    original = (run/'updates/000009/manifest.json').read_bytes()
    plan = retention.plan_runs([run])
    assert [path.name for path in plan.candidates] == ['000009']
    retention.apply_plan(plan)
    retired = run/'updates/000009'
    assert (retired/'manifest.json').read_bytes() == original
    assert not (retired/'optimizer.pt').exists()
    assert [entry['update'] for entry in artifacts.committed_records(run)] == [8, 9, 10, 11]
    assert artifacts.load_update(run/'latest').path.name == '000011'
    assert artifacts.load_update(run/'updates/000010').state.completed_updates == 10
    assert artifacts.export_policy(retired, tmp_path/'recovered') == tmp_path/'recovered'
    with pytest.raises(ValueError, match='retired'):
        artifacts.load_update(retired)
    branch = tmp_path/'branch'
    branch_experiment(run/'updates/000011', branch)
    assert not (branch/'updates/000009/optimizer.pt').exists()
    assert artifacts.load_update(branch/'latest').state.completed_updates == 11
    assert not retention.plan_runs([run]).candidates
    receipt = retired/'retention.json'
    receipt.unlink()
    with pytest.raises(ValueError):
        artifacts.committed_records(run)


def test_train_crash_preserves_previous_then_resume_finishes_retirement(tmp_path, monkeypatch, coordinator):
    from test_train import tiny_config
    trainer = coordinator
    from brass_rl.artifacts import load_update
    from brass_rl import retention
    run = tmp_path/'run'
    cfg = tiny_config()
    trainer.train(cfg, run, max_updates=1)
    prior = run/'updates/000001'
    original = trainer.publish_policies
    def fail(run_dir, journal):
        if journal[-1]['update'] == 2:
            raise OSError('public export failure fixture')
        return original(run_dir, journal)
    monkeypatch.setattr(trainer, 'publish_policies', fail)
    with pytest.raises(OSError, match='export failure'):
        trainer.train(cfg, run, resume=run/'latest', max_updates=2)
    assert load_update(prior).state.completed_updates == 1
    assert load_update(run/'latest').state.completed_updates == 2
    monkeypatch.setattr(trainer, 'publish_policies', original)
    trainer.train(cfg, run, resume=run/'latest', max_updates=2)
    assert not (prior/'optimizer.pt').exists()
    assert (prior/retention.RECEIPT).exists()


def test_sparse_branch_package_roundtrip_and_enrichment(tmp_path, training):
    from brass_rl import artifacts, retention
    from brass_rl.experiments import ensure_experiment, branch_experiment
    from brass_rl.transfer import pack, verify, import_package
    run = tmp_path/'a/training/runs/parent'
    run.mkdir(parents=True)
    model, optimizer, state = training
    ensure_experiment(run, state.config)
    for n in range(4):
        checkpoint = artifacts.save_update(run, model, optimizer,
            replace(state, completed_updates=n, league_snapshots=(), manifests={}, metrics=()))
        artifacts.export_policy(checkpoint, run/'policies'/checkpoint.name)
    old = tmp_path/'old.tar.gz'
    pack(run, old)
    child = run.parent/'child'
    branch_experiment(run/'updates/000002', child)
    plan = retention.plan_runs([run, child])
    assert all(p.name == '000001' for p in plan.candidates)
    retention.apply_plan(plan)
    archive = tmp_path/'sparse.tar.gz'
    pack(child, archive)
    assert verify(archive)['experiments'] == 2
    destination = tmp_path/'destination'
    imported = Path(import_package(archive, destination)['experiment_path'])
    assert import_package(archive, destination)['added_files'] == 0
    assert artifacts.load_update(imported/'latest').path.name == '000002'
    assert not (imported/'updates/000001/optimizer.pt').exists()
    enriched = Path(import_package(old, destination)['experiment_path'])
    assert artifacts.load_update(enriched/'updates/000001').state.completed_updates == 1


def test_retired_checkpoint_inspection_preserves_rng_and_checks_payloads(tmp_path, training):
    from brass_rl import artifacts, retention
    from test_artifacts import draws, assert_draws
    model, optimizer, state = training
    run = tmp_path/'run'
    for n in (1, 2, 3):
        artifacts.save_update(run, model, optimizer, replace(state, completed_updates=n))
    retention.apply_plan(retention.plan_runs([run]))
    retired = run/'updates/000002'
    before = artifacts.capture_rng_states()
    expected = draws()
    artifacts.restore_rng_states(before)
    inspected = artifacts.inspect_update_state(retired)
    assert inspected.completed_updates == 2 and inspected.league_snapshot_ids == ('initial',)
    assert_draws(draws(), expected)
    (retired/'controller.pt').unlink()
    with pytest.raises(ValueError):
        artifacts.inspect_update_state(retired)


def test_import_rejects_missing_full_adam_before_installing_receipt(tmp_path, training):
    from brass_rl import artifacts, retention
    from brass_rl.experiments import ensure_experiment
    from brass_rl.transfer import pack, import_package
    run = tmp_path/'a/training/runs/run'
    run.mkdir(parents=True)
    model, optimizer, state = training
    ensure_experiment(run, state.config)
    for n in range(3):
        artifacts.save_update(run, model, optimizer,
            replace(state, completed_updates=n, manifests={}, metrics=(), league_snapshots=()))
    old = tmp_path/'full.tar.gz'
    new = tmp_path/'sparse.tar.gz'
    pack(run, old)
    retention.apply_plan(retention.plan_runs([run]))
    pack(run, new)
    dest = tmp_path/'dest'
    target = Path(import_package(old, dest)['experiment_path'])
    middle = target/'updates/000001'
    (middle/'optimizer.pt').unlink()
    with pytest.raises(ValueError, match='incomplete|corrupt'):
        import_package(new, dest)
    assert not (middle/'retention.json').exists()


def _fixture_boundary(run, training, n):
    from brass_rl import artifacts
    model, optimizer, state = training
    state = replace(state, completed_updates=n, next_game_id=2*n, manifests={}, metrics=())
    return artifacts.save_update(run, model, optimizer, state)


@pytest.mark.parametrize('when', ['before', 'after'])
def test_incremental_sparse_import_crash_has_complete_resumable_prefix(tmp_path, training, monkeypatch, when):
    from brass_rl import artifacts, retention, transfer
    from brass_rl.experiments import ensure_experiment
    from brass_rl.train import train
    run = tmp_path/'a/training/runs/run'
    run.mkdir(parents=True)
    cfg = training[2].config
    ensure_experiment(run, cfg)
    for n in range(2):
        _fixture_boundary(run, training, n)
    old, new = tmp_path/'old.tar.gz', tmp_path/'new.tar.gz'
    transfer.pack(run, old)
    project = tmp_path/'destination'
    dest = Path(transfer.import_package(old, project)['experiment_path'])
    for n in (2, 3, 4):
        _fixture_boundary(run, training, n)
    # Both are off cadence: pin new latest and penultimate before the crash.
    retention.pin(run/'updates/000003')
    retention.pin(run/'updates/000004')
    retention.apply_plan(retention.plan_runs([run]))
    transfer.pack(run, new)
    exchange = transfer._exchange_directories
    def interrupt(left, right):
        if when == 'after':
            exchange(left, right)
        raise OSError('injected crash at exchange')
    monkeypatch.setattr(transfer, '_exchange_directories', interrupt)
    with pytest.raises((ValueError, OSError), match='exchange'):
        transfer.import_package(new, project)
    expected = 1 if when == 'before' else 4
    assert {int(p.name) for p in (dest/'updates').iterdir() if p.name.isdecimal()} == set(range(expected+1))
    assert train(cfg, dest, dest/'latest', max_updates=expected).preflight['optimizer'] == 'restored'
    if when == 'after':
        for n in (3, 4):
            boundary = dest/'updates'/('%06d' % n)
            assert (dest/'updates/pins'/(retention.file_sha(boundary/'manifest.json')+'.json')).is_file()
    monkeypatch.setattr(transfer, '_exchange_directories', exchange)
    transfer.import_package(new, project)
    assert transfer.import_package(new, project)['added_files'] == 0
    assert artifacts.load_update(dest/'latest').state.completed_updates == 4


def test_legacy_package_derives_parent_pin_before_normal_rolling(tmp_path, training, coordinator):
    from brass_rl import artifacts, transfer
    from brass_rl.experiments import ensure_experiment, branch_experiment
    import shutil
    run = tmp_path/'a/training/runs/parent'
    run.mkdir(parents=True)
    from test_train import tiny_config
    config = replace(training[2].config, league=tiny_config().league, games_per_update=2)
    training = (training[0], training[1], replace(training[2], config=config))
    experiment = ensure_experiment(run, config)
    for n in range(4):
        _fixture_boundary(run, training, n)
    child = run.parent/'child'
    branch_experiment(run/'updates/000003', child)
    # Historical v1 packages predate pin files. Parent tuples remain in experiment.json.
    for directory in (run, child):
        shutil.rmtree(directory/'updates/pins')
    bundle = tmp_path/'legacy.tar.gz'
    transfer.pack(child, bundle)
    dest = tmp_path/'destination'
    transfer.import_package(bundle, dest)
    parent = dest/'training/runs'/experiment['id']
    checkpoint = parent/'updates/000003'
    coordinator.train(training[2].config, parent, parent/'latest', max_updates=4)
    assert artifacts.load_update(checkpoint).state.completed_updates == 3


@pytest.mark.parametrize('role', ['root', 'base', 'pin', 'best'])
def test_pack_rejects_retirement_of_protected_boundary(tmp_path, training, role):
    from brass_rl import retention, transfer
    from brass_rl.experiments import ensure_experiment, atomic, encoded
    import json
    run = tmp_path/'run'
    run.mkdir()
    for n in (1, 2, 3):
        _fixture_boundary(run, training, n)
    experiment = ensure_experiment(run, training[2].config, state=replace(training[2], completed_updates=1))
    checkpoint = run/'updates'/('000001' if role == 'root' else '000002')
    if role == 'base':
        experiment['base_update'] = 2
        atomic(run/'experiment.json', encoded(experiment))
    if role == 'pin':
        retention.pin(checkpoint)
    if role == 'best':
        metrics = {'records': [{'development': {'best_id': json.loads((checkpoint/'metadata.json').read_bytes())['id']}}]}
        atomic(checkpoint/'metrics.json', encoded(metrics))
        manifest = json.loads((checkpoint/'manifest.json').read_bytes())
        manifest['content_hashes']['metrics.json'] = retention.file_sha(checkpoint/'metrics.json')
        atomic(checkpoint/'manifest.json', encoded(manifest))
    retention.retire(checkpoint, retention.file_sha(checkpoint/'manifest.json'))
    with pytest.raises(ValueError, match='protected checkpoint'):
        transfer.pack(run, tmp_path/'bad.tar.gz')


def test_active_study_verifier_reads_sparse_intermediate(tmp_path, training, monkeypatch):
    from brass_rl import artifacts, retention, four_branch_study as study, transfer
    from brass_rl.experiments import ensure_experiment
    model, optimizer, state = training
    run = tmp_path/'study/control'
    run.mkdir(parents=True)
    identity = ensure_experiment(run, state.config)['id']
    published = tmp_path/'models'
    for n in range(3):
        gids = [] if n == 0 else [2*n-2, 2*n-1]
        journal = {'experiment_id': identity, 'replay_ids': [str(g) for g in gids],
                   'train': [{'game_id': g} for g in gids]}
        checkpoint = artifacts.save_update(run, model, optimizer, replace(
            state, completed_updates=n, next_game_id=n*2, manifests=journal, metrics=()))
        artifacts.export_policy(checkpoint, run/'policies'/checkpoint.name)
        artifacts.export_policy(checkpoint, published/checkpoint.name)
    retention.apply_plan(retention.plan_runs([run]))
    monkeypatch.setattr(study, 'shared_roots', lambda: (tmp_path/'replays', published))
    monkeypatch.setattr(study, 'training_replays_by_game', lambda archive, ids: {int(g): {} for g in ids})
    monkeypatch.setattr(study, 'check_training_replay', lambda *args: None)
    monkeypatch.setattr(transfer, 'validate_replay', lambda record: None)
    declaration = {'base_update': 0, 'end_update': 2, 'first_game_id': 0, 'arms': ['control'],
                   'parent': {'model_id': 'source', 'experiment_id': 'parent'},
                   'training_deals': list(range(4)), 'games_per_arm': 4}
    source = run/'updates/000000'
    assert 'control' in study.verify_training(run.parent, declaration, source=source)
    (run/'updates/000001/controller.pt').write_bytes(b'corrupted fixture')
    with pytest.raises(ValueError):
        study.verify_training(run.parent, declaration, source=source)


def test_legacy_superseded_development_selection_survives_import_and_zero_game_resume(tmp_path, training):
    import shutil
    from brass_rl import artifacts, transfer
    from brass_rl.experiments import ensure_experiment
    from brass_rl.train import train
    run = tmp_path/'a/training/runs/run'
    run.mkdir(parents=True)
    model, optimizer, state = training
    ensure_experiment(run, state.config)
    for n in range(4):
        best = artifacts.policy_id(model, n) if n >= 2 else None
        rows = ({'development': {'best_id': best}},) if best else ()
        artifacts.save_update(run, model, optimizer, replace(
            state, completed_updates=n, next_game_id=2*n,
            development_best_id=best, manifests={}, metrics=rows))
    shutil.rmtree(run/'updates/pins')
    archive = tmp_path/'legacy.tar.gz'
    transfer.pack(run, archive)
    imported = Path(transfer.import_package(archive, tmp_path/'dest')['experiment_path'])
    train(state.config, imported, imported/'latest', max_updates=3)
    assert artifacts.load_update(imported/'updates/000002').state.completed_updates == 2


def test_import_rejects_contextually_damaged_local_pin_before_enrichment(tmp_path, training):
    from brass_rl import retention, transfer
    from brass_rl.experiments import ensure_experiment
    run = tmp_path/'a/training/runs/run'
    run.mkdir(parents=True)
    ensure_experiment(run, training[2].config)
    for n in range(3):
        _fixture_boundary(run, training, n)
    bundle = tmp_path/'full.tar.gz'
    transfer.pack(run, bundle)
    project = tmp_path/'destination'
    dest = Path(transfer.import_package(bundle, project)['experiment_path'])
    middle = dest/'updates/000001'
    retention.pin(middle)
    retention.retire(middle, retention.file_sha(middle/'manifest.json'))
    with pytest.raises(ValueError, match='protected checkpoint'):
        transfer.import_package(bundle, project)
    assert not (middle/'optimizer.pt').exists()


@pytest.mark.parametrize('dependency', ['former-best', 'sibling-fork', 'detached'])
def test_direct_legacy_resume_keeps_original_adam(tmp_path, training, dependency):
    import shutil
    from brass_rl import artifacts, retention
    from brass_rl.experiments import branch_experiment, ensure_experiment
    from brass_rl.train import train
    run = tmp_path/'training/runs/parent' if dependency != 'detached' else tmp_path/'detached'
    run.mkdir(parents=True)
    model, optimizer, state = training
    ensure_experiment(run, state.config)
    for n in range(4):
        best = artifacts.policy_id(model, n) if dependency == 'former-best' and n >= 2 else None
        rows = ({'development': {'best_id': best}},) if best else ()
        saved = artifacts.save_update(run, model, optimizer, replace(
            state, completed_updates=n, next_game_id=n*2,
            development_best_id=best, manifests={}, metrics=rows))
        if n == 2 and dependency == 'sibling-fork':
            child = run.parent/'child'
            branch_experiment(saved, child)
            shutil.rmtree(child/'updates/pins')
    if (run/'updates/pins').exists():
        shutil.rmtree(run/'updates/pins')  # Pre-retention v1 writers did not publish pins.
    previous = run/'updates/000002'
    expected = retention.file_sha(previous/'optimizer.pt')
    result = train(state.config, run, run/'latest', max_updates=3)
    assert result.completed_updates == 3
    assert (previous/'optimizer.pt').is_file(), 'direct resume retired a pre-retention Adam'
    assert retention.file_sha(previous/'optimizer.pt') == expected
    assert artifacts.load_update(previous).state.completed_updates == 2
    if dependency != 'detached':
        assert (run/'updates/pins'/(retention.file_sha(previous/'manifest.json')+'.json')).is_file()


@pytest.mark.parametrize('problem', ['unreadable', 'missing', 'file', 'empty'])
def test_destructive_inventory_requires_complete_readable_tree(tmp_path, training, monkeypatch, problem):
    import os
    from brass_rl import retention
    from brass_rl.__main__ import main
    root = tmp_path/'runs'
    run = root/'readable'
    for n in range(3):
        _fixture_boundary(run, training, n)
    checkpoints = list((run/'updates').glob('*/optimizer.pt'))
    before = {p: retention.file_sha(p) for p in checkpoints}
    if problem == 'unreadable':
        denied = root/'study'
        denied.mkdir()
        original = os.scandir
        def scandir(path):
            if path == str(denied) or path == denied:
                raise PermissionError('unreadable study fixture')
            return original(path)
        monkeypatch.setattr(os, 'scandir', scandir)
    elif problem == 'missing':
        root = tmp_path/'misspelled'
    elif problem == 'file':
        root = tmp_path/'file'
        root.touch()
    else:
        root = tmp_path/'empty'
        root.mkdir()
        main(['retention', '--runs-root', str(root)])
    with pytest.raises(SystemExit) as failure:
        main(['retention', '--runs-root', str(root), '--apply'])
    assert failure.value.code == 2
    assert all(p.is_file() and retention.file_sha(p) == h for p, h in before.items())
