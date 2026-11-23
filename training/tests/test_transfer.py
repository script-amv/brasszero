# SPDX-License-Identifier: Apache-2.0
"""Transfer must carry complete immutable history; verification never unpickles."""
import json
from pathlib import Path
import shutil
import tarfile

import pytest

from test_train import tiny_config, coordinator  # noqa: F401


def test_package_relocates_ancestors_models_and_imports_idempotently(coordinator, tmp_path, monkeypatch):
    from brass_rl.transfer import pack, verify, import_package
    from brass_rl.experiments import branch_experiment
    a, b = tmp_path/'a', tmp_path/'b'
    run = a/'training/runs/local'
    coordinator.train(tiny_config(), run, max_updates=0)
    branch = a/'training/runs/cloud'
    branch_experiment(run/'updates/000000', branch)
    coordinator.train(tiny_config(), branch, resume=branch/'latest', max_updates=0)
    archive = tmp_path/'result.tar.gz'
    info = pack(branch, archive)
    assert info['experiments'] == 2
    assert info['updates'] == 2
    assert info['unique_blob_bytes'] < info['logical_bytes']
    verified = verify(archive)
    assert verified['package_id'] == info['package_id']
    shutil.rmtree(a)
    imported = import_package(archive, b)
    assert imported['added_files'] > 0
    assert import_package(archive, b)['added_files'] == 0
    newrun = Path(imported['experiment_path'])
    restored = coordinator.train(tiny_config(), newrun, resume=newrun/'latest', max_updates=0)
    assert restored.preflight['restoration'] == 'full-state'
    assert len(list((b/'players/models').glob('*/metadata.json'))) == 1
    # Integrity conflict leaves all existing immutable bytes untouched.
    checkpoint = newrun/'updates/000000/optimizer.pt'
    checkpoint.unlink(); checkpoint.write_bytes(b'conflict')
    with pytest.raises(ValueError, match='conflict'):
        import_package(archive, b)
    assert checkpoint.read_bytes() == b'conflict'


def test_package_corruption_and_missing_history_rejected(coordinator, tmp_path):
    from brass_rl.transfer import pack
    run = tmp_path/'training/runs/local'
    coordinator.train(tiny_config(), run, max_updates=1)
    # Fake optimizer collection intentionally has no real game recordings.
    with pytest.raises(ValueError, match='replay|recording'):
        pack(run, tmp_path/'missing.tar.gz')
    shutil.rmtree(run/'updates/000000')
    with pytest.raises(ValueError, match='missing.*update|gap'):
        pack(run, tmp_path/'gap.tar.gz')


def test_transfer_actual_games_all_updates_and_same_identity_extension(tmp_path, monkeypatch):
    from dataclasses import replace
    from brass_ai.replay_archive import Archive
    from brass_rl.train import train
    from brass_rl.transfer import pack, verify, import_package
    config = replace(tiny_config(), games_per_update=1, ppo=replace(tiny_config().ppo, epochs=1))
    a, b = tmp_path/'mac', tmp_path/'cloud'
    monkeypatch.setenv('BRASS_REPLAY_ROOT', str(a/'replays'))
    run = a/'training/runs/local'
    train(config, run, max_updates=1)
    package = tmp_path/'outbound.tar.gz'
    info = pack(run, package)
    assert info['updates'] == 2 and info['replays'] == 1
    imported = import_package(package, b)
    moved = Path(imported['experiment_path'])
    monkeypatch.setenv('BRASS_REPLAY_ROOT', str(b/'replays'))
    train(config, moved, resume=moved/'latest', max_updates=2)
    returned = tmp_path/'return.tar.gz'
    assert pack(moved, returned)['updates'] == 3
    assert import_package(returned, a)['experiment_path'] == str(run)
    assert (run/'updates/000002/controller.pt').is_file()
    assert len(Archive(a/'replays').records()) == 2
    assert import_package(returned, a)['added_files'] == 0
    # Remove one blob while retaining the package declaration.
    broken = tmp_path/'broken.tar.gz'
    with tarfile.open(returned) as src, tarfile.open(broken, 'w:gz') as out:
        for member in src.getmembers()[1:]:
            out.addfile(member, src.extractfile(member))
    with pytest.raises(ValueError):
        verify(broken)


def test_uncommitted_prefix_is_transferred_as_interrupted(coordinator, tmp_path, monkeypatch):
    from brass_ai.replay_archive import Archive, Recorder, labeled_seats
    from brass_rl.transfer import pack, import_package
    root = tmp_path/'a'
    monkeypatch.setenv('BRASS_REPLAY_ROOT', str(root/'replays'))
    run = root/'training/runs/local'
    coordinator.train(tiny_config(), run, max_updates=0)
    identity = json.loads((run/'experiment.json').read_text())['id']
    recorder = Recorder(seed=123, seats=labeled_seats(['random']*4),
                        provenance={'category': 'training', 'experiment': identity, 'update': 1, 'game': 0})
    replay_id = recorder.game_id
    with pytest.raises(ValueError, match='active'):
        pack(run, tmp_path/'active.tar.gz')
    recorder.interrupt()
    package = tmp_path/'prefix.tar.gz'
    assert pack(run, package)['replays'] == 1
    import_package(package, tmp_path/'b')
    assert Archive(tmp_path/'b/replays').read(replay_id)['status'] == 'interrupted'


def test_import_conflict_preflight_leaves_other_files_unpublished(coordinator, tmp_path):
    from brass_rl.transfer import pack, import_package
    run = tmp_path/'a/training/runs/local'
    coordinator.train(tiny_config(), run, max_updates=0)
    package = tmp_path/'result.tar.gz'; pack(run, package)
    identity = json.loads((run/'experiment.json').read_text())['id']
    target = tmp_path/'b/training/runs'/identity/'weights'
    target.mkdir(parents=True)
    source = next((run/'weights').glob('*.pt'))
    (target/source.name).write_bytes(b'conflict')
    with pytest.raises(ValueError, match='conflict'):
        import_package(package, tmp_path/'b')
    assert not (tmp_path/'b/players/models').exists()
    assert not (target.parent/'updates').exists()


def test_existing_curated_model_identity_is_reused(coordinator, tmp_path):
    from brass_rl.transfer import pack, import_package
    run = tmp_path/'a/training/runs/local'
    coordinator.train(tiny_config(), run, max_updates=0)
    package = tmp_path/'result.tar.gz'; pack(run, package)
    b = tmp_path/'b'
    model = next(p for p in (tmp_path/'a/players/models').iterdir() if p.is_dir())
    shutil.copytree(model, b/'players/models/ppo-curated')
    import_package(package, b)
    assert import_package(package, b)['added_files'] == 0
    assert [p.name for p in (b/'players/models').iterdir() if p.is_dir()] == ['ppo-curated']


def test_package_rejects_undeclared_external_evaluation_opponent(coordinator, tmp_path):
    from brass_rl.transfer import pack
    run = tmp_path/'training/runs/local'
    coordinator.train(tiny_config(), run, max_updates=0)
    suite = run/'evaluation/dev/missing/suite.json'
    suite.parent.mkdir(parents=True)
    suite.write_text(json.dumps({'lineups': {'x': ['candidate', {'id':'policy-sha256-v1:missing'}, 'greedy', 'random']}}))
    with pytest.raises(ValueError, match='opponent dependency'):
        pack(run, tmp_path/'missing.tar.gz')


def test_import_failure_publishes_no_partial_run_and_retry_repairs(coordinator, tmp_path, monkeypatch):
    from brass_rl import transfer
    run = tmp_path/'a/training/runs/local'
    coordinator.train(tiny_config(), run, max_updates=0)
    package = tmp_path/'result.tar.gz'; transfer.pack(run, package)
    identity = json.loads((run/'experiment.json').read_text())['id']
    target = tmp_path/'b/training/runs'/identity
    replace = transfer.os.replace
    def fail(source, destination):
        if Path(destination) == target:
            raise OSError('injected import publication interruption')
        return replace(source, destination)
    monkeypatch.setattr(transfer.os, 'replace', fail)
    with pytest.raises(OSError, match='interruption'):
        transfer.import_package(package, tmp_path/'b')
    assert not target.exists()
    monkeypatch.setattr(transfer.os, 'replace', replace)
    transfer.import_package(package, tmp_path/'b')
    assert (target/'updates/000000/controller.pt').is_file()
    assert transfer.import_package(package, tmp_path/'b')['added_files'] == 0


@pytest.mark.parametrize('attack', ['traversal', 'symlink', 'undeclared', 'tamper'])
def test_malformed_package_fails_before_import(coordinator, tmp_path, attack):
    import io
    from brass_rl import transfer
    run = tmp_path/'a/training/runs/local'
    coordinator.train(tiny_config(), run, max_updates=0)
    package = tmp_path/'result.tar.gz'; transfer.pack(run, package)
    broken = tmp_path/'broken.tar.gz'
    with tarfile.open(package) as src, tarfile.open(broken, 'w:gz') as out:
        for item in src.getmembers():
            payload = src.extractfile(item).read()
            if item.name == 'manifest.json' and attack == 'traversal':
                manifest = json.loads(payload)
                key = next(iter(manifest['files']))
                manifest['files']['../escape'] = manifest['files'].pop(key)
                manifest['package_id'] = transfer.digest(transfer.encoded({k:v for k,v in manifest.items() if k != 'package_id'}))
                payload = transfer.encoded(manifest)
            elif item.name.startswith('blobs/') and attack == 'tamper':
                payload = b'X'+payload[1:]
            item.size = len(payload)
            out.addfile(item, io.BytesIO(payload))
        if attack in ('symlink', 'undeclared'):
            item = tarfile.TarInfo('evil')
            if attack == 'symlink':
                item.type = tarfile.SYMTYPE; item.linkname = '/tmp/escape'
            out.addfile(item, io.BytesIO(b''))
    with pytest.raises(ValueError):
        transfer.import_package(broken, tmp_path/'b')
    assert not (tmp_path/'b').exists()


def test_interrupted_game_mac_cloud_mac_monotonic_extension(coordinator, tmp_path, monkeypatch):
    import brass
    from brass_ai.replay_archive import Archive, Recorder, labeled_seats
    from brass_rl.transfer import pack, import_package
    a, b = tmp_path/'a', tmp_path/'b'
    monkeypatch.setenv('BRASS_REPLAY_ROOT', str(a/'replays'))
    run = a/'training/runs/local'
    coordinator.train(tiny_config(), run, max_updates=0)
    identity = json.loads((run/'experiment.json').read_text())['id']
    provenance = {'category':'training', 'experiment':identity, 'update':1, 'game':0}
    seats = labeled_seats(['random']*4)
    game = brass.Game(456)
    recorder = Recorder(seed=456, seats=seats, provenance=provenance)
    actor = game.current_player()
    recorder.append(actor, json.loads(game.observation(actor).legal_moves()[0]))
    recorder.interrupt()
    replay_id = recorder.game_id
    old_bytes = Archive(a/'replays').path(replay_id, '.jsonl').read_bytes()
    package = tmp_path/'out.tar.gz'; pack(run, package)
    imported = import_package(package, b)
    monkeypatch.setenv('BRASS_REPLAY_ROOT', str(b/'replays'))
    recorder = Recorder(seed=456, seats=seats, provenance=provenance)
    while not game.is_over():
        actor = game.current_player()
        recorder.append(actor, json.loads(game.observation(actor).legal_moves()[0]))
        game.apply_index(0)
    recorder.finish(game)
    returned = tmp_path/'back.tar.gz'; pack(Path(imported['experiment_path']), returned)
    import_package(returned, a)
    assert Archive(a/'replays').read(replay_id)['status'] == 'completed'
    assert any(p.read_bytes() == old_bytes for p in (a/'replays/history'/replay_id).iterdir())
    assert import_package(returned, a)['added_files'] == 0
    # Importing the older prefix never downgrades the completed game.
    import_package(package, a)
    assert Archive(a/'replays').read(replay_id)['status'] == 'completed'


def test_external_evaluation_opponent_is_included_from_catalog(coordinator, tmp_path):
    from dataclasses import replace
    from brass_rl.transfer import pack, import_package
    run = tmp_path/'a/training/runs/local'
    coordinator.train(tiny_config(), run, max_updates=0)
    other = coordinator.train(replace(tiny_config(), master_seed=123), tmp_path/'a/training/runs/other', max_updates=0)
    opponent = json.loads((other.latest_policy/'metadata.json').read_text())['id']
    suite = run/'evaluation/dev/external/suite.json'
    suite.parent.mkdir(parents=True)
    suite.write_text(json.dumps({'lineups': {'x': ['candidate', {'id':opponent}, 'greedy', 'random']}}))
    package = tmp_path/'result.tar.gz'
    assert pack(run, package)['models'] == 2
    import_package(package, tmp_path/'b')
    assert opponent in {json.loads(p.read_text())['id'] for p in (tmp_path/'b/players/models').glob('*/metadata.json')}


def test_abandoned_inprogress_header_is_idempotent_after_exclusive_lock(coordinator, tmp_path, monkeypatch):
    from brass_ai.replay_archive import Archive, Recorder, labeled_seats
    from brass_rl.transfer import pack, import_package
    a, b = tmp_path/'a', tmp_path/'b'
    monkeypatch.setenv('BRASS_REPLAY_ROOT', str(a/'replays'))
    run = a/'training/runs/local'
    coordinator.train(tiny_config(), run, max_updates=0)
    identity = json.loads((run/'experiment.json').read_text())['id']
    recorder = Recorder(seed=41, seats=labeled_seats(['random']*4),
                        provenance={'category':'training','experiment':identity,'update':1,'game':0})
    recorder._close()  # Abrupt exit leaves an in-progress header and unlocked lock file.
    path = Archive(a/'replays').path(recorder.game_id, '.jsonl')
    with path.open('ab') as stream:
        stream.write(b'{"torn":')
    package = tmp_path/'out.tar.gz'; pack(run, package)
    import_package(package, b)
    assert import_package(package, b)['added_files'] == 0
    assert Archive(b/'replays').read(recorder.game_id)['status'] == 'interrupted'


def test_adopted_history_reports_stubs_and_missing_models_but_future_is_strict(coordinator, tmp_path):
    from brass_rl.experiments import adopt_experiment
    from brass_rl.transfer import pack
    run = tmp_path/'training/runs/legacy'
    coordinator.train(tiny_config(), run, max_updates=1)
    (run/'experiment.json').unlink()
    before = (run/'config.json').read_bytes()
    adopt_experiment(run)
    assert (run/'config.json').read_bytes() == before
    stub = run/'ancestry/parent-journal/000000'
    stub.mkdir(parents=True)
    for name in ('manifest.json','metadata.json','metrics.json','manifests.json'):
        shutil.copyfile(run/'updates/000000'/name, stub/name)
    suite = run/'evaluation/dev/historical/suite.json'
    suite.parent.mkdir(parents=True)
    suite.write_text(json.dumps({'lineups':{'old':['candidate',{'id':'missing-old-policy'},'random','greedy']}}))
    report = pack(run, tmp_path/'history.tar.gz')
    assert any('journal_only_boundaries' in g for g in report['historical_gaps'])
    assert any('missing_historical_evaluation_model' in g for g in report['historical_gaps'])
    coordinator.train(tiny_config(), run, resume=run/'latest', max_updates=2)
    with pytest.raises(ValueError, match='training replay'):
        pack(run, tmp_path/'missing-future.tar.gz')


@pytest.fixture
def unordered_legacy_package(coordinator, tmp_path):
    """Two journal stubs/suites/rows expose filesystem-order-only diagnostics."""
    from brass_rl.experiments import adopt_experiment
    from brass_rl.transfer import pack
    run = tmp_path/'training/runs/legacy-order'
    coordinator.train(tiny_config(), run, max_updates=0)
    (run/'experiment.json').unlink()
    adopt_experiment(run)
    for name in ('z-last', 'a-first'):
        stub = run/'ancestry'/name/'000000'
        stub.mkdir(parents=True)
        shutil.copyfile(run/'updates/000000/manifest.json', stub/'manifest.json')
        suite = run/'evaluation'/name/'suite.json'
        suite.parent.mkdir(parents=True)
        suite.write_text(json.dumps({'lineups': {'old': ['candidate', {'id': name}, {'id': name}, 'random']}}))
        rows = suite.parent/'candidates/old/games'
        rows.mkdir(parents=True)
        for game in (1, 0):
            (rows/('%08d.json' % game)).write_text(json.dumps(
                {'update': 0, 'suite_id': name, 'candidate_id': 'old', 'game_id': game}))
    package = tmp_path/'ordered.tar.gz'
    pack(run, package)
    return package


def _rewrite_package_inventory(source, target, change):
    """Build a valid v1 envelope with altered declared diagnostics, same blobs."""
    import io
    from brass_rl.transfer import digest, encoded
    with tarfile.open(source) as src, tarfile.open(target, 'w:gz') as out:
        for member in src.getmembers():
            data = src.extractfile(member).read()
            if member.name == 'manifest.json':
                manifest = json.loads(data)
                change(manifest['inventory'])
                manifest['package_id'] = digest(encoded({k: v for k, v in manifest.items() if k != 'package_id'}))
                data = encoded(manifest)
                member.size = len(data)
            out.addfile(member, io.BytesIO(data))


def test_inventory_verification_is_independent_of_glob_order(unordered_legacy_package, monkeypatch):
    from brass_rl.transfer import verify
    import hashlib
    package = unordered_legacy_package
    before = hashlib.sha256(package.read_bytes()).hexdigest()
    expected = verify(package)
    original = Path.glob
    monkeypatch.setattr(Path, 'glob', lambda self, pattern: iter(reversed(list(original(self, pattern)))))
    actual = verify(package)
    assert actual == expected
    assert hashlib.sha256(package.read_bytes()).hexdigest() == before


def test_old_v1_inventory_order_accepted_without_rewriting(unordered_legacy_package, tmp_path):
    from brass_rl.transfer import verify
    import hashlib
    old = tmp_path/'old-v1-order.tar.gz'
    def reverse_diagnostics(inventory):
        inventory['historical_gaps'].reverse()
        for row in inventory['historical_gaps']:
            if 'journal_only_boundaries' in row:
                row['journal_only_boundaries'].reverse()
    _rewrite_package_inventory(unordered_legacy_package, old, reverse_diagnostics)
    before = hashlib.sha256(old.read_bytes()).hexdigest()
    assert verify(old)['updates'] == 1
    assert hashlib.sha256(old.read_bytes()).hexdigest() == before


@pytest.mark.parametrize('change', ['counter', 'remove_duplicate', 'journal_path', 'extra_diagnostic'])
def test_inventory_canonicalization_does_not_hide_content_changes(unordered_legacy_package, tmp_path, change):
    from brass_rl.transfer import verify
    broken = tmp_path/'changed-inventory.tar.gz'
    def alter(inventory):
        if change == 'counter':
            inventory['updates'] += 1
        elif change == 'remove_duplicate':
            rows = inventory['historical_gaps']
            duplicate = next(row for row in rows if 'missing_historical_evaluation_model' in row)
            assert rows.count(duplicate) == 2
            rows.remove(duplicate)
        elif change == 'journal_path':
            row = next(row for row in inventory['historical_gaps'] if 'journal_only_boundaries' in row)
            row['journal_only_boundaries'][0] += '-different'
        else:
            inventory['historical_gaps'].append({'unexpected': 'diagnostic'})
    _rewrite_package_inventory(unordered_legacy_package, broken, alter)
    with pytest.raises(ValueError, match='inventory/completeness mismatch'):
        verify(broken)


def test_inventory_normalization_preserves_duplicates_and_nested_sequences():
    from copy import deepcopy
    from brass_rl.transfer import _canonical_inventory
    row = {'journal_only_boundaries': ['z', 'a', 'z'],
           'unresolved_historical_parent': {'moves': [3, 1, 2]},
           'missing_training_replays': [2, 0, 1]}
    inventory = {'updates': 7, 'historical_gaps': [row, deepcopy(row)], 'other_sequence': [4, 2, 3]}
    before = deepcopy(inventory)
    normalized = _canonical_inventory(inventory)
    assert inventory == before  # Old package declaration is never rewritten.
    assert normalized['updates'] == 7
    assert normalized['other_sequence'] == [4, 2, 3]
    assert len(normalized['historical_gaps']) == 2
    for gap in normalized['historical_gaps']:
        assert gap['journal_only_boundaries'] == ['a', 'z', 'z']
        assert gap['unresolved_historical_parent']['moves'] == [3, 1, 2]
        assert gap['missing_training_replays'] == [2, 0, 1]


@pytest.mark.parametrize('malformed', [
    {}, '', [[('missing_evaluation_replay', 'example')]],
    [{'journal_only_boundaries': 'path'}],
    [{'journal_only_boundaries': {}}],
    [{'journal_only_boundaries': [123]}],
])
def test_inventory_normalization_rejects_malformed_diagnostic_types(malformed):
    from brass_rl.transfer import _canonical_inventory
    with pytest.raises(ValueError, match='inventory.*(list|object|string)'):
        _canonical_inventory({'historical_gaps': malformed})


@pytest.mark.parametrize('malformed', [{}, ''])
def test_rehashed_package_cannot_coerce_empty_diagnostics(coordinator, tmp_path, malformed):
    from brass_rl.transfer import pack, verify
    run = tmp_path/'training/runs/no-gaps'
    coordinator.train(tiny_config(), run, max_updates=0)
    source = tmp_path/'valid.tar.gz'
    assert pack(run, source)['historical_gaps'] == []
    broken = tmp_path/'malformed.tar.gz'
    _rewrite_package_inventory(source, broken, lambda inventory: inventory.update(historical_gaps=malformed))
    with pytest.raises(ValueError, match='inventory historical_gaps must be a list'):
        verify(broken)
