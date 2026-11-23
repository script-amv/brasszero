# SPDX-License-Identifier: Apache-2.0
"""Check the contents a JSON-only library consumer actually sees."""
import json

import pytest

from test_bc_current import seat_zero_collection  # noqa: F401
from test_train import coordinator, tiny_config  # noqa: F401


def read(path):
    return json.loads(path.read_bytes())


def test_local_and_imported_occurrences_survive_repair(coordinator, tmp_path, monkeypatch):
    from brass_rl.experiments import branch_experiment
    from brass_rl.transfer import pack, import_package
    source = tmp_path/'source'
    run = source/'training/runs/first'
    boundary = coordinator.train(tiny_config(), run, max_updates=0).latest_checkpoint
    policy = read(boundary/'metadata.json')['id']
    parent = read(run/'experiment.json')['id']
    child = run.parent/'bc-current'
    branch_experiment(boundary, child)
    coordinator.train(tiny_config(), child, child/'latest', max_updates=0)
    branch = read(child/'experiment.json')['id']
    models = source/'players/models'
    assert (models/'lineage.json').exists(), 'local publication must register every experiment'
    expected = {parent, branch}
    def check(models):
        lineage = read(models/'lineage.json')
        assert lineage['format'] == 'brass-library-lineage/1'
        assert {r['experiment_id'] for r in lineage['occurrences'][policy]} == expected
        assert {r['lineage'] for r in lineage['occurrences'][policy]} == {'experimental'}
        assert lineage['branches'][branch]['parent']['policy_id'] == policy
        assert lineage['mainline_tip'] is None
        assert read(models/'catalog.json')['other_preserved_snapshots'][0]['policy_id'] == policy
    check(models)
    out = tmp_path/'out.tar.gz'
    pack(child, out)
    dest = tmp_path/'dest'
    import_package(out, dest)
    models = dest/'players/models'
    check(models)
    before = (models/'lineage.json').read_bytes()
    (models/'lineage.json').unlink()
    (models/'catalog.json').unlink()
    assert import_package(out, dest)['added_files'] == 0
    check(models)
    assert (models/'lineage.json').read_bytes() == before
    assert (dest/'replays/index.jsonl').exists()


def test_explicit_adoption_and_competing_bc_children(coordinator, tmp_path, seat_zero_collection):  # noqa: F811
    from brass_rl.experiments import branch_experiment
    from test_bc_current import source_checkpoint
    from brass_rl.bc_current import fork_bc
    runs = tmp_path/'training/runs'
    base = source_checkpoint(coordinator, runs, max_updates=0)
    from brass_rl.library import adopt_mainline
    root_id = read(base/'metadata.json')['id']
    tip = adopt_mainline(tmp_path, base, expected_tip=None)['mainline_tip']
    assert tip['policy_id'] == root_id
    for name, experimental in [('supported', False), ('competing', False), ('opt-out', True)]:
        dest = runs/name
        fork_bc(base, dest, experimental=experimental)
        declaration = read(dest/'experiment.json')
        assert declaration['library_intent'] == ('experimental' if experimental else 'mainline')
    generic = runs/'ordinary'
    branch_experiment(base, generic)
    assert read(generic/'experiment.json')['library_intent'] == 'experimental'
    first = coordinator.train(tiny_config_for(runs/'supported'), runs/'supported', runs/'supported/latest', 1)
    second = coordinator.train(tiny_config_for(runs/'competing'), runs/'competing', runs/'competing/latest', 1)
    registry = read(tmp_path/'players/models/lineage.json')
    assert registry['mainline_tip']['experiment_id'] == read(runs/'supported/experiment.json')['id']
    second_id = read(second.latest_policy/'metadata.json')['id']
    assert any(r['experiment_id'] == read(runs/'competing/experiment.json')['id']
               and r['lineage'] == 'experimental' for r in registry['occurrences'][second_id])
    assert read(first.latest_policy/'metadata.json')['id'] == registry['mainline_tip']['policy_id']
    with pytest.raises(ValueError, match='predecessor'):
        adopt_mainline(tmp_path, second.latest_checkpoint, expected_tip=root_id)


def tiny_config_for(run):
    from brass_rl.config import TrainConfig
    return TrainConfig.from_json(run/'config.json')


def test_historical_gaps_and_sparse_boundaries(coordinator, tmp_path):
    from brass_rl.library import reconcile_library
    run = tmp_path/'training/runs/sparse'
    coordinator.train(tiny_config(), run, max_updates=3)
    import shutil
    shutil.rmtree(run/'updates/000001')
    before = (run/'updates/000003/manifest.json').read_bytes()
    report = reconcile_library(tmp_path)
    assert report['missing_training_games'] >= 2
    assert report['models'] == 4
    assert (run/'updates/000003/manifest.json').read_bytes() == before
    assert read(tmp_path/'players/models/lineage.json')['mainline_tip'] is None


def test_loose_kaggle_receipt_does_not_publish(tmp_path):
    from brass_rl.library import ingest_kaggle_return
    verified = tmp_path/'job/verified'
    verified.mkdir(parents=True)
    (verified/'verified.json').write_text('{}')
    with pytest.raises((ValueError, OSError)):
        ingest_kaggle_return(verified, tmp_path/'target')
    assert not (tmp_path/'target').exists()


def test_sealed_kaggle_return_uses_fresh_extraction_and_verified_parent(coordinator, tmp_path, monkeypatch):
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts/kaggle_control'))
    import prepare
    import shutil
    from brass_rl.library import ingest_kaggle_return
    from test_bc_current import source_checkpoint
    source = source_checkpoint(coordinator, tmp_path/'training/runs', max_updates=0)
    job = tmp_path/'job'
    verified = job/'verified'
    verified.mkdir(parents=True)
    dataset = job/'dataset'
    dataset.mkdir()
    initial = dataset/'state.bin'
    prepare.seed_bc(source, initial)
    seed = read(initial.with_suffix('.json'))
    declaration = dict(initial=seed, run='job-id', deadline=1000, revision='source', kernel='owner/job/1')
    (job/'job.json').write_text(json.dumps(declaration))
    shutil.copyfile(initial, verified/'continuation.tar.gz')
    receipt = dict(experiment=seed['experiment'], deadline=1000, revision='source', run='job-id',
                   manifest=seed['manifest'], initial_manifest=seed['manifest'], update=seed['update'],
                   sha256=seed['sha256'], initial_archive=seed['sha256'], reason='manual', outcome='normal')
    (verified/'return.json').write_text(json.dumps(receipt))
    recorded = dict(seed, control_id='job-id', kernel='owner/job/1', revision='source', deadline=1000,
                    reason='manual', outcome='normal', ack_at=None, requested_at=None)
    (verified/'verified.json').write_text(json.dumps(recorded))
    # A loose tree can change after download. It is never a source for ingestion.
    (verified/'unpacked').mkdir()
    (verified/'unpacked/extra').write_text('untrusted')
    registry = tmp_path/'players/models/lineage.json'
    before = registry.read_bytes()
    with pytest.raises(ValueError):
        ingest_kaggle_return(verified, tmp_path/'missing-parent-project')
    assert before == registry.read_bytes()
    result = ingest_kaggle_return(verified, tmp_path)
    assert result['added_files'] > 0 and result['experiment'] == seed['experiment']
    assert not (Path(result['experiment_path'])/'extra').exists()
    assert ingest_kaggle_return(verified, tmp_path)['added_files'] == 0
    assert read(registry)['branches'][seed['experiment']]['parent']['policy_id'] == read(source/'metadata.json')['id']
    before = registry.read_bytes()
    (verified/'continuation.tar.gz').write_bytes(b'damaged')
    with pytest.raises(ValueError):
        ingest_kaggle_return(verified, tmp_path)
    assert before == registry.read_bytes()

def test_action_exemplars_deduplicate_and_journal_gaps_require_matching_seed(coordinator, tmp_path):
    from helpers import native_game
    from brass_rl.library import reconcile_library
    run = tmp_path/'training/runs/trial'
    coordinator.train(tiny_config(), run, max_updates=1)
    examples = run/'replays'
    examples.mkdir(exist_ok=True)
    # Same game number, unrelated complete game. Its archive entry is retained,
    # but it cannot cover a journal decision made on another recorded deal.
    state = native_game('GameOver').to_json()
    payload = dict(format='brass-gui-game/1', start_state=state, moves=[], seats=[dict(name='Old', kind='human')]*4)
    for name in ('0.json', 'copy.json'):
        (examples/name).write_text(json.dumps(payload))
    incomplete = dict(payload, start_state=native_game().to_json())
    (examples/'prefix.json').write_text(json.dumps(incomplete))
    before = (examples/'0.json').read_bytes()
    first = reconcile_library(tmp_path)
    assert (first['legacy_files'], first['added_replays'], first['indexed_replays']) == (3, 1, 1)
    assert first['incomplete_legacy_files'] == ['training/runs/trial/replays/prefix.json']
    assert first['missing_training_games'] == 2
    second = reconcile_library(tmp_path)
    assert (second['added_replays'], second['indexed_replays'], second['missing_training_games']) == (0, 1, 2)
    assert (examples/'0.json').read_bytes() == before


def test_failure_after_registry_write_is_repaired_on_repeat_import(coordinator, tmp_path, monkeypatch):
    from brass_rl import model_inventory
    from brass_rl.transfer import import_package, pack
    source = tmp_path/'a/training/runs/trial'
    coordinator.train(tiny_config(), source, max_updates=0)
    package = tmp_path/'return.tgz'
    pack(source, package)
    project = tmp_path/'b'
    original = model_inventory.atomic
    def fail_catalog(path, data):
        if path.name == 'catalog.json':
            raise OSError('injected catalog interruption')
        original(path, data)
    with monkeypatch.context() as isolated:
        isolated.setattr(model_inventory, 'atomic', fail_catalog)
        with pytest.raises(OSError, match='catalog interruption'):
            import_package(package, project)
    assert (project/'players/models/lineage.json').exists()
    assert not list((project/'training/runs').glob('.import-*.json'))
    report = import_package(package, project)
    assert (report['added_models'], report['reused_models'], report['added_files']) == (0, 1, 0)
    assert read(project/'players/models/catalog.json')['other_preserved_snapshots']

def test_generic_import_registers_legacy_exemplars_in_the_same_operation(coordinator, tmp_path):
    from helpers import native_game
    from brass_rl.transfer import import_package, pack
    run = tmp_path/'a/training/runs/trial'
    coordinator.train(tiny_config(), run, max_updates=0)
    (run/'replays').mkdir(exist_ok=True)
    payload = dict(format='brass-gui-game/1', start_state=native_game('GameOver').to_json(),
                   moves=[], seats=[dict(name='Old', kind='human')]*4)
    (run/'replays/000010.json').write_text(json.dumps(payload))
    package = tmp_path/'packet.tgz'
    pack(run, package)
    destination = tmp_path/'b'
    report = import_package(package, destination)
    assert report['indexed_replays'] == 1
    assert import_package(package, destination)['indexed_replays'] == 1
    assert len(list((destination/'replays/games').glob('*.json.gz'))) == 1

def test_adopted_untagged_bc_resume_keeps_future_occurrences_mainline(coordinator, tmp_path, seat_zero_collection):  # noqa: F811
    from brass_rl.bc_current import fork_bc
    from brass_rl.library import adopt_mainline, reconcile_library
    from test_bc_current import source_checkpoint
    runs = tmp_path/'training/runs'
    parent = source_checkpoint(coordinator, runs, max_updates=0)
    adopt_mainline(tmp_path, parent, expected_tip=None)
    legacy = tmp_path/'historical'
    fork_bc(parent, legacy)
    meta = read(legacy/'experiment.json')
    del meta['library_intent']
    original = json.dumps(meta).encode()
    (legacy/'experiment.json').write_bytes(original)
    run = runs/'returned'
    legacy.rename(run)
    # Recreate an old return at a committed update with no portable intent.
    first = coordinator.train(tiny_config_for(run), run, run/'latest', 1)
    reconcile_library(tmp_path, runs=[run])
    origin = read(parent/'metadata.json')['id']
    assert read(tmp_path/'players/models/lineage.json')['mainline_tip']['policy_id'] == origin
    adopt_mainline(tmp_path, first.latest_checkpoint, expected_tip=origin)
    next_update = coordinator.train(tiny_config_for(run), run, run/'latest', 2)
    registry = read(tmp_path/'players/models/lineage.json')
    assert registry['mainline_tip']['policy_id'] == read(next_update.latest_policy/'metadata.json')['id']
    assert registry['mainline_tip']['experiment_id'] == meta['id']
    assert (run/'experiment.json').read_bytes() == original
