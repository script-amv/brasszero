# SPDX-License-Identifier: Apache-2.0
"""Controller boundaries using real recurrent PPO/Adam and tiny native observations."""
from dataclasses import replace
import json
import random

import numpy as np
import pytest
import torch

from brass_rl.artifacts import capture_rng_states, load_update
from brass_rl.config import LeagueConfig, ModelConfig, PPOConfig, TrainConfig
from helpers import shortfall_game


def assert_nested_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_nested_equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for first, second in zip(a, b):
            assert_nested_equal(first, second)
    else:
        assert a == b


def tiny_config():
    return TrainConfig(model=ModelConfig(entity_dim=4, context_dim=8, recurrent_dim=4,
                                        message_passing_rounds=1),
                       ppo=PPOConfig(epochs=2),
                       league=LeagueConfig((1., 0., 0.), snapshot_interval_updates=1, holdout_update=1),
                       games_per_update=2, concurrent_games=1, updates=4,
                       master_seed=0xE37CA450D1982FB6)


def tiny_collection(model, league, streams, start_game_id, games, concurrent_games, config):
    """Complete short optimizer fixtures, not Brass game/strength evidence.

    Real native two-choice observations/history and old-policy probabilities;
    global draws deliberately expose incorrect post-construction RNG restoration.
    Full Brass collection is separately exercised in the opt-in slow test.
    """
    from brass_ai.arena import GameResult
    from brass_ai.events import PublicEvent
    from brass_rl.agent import TorchAgent
    from brass_rl.rollout import Decision, Episode
    from brass_rl.schema import ComponentData
    components = ComponentData.load()
    result = []
    for offset in range(games):
        game_id = start_game_id + offset
        game = shortfall_game()
        initial = game.observation(0).public_json()
        agent = TorchAgent(model, components, streams.seed('policy', game_id, 0))
        agent.reset(0, initial)
        h0 = agent.memory_snapshot()
        move = game.observation(0).legal_moves()[0]
        game.apply_index(0)
        obs = game.observation(0)
        event = PublicEvent(1, 0, move, obs.public_json())
        agent.observe_public(event)
        h1 = agent.memory_snapshot()
        with torch.no_grad():
            prepared = agent.prepare(obs)
            choice = agent.choose(model([prepared.encoded], prepared.memory), 0)
        decision = Decision(game_id, 1, 0, obs, choice.index, choice.log_prob, choice.values)
        move = obs.legal_moves()[choice.index]
        game.apply_index(choice.index)
        last = PublicEvent(2, 0, move, game.observation(0).public_json())
        agent.observe_public(last)
        random.random()
        np.random.random()
        torch.rand(1)
        ranking = (0, 1, 2, 3) if offset % 2 == 0 else (1, 0, 2, 3)
        result.append(Episode(game_id, initial, (event, last),
                              {0: (h0, h1, agent.memory_snapshot())}, (decision,), league.sample(game_id),
                              GameResult(1, ('a', 'b', 'c', 'd'), ranking, (4, 3, 2, 1), 2)))
    return result


@pytest.fixture
def coordinator(monkeypatch):
    import brass_rl.train as module
    monkeypatch.setattr(module, 'collect_batch', tiny_collection)
    monkeypatch.setattr(module, 'retain_training_replays', lambda *a: ())
    return module


def rows(path):
    return [json.loads(line) for line in (path / 'metrics.jsonl').read_text().splitlines()]


def test_game_lines_report_played_version_and_only_new_committed_games(coordinator, tmp_path, capsys):
    run = tmp_path/'run'
    first = coordinator.train(tiny_config(), run, max_updates=1)
    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith('GAME ')]
    assert len(lines) == 2
    for game, line in enumerate(lines):
        assert 'game=%d ' % game in line
        assert 'model_update=0 saved_update=1' in line
        assert 'P1=AI:4VP/#1' in line if game == 0 else 'P1=AI:4VP/#2' in line
        assert 'P4=AI:1VP/#4' in line
        assert 'winner=P1' in line if game == 0 else 'winner=P2' in line
        assert json.loads((run/'experiment.json').read_text())['id'] in line
    coordinator.train(tiny_config(), run, resume=first.latest_checkpoint, max_updates=2)
    resumed = [line for line in capsys.readouterr().out.splitlines() if line.startswith('GAME ')]
    assert len(resumed) == 2
    assert all('model_update=1 saved_update=2' in line for line in resumed)
    assert 'game=2 ' in resumed[0] and 'game=3 ' in resumed[1]


@pytest.mark.parametrize('ranking,winner', [((1, 0, 2, 3), 'P2'), ((0, 0, 2, 3), 'none(tie:P1,P2)')])
def test_game_line_uses_native_tiebreaks_and_labels_opponents(coordinator, ranking, winner):
    game = {'game_id': 10, 'lineup': [
        {'kind': 'current', 'snapshot_id': None},
        {'kind': 'frozen', 'snapshot_id': 'training-000840'},
        {'kind': 'greedy', 'snapshot_id': None},
        {'kind': 'random', 'snapshot_id': None}],
        'outcome': {'vp': [50, 50, 40, 30], 'ranking': ranking}}
    line = coordinator._game_result_line(game, 'exp-example', 843, 844)
    assert 'P1=AI:50VP/#%d' % (ranking[0]+1) in line
    assert 'P2=Frozen(training-000840):50VP/#1' in line
    assert 'P3=Greedy:40VP/#3' in line
    assert 'P4=Random:30VP/#4' in line
    assert line.endswith('winner='+winner)


def test_failed_checkpoint_does_not_print_committed_game_lines(coordinator, monkeypatch, tmp_path, capsys):
    original = coordinator.save_update
    def fail(run, model, optimizer, state):
        if state.completed_updates == 1:
            raise OSError('injected save failure')
        return original(run, model, optimizer, state)
    monkeypatch.setattr(coordinator, 'save_update', fail)
    with pytest.raises(OSError, match='injected save failure'):
        coordinator.train(tiny_config(), tmp_path/'run', max_updates=1)
    assert not any(line.startswith('GAME ') for line in capsys.readouterr().out.splitlines())


def test_closed_game_log_pipe_does_not_stop_checkpointing(coordinator, monkeypatch, tmp_path):
    import builtins
    original = builtins.print
    def closed_pipe(*args, **kwargs):
        if args and isinstance(args[0], str) and args[0].startswith('GAME '):
            raise BrokenPipeError('viewer closed')
        return original(*args, **kwargs)
    monkeypatch.setattr(builtins, 'print', closed_pipe)
    result = coordinator.train(tiny_config(), tmp_path/'run', max_updates=1)
    assert result.completed_updates == 1
    assert load_update(result.latest_checkpoint).state.next_game_id == 2


def test_train_two_updates_and_resume_matches_uninterrupted(coordinator, tmp_path):
    config = tiny_config()
    full = coordinator.train(config, tmp_path / 'full')
    full_rng = capture_rng_states()
    split = coordinator.train(config, tmp_path / 'split', max_updates=2)
    assert split.completed_updates == 2
    assert split.state.config == config
    resumed = coordinator.train(config, tmp_path / 'split', resume=tmp_path / 'split' / 'latest', max_updates=4)
    assert resumed.state.next_game_id == full.state.next_game_id == 8
    assert resumed.choice_trace == full.choice_trace
    assert resumed.state.league_snapshot_ids == full.state.league_snapshot_ids
    assert resumed.state.league_history == full.state.league_history
    # Independent experiments intentionally have distinct provenance identities.
    assert {k: v for k, v in full.state.manifests.items() if k != 'experiment_id'} == {
        k: v for k, v in resumed.state.manifests.items() if k != 'experiment_id'}
    # Independent wall measurements differ; every other journal field stays exact.
    assert [{k: v for k, v in row.items() if k != 'timings_seconds'} for row in rows(tmp_path/'full')] == [
        {k: v for k, v in row.items() if k != 'timings_seconds'} for row in rows(tmp_path/'split')]
    assert_nested_equal(full.state.controller_rng_states, resumed.state.controller_rng_states)
    assert_nested_equal(full_rng, capture_rng_states())
    for key in full.model.state_dict():
        torch.testing.assert_close(full.model.state_dict()[key], resumed.model.state_dict()[key], rtol=0., atol=1e-6)
    a, b = load_update(full.latest_checkpoint), load_update(resumed.latest_checkpoint)
    assert_nested_equal(a.optimizer_state, b.optimizer_state)
    assert len(rows(tmp_path / 'split')) == 4
    assert all(len(json.loads((p/'metrics.json').read_text())['records']) <= 1
               for p in (tmp_path/'split'/'updates').iterdir() if p.name.isdecimal())
    assert full.state.development_best_id is None
    assert all(len(json.loads((p/'manifests.json').read_text())['train']) == 2
               for p in (tmp_path/'full'/'updates').iterdir() if p.name.isdecimal() and p.name != '000000')
    assert all(row['ppo']['optimizer_steps'] > 0 for row in rows(tmp_path/'full'))
    initial = torch.load(full.state.league_snapshots[0].weights_path, weights_only=True)
    assert any(not torch.equal(initial[k], v) for k, v in full.model.state_dict().items())


def test_interrupted_collection_restarts_last_boundary(coordinator, monkeypatch, tmp_path):
    config = tiny_config()
    coordinator.train(config, tmp_path / 'run', max_updates=1)
    def crash(*args):
        tiny_collection(*args)
        raise RuntimeError('injected collection interruption')
    monkeypatch.setattr(coordinator, 'collect_batch', crash)
    with pytest.raises(RuntimeError, match='injected'):
        coordinator.train(config, tmp_path / 'run', resume=tmp_path/'run'/'latest', max_updates=2)
    assert load_update(tmp_path/'run'/'latest').state.completed_updates == 1
    assert len(rows(tmp_path / 'run')) == 1
    monkeypatch.setattr(coordinator, 'collect_batch', tiny_collection)
    resumed = coordinator.train(config, tmp_path / 'run', resume=tmp_path/'run'/'latest', max_updates=2)
    reference = coordinator.train(config, tmp_path/'reference', max_updates=2)
    assert resumed.choice_trace == reference.choice_trace
    assert len(rows(tmp_path / 'run')) == 2


def test_first_collection_crash_has_resumable_update_zero(coordinator, monkeypatch, tmp_path):
    monkeypatch.setattr(coordinator, 'collect_batch', lambda *args: (_ for _ in ()).throw(RuntimeError('first collection')))
    with pytest.raises(RuntimeError, match='first collection'):
        coordinator.train(tiny_config(), tmp_path/'run')
    loaded = load_update(tmp_path/'run'/'latest')
    assert loaded.state.completed_updates == loaded.state.next_game_id == 0
    assert rows(tmp_path/'run') == []


def test_nonfinite_update_not_published(coordinator, monkeypatch, tmp_path):
    from brass_rl.ppo import UpdateMetrics
    config = tiny_config()
    coordinator.train(config, tmp_path/'run', max_updates=1)
    before = (tmp_path/'run'/'latest').read_bytes()
    monkeypatch.setattr(coordinator, 'update_policy', lambda *a: UpdateMetrics(total=float('nan')))
    with pytest.raises(ValueError, match='finite'):
        coordinator.train(config, tmp_path/'run', resume=tmp_path/'run'/'latest', max_updates=2)
    assert (tmp_path/'run'/'latest').read_bytes() == before
    assert load_update(tmp_path/'run'/'latest').state.completed_updates == 1
    assert len(rows(tmp_path/'run')) == 1


def test_ppo_abort_not_published(coordinator, monkeypatch, tmp_path):
    def abort(*args):
        raise ValueError('PPO update aborted and rolled back: nonfinite')
    monkeypatch.setattr(coordinator, 'update_policy', abort)
    with pytest.raises(ValueError, match='rolled back'):
        coordinator.train(tiny_config(), tmp_path/'run')
    assert load_update(tmp_path/'run'/'latest').state.completed_updates == 0


def test_stop_requested_finishes_update(coordinator, monkeypatch, tmp_path):
    stop = coordinator.StopFlag()
    def collect(*args):
        episodes = tiny_collection(*args)
        stop.request()
        return episodes
    monkeypatch.setattr(coordinator, 'collect_batch', collect)
    report = coordinator.train(tiny_config(), tmp_path/'run', stop_requested=stop)
    assert report.completed_updates == 1
    assert report.next_game_id == 2
    assert report.stop_reason == 'stop_requested'
    assert len(rows(tmp_path/'run')) == 1
    assert load_update(report.latest_checkpoint).state.completed_updates == 1


def test_metrics_reconstructed_from_committed_journal_without_rng_draws(coordinator, tmp_path):
    config = tiny_config()
    report = coordinator.train(config, tmp_path/'run', max_updates=2)
    original = (tmp_path/'run'/'metrics.jsonl').read_bytes()
    (tmp_path/'run'/'metrics.jsonl').write_text('interrupted cache\n')
    staging = tmp_path/'run'/'updates'/'.update-stale'
    staging.mkdir()
    (staging/'metrics.json').write_text('{"records":[{"update":999}]}')
    random.random()
    np.random.random()
    torch.rand(1)
    before = capture_rng_states()
    coordinator.rebuild_metrics(tmp_path/'run')
    assert (tmp_path/'run'/'metrics.jsonl').read_bytes() == original
    assert_nested_equal(before, capture_rng_states())
    resumed = coordinator.train(config, tmp_path/'run', resume=report.latest_checkpoint, max_updates=2)
    assert resumed.choice_trace == report.choice_trace
    assert len(rows(tmp_path/'run')) == 2


def test_run_directory_refuses_overwrite_and_historical_resume(coordinator, tmp_path):
    coordinator.train(tiny_config(), tmp_path/'run', max_updates=2)
    with pytest.raises(ValueError, match='resume'):
        coordinator.train(tiny_config(), tmp_path/'run')
    with pytest.raises(ValueError, match='latest'):
        coordinator.train(tiny_config(), tmp_path/'run', resume=tmp_path/'run'/'updates'/'000001')


@pytest.mark.slow
def test_full_brass_two_updates_and_resume(tmp_path):
    from brass_rl.train import train
    from pathlib import Path
    config = TrainConfig.from_json(Path('training/configs/local-smoke.json'))
    config = replace(config, master_seed=0xE37CA450D1982FB6)
    first = train(config, tmp_path/'full-brass', max_updates=1)
    final = train(config, tmp_path/'full-brass', resume=first.latest_checkpoint, max_updates=2)
    assert final.completed_updates == 2
    assert final.next_game_id == 4
    assert len(rows(tmp_path/'full-brass')) == 2
    assert all(np.isfinite(row['ppo']['total']) and row['ppo']['optimizer_steps'] > 0
               for row in rows(tmp_path/'full-brass'))
    assert [s.update for s in final.state.holdouts] == [1]
    assert [s.update for s in final.state.league_snapshots] == [0, 2]


def test_holdout_precedes_snapshot_once_and_store_refs_survive(coordinator, tmp_path):
    config = replace(tiny_config(), league=LeagueConfig((1., 0., 0.), snapshot_interval_updates=1, holdout_update=1))
    report = coordinator.train(config, tmp_path/'run', max_updates=2)
    assert [s.update for s in report.state.holdouts] == [1]
    assert [s.update for s in report.state.league_snapshots] == [0, 2]
    assert all(s.weights_path.is_file() and s.weights_path.parent == tmp_path/'run'/'weights'
               for s in report.state.league_snapshots + report.state.holdouts)
    resumed = coordinator.train(config, tmp_path/'run', resume=report.latest_checkpoint, max_updates=3)
    assert [s.update for s in resumed.state.holdouts] == [1]
    assert [s.update for s in resumed.state.league_snapshots] == [0, 2, 3]


def test_wall_time_margin_stops_before_collection_and_allows_resume_override(coordinator, tmp_path):
    config = tiny_config()
    config = replace(config, runtime=replace(config.runtime, max_wall_time_seconds=1., checkpoint_margin_seconds=2.))
    report = coordinator.train(config, tmp_path/'run')
    assert report.completed_updates == 0 and report.stop_reason == 'wall_time'
    extended = replace(config, runtime=replace(config.runtime, max_wall_time_seconds=None, checkpoint_margin_seconds=0.))
    assert coordinator.train(extended, tmp_path/'run', resume=report.latest_checkpoint, max_updates=1).completed_updates == 1



def test_resume_notes_surface_and_are_committed(coordinator, monkeypatch, tmp_path):
    config = tiny_config()
    report = coordinator.train(config, tmp_path/'run', max_updates=1)
    original = coordinator.load_update
    def load(*args, **kwargs):
        return replace(original(*args, **kwargs), rng_restore_notes=('synthetic nonexact restore note',))
    monkeypatch.setattr(coordinator, 'load_update', load)
    with pytest.warns(RuntimeWarning, match='synthetic nonexact'):
        resumed = coordinator.train(config, tmp_path/'run', resume=report.latest_checkpoint, max_updates=2)
    assert resumed.rng_restore_notes == ('synthetic nonexact restore note',)
    assert rows(tmp_path/'run')[-1]['rng_restore_notes'] == list(resumed.rng_restore_notes)
    assert load_update(resumed.latest_checkpoint).state.manifests['rng_restore_notes'] == list(resumed.rng_restore_notes)


def test_postrename_failure_resumes_committed_update_and_repairs_metrics(coordinator, monkeypatch, tmp_path):
    from brass_rl import artifacts
    config = tiny_config()
    coordinator.train(config, tmp_path/'run', max_updates=1)
    original = artifacts._pointer
    def fail_pointer(run, name, target):
        if name == 'latest' and target.name == '000002':
            raise OSError('injected postrename pointer failure')
        return original(run, name, target)
    monkeypatch.setattr(artifacts, '_pointer', fail_pointer)
    with pytest.raises(OSError, match='postrename'):
        coordinator.train(config, tmp_path/'run', resume=tmp_path/'run'/'latest', max_updates=2)
    assert len(rows(tmp_path/'run')) == 1
    assert load_update(tmp_path/'run'/'latest').state.completed_updates == 2
    monkeypatch.setattr(artifacts, '_pointer', original)
    resumed = coordinator.train(config, tmp_path/'run', resume=tmp_path/'run'/'latest', max_updates=2)
    assert len(rows(tmp_path/'run')) == 2
    assert [row['update'] for row in rows(tmp_path/'run')] == [1, 2]
    assert all(s.weights_path.is_file() for s in resumed.state.league_snapshots + resumed.state.holdouts)


def test_committed_metadata_corruption_rejected(coordinator, tmp_path):
    from brass_rl.artifacts import ArtifactError
    report = coordinator.train(tiny_config(), tmp_path/'run', max_updates=1)
    path = report.latest_checkpoint/'metrics.json'
    path.write_text('{"records":[]}')
    with pytest.raises(ArtifactError, match='journal metadata'):
        coordinator.rebuild_metrics(tmp_path/'run')


@pytest.mark.parametrize('section,field,value', [
    ('league', 'snapshot_interval_updates', 0), ('league', 'snapshot_interval_updates', True),
    ('league', 'holdout_update', 0), ('league', 'holdout_update', 1.5),
    ('runtime', 'checkpoint_margin_seconds', -1), ('runtime', 'checkpoint_margin_seconds', float('inf')),
])
def test_cadence_and_checkpoint_margin_configuration_validated(section, field, value):
    config = tiny_config()
    with pytest.raises(ValueError, match=field):
        replace(getattr(config, section), **{field: value})


def test_portable_policies_published_and_missing_export_repaired(coordinator, monkeypatch, tmp_path):
    from brass_rl.policy_io import inspect_policy
    config = tiny_config()
    report = coordinator.train(config, tmp_path/'run', max_updates=1)
    assert report.latest_policy == tmp_path/'run'/'policies'/'000001'
    assert inspect_policy(report.latest_policy).update == 1
    assert inspect_policy(tmp_path/'run'/'policies'/'000000').update == 0
    original = coordinator.export_policy
    def fail_export(source, destination):
        if source.name == '000002':
            raise OSError('injected export failure')
        return original(source, destination)
    monkeypatch.setattr(coordinator, 'export_policy', fail_export)
    with pytest.raises(RuntimeError, match='committed.*export'):
        coordinator.train(config, tmp_path/'run', resume=report.latest_checkpoint, max_updates=2)
    assert load_update(tmp_path/'run'/'latest').state.completed_updates == 2
    assert len(rows(tmp_path/'run')) == 2
    assert not (tmp_path/'run'/'policies'/'000002').exists()
    before = {p.name: p.read_bytes() for p in report.latest_policy.iterdir()}
    monkeypatch.setattr(coordinator, 'export_policy', original)
    repaired = coordinator.train(config, tmp_path/'run', resume=tmp_path/'run'/'latest', max_updates=2)
    assert inspect_policy(repaired.latest_policy).update == 2
    assert {p.name: p.read_bytes() for p in report.latest_policy.iterdir()} == before
    assert set(p.name for p in repaired.latest_policy.iterdir()) == {'policy.pt', 'metadata.json', 'manifest.json'}


def test_periodic_selection_is_atomic_and_resume_preserves_rng(coordinator, monkeypatch, tmp_path):
    import brass_rl.evaluate as evaluation
    def game(candidate,specs,seed,sampling):
        random.random(); np.random.random(); torch.rand(1)
        return dict(ranking=[0,1,2,3],vp=[10,5,2,0],plies=0,events=())
    monkeypatch.setattr(evaluation,'_play_game',game)
    config=replace(tiny_config(),league=replace(tiny_config().league,holdout_update=2),
                   eval=replace(tiny_config().eval,interval_updates=1,deals=1,replay_limit=0))
    full=coordinator.train(config,tmp_path/'full',max_updates=3)
    full_rng=capture_rng_states()
    first=coordinator.train(config,tmp_path/'split',max_updates=1)
    resumed=coordinator.train(config,tmp_path/'split',resume=first.latest_checkpoint,max_updates=3)
    assert_nested_equal(full_rng,capture_rng_states())
    assert_nested_equal(full.model.state_dict(),resumed.model.state_dict())
    assert_nested_equal(load_update(full.latest_checkpoint).optimizer_state,load_update(resumed.latest_checkpoint).optimizer_state)
    assert full.state.league_history == resumed.state.league_history
    assert full.state.development_best_id == resumed.state.development_best_id
    assert full.state.development_best_id is not None
    dev=[row['development'] for row in rows(tmp_path/'full')]
    assert not dev[0]['score']['holdout_available']
    assert dev[1]['score']['holdout_available']
    assert dev[0]['score']['suite_id'] != dev[1]['score']['suite_id'] == dev[2]['score']['suite_id']
    assert dev[0]['score']['selection_lineups'] == ['initial','mixed']
    assert dev[1]['score']['selection_lineups'] == ['holdout','initial','mixed']
    assert 'random' in dev[0]['summaries'] and 'earlier' in dev[0]
    assert load_update(tmp_path/'full'/'best').state.completed_updates == 2
    assert resumed.state.manifests['dev']['suite_id'] == dev[-1]['score']['suite_id']
    assert len(list((tmp_path/'split'/'evaluation'/'dev').rglob('games/*.json'))) == 56


def test_interrupted_evaluation_cannot_publish_best(coordinator,monkeypatch,tmp_path):
    import brass_rl.evaluate as evaluation
    config=replace(tiny_config(),eval=replace(tiny_config().eval,interval_updates=1,deals=1,replay_limit=0))
    def fail(*args):
        random.random(); np.random.random(); torch.rand(1)
        raise RuntimeError('injected eval interruption')
    monkeypatch.setattr(evaluation,'_play_game',fail)
    with pytest.raises(RuntimeError,match='eval interruption'):
        coordinator.train(config,tmp_path/'run',max_updates=1)
    loaded=load_update(tmp_path/'run'/'latest')
    assert loaded.state.completed_updates == 0 and loaded.state.development_best_id is None
    assert not (tmp_path/'run'/'best').exists()
    monkeypatch.setattr(evaluation,'_play_game',lambda *a: dict(ranking=[0,1,2,3],vp=[10,5,2,0],plies=0,events=()))
    report=coordinator.train(config,tmp_path/'run',resume=tmp_path/'run'/'latest',max_updates=1)
    assert report.state.development_best_id is not None


def test_evaluated_best_waits_for_completed_update_commit(coordinator,monkeypatch,tmp_path):
    import brass_rl.evaluate as evaluation
    config=replace(tiny_config(),eval=replace(tiny_config().eval,interval_updates=1,deals=1,replay_limit=0))
    calls=[]
    def game(*args):
        calls.append(1)
        return dict(ranking=[0,1,2,3],vp=[10,5,2,0],plies=0,events=())
    monkeypatch.setattr(evaluation,'_play_game',game)
    save=coordinator.save_update
    def fail(run,model,optimizer,state):
        if state.completed_updates == 1:
            raise RuntimeError('injected before journal commit')
        return save(run,model,optimizer,state)
    monkeypatch.setattr(coordinator,'save_update',fail)
    with pytest.raises(RuntimeError,match='journal commit'):
        coordinator.train(config,tmp_path/'run',max_updates=1)
    assert len(calls) == 24
    loaded=load_update(tmp_path/'run'/'latest')
    assert loaded.state.development_best_id is None and loaded.state.completed_updates == 0
    assert not (tmp_path/'run'/'best').exists()
    monkeypatch.setattr(coordinator,'save_update',save)
    report=coordinator.train(config,tmp_path/'run',resume=tmp_path/'run'/'latest',max_updates=1)
    assert len(calls) == 24  # Reuse all completed candidate/suite rows.
    assert report.state.development_best_id is not None
    assert load_update(tmp_path/'run'/'best').state.completed_updates == 1


def test_training_replay_labels_use_verified_collection_snapshots_before_eviction(coordinator,monkeypatch,tmp_path):
    import hashlib
    import brass_rl.replay as replay
    from brass_rl.league import Lineup,LineupEntry
    from brass_rl.policy_io import inspect_policy
    # Tiny optimizer fixtures are incomplete native histories. Exercise the real
    # controller/retention index while separate replay tests validate publication.
    monkeypatch.setattr(coordinator,'retain_training_replays',replay.retain_training_replays)
    monkeypatch.setattr(replay,'write_gui_replay',lambda *args,**kwargs: None)
    observed={}
    def collect(model,league,streams,start,games,concurrent,config):
        episodes=tiny_collection(model,league,streams,start,games,concurrent,config)
        snapshot=league.snapshots[-1]
        observed[start]=(snapshot.id,hashlib.sha256(snapshot.weights_path.read_bytes()).hexdigest())
        lineup=Lineup((LineupEntry('current',trainable=True),LineupEntry('frozen',snapshot.id),
                       LineupEntry('greedy'),LineupEntry('random')))
        return [replace(episode,lineup=lineup) for episode in episodes]
    monkeypatch.setattr(coordinator,'collect_batch',collect)
    config=replace(tiny_config(),games_per_update=1,ppo=replace(tiny_config().ppo,epochs=1),
                   league=replace(tiny_config().league,max_snapshots=3),
                   eval=replace(tiny_config().eval,training_replay_limit=1))
    report=coordinator.train(config,tmp_path/'run',max_updates=4)
    snapshot_id,digest=observed[3]
    assert snapshot_id == 'training-000003'
    assert snapshot_id not in report.state.league_snapshot_ids  # Cadence evicted the played opponent.
    index=json.loads((tmp_path/'run'/'replays'/'index.json').read_text())
    assert len(index) == 1
    assert index[0]['labels'] == [inspect_policy(tmp_path/'run'/'policies'/'000003').id,
                                 'training-000003:weights-sha256:'+digest,'greedy','random']


def test_journal_keeps_all_outcomes_timings_and_ties(coordinator, monkeypatch, tmp_path):
    import math
    original = tiny_collection
    def collect(*args):
        episodes = original(*args)
        episodes[1] = replace(episodes[1], result=replace(episodes[1].result,
                              ranking=(0, 0, 2, 3), vp=(4, 4, 2, 1)))
        return episodes
    monkeypatch.setattr(coordinator, 'collect_batch', collect)
    report = coordinator.train(tiny_config(), tmp_path/'run', max_updates=1)
    loaded = load_update(report.latest_checkpoint)
    games = loaded.state.manifests['train']
    assert len(games) == 2
    assert games[0]['outcome'] == {'ranking': [0, 1, 2, 3], 'vp': [4, 3, 2, 1],
                                 'strict_wins': [1, 0, 0, 0], 'tied_first': False}
    assert games[1]['outcome'] == {'ranking': [0, 0, 2, 3], 'vp': [4, 4, 2, 1],
                                 'strict_wins': [0, 0, 0, 0], 'tied_first': True}
    row = rows(tmp_path/'run')[0]
    assert set(row['timings_seconds']) == {'collection', 'ppo'}
    assert all(math.isfinite(v) and v >= 0 for v in row['timings_seconds'].values())
    before = (tmp_path/'run'/'metrics.jsonl').read_bytes()
    coordinator.train(tiny_config(), tmp_path/'run', resume=report.latest_checkpoint, max_updates=1)
    assert (tmp_path/'run'/'metrics.jsonl').read_bytes() == before


def test_failed_save_excludes_outcomes_and_timings(coordinator, monkeypatch, tmp_path):
    config = tiny_config()
    report = coordinator.train(config, tmp_path/'run', max_updates=1)
    before = rows(tmp_path/'run')
    original = coordinator.save_update
    def fail(*args):
        raise OSError('injected precommit failure')
    monkeypatch.setattr(coordinator, 'save_update', fail)
    with pytest.raises(OSError, match='precommit'):
        coordinator.train(config, tmp_path/'run', resume=report.latest_checkpoint, max_updates=2)
    coordinator.rebuild_metrics(tmp_path/'run')
    assert rows(tmp_path/'run') == before
    assert load_update(tmp_path/'run'/'latest').state.manifests['train'] == report.state.manifests['train']
    monkeypatch.setattr(coordinator, 'save_update', original)
    resumed = coordinator.train(config, tmp_path/'run', resume=report.latest_checkpoint, max_updates=2)
    assert len(rows(tmp_path/'run')) == 2
    assert [g['game_id'] for g in resumed.state.manifests['train']] == [2, 3]
    assert all('outcome' in g for g in resumed.state.manifests['train'])


@pytest.mark.parametrize('duration', [float('nan'), float('inf'), -1.])
def test_invalid_collection_timing_never_commits(coordinator, monkeypatch, tmp_path, duration):
    from types import SimpleNamespace
    clock = iter([0., 0., 0., duration, 0., 1.])
    monkeypatch.setattr(coordinator, 'time', SimpleNamespace(monotonic=lambda: next(clock)))
    with pytest.raises(ValueError, match='timings must be finite and nonnegative'):
        coordinator.train(tiny_config(), tmp_path/'run', max_updates=1)
    assert load_update(tmp_path/'run'/'latest').state.completed_updates == 0
    assert rows(tmp_path/'run') == []


def test_actual_complete_brass_outcome_is_retained(coordinator, monkeypatch, tmp_path):
    from brass_rl.rollout import collect_batch
    from brass_rl.returns import strict_outcomes
    results = []
    def collect(*args):
        episodes = collect_batch(*args)
        results.extend(ep.result for ep in episodes)
        return episodes
    monkeypatch.setattr(coordinator, 'collect_batch', collect)
    config = replace(tiny_config(), games_per_update=1, ppo=PPOConfig(epochs=1))
    report = coordinator.train(config, tmp_path/'run', max_updates=1)
    games = load_update(report.latest_checkpoint).state.manifests['train']
    assert len(games) == len(results) == 1
    assert results[0].moves > 100
    assert games[0]['outcome']['ranking'] == list(results[0].ranking)
    assert games[0]['outcome']['vp'] == list(results[0].vp)
    assert games[0]['outcome']['strict_wins'] == strict_outcomes(results[0].ranking).astype(int).tolist()


def test_legacy_journal_resume_leaves_prior_records_immutable(coordinator, monkeypatch, tmp_path):
    import hashlib
    from brass_rl.policy_io import json_bytes
    report = coordinator.train(tiny_config(), tmp_path/'run', max_updates=1)
    checkpoint = report.latest_checkpoint
    manifest = json.loads((checkpoint/'manifest.json').read_text())
    # Construct the actual earlier schema, then keep its payload bytes immutable.
    for filename, field in [('metrics.json', 'records'), ('manifests.json', 'train')]:
        data = json.loads((checkpoint/filename).read_text())
        if filename == 'manifests.json':
            data.pop('checkpoint_retention', None)
        for record in data[field]:
            record.pop('timings_seconds' if field == 'records' else 'outcome')
        payload = json_bytes(data)
        (checkpoint/filename).write_bytes(payload)
        manifest['content_hashes'][filename] = hashlib.sha256(payload).hexdigest()
    (checkpoint/'manifest.json').write_bytes(json_bytes(manifest))
    before = {p.name: p.read_bytes() for p in checkpoint.iterdir()}
    resumed = coordinator.train(tiny_config(), tmp_path/'run', resume=checkpoint, max_updates=2)
    assert before == {p.name: p.read_bytes() for p in checkpoint.iterdir()}
    from brass_rl.retention import ROLLING_POLICY
    assert resumed.state.manifests['checkpoint_retention'] == ROLLING_POLICY
    assert 'timings_seconds' not in rows(tmp_path/'run')[0]
    assert 'timings_seconds' in rows(tmp_path/'run')[1]
    assert all('outcome' in game for game in resumed.state.manifests['train'])
    coordinator.train(tiny_config(), tmp_path/'run', resume=tmp_path/'run/latest', max_updates=3)
    assert before == {p.name: p.read_bytes() for p in checkpoint.iterdir()}
    assert not (resumed.latest_checkpoint/'optimizer.pt').exists()
    assert (resumed.latest_checkpoint/'retention.json').is_file()


@pytest.mark.parametrize('captured', [False, True])
def test_periodic_recordings_include_dev_and_earlier_ids(captured, tmp_path):
    from types import SimpleNamespace
    from brass_ai.replay_archive import Archive, Recorder, labeled_seats, outcome
    from brass_rl.train import update_recordings
    from helpers import native_game
    archive = Archive(tmp_path/'archive')
    game = native_game('GameOver')
    result = outcome(game)
    ids = []
    for suite in [None, 'development', 'earlier']:
        category = 'evaluation' if suite else 'training'
        provenance = dict(category=category, experiment=suite or 'trial', game=0, run_experiment='trial', update=5)
        recorder = Recorder(archive, start_state=game.to_json(), seats=labeled_seats(['random']*4), provenance=provenance)
        recorder.finish(game)
        ids.append(recorder.game_id)
    episode = SimpleNamespace(replay_id=ids[0], game_id=0, result=SimpleNamespace(**result))
    evaluation_ids = sorted(ids[1:], reverse=True)
    batch = update_recordings(archive, [episode], 5, 'trial', 'trial', captured, evaluation_ids)
    expected = [episode.replay_id, *evaluation_ids] if captured else sorted(ids)
    assert [r['game_id'] for r in batch] == expected
