# SPDX-License-Identifier: Apache-2.0
"""One-variable full-state experiment and terminal-outcome diagnostics."""
from dataclasses import replace
from pathlib import Path
import hashlib
import json

import brass
import pytest
import torch

from brass_ai.arena import RefereeGame
from brass_rl.agent import TorchAgent
from brass_rl.artifacts import capture_rng_states, load_update
from brass_rl.league import League
from brass_rl.seeding import SeedStreams
from helpers import run_without_torch
from test_train import assert_nested_equal, coordinator, tiny_config  # noqa: F401


def fingerprint(root):
    return {p.relative_to(root): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file()}


def test_real_outcome_brier_uses_relative_seat_and_pre_event_half():
    from brass_rl.evaluate import terminal_brier
    samples = [{'ply': 0, 'era': 'canal', 'round': 4, 'values': [.75, .25, 0., 0.]},
               {'ply': 5, 'era': 'canal', 'round': 5, 'values': [0., 1., 0., 0.]}]
    result = terminal_brier(samples, [0, 1, 2, 3], seat=3, plies=10)
    assert result['outcome_relative'] == [0., 1., 0., 0.]
    assert result['first_self_brier'] == result['canal_opening_self_brier'] == .75**2
    assert result['self_brier'] == .75**2 / 2
    assert result['canal_opening_decisions'] == 1
    assert result['early_decisions'] == 1 and result['decisions'] == 2
    assert result['early_brier'] == (.75**2 + .75**2) / 4
    assert result['brier'] == result['early_brier'] / 2
    tie = terminal_brier(samples, [0, 0, 2, 3], seat=0, plies=10)
    assert tie['outcome_relative'] == [0.] * 4
    assert tie['brier'] == (.75**2 + .25**2 + 1) / 8


def test_value_recording_keeps_native_choices_private_rng_and_memory(tmp_path):
    from brass_rl.evaluate import RecordingCandidate
    from brass_rl.model import BrassPolicy
    from brass_rl.schema import ComponentData
    components = ComponentData.load()
    model = BrassPolicy(tiny_config().model, components.schema, components)
    ref = RefereeGame(52, [brass.RandomBot(i) for i in range(4)])
    seat, obs = ref.pending()
    plain = TorchAgent(model, components, 128)
    traced = RecordingCandidate(model, components, 128)
    for a in (plain, traced):
        a.reset(seat, obs.public_json())
    global_rng = capture_rng_states()
    for _ in range(12):
        actor, obs = ref.pending()
        if actor == seat:
            with torch.no_grad():
                p = plain.prepare(obs)
                choice = plain.choose(model([p.encoded], p.memory, candidate_chunk=1024), 0)
            if traced.samples:
                with torch.no_grad():
                    p = traced.prepare(obs)
                    assert traced.choose(model([p.encoded], p.memory, candidate_chunk=1024), 0) == choice
            else:
                assert traced.act(obs) == choice.index
            assert traced.samples[-1] == {'ply': plain.ply, 'values': list(choice.values),
                                           'round': plain._public['round'], 'era': plain._public['era']}
            assert torch.equal(plain._generator.get_state(), traced._generator.get_state())
            index = choice.index
        else:
            index = 0
        event = ref.apply_index(index)
        for a in (plain, traced):
            a.observe_public(event)
        assert torch.equal(plain.memory_snapshot(), traced.memory_snapshot())
    assert len(traced.samples) >= 2
    assert_nested_equal(global_rng, capture_rng_states())


def test_compact_lambda_forks_have_identical_state_and_resume(coordinator, tmp_path, monkeypatch):
    from brass_rl.learning_study import fork_arm
    from brass_rl.policy_io import ArtifactError
    original = tmp_path/'original'
    parent_cfg = tiny_config()
    parent_cfg = replace(parent_cfg, eval=replace(parent_cfg.eval, interval_updates=2))
    def choose_prior(model, update, league, run_dir, cfg, journal):
        identity = json.loads((run_dir/'updates/000001/metadata.json').read_text())['id']
        return {'best_id': identity, 'manifest': {}}
    monkeypatch.setattr(coordinator, 'periodic_development', choose_prior)
    first = coordinator.train(parent_cfg, original, max_updates=2)
    source = first.latest_checkpoint
    before = fingerprint(original)
    loaded = load_update(source)
    assert loaded.state.development_best_id is not None
    arm_configs = []
    traces = []
    for arm, lam in [('control', .95), ('candidate', .99)]:
        dest = tmp_path/arm
        fork_arm(source, dest, lam)
        child = load_update(dest/'latest')
        meta = json.loads((dest/'experiment.json').read_text())
        assert meta['parent']['checkpoint_sha256'] == before[source.relative_to(original)/'manifest.json']
        assert meta['parent']['model_id'] == json.loads((source/'metadata.json').read_text())['id']
        assert meta['base_update'] == 2 and meta['historical_gaps']
        assert meta['config_overrides']['ppo.gae_lambda'] == {'from': .95, 'to': lam}
        assert [p.name for p in (dest/'updates').iterdir()] == ['000002']
        assert child.state.development_best_id is None
        assert not child.state.metrics and not child.state.manifests.get('train')
        assert child.state.next_game_id == loaded.state.next_game_id == 4
        assert child.state.master_seed == loaded.state.master_seed
        assert_nested_equal(loaded.model.state_dict(), child.model.state_dict())
        assert_nested_equal(loaded.optimizer_state, child.optimizer_state)
        for name in ('rng_states','controller_rng_states','league_history','league_snapshot_ids'):
            assert_nested_equal(getattr(loaded.state, name), getattr(child.state, name))
        def league(s):
            return League.restore(SeedStreams(s.master_seed), s.league_snapshots, s.config.league,
                                  s.holdouts, s.league_history)
        assert league(child.state).sample(4) == league(loaded.state).sample(4)
        assert child.state.config.ppo.gae_lambda == lam
        assert not child.state.config.eval.interval_updates
        arm_configs.append(child.state.config)
        traces.append(coordinator.train(child.state.config, dest, resume=dest/'latest', max_updates=3).choice_trace)
        with pytest.raises(ArtifactError, match='config mismatch'):
            load_update(dest/'latest', config=replace(child.state.config, ppo=replace(child.state.config.ppo, gae_lambda=.5)))
    assert arm_configs[0] == replace(arm_configs[1], ppo=arm_configs[0].ppo)
    assert arm_configs[0].ppo == replace(arm_configs[1].ppo, gae_lambda=.95)
    assert traces[0] == traces[1] and len(traces[0]) == 2
    after = fingerprint(original)
    source_pin = Path('updates/pins')/(before[source.relative_to(original)/'manifest.json']+'.json')
    assert after.pop(source_pin) == hashlib.sha256((original/source_pin).read_bytes()).hexdigest()
    assert json.loads((original/source_pin).read_text())['update'] == int(source.name)
    assert after == before


def test_study_limits_reserve_time_and_bytes_before_next_unit():
    from brass_rl.learning_study import budget_reason
    assert budget_reason(719, 0, 120) is None
    assert budget_reason(120, 0, 120) == 'time'
    assert budget_reason(2000, 1024**3 - 60 * 1024**2 + 1, 120) == 'disk'


def test_learning_rate_branch_retains_adam_moments_and_uses_declared_step(coordinator, tmp_path):
    from brass_rl.learning_study import fork_arm
    original = tmp_path/'parent'
    source = coordinator.train(tiny_config(), original, max_updates=1).latest_checkpoint
    before = fingerprint(original)
    parent = load_update(source)
    rate = parent.state.config.ppo.learning_rate / 10
    branch = tmp_path/'slower'
    fork_arm(source, branch, .95, learning_rate=rate)
    child = load_update(branch/'latest')
    assert child.state.config.ppo.learning_rate == rate
    assert child.optimizer_state['param_groups'][0]['lr'] == rate
    assert_nested_equal(parent.optimizer_state['state'], child.optimizer_state['state'])
    assert_nested_equal(parent.model.state_dict(), child.model.state_dict())
    for name in ('rng_states', 'controller_rng_states', 'league_history', 'next_game_id'):
        assert_nested_equal(getattr(parent.state, name), getattr(child.state, name))
    meta = json.loads((branch/'experiment.json').read_text())
    assert meta['config_overrides']['ppo.learning_rate'] == {'from': rate*10, 'to': rate}
    assert meta['initial_state_verified']['adam']
    coordinator.train(child.state.config, branch, resume=branch/'latest', max_updates=2)
    assert load_update(branch/'latest').optimizer_state['param_groups'][0]['lr'] == rate
    after = fingerprint(original)
    source_pin = Path('updates/pins')/(before[source.relative_to(original)/'manifest.json']+'.json')
    assert after.pop(source_pin) == hashlib.sha256((original/source_pin).read_bytes()).hexdigest()
    assert json.loads((original/source_pin).read_text())['update'] == int(source.name)
    assert after == before


def test_study_help_without_torch():
    result = run_without_torch("""
import runpy
sys.argv = ['learning_study', '--help']
runpy.run_module('brass_rl.learning_study', run_name='__main__')
""")
    assert result.returncode == 0 and 'run' in result.stdout


def test_value_scores_average_each_game_then_each_deal():
    from brass_rl.learning_study import value_summary
    rows = []
    for deal, error, decisions in ((0, .1, 1), (1, .4, 10)):
        for rotation in range(4):
            rows.append({'deal_id': deal, 'lineup': 'greedy', 'rotation': rotation,
                         'terminal_brier': {'brier': error, 'decisions': decisions}})
    out = value_summary(rows)['greedy']
    assert out['independent_deals'] == 2
    assert out['deal_means']['0']['brier'] == .1
    assert out['mean']['brier'] == .25
    assert out['decisions'] == 44


def test_live_eval_resumes_rows_with_unavailable_legacy_scores(tmp_path, monkeypatch):
    from brass_rl import evaluate as module
    from brass_rl.evaluate import EvalSuite, evaluate
    from brass_rl.policy_io import inspect_policy
    # Only native UI integration tests play full games. Here two scalar rows
    # suffice to check the callback is only called for genuinely missing games.
    from brass_rl.artifacts import export_model_policy
    from brass_rl.model import BrassPolicy
    from brass_rl.schema import ComponentData
    components = ComponentData.load()
    policy = export_model_policy(BrassPolicy(tiny_config().model, components.schema, components), 0, tmp_path/'p')
    identity = inspect_policy(policy).id
    declared = EvalSuite('dev', 100, 1, {'greedy': ['candidate'] + ['greedy']*3}, identity, 0)
    budget = iter([None, 'stop'])
    def check():
        if next(budget) is not None:
            raise RuntimeError('budget')
    monkeypatch.setattr(module, '_play_game', lambda *a: dict(ranking=[0, 1, 2, 3], vp=[4, 3, 2, 1],
                                                             plies=0, events=()))
    with pytest.raises(RuntimeError, match='budget'):
        evaluate(policy, declared, tmp_path/'eval', before_game=check)
    files = list((tmp_path/'eval').rglob('games/*.json'))
    assert len(files) == 1
    old = files[0].read_bytes()
    called = []
    result = evaluate(policy, declared, tmp_path/'eval', before_game=lambda: called.append(1))
    assert len(called) == 3 and len(result.results) == 4
    assert result.results[0]['terminal_brier'] is None and files[0].read_bytes() == old


def test_stopped_study_resumes_both_arms_and_binds_equal_endpoints(coordinator, tmp_path, monkeypatch):
    """Tiny native training fixtures and synthetic eval rows test restart bookkeeping."""
    import time
    from brass_rl import learning_study as study
    from brass_rl import evaluate as evaluator
    from brass_rl.evaluate import EvalSuite
    school = tmp_path/'school'
    first = coordinator.train(tiny_config(), school, max_updates=1)
    root = tmp_path/'experiment/study'
    root.mkdir(parents=True)
    for name, lam in study.ARMS.items():
        study.fork_arm(first.latest_checkpoint, root/name, lam)
    monkeypatch.setattr(study, 'BASE', 1)
    monkeypatch.setattr(study, 'END', 2)
    monkeypatch.setattr(study, 'SCHOOL', school)
    monkeypatch.setattr(study, 'shared_bytes', lambda: 0)
    monkeypatch.setattr(study, 'caffeinate', lambda pid: None)
    def suite(master, candidate):
        return EvalSuite('dev', master, 1, {'greedy': ['candidate'] + ['greedy']*3}, candidate, 0)
    monkeypatch.setattr(study, 'suite', suite)
    monkeypatch.setattr(evaluator, '_play_game', lambda *args: {
        'events': (), 'ranking': [0,1,2,3], 'vp': [4,3,2,1], 'plies': 1,
        'terminal_brier': {'decisions': 1, 'brier': .25}})
    manifest = {'shared_baseline_bytes': 0, 'suite': suite(42, 'unused').manifest()}
    start = time.time()
    stopped = study.run(root, manifest, lambda: True, start)
    assert stopped['reason'] == 'manual'
    assert stopped['completed_updates'] == {'control': 1, 'candidate': 1}
    assert not (root/'endpoints.json').exists()
    done = study.run(root, manifest, lambda: False, start + 100)
    assert done['status'] == 'complete'
    assert done['deadline_epoch'] == start + 7200
    endpoints = study.read(root/'endpoints.json')
    assert endpoints['matched_update'] == 2
    assert set(endpoints['candidates']) == set(study.read(root/'result.json')['policies'])
    assert set(endpoints['candidates']) == {'control', 'candidate', 'parent'}
    before = fingerprint(root/'eval')
    study.run(root, manifest, lambda: False, start + 200)
    assert fingerprint(root/'eval') == before


def test_four_forks_start_with_parent_public_bytes_and_private_state(coordinator, tmp_path):
    from brass_rl.four_branch_study import ARMS, arm_config
    from brass_rl.learning_study import fork_arm
    from brass_rl.policy_io import ArtifactError
    src = coordinator.train(tiny_config(), tmp_path/'school', max_updates=1).latest_checkpoint
    parent = load_update(src)
    identities = set()
    for name in ARMS:
        child = tmp_path/name
        config = arm_config(parent.state.config, name)
        meta = fork_arm(src, child, config.ppo.gae_lambda, study_config=config)
        identities.add(meta['id'])
        assert all(meta['initial_state_verified'].values())
        restored = load_update(child/'latest', config=config)
        for f in ('policy.pt', 'metadata.json'):
            assert (src/f).read_bytes() == (restored.path/f).read_bytes()
        assert_nested_equal(parent.optimizer_state, restored.optimizer_state)
        assert_nested_equal(parent.state.controller_rng_states, restored.state.controller_rng_states)
        assert_nested_equal(parent.state.rng_states, restored.state.rng_states)
        assert meta['parent']['model_id'] == meta['initial_model_id']
        if name == 'C':
            with pytest.raises(ArtifactError):
                load_update(child/'latest', config=arm_config(parent.state.config, 'control'))
    assert len(identities) == 4
