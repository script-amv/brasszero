# SPDX-License-Identifier: Apache-2.0
"""Profiler counts native legal lists once and protects published artifacts."""
import pytest

from brass_rl.config import LeagueConfig, ModelConfig, PPOConfig, TrainConfig


def small_config():
    return TrainConfig(model=ModelConfig(entity_dim=4, context_dim=8, recurrent_dim=4),
                       ppo=PPOConfig(epochs=1), league=LeagueConfig(lineup_probabilities=(1,0,0)),
                       games_per_update=1, concurrent_games=1, updates=1, master_seed=897613245)


def test_profile_counts_real_candidates(monkeypatch):
    from brass_ai.arena import RefereeGame
    from brass_rl.profile import profile_pipeline
    original = RefereeGame.pending
    measured = {}
    def pending(self):
        result = original(self)
        measured[(self._seed,self._ply)] = result[1].num_legal()
        return result
    monkeypatch.setattr(RefereeGame, 'pending', pending)
    report = profile_pipeline(small_config(), 1)
    assert report.games == 1 and report.decisions == len(measured) > 100
    assert report.candidates_total == sum(measured.values())
    assert report.candidates_max == max(measured.values())
    assert report.largest_position['candidates'] == max(measured.values())
    assert report.optimizer_steps > 0
    assert sum(report.stage_seconds.values()) == pytest.approx(report.wall_seconds)
    assert all(t >= 0 for t in report.stage_seconds.values())
    assert report.disk_bytes['updates'] > report.disk_bytes['policies'] > 0


def test_memory_limit_reports_largest_position(monkeypatch):
    from brass_rl.profile import profile_pipeline
    from brass_rl.rollout import CollectionMemoryError
    import brass_rl.rollout as rollout
    original = rollout._native_bytes
    def oversized(obs):
        return original(obs)+1024**3
    monkeypatch.setattr(rollout, '_native_bytes', oversized)
    with pytest.raises(CollectionMemoryError, match=r'largest position: game=0, ply=0, candidates=[1-9]'):
        profile_pipeline(small_config(), 1)


@pytest.mark.parametrize('target', ['policy.pt', 'metadata.json', 'manifest.json', '.', 'profile.json', 'nested/profile.json'])
def test_cli_profile_preserves_checkpoints(tmp_path, target, capsys, monkeypatch):
    from brass_rl.__main__ import main
    monkeypatch.setattr('brass_rl.profile.profile_pipeline', lambda *a: pytest.fail('must reject output before expensive training'))
    from brass_rl.artifacts import export_model_policy
    from brass_rl.model import BrassPolicy
    from brass_rl.schema import ComponentData
    c = ComponentData.load()
    policy = export_model_policy(BrassPolicy(small_config().model, c.schema, c), 0, tmp_path/'published')
    before = {p.name:p.read_bytes() for p in policy.iterdir()}
    config = tmp_path/'config.json'
    config.write_text('{}')
    with pytest.raises(SystemExit) as error:
        main(['profile','--config',str(config),'--games','1','--out',str(policy/target)])
    assert error.value.code == 2
    assert 'exists' in capsys.readouterr().err
    assert before == {p.name:p.read_bytes() for p in policy.iterdir()}


def test_profile_source_revision_is_independent_of_launch_directory(tmp_path, monkeypatch):
    from pathlib import Path
    import subprocess
    from brass_rl.profile import _version
    root = Path(__file__).resolve().parents[2]
    expected = (subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
                if (root/'.git').exists() else
                (__import__('os').environ.get('BRASS_SOURCE_REVISION') or
                 'unavailable (source archive; record archive revision separately)'))
    monkeypatch.chdir(tmp_path)
    assert _version(['git','rev-parse','HEAD']) == expected
