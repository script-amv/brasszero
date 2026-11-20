# SPDX-License-Identifier: Apache-2.0
"""CLI remains lazy and rejects invalid search input before optional loading."""
import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]


def run(*args):
    env = dict(os.environ, PYTHONPATH='engine/tools:training:players/search')
    return subprocess.run([str(ROOT/'engine/bindings/.venv/bin/python'), '-m', 'brass_rl', *args],
                          cwd=ROOT, env=env, text=True, capture_output=True)


def test_search_help_without_torch():
    for command in ('search-eval', 'search-arena'):
        result = run(command, '--help')
        assert result.returncode == 0, result.stderr
        assert '--config' in result.stdout


def test_search_invalid_config_before_loading(tmp_path):
    config = tmp_path/'bad.json'; config.write_text('{"max_rollouts": -1}')
    result = run('search-eval', '--checkpoint', '/missing', '--config', str(config), '--out', str(tmp_path/'out'))
    assert result.returncode == 2 and 'max_rollouts' in result.stderr
    assert 'optional RL dependencies' not in result.stderr
    assert not (tmp_path/'out').exists()


def test_cli_resume_master_and_signal_boundary(tmp_path, monkeypatch, capsys):
    import signal
    from types import SimpleNamespace
    from brass_rl import __main__ as cli
    from brass_rl import search_evaluate
    from brass_rl import policy_io
    from brass_search import identity
    out = tmp_path/'run'; out.mkdir()
    (out/'suite.json').write_text('{"master_seed": 18446744073709551615}')
    from brass_rl.search_budget import WallBudget
    WallBudget(out, 1800)
    config = tmp_path/'config.json'; config.write_text('{"max_rollouts": 0}')
    monkeypatch.setattr(policy_io, 'inspect_policy', lambda path: SimpleNamespace(id='plain'))
    monkeypatch.setattr(identity, 'player_manifest', lambda *args: {'id': 'search'})
    def evaluate(checkpoint, config, out, **kwargs):
        assert kwargs['master_seed'] == 2**64-1
        assert not kwargs['stop_requested']()
        signal.getsignal(signal.SIGINT)(signal.SIGINT, None)
        assert kwargs['stop_requested']()
        return {'stop_reason': 'requested'}
    monkeypatch.setattr(search_evaluate, 'evaluate_search', evaluate)
    handler = signal.getsignal(signal.SIGINT)
    cli.main(['search-eval','--checkpoint','/unused','--config',str(config),'--out',str(out)])
    assert signal.getsignal(signal.SIGINT) == handler
    assert 'requested' in capsys.readouterr().out


@pytest.mark.parametrize('boundary', ['reserve', 'optional_import', 'output', 'refund'])
def test_cli_sigterm_covers_reserved_invocation(tmp_path, monkeypatch, boundary):
    import builtins
    import json
    import signal
    from brass_rl import __main__ as cli
    from brass_rl import search_budget, search_evaluate

    out = tmp_path/'run'
    config = tmp_path/'config.json'
    config.write_text('{"max_rollouts":0}')
    previous = signal.getsignal(signal.SIGTERM)
    observed_stop = []

    def request_stop():
        assert signal.getsignal(signal.SIGTERM) != previous, boundary
        signal.raise_signal(signal.SIGTERM)

    atomic_json = search_budget.atomic_json
    def write_budget(path, state):
        atomic_json(path, state)
        if state['reservation'] is not None and boundary == 'reserve':
            request_stop()
        elif state['measured_seconds'] > 0 and boundary == 'refund':
            request_stop()

    original_import = builtins.__import__
    def guarded_import(name, *args, **kwargs):
        if name == 'search_evaluate':
            state = json.loads((out/'budget.json').read_text())
            assert state['remaining_seconds'] == 0 and state['reservation']['seconds'] == 1800
            if boundary == 'optional_import':
                request_stop()
        return original_import(name, *args, **kwargs)

    def evaluate(*args, stop_requested, **kwargs):
        observed_stop.append(stop_requested)
        assert stop_requested() == (boundary in ('reserve', 'optional_import'))
        return {'completed_games': 0}

    def output(value):
        assert json.loads(value)['completed_games'] == 0
        if boundary == 'output':
            request_stop()

    monkeypatch.setattr(search_budget, 'atomic_json', write_budget)
    monkeypatch.setattr(builtins, '__import__', guarded_import)
    monkeypatch.setattr(search_evaluate, 'evaluate_search', evaluate)
    monkeypatch.setattr(cli, 'print', output, raising=False)
    cli.main(['search-eval', '--checkpoint', '/unused', '--config', str(config), '--out', str(out)])
    assert observed_stop[0]() and signal.getsignal(signal.SIGTERM) == previous
    state = json.loads((out/'budget.json').read_text())
    assert state['reservation'] is None and 0 < state['remaining_seconds'] < 1800


def test_search_arena_seeded_match_ignores_random_session_but_play_retains_it(tmp_path,monkeypatch):
    import uuid
    from types import SimpleNamespace
    from brass_rl import __main__ as cli,policy_io
    import brass_search
    from brass_search import identity
    import brass_ai.__main__ as arena_cli
    import brass_ai.arena as arena
    from brass_ai.replay_archive import recording_metadata,digest
    config=tmp_path/'config.json';config.write_text('{"max_rollouts":0}')
    monkeypatch.setattr(policy_io,'inspect_policy',lambda *a:None)
    monkeypatch.setattr(identity,'player_manifest',lambda *a:{'id':'search:test','base_policy_id':'policy:test','base_update':600,'config':{'max_rollouts':0}})
    monkeypatch.setattr(brass_search,'register_search_checkpoint',lambda *a,**k:None)
    seen=[]
    monkeypatch.setattr(arena,'play_game',lambda *a,**k:seen.append(recording_metadata()))
    def terminal(args):
        if args[0]=='match':arena.run_match(['random']*4,1,6)
        else:seen.append(recording_metadata())
    monkeypatch.setattr(arena_cli,'main',terminal)
    for session in ('first-uuid','second-uuid'):
        monkeypatch.setattr(uuid,'uuid4',lambda session=session:SimpleNamespace(hex=session))
        cli.main(['search-arena','--agent','searched=/unused','--config',str(config),'--','match'])
    assert seen[:4]==seen[4:]
    assert [digest(x) for x in seen[:4]]==[digest(x) for x in seen[4:]]
    assert all('session' not in x and 'search_players' in x for x in seen)
    seen.clear()
    for session in ('first-uuid','second-uuid'):
        monkeypatch.setattr(uuid,'uuid4',lambda session=session:SimpleNamespace(hex=session))
        cli.main(['search-arena','--agent','searched=/unused','--config',str(config),'--','play'])
    assert [x['session'] for x in seen]==['first-uuid','second-uuid']
