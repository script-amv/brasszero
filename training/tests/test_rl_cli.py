# SPDX-License-Identifier: Apache-2.0
"""CLI failures stay actionable, and help/import stay torch-free."""
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
CORE_PYTHON = Path(os.environ.get("BRASS_CORE_PYTHON", ROOT/"engine/bindings"/".venv"/"bin"/"python"))


def test_core_interpreter_is_actually_torch_free():
    result = subprocess.run([str(CORE_PYTHON), "-c",
                             'import importlib.util; import brass; assert importlib.util.find_spec("torch") is None'],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def command(*args, python=sys.executable):
    return subprocess.run([str(python), '-m', 'brass_rl', *map(str, args)], cwd=ROOT,
                          env=dict(os.environ, PYTHONPATH='engine/tools:training'), capture_output=True, text=True)


def test_cli_bad_config_error(tmp_path):
    config = tmp_path/'bad.json'
    config.write_text('{"games_per_update":0}')
    result = command('train', '--config', config, '--out', tmp_path/'run')
    assert result.returncode != 0
    assert 'games_per_update' in result.stderr and 'error:' in result.stderr
    assert 'Traceback' not in result.stderr
    assert not (tmp_path/'run').exists()


def test_cli_help_without_torch():
    result = command('--help', python=CORE_PYTHON)
    assert result.returncode == 0, result.stderr
    assert 'runtime' in result.stdout and 'arena' in result.stdout and 'train' in result.stdout


def test_cli_runtime():
    import json
    result = command('runtime')
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['parameters_changed'] is True


def test_cli_arena_rejects_bad_checkpoint_argument():
    result = command('arena', '--agent', 'missing-equals', '--', 'match')
    assert result.returncode != 0
    assert 'NAME=CHECKPOINT' in result.stderr
    assert 'Traceback' not in result.stderr


def test_cli_missing_training_dependencies_explains_separate_environment(tmp_path):
    result = command('train', '--config', ROOT/'training'/'configs'/'local-smoke.json', '--out', tmp_path/'run',
                     python=CORE_PYTHON)
    assert result.returncode != 0
    assert 'optional RL dependencies' in result.stderr and 'separate training environment' in result.stderr
    assert 'Traceback' not in result.stderr
    assert not (tmp_path/'run').exists()


def test_cli_eval_manifest_error_is_actionable(tmp_path):
    suite=tmp_path/'suite.json'
    suite.write_text('{"stream":"train"}')
    result=command('eval','--checkpoint',tmp_path/'missing','--suite',suite,'--out',tmp_path/'eval')
    assert result.returncode != 0
    assert 'error:' in result.stderr and 'invalid choice' not in result.stderr
    assert 'Traceback' not in result.stderr


def test_cli_eval_help_without_torch():
    result=command('eval','--help',python=CORE_PYTHON)
    assert result.returncode == 0 and '--suite' in result.stdout
