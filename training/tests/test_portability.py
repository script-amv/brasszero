# SPDX-License-Identifier: Apache-2.0
"""CPU transport is local QA; the Linux fixture gate requires real Linux evidence."""
import json
import os
from pathlib import Path
import subprocess

import brass
import numpy as np
import pytest
import torch

from brass_rl.agent import load_agent
from brass_rl.artifacts import export_model_policy
from brass_rl.config import ModelConfig
from brass_rl.distributions import policy_probabilities
from brass_rl.model import BrassPolicy
from brass_rl.schema import ComponentData


def probabilities(agent, fixture):
    agent.reset(fixture['seat'], fixture['initial_public_json'])
    from brass_ai.events import PublicEvent
    for event in fixture['events']:
        agent.observe_public(PublicEvent(**event))
    observation = brass.Observation.from_json(fixture['observation_json'])
    prepared = agent.prepare(observation)
    with torch.no_grad():
        return policy_probabilities(agent.model([prepared.encoded],prepared.memory)).cpu().numpy()


def test_portable_state_dict_cpu_round_trip(tmp_path, monkeypatch):
    c = ComponentData.load()
    model = BrassPolicy(ModelConfig(entity_dim=4,context_dim=8,recurrent_dim=4),c.schema,c)
    export = export_model_policy(model,0,tmp_path/'portable')
    calls = []
    load = torch.load
    def cpu_load(*args, **kwargs):
        calls.append(kwargs.get('map_location'))
        return load(*args, **kwargs)
    monkeypatch.setattr(torch,'load',cpu_load)
    agent = load_agent(export,1,device='cpu')
    assert calls == ['cpu']
    assert agent.model.config == model.config
    assert all(torch.equal(t,agent.model.state_dict()[k]) for k,t in model.state_dict().items())


@pytest.mark.slow
def test_linux_exported_history_probabilities_on_mac():
    import platform
    if platform.system() != 'Darwin':
        pytest.skip('Linux -> Mac gate must execute on Mac')
    path = Path(os.environ.get('BRASS_LINUX_FIXTURE','training/runs/linux-portability/fixture.json'))
    if not path.exists():
        pytest.skip('pending real recorded Linux export/history/probability fixture; no Linux runtime supplied')
    fixture = json.loads(path.read_text())
    assert fixture['producer']['system'] == 'Linux'
    assert fixture['producer']['source_revision']
    assert fixture['events'], 'fixture must exercise nonempty recurrent history'
    agent = load_agent(path.parent/fixture['policy_path'],1,device='cpu')
    assert agent.model.config == ModelConfig(**fixture['model_config'])
    np.testing.assert_allclose(probabilities(agent,fixture),fixture['probabilities'],rtol=0.,atol=1e-4)


def test_bootstrap_rejects_bad_arguments_without_installing():
    result = subprocess.run(['bash','training/scripts/cloud_bootstrap.sh','--unknown'],capture_output=True,text=True)
    assert result.returncode == 2
    assert 'Usage:' in result.stderr


def test_source_archive_contains_only_tracked_training_source(tmp_path):
    import tarfile
    if not (Path(__file__).resolve().parents[2]/'.git').exists():
        pytest.skip('source packaging runs from a Git checkout; this extracted archive intentionally has no .git')
    target = tmp_path/'source.tar.gz'
    result = subprocess.run(['bash','training/scripts/cloud_bootstrap.sh','--archive',str(target)],capture_output=True,text=True)
    assert result.returncode == 0, result.stderr
    with tarfile.open(target) as archive:
        names = {member.name.rstrip('/') for member in archive.getmembers() if member.isfile()}
    assert {'Cargo.lock','engine/core/src/lib.rs','engine/tools/brass_ai/fingerprints.py','training/brass_rl/train.py', 'engine/bindings/tests/fixtures/components.json'} <= names
    assert all(name.split('/')[0] in {'Cargo.toml','Cargo.lock','engine','players','training'} for name in names)
    assert not any(name.startswith('gui/') for name in names)
    assert not any(set(Path(name).parts)&{'.git','.venv','runs','oracles','HANDOFF.md'} for name in names)
    before = target.read_bytes()
    second = subprocess.run(['bash','training/scripts/cloud_bootstrap.sh','--archive',str(target)],capture_output=True,text=True)
    assert second.returncode != 0 and target.read_bytes() == before


def bootstrap_fixture(tmp_path, monkeypatch):
    """Shell control-flow simulation only: does not claim Linux/install evidence."""
    import shutil
    root = tmp_path/'source'
    script = root/'training/scripts/cloud_bootstrap.sh'
    script.parent.mkdir(parents=True)
    shutil.copyfile(Path(__file__).resolve().parents[1]/'scripts/cloud_bootstrap.sh', script)
    commands = tmp_path/'bin'
    commands.mkdir()
    log = tmp_path/'commands.jsonl'
    stub = '''#!%s
import json, os, pathlib, sys
with open(os.environ['BOOTSTRAP_LOG'], 'a') as log:
    log.write(json.dumps([sys.argv[0], *sys.argv[1:]])+'\\n')
name = pathlib.Path(sys.argv[0]).name
if name == 'uname': print('Linux')
elif sys.argv[1:3] == ['-m', 'venv']:
    path = pathlib.Path(sys.argv[3]); (path/'bin').mkdir(parents=True)
    for executable in ('python', 'maturin'):
        target = path/'bin'/executable; target.write_text(pathlib.Path(__file__).read_text()); target.chmod(0o755)
elif sys.argv[1:2] == ['-c'] and 'a,b =' in sys.argv[2]:
    code = sys.argv[2]; sys.argv = ['-c', *sys.argv[3:]]; exec(code)
elif sys.argv[1:2] == ['-c'] and 'sys.prefix' in sys.argv[2]:
    print(pathlib.Path(sys.argv[0]).resolve().parents[1])
''' % __import__('sys').executable
    for name in ('uname', 'cargo', 'rustc', 'python3.10'):
        executable = commands/name
        executable.write_text(stub)
        executable.chmod(0o755)
    monkeypatch.setenv('PATH', str(commands)+os.pathsep+os.environ['PATH'])
    monkeypatch.setenv('BOOTSTRAP_LOG', str(log))
    return root, script, log


def test_bootstrap_provisions_distinct_core_and_rl_environments(tmp_path, monkeypatch):
    root, script, log = bootstrap_fixture(tmp_path, monkeypatch)
    result = subprocess.run(['bash', str(script), '--venv', 'rl-env', '--core-venv', 'core-env'],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    commands = [json.loads(line) for line in log.read_text().splitlines()]
    assert [c[3] for c in commands if c[1:3] == ['-m', 'venv']] == ['core-env', 'rl-env']
    core = [c for c in commands if c[0].endswith('core-env/bin/python')]
    assert any(c[1:4] == ['-m', 'pip', 'install'] and 'maturin==1.9.6' in c for c in core)
    assert any('find_spec("torch") is None' in ' '.join(c) for c in core)
    assert sum(c[0].endswith('/bin/maturin') for c in commands) == 2
    assert any('brass_rl' in c and 'runtime' in c for c in commands)


@pytest.mark.parametrize('existing', ['rl-env', 'core-env'])
def test_bootstrap_refuses_either_existing_environment_before_creation(tmp_path, monkeypatch, existing):
    root, script, log = bootstrap_fixture(tmp_path, monkeypatch)
    (root/existing).mkdir()
    result = subprocess.run(['bash', str(script), '--venv', 'rl-env', '--core-venv', 'core-env'],
                            capture_output=True, text=True)
    assert result.returncode != 0 and 'refusing' in result.stderr
    assert not any(json.loads(line)[1:3] == ['-m', 'venv'] for line in log.read_text().splitlines())


def test_bootstrap_preflight_installs_nothing(tmp_path, monkeypatch):
    root, script, log = bootstrap_fixture(tmp_path, monkeypatch)
    result = subprocess.run(['bash', str(script), '--check', '--core-venv', 'core-env'],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert 'Preflight only' in result.stdout
    assert not any('-m' in json.loads(line) for line in log.read_text().splitlines())


@pytest.mark.parametrize('core', ['rl-env', 'rl-env/nested', '.'])
def test_bootstrap_rejects_shared_or_nested_environments(tmp_path, monkeypatch, core):
    root, script, log = bootstrap_fixture(tmp_path, monkeypatch)
    result = subprocess.run(['bash', str(script), '--venv', 'rl-env', '--core-venv', core],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert not any(json.loads(line)[1:3] == ['-m', 'venv'] for line in log.read_text().splitlines())


@pytest.mark.parametrize('option', ['--python', '--venv', '--core-venv', '--archive'])
def test_bootstrap_rejects_missing_option_values(option):
    result = subprocess.run(['bash', 'training/scripts/cloud_bootstrap.sh', option], capture_output=True, text=True)
    assert result.returncode == 2 and 'Usage:' in result.stderr


def test_bootstrap_requires_git_for_complete_provenance_test_gate(tmp_path, monkeypatch):
    root, script, log = bootstrap_fixture(tmp_path, monkeypatch)
    commands = root.parent/'bin'
    import shutil
    (commands/'dirname').symlink_to(shutil.which('dirname'))
    monkeypatch.setenv('PATH', str(commands))
    result = subprocess.run(['/bin/bash', str(script), '--check'], capture_output=True, text=True)
    assert result.returncode == 2 and 'Git' in result.stderr
