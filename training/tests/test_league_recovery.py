# SPDX-License-Identifier: Apache-2.0
import hashlib
import importlib.util
import json
from pathlib import Path
import tarfile

import pytest


def script(name):
    path = Path(__file__).resolve().parents[1]/'scripts'/(name+'.py')
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sealed_provider_partial_restores_same_committed_prefix(tmp_path,monkeypatch):
    launcher = script('kaggle_league')
    builder = script('prepare_league_recovery')
    original = tmp_path/'launch'
    (original/'kernel').mkdir(parents=True)
    (original/'dataset').mkdir()
    input_bytes = b'original sealed scientific inputs'
    sha = hashlib.sha256(input_bytes).hexdigest()
    (original/'dataset/league-input.json').write_bytes(input_bytes)
    (original/'kernel/run.py').write_text("INPUT_SHA = '%s'\nRECOVERY_SHA = 'fresh'\n"%sha)
    (original/'kernel/kernel-metadata.json').write_text(json.dumps(dict(id='user/study',dataset_sources=['user/original'])))
    prior = tmp_path/'prior/league-return'
    prior.mkdir(parents=True)
    (prior/'input-provenance.json').write_bytes(input_bytes)
    ids = dict(old='old-experiment',new='new-experiment')
    for arm,i in ids.items():
        (prior/'study'/arm).mkdir(parents=True)
        (prior/'study'/arm/'experiment.json').write_text(json.dumps({'id':i}))
    (prior/'study/arm-identities.json').write_text(json.dumps(ids))
    boundary = Path('study/old/updates/000794/manifests.json')
    (prior/boundary).parent.mkdir(parents=True)
    (prior/boundary).write_bytes(b'preserved completed update')
    archive = tmp_path/'partial.tar.gz'
    with tarfile.open(archive,'x:gz') as f:
        f.add(prior,arcname='league-return')
    receipt = builder.prepare(original,archive,tmp_path/'recovery')
    dataset = tmp_path/'recovery/dataset'
    assert receipt['arms']==ids
    monkeypatch.setattr(launcher,'INPUT_SHA',sha)
    monkeypatch.setattr(launcher,'RECOVERY_SHA',hashlib.sha256((dataset/'league-recovery.json').read_bytes()).hexdigest())
    work = tmp_path/'fresh/league-return'
    launcher.restore_provider_input(work,dataset)
    assert (work/boundary).read_bytes()==(prior/boundary).read_bytes()
    # A repeat on this new working directory must preserve newly committed data.
    continued = work/'study/old/updates/000795'
    continued.mkdir()
    launcher.restore_provider_input(work,dataset)
    assert continued.exists() and launcher.arm_identities(work)==ids
    # The same experiment IDs at a shorter boundary cannot satisfy the seal.
    (work/boundary).unlink()
    with pytest.raises(ValueError,match='sealed prefix'):
        launcher.restore_provider_input(work,dataset)
    (work/boundary).write_bytes(b'different immutable checkpoint payload')
    with pytest.raises(ValueError,match='sealed prefix'):
        launcher.restore_provider_input(work,dataset)
    (work/boundary).write_bytes((prior/boundary).read_bytes())
    (work/'study/new/experiment.json').write_text(json.dumps({'id':'reforked'}))
    with pytest.raises(ValueError,match='arm identity'):
        launcher.restore_provider_input(work,dataset)
    monkeypatch.setattr(launcher,'INPUT_SHA','other')
    with pytest.raises(ValueError,match='sealed input'):
        launcher.restore_provider_input(work,dataset)
