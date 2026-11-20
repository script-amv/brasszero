# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import sys
import time

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/kaggle_control'))
import job
from common import extract, seal


@pytest.fixture
def monitor(tmp_path):
    state, signals = tmp_path/'state', tmp_path/'signals'
    (state/'run').mkdir(parents=True)
    signals.mkdir()
    return job.Monitor({},843,time.monotonic()+3600,state,signals,tmp_path)


def test_control_loss_latches_without_reset(monitor,monkeypatch):
    def offline(*args):
        raise OSError('offline')
    monkeypatch.setattr(job,'control',offline)
    monitor.last_contact -= 181
    monitor.observe()
    monitor.tick()
    assert (monitor.signals/'stop').exists() and monitor.reason == 'control_unavailable'
    monkeypatch.setattr(job,'control',lambda *args: {'requested':False})
    monitor.last_poll -= 36
    monitor.tick()
    monitor.advance('training')
    assert monitor.status()['phase'] == 'ack_stopping'
    assert monitor.confirmed.is_set()


def test_monotonic_deadline_and_inflight_update(monitor,monkeypatch):
    calls = []
    def control(cap,action,row=None):
        calls.append((action,row))
        return {'requested':False}
    monkeypatch.setattr(job,'control',control)
    (monitor.signals/'active').touch()
    monitor.observe()
    monitor.tick()
    monitor.tick()
    assert [call[0] for call in calls] == ['poll','progress']
    monitor.deadline = time.monotonic()-1
    monitor.observe()
    assert monitor.reason == 'deadline'
    monitor.tick()
    assert calls[-1][1]['phase'] == 'ack_stopping'


def test_setup_stop_exports_exact_resume_source(tmp_path,monkeypatch):
    import common
    data, scratch, output = [tmp_path/n for n in ('input','scratch','output')]
    for d in (data,scratch,output):
        d.mkdir()
    state = tmp_path/'seed'
    boundary = state/'run/updates/000843'
    boundary.mkdir(parents=True)
    (boundary/'optimizer.pt').write_bytes(b'fixture')
    (boundary/'manifest.json').write_bytes(b'{}')
    policy = state/'run/policies/000843'
    policy.mkdir(parents=True)
    (policy/'manifest.json').write_bytes(b'{}')
    (state/'run/latest').write_text('{"update":"000843"}')
    (state/'run/experiment.json').write_text('{"id":"qa"}')
    archive_sha = seal(state,data/'state.bin')
    common.write(data/'runner.json',{'run':'test'})
    common.write(data/'run-input.json',dict(files={'state.bin':archive_sha,'runner.json':common.digest(data/'runner.json')},
        run='test',initial=dict(update=843,experiment='qa',manifest=common.digest(boundary/'manifest.json')),
        deadline=time.time()+3600,revision='qa'))
    phases = []
    def control(cap,action,row=None):
        if row:
            phases.append(row['phase'])
        return {'requested':True}
    monkeypatch.setattr(job,'control',control)
    job.run(data/'run-input.json',scratch,time.time(),output)
    result = common.read(output/'return.json')
    assert result['reason'] == 'manual' and result['update'] == 843
    assert {'ack_stopping','checkpoint_saved','exporting','export_ready'} <= set(phases)
    extract(output/'continuation.tar.gz',tmp_path/'returned')
    assert (tmp_path/'returned/state/run/updates/000843/optimizer.pt').read_bytes() == b'fixture'
    assert {p.name for p in output.iterdir()} == {'return.json','continuation.tar.gz'}


def test_progress_attempt_budget_survives_lost_responses(monitor,monkeypatch):
    attempts = []
    now = [0]
    monkeypatch.setattr(job.time,'monotonic',lambda:now[0])
    def lost(cap,action,row=None):
        if action == 'poll':
            return {'requested':False}
        attempts.append(row['seq'])
        raise OSError('write accepted, response lost')
    monkeypatch.setattr(job,'control',lost)
    for t in range(0,12*3600,5):
        now[0] = t
        monitor.tick()
    assert len(attempts) == 360


def test_admission_margin_normal_and_slow():
    from train_run import time_for_update
    provider_end = 10000
    cutoff = provider_end-job.RESERVE-job.UPDATE_MARGIN
    assert time_for_update(cutoff-1,cutoff,50)
    assert provider_end-(cutoff-1+50) >= 1800
    assert not time_for_update(cutoff,cutoff,50)
    assert not time_for_update(cutoff-100,cutoff,400)


def test_owner_highwater_and_first_stop(tmp_path):
    from owner import remember,save_stop
    local = {'run':'qa'}
    first = save_stop(tmp_path,dict(run='qa'),1,2)
    assert save_stop(tmp_path,dict(run='qa'),6,7) == first
    for index,phase in enumerate(('training','ack_stopping','training','export_ready')):
        seen = remember(tmp_path,local,dict(requested=index==1,progress=dict(phase=phase,seq=index)))
        assert seen['requested'] is True
        if index == 2:
            assert seen['progress']['phase'] == 'ack_stopping'
    assert seen['progress']['phase'] == 'export_ready'


def test_archive_limits_and_internal_links(tmp_path,monkeypatch):
    import os
    from types import SimpleNamespace
    import common
    root = tmp_path/'data'
    root.mkdir()
    (root/'payload').write_bytes(b'0123456789')
    os.link(root/'payload',root/'copy')
    archive = tmp_path/'state.bin'
    seal(root,archive)
    extract(archive,tmp_path/'valid')
    assert os.path.samefile(tmp_path/'valid/state/copy',tmp_path/'valid/state/payload')
    with pytest.raises(ValueError):
        extract(archive,tmp_path/'many',max_files=1)
    with pytest.raises(ValueError):
        extract(archive,tmp_path/'huge',max_bytes=8)
    monkeypatch.setattr(common.shutil,'disk_usage',lambda p: SimpleNamespace(free=128*1024**2+4))
    with pytest.raises(ValueError,match='space limit'):
        extract(archive,tmp_path/'full')


def test_gate_rejects_restart_or_deadline_extension(tmp_path,monkeypatch):
    import owner
    from prepare import campaign_gate
    from common import write
    prior = dict(stage='proof',run='first',revision='rev',deadline=99,kernel='acct/proof/1',
                 initial=dict(update=843,experiment='e'))
    write(tmp_path/'job.json',prior)
    write(tmp_path/'stop-receipt.json',dict(run='first',requested=True,requested_at=1,purpose='proof'))
    archive = tmp_path/'verified/continuation.tar.gz'
    receipt = tmp_path/'verified/verified.json'
    result = dict(update=844,next_game_id=1688,replay_count=2,reason='manual',ack_at=2000,requested_at=1000,
                  outcome='normal',kernel='acct/proof/1',control_id='first',experiment='e')
    monkeypatch.setattr(owner,'download',lambda *args:result)
    monkeypatch.setattr(owner,'provider_status',lambda k:'COMPLETE')
    assert campaign_gate(tmp_path,archive,receipt,99,'rev') == 'first'
    for p,d,r in ((tmp_path/'original-seed.bin',99,'rev'),(archive,100,'rev'),(archive,99,'other')):
        with pytest.raises(ValueError):
            campaign_gate(tmp_path,p,receipt,d,r)
    for k,value in (('update',843),('outcome','recovered'),('reason','deadline'),('control_id','other')):
        actual = result[k]
        result[k] = value
        with pytest.raises(ValueError):
            campaign_gate(tmp_path,archive,receipt,99,'rev')
        result[k] = actual
    write(tmp_path/'operator-halt.json',dict(run='first'))
    with pytest.raises(ValueError):
        campaign_gate(tmp_path,archive,receipt,99,'rev')


def test_single_successor_and_owner_stop_race(tmp_path):
    from prepare import claim_successor
    from owner import stop_target
    from common import write
    proof = tmp_path/'proof'
    proof.mkdir()
    parent = dict(run='original',stage='proof')
    chosen = tmp_path/'long'
    chosen.mkdir()
    successor = dict(run='next',proof_run='original')
    write(chosen/'job.json',successor)
    claim_successor(proof,chosen,successor)
    with pytest.raises(ValueError,match='already'):
        claim_successor(proof,tmp_path/'second',dict(run='second'))
    assert stop_target(proof,parent) == chosen
    early = tmp_path/'early'
    early.mkdir()
    assert stop_target(early,parent) == early
    with pytest.raises(ValueError,match='owner ended'):
        claim_successor(early,chosen,successor)


@pytest.mark.parametrize('order',['stop_first','claim_first','existing_halt'])
def test_recovered_proof_campaign_routes_real_stop(tmp_path,monkeypatch,order):
    from campaign import adopt_retry
    from common import digest,write
    import owner
    from prepare import claim_successor
    base = tmp_path/'jobs'
    base.mkdir()
    one,two,longdir = [base/s for s in ('proof','retry','long')]
    for d in (one,two,longdir):
        d.mkdir()
    experiment = 'exp'
    write(base/'campaign.json',dict(deadline=99,experiment=experiment))
    (one/'verified').mkdir()
    (one/'verified/continuation.tar.gz').write_bytes(b'state')
    summary = dict(sha256=digest(one/'verified/continuation.tar.gz'),control_id='one',kernel='k/one/1',
                   experiment=experiment,outcome='recovered',reason='setup_error',update=843,replay_count=0)
    write(one/'verified/verified.json',summary)
    (two/'dataset').mkdir()
    (two/'dataset/state.bin').write_bytes(b'state')
    first = dict(run='one',stage='proof',deadline=99,kernel='k/one/1',initial=dict(experiment=experiment))
    retry = dict(run='two',stage='proof',deadline=99,revision='frozen',initial=summary)
    child = dict(run='three',stage='long',deadline=99,revision='frozen',proof_run='two',
                 initial=dict(experiment=experiment,control_id='two'))
    for d,j in ((one,first),(two,retry),(longdir,child)):
        write(d/'job.json',j)
        cap = tmp_path/'.local/share/brass-kaggle-control'/j['run']/'owner.json'
        cap.parent.mkdir(parents=True)
        write(cap,dict(run=j['run'],token='fixture'))
        cap.chmod(0o600)
    if order=='existing_halt':
        write(one/'operator-halt.json',dict(run='one'))
    state = adopt_retry(one,two)
    if order=='existing_halt':
        assert state['halted_by']=='preexisting-operator-halt'
        with pytest.raises(ValueError):
            claim_successor(two,longdir,child)
        return
    if order=='claim_first':
        claim_successor(two,longdir,child)
    sent=[]
    def send(cap,action,payload):
        assert action=='stop'
        sent.append(cap['run'])
        return dict(run=cap['run'],requested=True,**payload)
    monkeypatch.setattr(owner,'control',send)
    monkeypatch.setattr(Path,'home',lambda:tmp_path)
    for entry in ((one,two,longdir) if order=='claim_first' else (one,two)):
        monkeypatch.setattr(sys,'argv',['owner.py',str(entry),'stop'])
        owner.main()
    assert set(sent)=={'three' if order=='claim_first' else 'two'}
    with pytest.raises(ValueError):
        claim_successor(two,longdir,child)
