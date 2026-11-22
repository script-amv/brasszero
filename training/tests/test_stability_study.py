# SPDX-License-Identifier: Apache-2.0
"""Recovery, sealed resource decisions and matched-deal arithmetic without scientific games."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


def test_resume_schedules_only_uncommitted_control_games():
    from brass_rl.stability_study import next_work
    assert next_work({'control': 0, 'candidate': 0}, 128) == ('control', 8)
    assert next_work({'control': 6, 'candidate': 0}, 128) == ('control', 2)
    assert next_work({'control': 8, 'candidate': 0}, 128) == ('candidate', 8)
    assert next_work({'control': 10, 'candidate': 8}, 128) == ('control', 6)
    assert next_work({'control': 128, 'candidate': 128}, 128) is None
    for a, b in [(16, 0), (4, 8), (7, 0), (134, 128)]:
        with pytest.raises(ValueError):
            next_work({'control': a, 'candidate': b}, 128)


def test_resource_only_extension_and_reserves():
    from brass_rl.stability_study import extension, bound
    gib = 1024**3
    # 16 more paired blocks at 300s each, plus a fixed 90-minute evaluation reserve.
    assert extension(remaining=11_000, pair_seconds=300, pair_bytes=10**7, used=gib, free=5*gib) == 256
    assert extension(remaining=10_000, pair_seconds=300, pair_bytes=10**7, used=gib, free=5*gib) == 128
    assert extension(remaining=25_000, pair_seconds=300, pair_bytes=gib/5, used=gib, free=9*gib) == 128
    assert bound(remaining=300, needed=301, used=0, free=9*gib) == 'time'
    assert bound(remaining=100, needed=1, used=4*gib-1, free=9*gib) == 'disk'
    assert bound(remaining=100, needed=1, used=0, free=2*gib+1) == 'free_disk'


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protocol_fixture(tmp_path):
    project = tmp_path
    out = project/'training/runs/ppo-stability-20261122'
    pre = project/'preregistration.json'
    dump(pre, {'task_deadline_utc': '2026-11-22T21:59:00+00:00'})
    protocol = dict(project=str(project), out=str(out), preregistration=str(pre), preregistration_sha256=sha(pre),
                    deadline_utc='2026-11-22T21:59:00+00:00', base_update=4055, first_game_id=8110)
    dump(out/'protocol.json', protocol)
    progress = dict(state='running', phase='training', pid=2**30, experiment_ids={'control': 'exp-control', 'candidate':'exp-candidate'},
                    committed_games={'control': 0, 'candidate': 0}, protocol_sha256=sha(out/'protocol.json'))
    dump(out/'progress.json', progress)
    return project, out, pre


def test_stdlib_status_and_stop_use_latest_commit_after_abrupt_exit(tmp_path):
    project, out, pre = protocol_fixture(tmp_path)
    dump(out/'candidate/experiment.json', {'id':'exp-candidate'})
    dump(out/'candidate/updates/004055/metrics.json', {'records':[]})
    base = out/'control/updates/004055'
    dump(base/'metrics.json', {'records': []})
    dump(out/'control/experiment.json', {'id': 'exp-control'})
    nxt = base.with_name('004056')
    dump(nxt/'metrics.json', {'records': [{'first_game_id': 8110, 'next_game_id': 8112, 'games': 2, 'update': 4056}]})
    dump(nxt/'manifest.json', {'update': 4056, 'content_hashes': {'metrics.json': sha(nxt/'metrics.json')}})
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]))
    args = ['--out', str(out), '--project', str(project), '--preregistration', str(pre)]
    def command(name):
        return subprocess.run([sys.executable, '-S', '-m', 'brass_rl.stability_study', name, *args],
                              env=env, text=True, capture_output=True)
    result = command('status')
    assert result.returncode == 0, result.stderr
    status = json.loads(result.stdout)
    assert status['state'] == 'interrupted' and status['committed_games'] == {'control': 2, 'candidate': 0}
    assert command('stop').returncode == 0 and (out/'STOP').is_file()
    assert (nxt/'metrics.json').is_file()
    pre.write_text('{}')
    assert command('status').returncode != 0
    (out/'progress.json').unlink()
    assert command('resume').returncode != 0


def test_progress_cannot_rebind_deadline(tmp_path):
    from brass_rl.stability_study import read_control
    project, out, pre = protocol_fixture(tmp_path)
    protocol = json.loads((out/'protocol.json').read_bytes())
    protocol['deadline_utc'] = '2026-11-24T00:00:00+00:00'
    dump(out/'protocol.json', protocol)
    with pytest.raises(ValueError):
        read_control(out, project, pre)


def rows(arm, offset):
    rows = []
    for deal, delta in enumerate((0, 10)):
        for rotation, seat in enumerate((0, 3, 2, 1)):
            # Includes one shared first in each block that is never a sole win.
            vp = [3]*4
            ranks = [3]*4
            ranks[seat] = (0, 0, 1, 2)[rotation]
            ranks[(seat+1)%4] = 0 if rotation else 1
            vp[seat] = 10+delta+rotation+offset
            rows.append(dict(candidate_id=arm, candidate_seat=seat, rotation=rotation, deal_id=deal,
                             deal_seed=100+deal, lineup='greedy', seat_ids=[arm if i == seat else 'greedy' for i in range(4)],
                             seat_updates=[123 if i == seat else None for i in range(4)], sampling=[deal, rotation, 2, 3], vp=vp, ranking=ranks))
    return rows


def test_whole_deal_pairing_and_native_strict_wins():
    from brass_rl.stability_study import lineup_summary, paired_difference
    a = rows('candidate', 3)
    b = rows('control', 0)
    summary = lineup_summary(b)
    assert (summary['mean_vp'], summary['median_vp'], summary['q25_vp'], summary['max_vp']) == (16.5, 16.5, 11.75, 23)
    assert summary['sole_wins'] == 2 and summary['mean_place'] == 1.75
    # Both whole blocks have the same four-rotation mean delta despite variable individual deltas.
    for i, row in enumerate(a):
        row['vp'][row['candidate_seat']] += (-6, -2, 2, 6)[i%4]
    result = paired_difference(a, list(reversed(b)))
    assert result['mean_vp'] == {'difference': 3, 'ci95': [3, 3]}
    assert result['independent_deals'] == 2 and result['draws'] == 10000
    b[3]['sampling'][0] += 1
    with pytest.raises(ValueError):
        paired_difference(a, b)
    with pytest.raises(ValueError):
        paired_difference(a[:-1], b[:-1])


def test_training_telemetry_weights_final_measurements_by_decision_counts():
    from brass_rl.stability_study import training_summary
    first = dict(actor_steps=2, critic_steps=3, kl_checks=2, monitor_policy_decision_visits=4,
                 actor_fragment_visits=3, critic_fragment_visits=4, actor_policy_visits=9, critic_value_visits=12,
                 monitor_policy_decisions=2, actor_four_game_steps=0, critic_four_game_steps=0,
                 kl_stopped=True, final_kl=.1, final_clip_fraction=.4, final_policy_decisions=10,
                 value_decisions=5, critic_mse_before=.8, critic_mse_after=.4, actor_stop_reason='minibatch_kl')
    second = dict(first, kl_stopped=False, final_kl=.3, final_clip_fraction=.2, final_policy_decisions=30,
                  critic_mse_after=.1, value_decisions=15, actor_stop_reason='epochs_complete')
    data = [dict(ppo=p, games=2, update=2+i, timings_seconds={'collection': 10, 'ppo': 5}) for i,p in enumerate((first, second))]
    report = training_summary(data)
    assert report['actor_steps'] == 4 and report['critic_steps'] == 6 and report['kl_stop_fraction'] == .5
    assert report['final_kl'] == pytest.approx(.25) and report['final_clip_fraction'] == pytest.approx(.25)
    assert report['critic_mse_after'] == pytest.approx(.175)
    assert report['final_policy_decisions'] == 40 and report['monitor_policy_decision_visits'] == 8


def test_source_payload_and_source_manifest_identity_are_both_checked(tmp_path):
    from brass_rl.stability_study import check_files
    data = tmp_path/'weights'
    data.write_bytes(b'original')
    manifest = tmp_path/'manifest.json'
    dump(manifest, {'content_hashes': {'weights': sha(data)}})
    expected = {'weights': sha(data), 'manifest.json': sha(manifest)}
    check_files(tmp_path, expected)
    data.write_bytes(b'altered!')
    with pytest.raises(ValueError):
        check_files(tmp_path, expected)
    data.write_bytes(b'original')
    dump(manifest, {})
    with pytest.raises(ValueError):
        check_files(tmp_path, expected)


def test_schedule_rotates_seats_and_matches_physical_sampling():
    from brass_rl.stability_study import schedule
    manifest = dict(master_seed=3915736456090994142, deal_seeds=[100, 200],
                    stream='final', lineups={'greedy':['candidate','greedy','greedy','greedy'],
                                              'fixed':['candidate',{'id':'a'},{'id':'b'},{'id':'c'}]})
    a = list(schedule(manifest))
    assert len(a) == 16 and [g['game_id'] for g in a] == list(range(16))
    assert [g['candidate_seat'] for g in a[:4]] == [0,3,2,1]
    assert a[1]['specs'] == [{'id':'a'},{'id':'b'},{'id':'c'},'candidate']
    assert a[4]['deal_seed'] == a[12]['deal_seed'] == 200
    assert all(len(set(g['sampling'])) == 4 for g in a)
    assert [g['sampling'] for g in a] == [g['sampling'] for g in schedule(dict(manifest,candidate_id='other'))]


def test_persisted_extension_survives_interrupt_and_refuses_lost_decision(tmp_path):
    from brass_rl.stability_study import Study
    out = tmp_path/'study'
    out.mkdir()
    protocol = dict(archive=str(tmp_path/'archive'), models=str(tmp_path/'models'),
                    shared_baseline_bytes={'archive':0,'models':0}, deadline_utc='2100-01-01T00:00:00+00:00')
    state = dict(committed_games={'control':128,'candidate':128}, extension_decision=None,
                 training_intervals={'control':[dict(games=8,seconds=160)],'candidate':[dict(games=8,seconds=450)]})
    runner = Study(out, protocol, state)
    assert runner.select_endpoint() == 256
    recorded = (out/'endpoint-decision.json').read_bytes()
    state['extension_decision'] = None  # after decision publication, before progress publication
    runner.deadline = 0
    assert runner.select_endpoint() == 256
    assert (out/'endpoint-decision.json').read_bytes() == recorded
    (out/'endpoint-decision.json').unlink()
    with pytest.raises(ValueError):
        runner.select_endpoint()


def test_score_free_endpoint_choice_and_no_restart_if_stop_pending_at_primary(tmp_path):
    from brass_rl.stability_study import Halt, Study
    out = tmp_path/'study'
    out.mkdir()
    protocol = dict(archive=str(tmp_path/'archive'), models=str(tmp_path/'models'),
                    shared_baseline_bytes={'archive':0,'models':0}, deadline_utc='2100-01-01T00:00:00+00:00')
    runner = Study(out, protocol, dict(committed_games={'control':128,'candidate':128},
                   training_intervals={'control':[],'candidate':[]}))
    (out/'STOP').touch()
    with pytest.raises(Halt, match='manual'):
        runner.train()
    assert not (out/'endpoint-decision.json').exists()


def test_partial_evaluation_reports_exact_coverage_after_restart(tmp_path):
    from brass_rl.stability_study import evaluation_coverage
    games = tmp_path/'evaluation/source/games'
    for i in range(6):
        dump(games/f'{i:08d}.json', dict(lineup='fixed', deal_id=i//4, rotation=i%4))
    counts = evaluation_coverage(tmp_path, ['greedy','fixed'])
    assert counts['source']['fixed'] == dict(games=6, complete_deals=[0], partial_rotations={'1':[0,1]})
    assert counts['control']['greedy'] == dict(games=0, complete_deals=[], partial_rotations={})


def test_reports_distinct_step_probe_and_full_epoch_denominators():
    from brass_rl.stability_study import training_summary
    ppo = {k:3 for k in ('actor_steps','critic_steps','actor_policy_visits','critic_value_visits',
                          'actor_fragment_visits','critic_fragment_visits','monitor_policy_decisions',
                          'actor_four_game_steps','critic_four_game_steps','value_decisions')}
    ppo.update(kl_checks=3, kl_stopped=True, monitor_policy_decision_visits=24, monitor_games=8, monitor_sequences=8,
               kl_check_interval_actor_steps=1, monitor_kl=.02, monitor_clip_fraction=.05, full_epoch_kl_checks=1,
               full_epoch_policy_decision_visits=30, last_full_epoch_kl=.04, final_kl=.04, final_clip_fraction=.1,
               final_policy_decisions=30, critic_mse_before=.1, critic_mse_after=.09, actor_stop_reason='full_epoch_kl')
    report = training_summary([dict(ppo=ppo, games=8, update=4080, timings_seconds={'collection':1,'ppo':2})])
    assert report['full_epoch_kl_checks'] == 1 and report['full_epoch_policy_decision_visits'] == 30
    check, = report['kl_measurements']
    assert (check['update'],check['step_probe_kl'],check['last_full_epoch_kl'],check['full_final_kl']) == (4080,.02,.04,.04)
    assert check['monitor_games'] == 8 and check['full_final_decisions'] == 30


def test_paired_report_refuses_different_physical_opponent():
    from brass_rl.stability_study import paired_difference
    a, b = rows('candidate', 0), rows('control', 0)
    for r in a+b:
        r['seat_ids'] = ['fixed']*4
        r['seat_updates'] = [688]*4
        r['seat_ids'][r['candidate_seat']] = r['candidate_id']
    b[-1]['seat_ids'][3] = 'different-opponent'
    with pytest.raises(ValueError):
        paired_difference(a,b)


def test_lost_arm_identity_cannot_be_adopted_during_training(tmp_path):
    from brass_rl.stability_study import committed
    project, out, pre = protocol_fixture(tmp_path)
    for arm in ('candidate','control'):
        dump(out/arm/'experiment.json', {'id':'exp-'+arm})
        dump(out/arm/'updates/004055/metrics.json', {'records':[]})
    state = json.loads((out/'progress.json').read_bytes())
    del state['experiment_ids']['candidate']
    with pytest.raises(ValueError):
        committed(out, state)


def test_endpoint_requires_each_intermediate_actor_copy_to_match_catalog(tmp_path):
    from brass_rl.stability_study import check_publications
    arm = tmp_path/'training/runs/study/control'
    entries = {}
    for update in (4056,4057):
        run_export = arm/'policies'/f'{update:06d}'
        playable = tmp_path/'players/models'/str(update)
        for path in (run_export,playable):
            dump(path/'metadata.json', {'id':str(update),'update':update})
            (path/'policy.pt').write_bytes(str(update).encode())
            dump(path/'manifest.json', {'content_hashes':{k:sha(path/k) for k in ('policy.pt','metadata.json')}})
        entries[str(update)] = dict(path=playable.relative_to(tmp_path).as_posix(), lineage='experimental',
                                    source_export=run_export.relative_to(tmp_path).as_posix(),
                                    files={k:sha(playable/k) for k in ('policy.pt','manifest.json','metadata.json')})
    assert check_publications(tmp_path,arm,[4056,4057],entries) == 2
    (tmp_path/'players/models/4056/policy.pt').write_bytes(b'other')
    with pytest.raises(ValueError):
        check_publications(tmp_path,arm,[4056,4057],entries)
