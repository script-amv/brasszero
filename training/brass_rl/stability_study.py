# SPDX-License-Identifier: Apache-2.0
"""Bounded, pre-registered local PPO comparison. Status and stop use only stdlib."""
import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
from statistics import mean, median
import subprocess
import tempfile
import time

from .bounded import GIB, MIB, disk_bytes
from .experiments import sync_dir, writer_lock
from .learning_study import fixed, parent_identity, read, write


NAME = 'ppo-stability-20261122'
ARMS = ('control', 'candidate')
MODELS = ('source',) + ARMS
BASE, FIRST, PRIMARY, EXTENDED = 4055, 8110, 128, 256
TIME_RESERVE, DISK_RESERVE = 90*60, 64*MIB
REG_SHA = '9991c55244f74bd97dbae56cf19b75580a87c228e67d554674750adf131c400e'
SOURCE_SHA = '0edc774d71641b2c22c362e865c0801aff98cc2579fd3a3d01d9752c12adae1e'
SOURCE_ID = 'policy-sha256-v1:fb172a79421038fc6ca5d829ef5c0e315531c140a1523d1508ea5f8dedfde0c7'
OPPONENTS = {
    688: 'policy-sha256-v1:e0059c09239366d9ab33b1d1e461d19b22ca5651d70c19fa2ee3454e0f90ae0c',
    738: 'policy-sha256-v1:223cf6014af1a1e800b914b373f7ed555be3a870525a66879487c9a2f25c4201',
    793: 'policy-sha256-v1:fc0cd4367c80203134660cdb3397e282370745658b1fbea4f3f1ff4a6ab5bf49',
}
EVIDENCE = {'qa-guard-addendum.json': '4cecd1e190f84a159c861a38a3cb2e5840b6f20967932218fc3c5b19a80d5bb7',
            'source-preflight.json': '881d2032f82161d8b58f7abdc5e35381050025a8be7f43ad851dd78df14d564a',
            'extra-freshness.json': '7da097d860434cd2b67847c3d63e56aaf8f929807c1db2a549db598e7564ca61'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def next_work(counts, endpoint):
    """Numeric commits can split a control block at two-game boundaries."""
    c, t = (counts[name] for name in ARMS)
    if not (0 <= t <= c <= endpoint and t % 8 == 0 and c % 2 == 0 and c-t <= 8):
        raise ValueError('committed game counts violate paired block schedule')
    if t == endpoint:
        return None
    return ('control', t+8-c) if c < t+8 else ('candidate', 8)


def bound(remaining, needed, used, free, growth=0):
    if remaining <= needed:
        return 'time'
    if used + growth + DISK_RESERVE >= 4*GIB:
        return 'disk'
    if free - growth - DISK_RESERVE <= 2*GIB:
        return 'free_disk'
    return None


def extension(remaining, pair_seconds, pair_bytes, used, free):
    reason = bound(remaining, 16*pair_seconds+TIME_RESERVE, used, free, 16*pair_bytes)
    return PRIMARY if reason else EXTENDED


def read_control(out, project, pre):
    protocol = read(out/'protocol.json')
    state = read(out/'progress.json')
    if (state['protocol_sha256'] != sha(out/'protocol.json') or protocol['out'] != str(out)
            or protocol['project'] != str(project) or protocol['preregistration'] != str(pre)
            or protocol['preregistration_sha256'] != sha(pre)
            or protocol['deadline_utc'] != read(pre)['task_deadline_utc']):
        raise ValueError('study identity, deadline or registration changed')
    return protocol, state


def committed(out, state):
    counts = {}
    for arm in ARMS:
        run = out/arm
        if arm not in state['experiment_ids'] and state['phase'] != 'initializing':
            raise ValueError('lost registered arm identity: '+arm)
        if not run.exists():
            if arm in state['experiment_ids'] or state['committed_games'][arm]:
                raise ValueError('missing previously committed arm: '+arm)
            counts[arm] = 0
            continue
        exp = read(run/'experiment.json')['id']
        if state['experiment_ids'].get(arm, exp) != exp:
            raise ValueError('arm identity changed: '+arm)
        path = max((p for p in (run/'updates').iterdir() if p.is_dir() and p.name.isdecimal()),
                   key=lambda p: int(p.name))
        if arm not in state['experiment_ids'] and int(path.name) != BASE:
            raise ValueError('unregistered arm already contains training games')
        if int(path.name) == BASE:
            counts[arm] = 0
        else:
            manifest = read(path/'manifest.json')
            if manifest['update'] != int(path.name) or sha(path/'metrics.json') != manifest['content_hashes']['metrics.json']:
                raise ValueError('committed boundary metadata mismatch')
            row, = read(path/'metrics.json')['records']
            count = row['next_game_id']-FIRST
            step = 2 if arm == 'control' else 8
            if (row['update'] != manifest['update'] or row['games'] != step
                    or count != (row['update']-BASE)*step or row['first_game_id']+step != row['next_game_id']):
                raise ValueError('unexpected committed game range')
            counts[arm] = count
        if counts[arm] < state['committed_games'][arm]:
            raise ValueError('arm rolled back behind observed commit')
    next_work(counts, (state.get('extension_decision') or {}).get('endpoint', PRIMARY))
    return counts


def status(out, project, pre):
    protocol, state = read_control(out, project, pre)
    state['committed_games'] = committed(out, state)
    if state['state'] == 'running':
        try:
            os.kill(state['pid'], 0)
            # An unrelated process can inherit an old PID, but cannot inherit the study lock.
            with writer_lock(out):
                state['state'] = 'interrupted'
        except ProcessLookupError:
            state['state'] = 'interrupted'
        except ValueError:
            pass  # Live writer owns the reservation.
    if 'registration' in protocol:
        state['evaluation_coverage'] = evaluation_coverage(out, protocol['registration']['evaluation_lineups'])
    if state['state'] == 'complete' and not (out/'report.json').is_file():
        raise ValueError('missing completed scientific report')
    return dict(state, deadline_utc=protocol['deadline_utc'], stop_requested=(out/'STOP').exists())


def quantile(values, q):
    values = sorted(values)
    x = (len(values)-1)*q
    lo = int(x)
    return values[lo] + (x-lo)*(values[math.ceil(x)]-values[lo])


def measures(row):
    seat = row['candidate_seat']
    rank = row['ranking'][seat]
    return row['vp'][seat], int(rank == 0 and row['ranking'].count(0) == 1), rank+1


def lineup_summary(rows):
    vp, wins, places = zip(*(measures(r) for r in rows))
    return dict(games=len(rows), independent_deals=len({r['deal_seed'] for r in rows}),
                mean_vp=mean(vp), median_vp=median(vp), q25_vp=quantile(vp, .25), max_vp=max(vp),
                sole_wins=sum(wins), sole_win_rate=mean(wins), mean_place=mean(places))


def paired_difference(left, right):
    def indexed(rows):
        pairs = {(r['deal_id'], r['rotation']): r for r in rows}
        if len(pairs) != len(rows):
            raise ValueError('duplicate evaluation slot')
        return pairs
    a, b = indexed(left), indexed(right)
    deals = sorted({d for d, _ in a})
    if not deals or a.keys() != b.keys() or set(a) != {(d,r) for d in deals for r in range(4)}:
        raise ValueError('paired bootstrap requires complete four-rotation blocks')
    blocks = []
    for d in deals:
        diffs = []
        for r in range(4):
            x, y = a[d,r], b[d,r]
            if (any(x[k] != y[k] for k in ('deal_seed', 'lineup', 'sampling', 'candidate_seat'))
                    or x['candidate_seat'] != (-r)%4 or x['deal_seed'] != a[d,0]['deal_seed']):
                raise ValueError('paired games differ in deal, seat, lineup or sampling')
            for key in ('seat_ids','seat_updates'):
                if any(x[key][seat] != y[key][seat] for seat in range(4) if seat != x['candidate_seat']):
                    raise ValueError('paired physical opponent identity or update differs')
            diffs.append([v-w for v,w in zip(measures(x), measures(y))])
        blocks.append(tuple(map(mean, zip(*diffs))))
    if len({a[d,0]['deal_seed'] for d in deals}) != len(deals):
        raise ValueError('repeated independent deal')
    rng = random.Random(20261122)
    columns = list(zip(*blocks))
    boot = [[], [], []]
    for _ in range(10000):
        indices = rng.choices(range(len(blocks)), k=len(blocks))
        for samples, column in zip(boot, columns):
            samples.append(sum(column[i] for i in indices)/len(indices))
    result = dict(independent_deals=len(deals), draws=10000, seed=20261122)
    for key, column, samples in zip(('mean_vp','sole_win_rate','mean_place'), columns, boot):
        result[key] = dict(difference=mean(column), ci95=[quantile(samples,.025), quantile(samples,.975)])
    return result


def evaluation_coverage(out, lineups):
    coverage = {}
    for name in MODELS:
        slots = {lineup:{} for lineup in lineups}
        for f in (out/'evaluation'/name/'games').glob('*.json'):
            r = read(f)
            rotations = slots[r['lineup']].setdefault(r['deal_id'], set())
            if r['rotation'] not in range(4) or r['rotation'] in rotations or r['deal_id'] not in range(16):
                raise ValueError('invalid recorded evaluation slot')
            rotations.add(r['rotation'])
        coverage[name] = {lineup:dict(games=sum(map(len,deals.values())),
                               complete_deals=sorted(d for d,rs in deals.items() if len(rs) == 4),
                               partial_rotations={str(d):sorted(rs) for d,rs in sorted(deals.items()) if len(rs)<4})
                          for lineup,deals in slots.items()}
    return coverage


def training_summary(rows):
    if not rows:
        return dict(updates=0, games=0)
    ppo = [r['ppo'] for r in rows]
    totals = {k: sum(p[k] for p in ppo) for k in (
        'actor_steps', 'critic_steps', 'kl_checks', 'monitor_policy_decision_visits', 'actor_fragment_visits',
        'critic_fragment_visits', 'actor_policy_visits', 'critic_value_visits', 'actor_four_game_steps',
        'critic_four_game_steps', 'final_policy_decisions', 'value_decisions')}
    for key, weight in [('final_kl','final_policy_decisions'), ('final_clip_fraction','final_policy_decisions'),
                        ('critic_mse_before','value_decisions'), ('critic_mse_after','value_decisions')]:
        totals[key] = sum(p[key]*p[weight] for p in ppo)/totals[weight] if totals[weight] else None
    for key in ('full_epoch_kl_checks', 'full_epoch_policy_decision_visits'):
        totals[key] = sum(p.get(key, 0) for p in ppo)
    totals['kl_measurements'] = [dict(update=row['update'], step_probe_kl=p.get('monitor_kl'),
        step_probe_clip_fraction=p.get('monitor_clip_fraction'), step_probe_decisions=p['monitor_policy_decisions'],
        step_probe_interval=p.get('kl_check_interval_actor_steps'), monitor_games=p.get('monitor_games'),
        monitor_sequences=p.get('monitor_sequences'), full_epoch_checks=p.get('full_epoch_kl_checks', 0),
        last_full_epoch_kl=p['last_full_epoch_kl'] if p.get('full_epoch_kl_checks') else None,
        full_final_kl=p['final_kl'], full_final_clip_fraction=p['final_clip_fraction'],
        full_final_decisions=p['final_policy_decisions']) for row,p in zip(rows,ppo)]
    times = {k:sum(r['timings_seconds'][k] for r in rows) for k in ('collection','ppo')}
    games = sum(r['games'] for r in rows)
    totals.update(updates=len(rows), games=games, kl_stop_fraction=mean(int(p['kl_stopped']) for p in ppo),
                  actor_stop_reasons=dict(Counter(p['actor_stop_reason'] for p in ppo)), timings_seconds=times,
                  games_per_compute_hour=games*3600/sum(times.values()) if sum(times.values()) else None)
    return totals


def check_files(directory, hashes):
    for name, expected in hashes.items():
        if sha(directory/name) != expected:
            raise ValueError('sealed bytes changed: '+str(directory/name))



def catalog_entries(inventory):
    entries = {}
    for section in ('models','other_preserved_snapshots'):
        for row in inventory[section]:
            if row['policy_id'] in entries:
                raise ValueError('duplicate playable policy ID in catalog')
            entries[row['policy_id']] = (section,row)
    return entries


def publication_identity(section, entry):
    # Keep the pre-existing classification and original publication provenance.
    return dict(section=section, **{k:entry.get(k) for k in
                ('lineage','role','branch','source_export','provenance','path','files')})


def check_publications(project, arm, updates, catalog, initial, experiments, *, partial=False):
    count, issues = 0, []
    for number in updates:
        try:
            export = arm/'policies'/('%06d'%number)
            meta = read(export/'metadata.json')
            identity = meta['id']
            section, entry = catalog[identity]
            files = entry['files']
            if meta['update'] != number or entry['update'] != number or set(files) != {'metadata.json','policy.pt','manifest.json'}:
                raise ValueError('invalid catalog metadata: '+str(export))
            if identity in initial:
                if publication_identity(section,entry) != initial[identity]:
                    raise ValueError('pre-existing policy publication changed')
            else:
                # Two arms can publish the same actor. Verify the original
                # study arm's source export and experiment in that case.
                branch = entry['branch']
                if section != 'other_preserved_snapshots' or branch not in experiments:
                    raise ValueError('new actor missing from experimental catalog section')
                owner = arm.parent/branch
                original = owner/'policies'/export.name
                if (entry['lineage'] != 'experimental' or entry['role'] != branch
                        or entry['source_export'] != original.relative_to(project).as_posix()
                        or entry['provenance']['run'] != owner.relative_to(project).as_posix()
                        or entry['provenance']['experiment_id'] != experiments[branch]):
                    raise ValueError('study actor catalog provenance mismatch')
                if original != export:
                    check_files(original,files)
            check_files(export,files)
            check_files(project/entry['path'],files)
            count += 1
        except (OSError, KeyError, ValueError) as error:
            if not partial:
                raise
            issues.append(dict(update=number,error=str(error)))
    return count,issues


def portable(path, identity=None):
    from .policy_io import inspect_policy
    meta = inspect_policy(path)
    if identity and meta.id != identity:
        raise ValueError('portable policy identity mismatch: '+str(path))
    return dict(path=str(path), id=meta.id, update=meta.update, files=dict(meta.content_hashes))


def suite(protocol, candidate_id):
    from .evaluate import EvalSuite
    peers = [{'id': p['id'], 'path': p['path']} for p in protocol['opponents']]
    return EvalSuite('final', protocol['registration']['evaluation_master_seed'], 16,
                     {'greedy':['candidate']+['greedy']*3, 'fixed688-738-793':['candidate']+peers},
                     candidate_id, replay_limit=0, outcomes_only=True).manifest()


def schedule(manifest):
    from .seeding import SeedStreams
    streams = SeedStreams(manifest['master_seed'])
    for index, (name, entries) in enumerate(sorted(manifest['lineups'].items())):
        for deal, seed in enumerate(manifest['deal_seeds']):
            for rotation in range(4):
                specs = entries[rotation:]+entries[:rotation]
                yield dict(game_id=(index*len(manifest['deal_seeds'])+deal)*4+rotation, deal_id=deal, deal_seed=seed,
                           lineup=name, rotation=rotation, specs=specs, candidate_seat=specs.index('candidate'),
                           sampling=[streams.seed('final.policy', index, deal, rotation, s) for s in range(4)])


def revision():
    root = Path(__file__).resolve().parents[2]
    return subprocess.check_output(['git', '-C', str(root), 'rev-parse', '--verify', 'HEAD'], text=True).strip()


def declare(out, project, pre):
    from .artifacts import load_update
    from .stability_run import PROTOCOL
    from .seeding import SeedStreams
    if sha(pre) != REG_SHA:
        raise ValueError('unrecognized pre-registration')
    check_files(pre.parent, EVIDENCE)
    reg = read(pre)
    source = project/'training/runs/vast-legacy1919-20261122/updates/004055'
    parent = parent_identity(source)
    if parent['checkpoint_sha256'] != SOURCE_SHA or parent['model_id'] != SOURCE_ID:
        raise ValueError('source 4055 identity mismatch')
    source_files = dict(read(source/'manifest.json')['content_hashes'], **{'manifest.json': SOURCE_SHA})
    check_files(source, source_files)
    checkpoint = load_update(source).state
    if (checkpoint.master_seed, checkpoint.next_game_id, checkpoint.completed_updates) != (reg['training_master'], FIRST, BASE):
        raise ValueError('source checkpoint counters or master mismatch')
    config = json.loads(json.dumps(asdict(checkpoint.config)))
    streams = SeedStreams(reg['training_master'])
    if reg['training_deals'] != [streams.seed('train.deal', i) for i in range(FIRST, FIRST+EXTENDED)]:
        raise ValueError('planned training deal stream mismatch')
    catalog = read(project/'players/models/catalog.json')
    known = {row['policy_id']: row for section in ('models', 'other_preserved_snapshots') for row in catalog[section]}
    peers = []
    for update, identity in OPPONENTS.items():
        p = portable(project/known[identity]['path'], identity)
        if p['update'] != update:
            raise ValueError('opponent update mismatch')
        peers.append(p)
    source_actor = portable(source.parents[1]/'policies'/source.name, SOURCE_ID)
    if source_actor['files']['policy.pt'] != parent['policy_sha256']:
        raise ValueError('source actor differs from the checkpoint')
    protocol = dict(format='brass-stability-study/1', stability_protocol=PROTOCOL, project=str(project), out=str(out), preregistration=str(pre),
                    preregistration_sha256=REG_SHA, evidence_sha256=EVIDENCE, registration=reg,
                    deadline_utc=reg['task_deadline_utc'], base_update=BASE, first_game_id=FIRST,
                    source=str(source), source_files=source_files, checkpoint_config=config, parent=parent, source_actor=source_actor,
                    opponents=peers, revision=revision(), minimum_free_bytes=2*GIB, evaluation_reserve_seconds=TIME_RESERVE,
                    atomic_write_reserve_bytes=DISK_RESERVE, bootstrap_draws=10000, bootstrap_seed=20261122,
                    archive=str(project/'replays'), models=str(project/'players/models'))
    manifest = suite(protocol, SOURCE_ID)
    if manifest['deal_seeds'] != reg['evaluation_deal_seeds']:
        raise ValueError('final suite differs from registered deals')
    protocol['initial_publications'] = {identity:publication_identity(section,row)
        for identity,(section,row) in catalog_entries(catalog).items() if BASE < row['update'] <= BASE+EXTENDED//2}
    protocol['packet'] = str(pre.parent)
    protocol['source_suite'] = manifest
    protocol['shared_baseline_bytes'] = {k:disk_bytes(Path(protocol[k])) for k in ('archive','models')}
    return protocol


def check_inputs(protocol):
    from .stability_run import PROTOCOL
    if (protocol['revision'] != revision() or protocol['preregistration_sha256'] != REG_SHA
            or protocol['stability_protocol'] != PROTOCOL):
        raise ValueError('resume requires the sealed source revision and registration')
    check_files(Path(protocol['preregistration']).parent, EVIDENCE)
    source = Path(protocol['source'])
    check_files(source, protocol['source_files'])
    if parent_identity(source) != protocol['parent'] or protocol['parent']['checkpoint_sha256'] != SOURCE_SHA:
        raise ValueError('source parent changed')
    for policy in [protocol['source_actor'], *protocol['opponents']]:
        check_files(Path(policy['path']), policy['files'])
    if suite(protocol, SOURCE_ID) != protocol['source_suite']:
        raise ValueError('final evaluator fingerprint or suite changed')


def validate_arm(out, protocol, name):
    from .experiments import read_experiment
    run = out/name
    meta = read_experiment(run)
    cfg = dict(protocol['checkpoint_config'])
    cfg['games_per_update'] = 8 if name == 'candidate' else 2
    if (read(run/'config.json') != cfg or cfg['master_seed'] != protocol['registration']['training_master']
            or meta['parent'] != protocol['parent'] or meta['stability_arm'] != name
            or meta['initial_next_game_id'] != FIRST or meta['base_update'] != BASE
            or meta['stability_protocol'] != protocol['stability_protocol'] or not meta['initial_state_verified']
            or not all(meta['initial_state_verified'].values())):
        raise ValueError('arm identity, source, or training configuration mismatch')
    return meta['id']


def audit_training(out, protocol, state, arm, archive):
    """Advance only through native-verified, committed replay ranges."""
    from brass_ai.replay_archive import engine_id
    from .artifacts import committed_records
    from .transfer import validate_replay
    run = out/arm
    expected = 2 if arm == 'control' else 8
    journal = committed_records(run)
    if [j['update'] for j in journal] != list(range(BASE, BASE+len(journal))):
        raise ValueError('incomplete arm journal')
    previous = state['verified_games'][arm]
    if previous > (len(journal)-1)*expected:
        raise ValueError('verified training history lost')
    for entry in journal[1+previous//expected:]:
        number = entry['update']
        manifest = entry['manifests']
        games = manifest['train']
        first = FIRST+(number-BASE-1)*expected
        if (manifest['experiment_id'] != state['experiment_ids'][arm] or manifest['stability_arm'] != arm
                or [g['game_id'] for g in games] != list(range(first, first+expected))
                or len(manifest['replay_ids']) != expected or len(set(manifest['replay_ids'])) != expected):
            raise ValueError('incomplete training game or replay range')
        by_game = {g['game_id']:g for g in games}
        for replay_id in manifest['replay_ids']:
            rec = archive.read(replay_id)
            origin = rec['provenance']
            g = by_game.pop(origin['game'])
            seed = protocol['registration']['training_deals'][g['game_id']-FIRST]
            if (rec['status'] != 'completed' or rec['category'] != 'training' or rec['game_id'] != replay_id
                    or rec['engine_id'] != engine_id() or origin['experiment'] != manifest['experiment_id']
                    or origin['update'] != number
                    or rec['initial'] != {'seed':seed} or g['deal_seed'] != seed or len(rec['moves']) != g['plies']
                    or any(rec['result'][key] != g['outcome'][key] for key in ('vp','ranking'))):
                raise ValueError('training replay provenance or outcome mismatch')
            validate_replay(rec)
        state['verified_games'][arm] += expected
    return [r for entry in journal[1:] for r in entry['metrics']]


def evaluation_slot(directory, slot, manifest, endpoint, opponents):
    from brass_ai.replay_archive import labeled_seats
    peers = {p['id']:p for p in opponents}
    seats = [endpoint if spec == 'candidate' else None if spec == 'greedy' else peers[spec['id']]
             for spec in slot['specs']]
    labels = [p['id'] if p else 'greedy' for p in seats]
    ids = [seat['bot'] for seat in labeled_seats(labels)]
    updates = [p['update'] if p else None for p in seats]
    lineage = dict(slot, suite_id=manifest['suite_id'], candidate_id=endpoint['id'], update=endpoint['update'],
                   checkpoint_sha256=endpoint['files']['policy.pt'], labels=labels, seat_ids=ids, seat_updates=updates)
    del lineage['specs']
    origin = dict(category='evaluation', outcomes_only=True, mode='sample', experiment=manifest['suite_id'],
                  run_experiment=endpoint['experiment_id'], experiment_path=str(directory), game=slot['game_id'],
                  deal=slot['deal_id'], rotation=slot['rotation'], lineup=slot['lineup'], update=endpoint['update'],
                  candidate_id=endpoint['id'], seat_ids=ids, seat_updates=updates)
    return directory/'games'/('%08d.json' % slot['game_id']), lineage, origin


def recover_evaluation(out, frozen, opponents, archive, can_verify, *, expected_suites=None):
    """Reconcile completed archives before reporting a stop. Never play a game."""
    from .combination_evaluation import outcome_id, outcome_row
    def missing():
        raise ValueError('completed evaluation archive disappeared during recovery')
    pending = []
    for name, endpoint in frozen.items():
        directory = out/'evaluation'/name
        if not directory.exists():
            continue
        manifest = read(directory/'suite.json')
        if expected_suites is not None and manifest != expected_suites[name]:
            raise ValueError('stored suite differs from frozen evaluation')
        for slot in schedule(manifest):
            path = directory/'games'/('%08d.json' % slot['game_id'])
            if path.exists():
                continue
            path, lineage, origin = evaluation_slot(directory,slot,manifest,endpoint,opponents)
            identity = outcome_id(lineage,origin)
            if archive.path(identity).is_file():
                if can_verify():
                    outcome_row(archive,path,lineage,origin,(),slot['sampling'],missing)
                else:
                    pending.append(dict(model=name,lineup=slot['lineup'],deal_id=slot['deal_id'],
                                        rotation=slot['rotation'],game_id=slot['game_id'],replay_id=identity))
    return pending


def evaluate_endpoint(out, name, endpoint, protocol, archive, before, observed, completed):
    from .combination_evaluation import outcome_row
    from .policy_io import load_policy
    directory = out/'evaluation'/name
    manifest = suite(protocol, endpoint['id'])
    if manifest['deal_seeds'] != protocol['registration']['evaluation_deal_seeds']:
        raise ValueError('endpoint suite deal mismatch')
    if directory.exists() and not (directory/'suite.json').is_file():
        raise ValueError('existing evaluation missing sealed suite')
    fixed(directory/'suite.json', manifest)
    catalog = {}
    for p in [endpoint, *protocol['opponents']]:
        policy = load_policy(Path(p['path']), portable_only=True)
        meta = policy[2]
        if meta.id != p['id'] or dict(meta.content_hashes) != p['files']:
            raise ValueError('frozen evaluation policy changed')
        catalog[p['id']] = policy
    results = []
    for slot in schedule(manifest):
        path, lineage, origin = evaluation_slot(directory,slot,manifest,endpoint,protocol['opponents'])
        policies = [None if label == 'greedy' else catalog[label] for label in lineage['labels']]
        observed(slot)
        row = outcome_row(archive, path, lineage, origin, policies, slot['sampling'], before)
        results.append(row)
        completed(row)
    return results


class Halt(Exception):
    """A durable graceful stop has reached a game or complete update boundary."""


def storage(out, protocol):
    shared = {key: max(0, disk_bytes(Path(protocol[key]))-base)
              for key, base in protocol['shared_baseline_bytes'].items()}
    used = disk_bytes(out)
    packet = disk_bytes(Path(protocol['packet'])) if 'packet' in protocol else 0
    return dict(study_tree_bytes=used, packet_bytes=packet, shared_growth_bytes=shared, used=used+packet+sum(shared.values()),
                free=shutil.disk_usage(out).free)


class Study:
    def __init__(self, out, protocol, state, operator_stop=None):
        from brass_ai.replay_archive import Archive
        self.out, self.protocol, self.state = out, protocol, state
        self.deadline = datetime.fromisoformat(protocol['deadline_utc']).timestamp()
        self.archive = Archive(Path(protocol['archive']))
        self.operator_stop = operator_stop
        self.signal = False
        self.rows = {arm: [] for arm in ARMS}

    def save(self, **updates):
        self.state.update(updated_epoch=time.time(), **updates)
        write(self.out/'progress.json', self.state)

    def guard(self, needed=0, growth=0):
        if self.signal or (self.out/'STOP').exists() or (self.operator_stop and self.operator_stop.exists()):
            raise Halt('manual')
        usage = storage(self.out, self.protocol)
        self.state['disk'] = usage
        reason = bound(self.deadline-time.time(), needed, usage['used'], usage['free'], growth)
        if reason:
            raise Halt(reason)

    def observe_training(self):
        counts = committed(self.out, self.state)
        for arm in ARMS:
            if arm in self.state['experiment_ids']:
                self.rows[arm] = audit_training(self.out, self.protocol, self.state, arm, self.archive)
                if self.state['verified_games'][arm] != counts[arm]:
                    raise ValueError('training audit did not cover committed range')
        self.save(committed_games=counts)

    def prepare(self, resume):
        from .stability_run import fork_stability, train_stability
        identities = self.state['experiment_ids']
        for arm in ARMS:
            self.guard(60)
            run = self.out/arm
            if not run.exists():
                if arm in identities or self.state['phase'] != 'initializing':
                    raise ValueError('previous arm missing; refusing to start over')
                fork_stability(Path(self.protocol['source']), run, arm == 'candidate')
            identity = validate_arm(self.out, self.protocol, arm)
            if (identities.get(arm, identity) != identity or identity == self.protocol['parent']['experiment_id']
                    or identity in [v for k,v in identities.items() if k != arm]):
                raise ValueError('stability arms require their original independent experiment IDs')
            identities[arm] = identity
            self.save()
            if resume:
                # A commit can precede latest-pointer, export, retention or catalog publication.
                train_stability(run, 0)
        self.observe_training()

    def rates(self):
        # Include whole call time (publication and validation too). If killed before a
        # timing sample was saved, committed collection/PPO durations remain available.
        rates = {}
        for arm, fallback in [('control',30), ('candidate',120)]:
            work = self.state['training_intervals'][arm]
            rows = self.rows[arm]
            observed = [i['seconds']/i['games'] for i in work if i['games']]
            observed.extend(sum(r['timings_seconds'].values())/r['games'] for r in rows)
            rates[arm] = 1.25*max(observed, default=fallback)
        return rates

    def select_endpoint(self):
        decision_file = self.out/'endpoint-decision.json'
        decision = self.state.get('extension_decision')
        if decision_file.exists():
            sealed = read(decision_file)
            if decision is not None and decision != sealed:
                raise ValueError('stored endpoint decisions disagree')
            if extension(**sealed['resources']) != sealed['endpoint']:
                raise ValueError('sealed endpoint violates its resource rule')
            decision = sealed
        elif decision is not None:
            raise ValueError('missing immutable endpoint decision')
        elif self.state['committed_games'] == dict.fromkeys(ARMS, PRIMARY):
            disk = storage(self.out, self.protocol)
            resources = dict(remaining=self.deadline-time.time(), pair_seconds=8*sum(self.rates().values()),
                             pair_bytes=max(8*MIB, 1.25*disk['used']/16), used=disk['used'], free=disk['free'])
            decision = dict(endpoint=extension(**resources), resources=resources, epoch=time.time())
            fixed(decision_file, decision)
        if decision and min(self.state['committed_games'].values()) < PRIMARY:
            raise ValueError('extension chosen before both primary endpoints')
        if decision and self.state.get('extension_decision') != decision:
            self.save(extension_decision=decision)
        return decision['endpoint'] if decision else PRIMARY

    def train(self):
        while True:
            self.guard()
            target = self.select_endpoint()
            counts = self.state['committed_games']
            work = next_work(counts, target)
            if work is None:
                return target
            arm, additional = work
            rates = self.rates()
            pair_cost = sum((counts['candidate']+8-counts[k])*rates[k] for k in ARMS)
            self.save(phase='training', active_arm=arm, block_target=counts['candidate']+8)
            disk = self.state['disk']
            blocks = max(1, counts['candidate']//8)
            pair_growth = max(32*MIB, 1.25*disk['used']/blocks)
            self.guard(TIME_RESERVE+pair_cost, pair_growth)
            reason = None

            def stop_at_boundary():
                nonlocal reason
                try:
                    self.guard(TIME_RESERVE+rates[arm]*(2 if arm == 'control' else 8))
                except Halt as exc:
                    reason = str(exc)
                return reason is not None

            from .stability_run import train_stability
            start = time.monotonic()
            report = train_stability(self.out/arm, additional, stop_requested=stop_at_boundary)
            done = report.next_game_id-FIRST-counts[arm]
            seconds = time.monotonic()-start
            if done:
                self.state['training_intervals'][arm].append(dict(games=done, seconds=seconds))
            self.observe_training()
            if reason or done < additional:
                raise Halt(reason or report.stop_reason)
            if done != additional:
                raise ValueError('trainer exceeded planned game range')

    def publications(self, *, partial):
        p = self.protocol
        project = Path(p['project'])
        try:
            catalog = catalog_entries(read(project/'players/models/catalog.json'))
        except (OSError, KeyError, ValueError) as error:
            if not partial:
                raise
            issues = {arm:[dict(update=r['update'],error=str(error)) for r in self.rows[arm]] for arm in ARMS}
            self.save(verified_actor_publications=dict.fromkeys(ARMS,0), actor_publication_issues=issues)
            return
        verified, issues = {}, {}
        for arm in ARMS:
            verified[arm], issues[arm] = check_publications(project,self.out/arm,
                [r['update'] for r in self.rows[arm]],catalog,p['initial_publications'],self.state['experiment_ids'],
                partial=partial)
        self.save(verified_actor_publications=verified, actor_publication_issues=issues)

    def endpoints(self, endpoint):
        if self.state['committed_games'] != dict.fromkeys(ARMS, endpoint):
            raise ValueError('equal frozen endpoints required')
        identities = self.state['experiment_ids']
        self.publications(partial=False)
        frozen = {'source': dict(self.protocol['source_actor'], experiment_id=self.protocol['parent']['experiment_id'])}
        for arm in ARMS:
            number = self.rows[arm][-1]['update']
            run = self.out/arm
            identity = read(run/'updates'/('%06d'%number)/'metadata.json')['id']
            frozen[arm] = dict(portable(run/'policies'/('%06d'%number), identity), experiment_id=identities[arm])
        path = self.out/'frozen-policies.json'
        if (self.out/'evaluation').exists() and not path.exists():
            raise ValueError('missing endpoint declaration for existing evaluation')
        fixed(path, frozen)
        return frozen

    def evaluate(self, frozen):
        results = {}
        for name in MODELS:
            self.save(phase='evaluation', active_arm=name)
            def observed(slot):
                self.guard()
                self.save(evaluation_slot={k:slot[k] for k in ('game_id','lineup','deal_id','rotation')})
            def before():
                self.guard(max(30., 1.25*self.state.get('slowest_eval_seconds', 0)))
            def completed(row):
                if row['seconds'] is not None:
                    self.state['slowest_eval_seconds'] = max(row['seconds'], self.state.get('slowest_eval_seconds', 0))
            results[name] = evaluate_endpoint(self.out, name, frozen[name], self.protocol, self.archive,
                                              before, observed, completed)
        stats, comparisons = {}, {}
        for lineup in self.protocol['registration']['evaluation_lineups']:
            groups = {name:[r for r in rows if r['lineup'] == lineup] for name,rows in results.items()}
            if any(len(rows) != 64 for rows in groups.values()):
                raise ValueError('final lineup lacks all 16 deals and four seats')
            stats[lineup] = {name:lineup_summary(rows) for name,rows in groups.items()}
            comparisons[lineup] = {a+'-'+b:paired_difference(groups[a],groups[b]) for a,b in (
                ('candidate','control'),('candidate','source'),('control','source'))}
        for p in frozen.values():
            check_files(Path(p['path']), p['files'])
        return dict(lineups=stats, paired_differences=comparisons)

    def finish(self, final, reason=None):
        p, s = self.protocol, self.state
        check_inputs(p)
        if final is None:
            self.publications(partial=True)
        disk = storage(self.out, p)
        frozen_file = self.out/'frozen-policies.json'
        frozen = read(frozen_file) if frozen_file.exists() else None
        for policy in (frozen or {}).values():
            check_files(Path(policy['path']), policy['files'])
        pending = []
        if final is None and frozen:
            def can_verify():
                available = storage(self.out,p)
                return bound(self.deadline-time.time(),5,available['used'],available['free']) is None
            expected = {k:suite(p,v['id']) for k,v in frozen.items()}
            pending = recover_evaluation(self.out,frozen,p['opponents'],self.archive,can_verify,expected_suites=expected)
        coverage = evaluation_coverage(self.out, p['registration']['evaluation_lineups'])
        evaluation_timings = {}
        for name in MODELS:
            durations = [read(path)['seconds'] for path in (self.out/'evaluation'/name/'games').glob('*.json')]
            observed = [duration for duration in durations if duration is not None]
            awaiting = sum(item['model'] == name for item in pending)
            evaluation_timings[name] = dict(known_games=len(observed), unknown_games=len(durations)-len(observed)+awaiting,
                                            recorded_game_seconds=sum(observed))
        observed_deals = {int(d) for by_lineup in coverage.values() for group in by_lineup.values()
                          for d in [*group['complete_deals'], *group['partial_rotations']]}
        all_deals = observed_deals | {row['deal_id'] for row in pending}
        report = dict(complete=final is not None, stop_reason=reason, phase=s['phase'], active_arm=s.get('active_arm'),
                      revision=p['revision'], protocol_sha256=s['protocol_sha256'], preregistration_sha256=REG_SHA,
                      source=p['parent'], frozen_policies=frozen, stability_protocol=p['stability_protocol'],
                      evaluation_timing=evaluation_timings,
                      base_game=FIRST, committed_games=s['committed_games'], native_verified_training=s['verified_games'],
                      training={arm:training_summary(self.rows[arm]) for arm in ARMS},
                      training_intervals=s['training_intervals'], extension_decision=s.get('extension_decision'),
                      native_verified_evaluation=coverage, recorded_pending_native_verification=pending,
                      planned_games_per_model_per_lineup=64,
                      planned_independent_deals=16, observed_independent_deals=len(all_deals),
                      native_verified_independent_deals=len(observed_deals),
                      verified_actor_publications=s['verified_actor_publications'], actor_publication_issues=s['actor_publication_issues'],
                      evaluation=final, deadline_utc=p['deadline_utc'],
                      elapsed_wall_seconds=time.time()-s['started_epoch'], disk=disk, training_repetitions=1,
                      interpretation='One training repetition tests the combined treatment. Effects of individual '
                      'components and general playing strength remain unresolved. Snapshot cadence is unchanged per '
                      'update and differs per game. Critic MSE describes fitting the collected training games. '
                      'The two lineups reuse the same sixteen deals. QA games are outside all study counts.')
        write(self.out/'report.json', report)
        self.save(state='complete' if final is not None else 'stopped', reason=reason, disk=disk, pid=None,
                  active_arm=None if final is not None else s.get('active_arm'),
                  phase='complete' if final is not None else s['phase'], report_path=str(self.out/'report.json'))


def initialize_study(out, protocol):
    # Publish both files in one directory rename before any arm can exist.
    with tempfile.TemporaryDirectory(prefix='.stability-init-',dir=out.parent) as tmp:
        stage = Path(tmp)/'study'
        fixed(stage/'protocol.json', protocol)
        state = dict(state='running', phase='initializing', active_arm=None, pid=os.getpid(), started_epoch=time.time(),
                     protocol_sha256=sha(stage/'protocol.json'), experiment_ids={}, committed_games=dict.fromkeys(ARMS,0),
                     verified_games=dict.fromkeys(ARMS,0), training_intervals={k:[] for k in ARMS}, extension_decision=None)
        write(stage/'progress.json',state)
        os.rename(stage,out)
        sync_dir(out.parent)
    return state


def execute(command, out, project, pre, operator_stop=None):
    if out != project/'training/runs'/NAME:
        raise ValueError('study must use its dedicated project training/runs directory')
    archive = str(project/'replays')
    if Path(os.environ.get('BRASS_REPLAY_ROOT', archive)).resolve() != Path(archive):
        raise ValueError('study requires the canonical project replay archive')
    os.environ['BRASS_REPLAY_ROOT'] = archive
    with writer_lock(out):
        resume = command == 'resume'
        if resume:
            protocol, state = read_control(out, project, pre)
            if state['state'] == 'complete':
                return status(out, project, pre)
        elif out.exists():
            raise ValueError('existing study; use explicit resume')
        if not resume:
            if sha(pre) != REG_SHA:
                raise ValueError('unrecognized pre-registration')
            reg = read(pre)
            why = bound(datetime.fromisoformat(reg['task_deadline_utc']).timestamp()-time.time(), 60, 0,
                        shutil.disk_usage(out.parent).free)
            if why:
                raise ValueError('study cannot start: '+why)
        checkout = Path(__file__).resolve().parents[2]
        dirty = subprocess.check_output(['git','-C',str(checkout),'status','--porcelain','--',
                                         'training/brass_rl','engine','players/baselines'], text=True)
        if dirty:
            raise ValueError('study requires committed engine and training source')
        import torch
        torch.set_num_threads(1)
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
        if not resume:
            protocol = declare(out, project, pre)
            state = initialize_study(out,protocol)
        check_inputs(protocol)
        if resume:
            committed(out, state)
            if operator_stop is None:
                (out/'STOP').unlink(missing_ok=True)
        study = Study(out, protocol, state, operator_stop)
        study.save(state='running', pid=os.getpid(), reason=None)
        def request_stop(signum, frame):
            study.signal = True
        original = {sig:signal.signal(sig, request_stop) for sig in (signal.SIGTERM, signal.SIGINT)}
        try:
            study.prepare(resume)
            endpoint = study.train()
            frozen = study.endpoints(endpoint)
            study.finish(study.evaluate(frozen))
        except Halt as stopped:
            study.observe_training()
            study.finish(None, str(stopped))
        except BaseException as failed:
            study.save(state='failed', reason=repr(failed), pid=None)
            raise
        finally:
            for sig, handler in original.items():
                signal.signal(sig, handler)
        return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('run','status','stop','resume'))
    for option in ('project','out','preregistration'):
        parser.add_argument('--'+option, required=True, type=Path)
    parser.add_argument('--operator-stop', type=Path, help='external stop marker owned by a serialized supervisor')
    args = parser.parse_args(argv)
    out, project, pre = (p.resolve() for p in (args.out, args.project, args.preregistration))
    if args.command in ('status','stop'):
        result = status(out, project, pre)
        if args.command == 'stop':
            (out/'STOP').touch()
            result['stop_requested'] = True
    else:
        result = execute(args.command, out, project, pre, args.operator_stop)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
