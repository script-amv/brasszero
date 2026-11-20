# SPDX-License-Identifier: Apache-2.0
"""Local four-arm full-state study. Run or resume only the sealed 400 + 320 games."""
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

from .bounded import GIB, MIB, caffeinate, disk_bytes
from .config import TrainConfig
from .experiments import atomic, encoded, read_experiment, writer_lock
from .learning_study import (ROOT, SCHOOL, PROFILE, StudyStopped, fixed, fork_arm,
                             parent_identity, read, write)
from .seeding import SeedStreams

STUDY = ROOT/'training/runs/four-branch-study-2026-11-19'
SOURCE = SCHOOL/'updates/000688'
ARMS = ('control', 'A', 'B', 'C')
BASE, END, FIRST_GAME, GAMES = 688, 738, 1376, 100
FINAL_MASTER = 13591682439352577831
LIMIT, RESERVE = 8*GIB, 128*MIB


def arm_config(parent, arm):
    if arm not in ARMS:
        raise ValueError('unknown study arm')
    league, ppo = parent.league, parent.ppo
    if arm == 'A':
        league = replace(league, lineup_probabilities=(.8, .2, 0), lineup_schedule='four_of_five')
    elif arm == 'B':
        ppo = replace(ppo, critic_target='terminal')
    elif arm == 'C':
        ppo = replace(ppo, reward='score')
    return replace(parent, ppo=ppo, league=league, eval=replace(parent.eval, interval_updates=0),
                   runtime=replace(parent.runtime, max_wall_time_seconds=None))


def suite(candidate):
    from .evaluate import EvalSuite
    ref = {'id': read(SOURCE/'metadata.json')['id'], 'path': str(SOURCE)}
    return EvalSuite('final', FINAL_MASTER, 8, {'greedy': ['candidate']+['greedy']*3,
                     'frozen688': ['candidate']+[ref]*3}, candidate, replay_limit=0, outcomes_only=True)


def shared_roots():
    archive = Path(os.environ.get('BRASS_REPLAY_ROOT', ROOT/'replays')).resolve()
    return archive, ROOT/'players/models'


def freshness_evidence(root, evaluation, training, *, expected_deals=8, extra_protocols=()):
    """Compare actual integer seeds with all available recorded games and suite manifests."""
    from brass_ai.replay_archive import Archive
    known, files = set(), []
    declarations = set((ROOT/'training/runs').rglob('suite.json'))
    declarations.update(extra_protocols)
    for path in sorted(declarations):
        if root in path.parents:
            continue
        manifest = read(path)
        known.update(manifest.get('deal_seeds', ()))
        known.update(manifest.get('training_deals', ()))
        for key in ('suite', 'evaluation', 'head_to_head'):
            section = manifest.get(key)
            if isinstance(section, dict):
                known.update(section.get('deal_seeds', ()))
        files.append({'file': str(path.relative_to(ROOT)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    archive = Archive(shared_roots()[0])
    recorded = 0
    for path in archive.games.glob('*.json*'):
        initial = archive.read(path.name.split('.')[0])['initial']
        if initial.get('seed') is not None:
            known.add(initial['seed'])
            recorded += 1
    if (len(evaluation) != expected_deals or len(set(evaluation)) != expected_deals
            or len(set(training)) != GAMES or set(evaluation) & (known | set(training))):
        raise ValueError('repeated or previously used evaluation deal')
    return {'prior_suite_files': files, 'prior_seeded_recordings': recorded, 'distinct_prior_seeds': len(known),
            'prior_seeds_sha256': hashlib.sha256(encoded(sorted(known))).hexdigest(),
            'shared_training_deals': GAMES, 'evaluation_intersection': []}


def historical_training_evidence(root, evaluation, master, *, next_game=FIRST_GAME+GAMES, include_ancestry=False):
    """Include retained checkpoint manifests even when a game predates durable recording."""
    known = {SeedStreams(master).seed('train.deal', gid) for gid in range(next_game)}
    files = 0
    pattern = 'manifests.json' if include_ancestry else 'updates/*/manifests.json'
    for path in (ROOT/'training/runs').rglob(pattern):
        if root in path.parents:
            continue
        manifest = read(path)
        known.update(row['deal_seed'] for row in manifest.get('train', ()) if 'deal_seed' in row)
        if include_ancestry:
            for key in ('dev', 'final'):
                suite = manifest.get(key)
                if isinstance(suite, dict):
                    known.update(suite.get('deal_seeds', ()))
        files += 1
    if known.intersection(evaluation):
        raise ValueError('evaluation overlaps historical training deals')
    return {'checkpoint_manifests': files, 'distinct_deals': len(known),
            'deal_digest': hashlib.sha256(encoded(sorted(known))).hexdigest(), 'evaluation_intersection': []}


def initialize(root):
    from .artifacts import load_update
    parent = parent_identity(SOURCE)
    if parent['checkpoint_sha256'] != PROFILE['parent_manifest_sha256']:
        raise ValueError('source checkpoint manifest changed')
    cfg = TrainConfig.from_json(SCHOOL/'config.json')
    source = load_update(SOURCE, config=cfg)
    if (source.state.next_game_id != FIRST_GAME or source.state.completed_updates != BASE or
            cfg.runtime.device != 'cpu' or cfg.runtime.torch_threads != 1 or cfg.games_per_update != 2 or
            cfg.ppo.gae_lambda != .95 or cfg.ppo.reward != 'strict_win' or cfg.ppo.critic_target != 'gae' or
            not cfg.model.use_history or cfg.master_seed != source.state.master_seed):
        raise ValueError('incompatible starting state')
    streams = SeedStreams(source.state.master_seed)
    deals = [streams.seed('train.deal', g) for g in range(FIRST_GAME, FIRST_GAME+GAMES)]
    protocol = suite(parent['model_id']).manifest()
    filename = root/'study.json'
    old = read(filename) if filename.exists() else None
    archive, models = shared_roots()
    definition = dict(format='brass-four-branch/1', parent=parent, arms=list(ARMS), base_update=BASE, end_update=END,
                      first_game_id=FIRST_GAME, games_per_arm=GAMES, games_per_update=cfg.games_per_update,
                      training_master=source.state.master_seed, training_deals=deals, evaluation=protocol,
                      frozen_lineup_game_ids=[g for g in range(FIRST_GAME, FIRST_GAME+GAMES) if g % 5 == 0],
                      storage_limit_bytes=LIMIT, reserve_bytes=RESERVE, archive=str(archive), models=str(models),
                      shared_baseline_bytes=old['shared_baseline_bytes'] if old else disk_bytes(archive, models),
                      freshness=old['freshness'] if old else freshness_evidence(root, protocol['deal_seeds'], deals))
    fixed(filename, definition)
    history_path = root/'historical-training-seeds.json'
    if not history_path.exists():
        fixed(history_path, historical_training_evidence(root, protocol['deal_seeds'], source.state.master_seed))
    prepare_branches(root, definition, SOURCE, cfg, arm_config)
    return definition


def prepare_branches(root, definition, source, cfg, config_for, *, checkpoint=None):
    from .artifacts import load_update
    from .train import publish_policies
    parent = definition['parent']
    ids = {}
    for arm in definition['arms']:
        if checkpoint is not None:
            checkpoint(arm)
        dest = root/arm
        expected = config_for(cfg, arm)
        if not dest.exists():
            storage(root, definition, check=True)
            fork_arm(source, dest, .95, study_config=expected)
        meta = read_experiment(dest)
        initial = load_update(dest/'updates'/('%06d' % definition['base_update']), config=expected)
        if (meta['parent'] != parent or not all(meta['initial_state_verified'].values()) or
                meta['initial_model_id'] != parent['model_id'] or initial.state.next_game_id != definition['first_game_id'] or
                initial.state.manifests['experiment_id'] != meta['id'] or TrainConfig.from_json(dest/'config.json') != expected):
            raise ValueError('branch ancestry/config/identity mismatch')
        ids[arm] = meta['id']
        publish_policies(dest)
        if checkpoint is not None:
            checkpoint(arm)
    if len(set(ids.values())) != len(definition['arms']) or parent['experiment_id'] in ids.values():
        raise ValueError('experiment identities must be independent')
    fixed(root/'branches.json', ids)


def storage(root, manifest, *, check=False):
    study = disk_bytes(root)
    shared = disk_bytes(*shared_roots())
    usage = {'study_logical_bytes': study, 'shared_growth_logical_bytes': max(0, shared-manifest['shared_baseline_bytes']),
             'free_bytes': shutil.disk_usage(root).free}
    usage['combined_logical_bytes'] = study + usage['shared_growth_logical_bytes']
    if check and (usage['combined_logical_bytes'] + RESERVE > LIMIT or usage['free_bytes'] < RESERVE):
        raise StudyStopped('disk reserve reached; committed work retained')
    return usage


def latest(root, arm):
    return max(int(p.name) for p in (root/arm/'updates').iterdir() if p.is_dir() and p.name.isdecimal())


def verify_public_copy(local, published):
    for name in ('policy.pt', 'metadata.json', 'manifest.json'):
        if (local/name).read_bytes() != (published/name).read_bytes():
            raise ValueError('published policy differs from immutable boundary export')


def check_training_replay(replay, game, gid, seed, experiment, number, collection_id, history):
    # The recorder binds the current policy in the actual seats. Frozen seats
    # carry the source weight digest and native baselines carry their own names.
    if (game['game_id'] != gid or game['deal_seed'] != seed
            or replay['initial']['seed'] != game['deal_seed'] or replay['result']['vp'] != game['outcome']['vp']
            or replay['result']['ranking'] != game['outcome']['ranking']
            or replay['provenance']['experiment'] != experiment or replay['provenance']['update'] != number):
        raise ValueError('training replay/deal/provenance mismatch')
    labels = []
    for seat in game['lineup']:
        if seat['kind'] == 'current':
            labels.append(collection_id)
        elif seat['kind'] == 'frozen':
            historical = history[seat['snapshot_id']]
            labels.append('%s-%06d:weights-sha256:%s' %
                          (historical['role'], historical['update'], historical['sha256']))
        else:
            labels.append(seat['kind'])
    if [seat['name'] for seat in replay['seats']] != labels or len(replay['moves']) != game['plies']:
        raise ValueError('training replay seat identity or length differs')


def training_replays_by_game(archive, ids):
    records = {}
    for replay_id in ids:
        record = archive.read(replay_id)
        if record['game_id'] != replay_id:
            raise ValueError('training replay ID differs from its checkpoint reference')
        records[record['provenance']['game']] = record
    return records


def verify_training(root, manifest, *, source=SOURCE, config_for=None):
    base, end = manifest['base_update'], manifest['end_update']
    first_game, arms = manifest['first_game_id'], manifest['arms']
    source_config = TrainConfig.from_json(source.parents[1]/'config.json') if config_for else None
    from brass_ai.replay_archive import Archive
    from .artifacts import inspect_update_state, load_update
    from .policy_io import inspect_policy
    from .transfer import validate_replay
    archive = Archive(shared_roots()[0])
    endpoints = {'parent': {'path': str(source), 'id': manifest['parent']['model_id'],
                            'experiment_id': manifest['parent']['experiment_id'], 'value_semantics': 'strict_win'}}
    seen = set()
    published = {read(p)['id']: p.parent for p in shared_roots()[1].glob('*/metadata.json')}
    for arm in arms:
        expected_cfg = config_for(source_config, arm) if config_for else None
        exposure_arm = expected_cfg.league.lineup_schedule == 'four_of_five' if expected_cfg else arm == 'A'
        path = root/arm
        experiment = read_experiment(path)['id']
        updates = sorted(int(p.name) for p in (path/'updates').iterdir() if p.name.isdecimal())
        if updates != list(range(base, end+1)):
            raise ValueError('all 51 complete boundaries are required before evaluation')
        identities = []
        for number in updates:
            boundary = path/'updates'/('%06d' % number)
            public = path/'policies'/boundary.name
            if number in (base, end) or number % 10 == 0:
                state = load_update(boundary, config=expected_cfg).state
            else:
                state = inspect_update_state(boundary, config=expected_cfg)
            identity = inspect_policy(public)
            verify_public_copy(public, published[identity.id])
            if (identity.id != read(boundary/'metadata.json')['id'] or state.completed_updates != number or
                    state.next_game_id != first_game + 2*(number-base) or state.manifests['experiment_id'] != experiment):
                raise ValueError('invalid retained boundary or public policy')
            if number > base:
                games = state.manifests['train']
                ids = state.manifests['replay_ids']
                if len(games) != 2 or len(ids) != 2 or len(set(ids)) != 2 or seen.intersection(ids):
                    raise ValueError('training requires two distinct retained games per update')
                records = training_replays_by_game(archive, ids)
                for i, game in enumerate(games):
                    gid = first_game + 2*(number-base-1) + i
                    replay = records[gid]
                    check_training_replay(replay, game, gid, manifest['training_deals'][gid-first_game],
                                          experiment, number, identities[-1]['id'], state.league_history)
                    if exposure_arm:
                        expected = ['current']+['frozen']*3 if gid % 5 == 0 else ['current']*4
                        if sorted(e['kind'] for e in game['lineup']) != sorted(expected):
                            raise ValueError('80/20 exposure mismatch')
                    validate_replay(replay)
                seen.update(ids)
            identities.append({'update': number, 'id': identity.id, 'sha256': identity.content_hashes['policy.pt']})
        fixed(path/'verified-policies.json', identities)
        score_reward = expected_cfg.ppo.reward == 'score' if expected_cfg else arm == 'C'
        endpoints[arm] = {'path': str(public), 'id': identity.id, 'experiment_id': experiment,
                          'value_semantics': 'expected_terminal_score_reward' if score_reward else 'strict_win'}
    if len(seen) != len(arms)*manifest['games_per_arm']:
        raise ValueError('incomplete training recording audit')
    fixed(root/'endpoints.json', endpoints)
    return endpoints


def evaluation_origin(arm, endpoint, *, study=STUDY):
    # Evaluating the archived source in this study must not append study games
    # to an eventual transfer package for the historical school experiment.
    owner = study.name+'-parent-evaluation' if arm == 'parent' else endpoint['experiment_id']
    return {'run_experiment': owner, 'source_experiment': endpoint['experiment_id'], 'study_arm': arm}


def check_evaluation_replay(recording, row, origin):
    # With four copies of 688, seat labels alone cannot distinguish rotations.
    # Bind the scalar row to the recorder's independent, per-game coordinates.
    expected = dict(origin, game=row['game_id'], deal=row['deal_id'], rotation=row['rotation'],
                    lineup=row['lineup'], update=row['update'], candidate_id=row['candidate_id'],
                    experiment=row['suite_id'])
    if (any(recording['provenance'].get(k) != v for k, v in expected.items())
            or recording['game_id'] != row['replay_id'] or recording['initial']['seed'] != row['deal_seed']
            or recording['result']['vp'] != row['vp'] or recording['result']['ranking'] != row['ranking']
            or [s['name'] for s in recording['seats']] != row['labels'] or len(recording['moves']) != row['plies']):
        raise ValueError('evaluation replay/provenance mismatch')


def final_evaluation(root, endpoints, manifest, before_game, *, suite_for=suite,
                     candidates=('parent',)+ARMS, study=STUDY, evaluator=None):
    from brass_ai.replay_archive import Archive, recording_context
    from .evaluate import evaluate
    from .four_branch_report import summarize
    from .transfer import validate_replay
    archive = Archive(shared_roots()[0])
    rows, ids = {}, set()
    for arm, endpoint in endpoints.items():
        declared = suite_for(endpoint['id'])
        protocol = declared.manifest()
        if any(value != manifest['evaluation'][k] for k, value in protocol.items()
               if k not in ('candidate_id', 'suite_id')):
            raise ValueError('frozen evaluation protocol changed')
        origin = evaluation_origin(arm, endpoint, study=study)
        with recording_context(**origin):
            evaluate_candidate = evaluator or evaluate
            report = evaluate_candidate(Path(endpoint['path']), declared, root/'evaluation'/arm,
                                        before_game=lambda: before_game(arm))
        for row in report.results:
            recording = archive.read(row['replay_id'])
            if recording['game_id'] in ids:
                raise ValueError('duplicate evaluation replay')
            check_evaluation_replay(recording, row, origin)
            validate_replay(recording)
            ids.add(recording['game_id'])
        rows[arm] = report.results
    if len(ids) != len(candidates)*64:
        raise ValueError('incomplete set of unique frozen evaluation games')
    return summarize(rows, candidates=candidates)


def run(root, manifest, stop):
    from .four_branch_report import markdown_report
    return run_study(root, manifest, stop, arms=ARMS, base=BASE, end=END, verify=verify_training,
                     evaluate=final_evaluation, render=markdown_report)


def run_study(root, manifest, stop, *, arms, base, end, verify, evaluate, render, initialize=None, init_only=False):
    status_path = root/'progress.json'
    prior = read(status_path) if status_path.exists() else {}
    if prior.get('status') == 'complete':
        if not (root/'result.json').is_file():
            raise ValueError('completed study missing result')
        return prior
    started, tick = time.time(), time.monotonic()
    status = dict(prior, pid=os.getpid(), status='running', reason=None, phase='initialization' if initialize else 'training',
                  first_started_epoch=prior.get('first_started_epoch', started),
                  invocation_started_epoch=started, invocation_count=prior.get('invocation_count', 0)+1)
    prior_active = prior.get('active_seconds', 0)
    if prior.get('status') == 'running':
        status['interrupted_invocations'] = prior.get('interrupted_invocations', 0)+1
        if initialize is not None:
            gaps = list(prior.get('interrupted_intervals', []))
            gaps.append(dict(last_phase=prior.get('phase'), last_arm=prior.get('arm'),
                             last_observed_epoch=prior['updated_epoch'], next_started_epoch=started))
            status['interrupted_intervals'] = gaps

    def refresh(arm=None):
        status.update(arm=arm, updated_epoch=time.time(), active_seconds=prior_active+time.monotonic()-tick,
                      completed_updates={a: latest(root, a) if (root/a/'updates').is_dir() else None for a in arms},
                      evaluation_games=sum(1 for _ in (root/'evaluation').glob('*/candidates/*/games/*.json')))
        if initialize is not None:
            status['elapsed_seconds'] = status['updated_epoch']-status['first_started_epoch']
        if (root/'head-to-head').is_dir():
            status['head_to_head_games'] = sum(1 for _ in (root/'head-to-head/games').glob('*.json'))
            status['evaluation_games'] += status['head_to_head_games']
        status['new_training_games'] = sum(2*(u-base) for u in status['completed_updates'].values() if u is not None)
        write(status_path, status)

    def check(arm, declaration=None):
        nonlocal manifest
        if declaration is not None:
            manifest = declaration
        refresh(arm)
        if stop() or (root/'stop').exists():
            raise StudyStopped('manual stop at complete boundary')
        if manifest is not None:
            storage(root, manifest, check=True)

    power = None
    try:
        check(None)
        power = caffeinate(os.getpid())
        if initialize is not None:
            manifest = initialize(root, checkpoint=check)
            check(None)
        if init_only:
            status.update(status='initialized', phase='initialized')
            return status
        from .train import train
        status['phase'] = 'training'
        for arm in arms:
            check(arm)
            dest = root/arm
            if latest(root, arm) < end:
                cfg = TrainConfig.from_json(dest/'config.json')
                report = train(cfg, dest, resume=dest/'latest', max_updates=end, stop_requested=lambda: check(arm))
                if report.completed_updates != end:
                    raise StudyStopped('incomplete arm: '+report.stop_reason)
                del report
        check(None)
        status['phase'] = 'training_verification'
        refresh()
        endpoints = verify(root, manifest)
        status['phase'] = 'evaluation'
        check('parent')
        result = evaluate(root, endpoints, manifest, check)
        refresh()
        result.update(endpoints=endpoints, training_games=len(arms)*2*(end-base),
                      evaluation_games=status['evaluation_games'],
                      active_seconds=status['active_seconds'], elapsed_seconds=time.time()-status['first_started_epoch'],
                      disk=storage(root, manifest))
        if status.get('interrupted_intervals'):
            result['interrupted_intervals'] = status['interrupted_intervals']
        import subprocess
        result['disk']['study_allocated_kib_du'] = int(subprocess.check_output(['du', '-sk', str(root)]).split()[0])
        write(root/'result.json', result)
        atomic(root/'result.md', render(result).encode('utf-8'))
        status.update(status='complete', phase='complete')
    except StudyStopped as exc:
        status.update(status='stopped', reason=str(exc))
    except BaseException as exc:
        status.update(status='failed', reason=repr(exc))
        raise
    finally:
        if power is not None:
            power.terminate()
            power.wait()
        if manifest is not None:
            status['disk'] = storage(root, manifest)
        refresh(status.get('arm'))
    return status


def main(argv=None):
    study_cli(argv, STUDY, __spec__.name, initialize, run, __doc__)


def study_cli(argv, root, module, initialize, run, description, *, managed_init=False):
    import sys
    import subprocess
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('action', choices=('init', 'run', 'launch', 'status', 'stop'))
    action = parser.parse_args(argv).action
    if action == 'status':
        path = root/'progress.json'
        print(json.dumps(read(path) if path.exists() else {'status': 'not started'}, indent=2))
        return
    root.mkdir(parents=True, exist_ok=True)
    if action == 'stop':
        (root/'stop').touch()
        return
    if action == 'launch':
        if (root/'stop').exists():
            raise SystemExit('clear the requested stop before resuming')
        with (root/'controller.log').open('ab') as log:
            proc = subprocess.Popen([sys.executable, '-u', '-m', module, 'run'], cwd=ROOT,
                                    stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
        print(json.dumps({'spawned_pid': proc.pid, 'log': str(root/'controller.log')}), flush=True)
        return
    with writer_lock(root):
        from .__main__ import _graceful_signals
        if managed_init:
            from .__main__ import _SearchStop
            stop = _SearchStop()
            with _graceful_signals(stop):
                print(json.dumps(run(root, None, stop, init_only=action == 'init'), sort_keys=True), flush=True)
            return
        import torch
        from .train import StopFlag
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        stop = StopFlag()
        with _graceful_signals(stop):
            manifest = initialize(root)
            if action == 'run':
                print(json.dumps(run(root, manifest, stop), sort_keys=True), flush=True)
            else:
                print(json.dumps({'status': 'initialized', 'parent': manifest['parent']}), flush=True)


if __name__ == '__main__':
    main()
