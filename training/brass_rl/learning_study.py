# SPDX-License-Identifier: Apache-2.0
"""Bounded, matched GAE .95/.99 continuation from school update 688."""
import argparse
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
import os
import shutil
from pathlib import Path
import tempfile
import time

from .bounded import GIB, MIB, caffeinate, disk_bytes
from .config import TrainConfig
from .experiments import (atomic, encoded, ensure_experiment, link_or_copy, parent_identity, read_experiment, same_state,
                          sync_dir, writer_lock)

ROOT = Path(__file__).resolve().parents[2]
PROFILE = json.loads((ROOT/'training/configs/gae-study.json').read_text())
SCHOOL = ROOT/PROFILE['source_run']
STUDY = ROOT/'training/runs/learning-improvement-2026-11-19'
BASE, END = PROFILE['base_update'], PROFILE['end_update']
ARMS = PROFILE['arms']
ARM_SECONDS, GLOBAL_SECONDS = 2400, 7200
RESERVE_BYTES, RESERVE_SECONDS = 60*MIB, 120


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    atomic(path, encoded(value))


def fixed(path, value):
    if path.exists():
        if encoded(read(path)) != encoded(value):
            raise ValueError('immutable study declaration changed: '+str(path))
    else:
        write(path, value)


def fork_arm(source, dest, lam, *, learning_rate=None, study_config=None, dynamic_members=None):
    """Publish one verified full-state boundary; earlier history lives in parent."""
    import torch
    from .artifacts import capture_rng_states, load_update, restore_rng_states, save_update
    source, dest = Path(source).resolve(), Path(dest).resolve()
    if not source.name.isdecimal() or lam not in ARMS.values():
        raise ValueError('study needs an explicit boundary and one of the two declared lambdas')
    parent = parent_identity(source)
    initial_rng = capture_rng_states()
    with writer_lock(source.parent.parent), writer_lock(dest):
        if dest.exists():
            raise ValueError('study arm already exists')
        try:
            original = load_update(source, for_training=True)
            s = original.state
            rate = s.config.ppo.learning_rate if learning_rate is None else learning_rate
            cfg = replace(s.config, ppo=replace(s.config.ppo, gae_lambda=lam, learning_rate=rate),
                          eval=replace(s.config.eval, interval_updates=0),
                          runtime=replace(s.config.runtime, max_wall_time_seconds=ARM_SECONDS,
                                          checkpoint_margin_seconds=RESERVE_SECONDS))
            overrides = {key: {'from': old, 'to': new} for key, old, new in (
                ('ppo.gae_lambda', s.config.ppo.gae_lambda, lam),
                ('eval.interval_updates', s.config.eval.interval_updates, 0),
                ('runtime.max_wall_time_seconds', s.config.runtime.max_wall_time_seconds, ARM_SECONDS),
                ('runtime.checkpoint_margin_seconds', s.config.runtime.checkpoint_margin_seconds, RESERVE_SECONDS))}
            if learning_rate is not None:
                overrides['ppo.learning_rate'] = {'from': s.config.ppo.learning_rate, 'to': rate}
            if study_config is not None:
                if learning_rate is not None or study_config.ppo.learning_rate != s.config.ppo.learning_rate:
                    raise ValueError("configured study fork must preserve Adam learning rate")
                cfg = study_config
                overrides = {}
                for section, fields in asdict(cfg).items():
                    previous = asdict(s.config)[section]
                    if isinstance(fields, dict):
                        for key, value in fields.items():
                            if value != previous[key]:
                                overrides[section+'.'+key] = {'from': previous[key], 'to': value}
                    elif fields != previous:
                        overrides[section] = {'from': previous, 'to': fields}
            with tempfile.TemporaryDirectory(prefix='.study-fork-', dir=dest.parent) as tmp:
                staging = Path(tmp)/'run'
                staging.mkdir()
                meta = ensure_experiment(staging, cfg, parent=parent)
                for snapshot in s.league_snapshots + s.holdouts:
                    target = staging/'weights'/snapshot.weights_path.name
                    if not target.exists():
                        link_or_copy(snapshot.weights_path, target)
                optimizer = torch.optim.Adam(original.model.parameters(), lr=cfg.ppo.learning_rate)
                optimizer.load_state_dict(original.optimizer_state)
                # Adam.load_state_dict restores the saved learning rate too.
                # Change only that declared hyperparameter, retaining moments/steps.
                expected_adam = deepcopy(original.optimizer_state)
                if learning_rate is not None:
                    optimizer.param_groups[0]['lr'] = rate
                    expected_adam['param_groups'][0]['lr'] = rate
                boundary = replace(s, config=cfg, development_best_id=None, metrics=(),
                                   manifests={'experiment_id': meta['id'], 'replay_ids': []})
                if dynamic_members is not None:
                    from .dynamic_league import DynamicLeague
                    from .seeding import SeedStreams
                    if study_config != s.config or s.dynamic_league is not None:
                        raise ValueError('league migration requires unchanged legacy source config')
                    dynamic = DynamicLeague.start(SeedStreams(s.master_seed), dynamic_members, s.holdouts,
                                                  history_extent=s.completed_updates+1,
                                                  next_game_id=s.next_game_id)
                    boundary = replace(boundary, league_snapshots=dynamic.snapshots,
                                       league_history=dynamic.history, dynamic_league=dynamic.state)
                    meta['league_migration'] = {
                        'from': 'legacy', 'to': dynamic.state['format'],
                        'source_snapshots': read(source/'manifest.json')['snapshots'],
                        'source_history_sha256': hashlib.sha256(encoded(s.league_history)).hexdigest()}
                restore_rng_states(s.rng_states, device=cfg.runtime.device)
                checkpoint = save_update(staging, original.model, optimizer, boundary,
                                         inherited_policy=source if study_config is not None else None)
                child = load_update(checkpoint)
                verified = {
                    'model': same_state(original.model.state_dict(), child.model.state_dict()),
                    'adam': same_state(expected_adam, child.optimizer_state),
                    'global_rng': same_state(s.rng_states, child.state.rng_states),
                    'controller_rng': same_state(s.controller_rng_states, child.state.controller_rng_states),
                    'league': (s.league_history == child.state.league_history and
                               read(source/'manifest.json')['snapshots'] == read(checkpoint/'manifest.json')['snapshots']),
                    'counters_and_master': (s.master_seed, s.completed_updates, s.next_game_id) ==
                            (child.state.master_seed, child.state.completed_updates, child.state.next_game_id)}
                if dynamic_members is not None:
                    del verified['league']
                    verified['declared_league_migration'] = (
                        boundary.dynamic_league == child.state.dynamic_league and
                        boundary.league_history == child.state.league_history and
                        boundary.league_snapshot_ids == child.state.league_snapshot_ids and
                        [h.id for h in s.holdouts] == [h.id for h in child.state.holdouts])
                if study_config is not None:
                    verified['public_identity'] = parent['model_id'] == read(checkpoint/'metadata.json')['id']
                if not all(verified.values()):
                    raise ValueError('fork state differs from source: '+str(verified))
                meta.update(mode='full-state-branch', base_update=s.completed_updates, config_overrides=overrides,
                            historical_gaps=['Inherited prefix before update %d is referenced in the parent; '
                                             'its retained history and original gaps remain there.' % s.completed_updates],
                            source_resume_notes=list(original.rng_restore_notes),
                            parent_development_best_id=s.development_best_id, initial_state_verified=verified,
                            initial_model_id=read(checkpoint/'metadata.json')['id'],
                            initial_policy_sha256=read(checkpoint/'manifest.json')['content_hashes']['policy.pt'],
                            initial_next_game_id=s.next_game_id)
                write(staging/'experiment.json', meta)
                from .retention import pin
                pin(source)
                os.replace(staging, dest)
                sync_dir(dest.parent)
                return meta
        finally:
            restore_rng_states(initial_rng)


def budget_reason(remaining_seconds, used_bytes, reserve_seconds):
    if remaining_seconds <= reserve_seconds:
        return 'time'
    if used_bytes + RESERVE_BYTES > GIB:
        return 'disk'
    return None


def shared_bytes():
    # Common archive and public policy publication happen outside the study.
    return disk_bytes(Path(os.environ.get('BRASS_REPLAY_ROOT', ROOT/'replays')), ROOT/'players/models')


def suite(master, candidate):
    from .evaluate import EvalSuite
    frozen = SCHOOL/'updates/000600'
    ref = {'id': read(frozen/'metadata.json')['id'], 'path': str(frozen)}
    return EvalSuite('dev', master, 4, {'greedy': ['candidate'] + ['greedy']*3,
                       'frozen600': ['candidate'] + [ref]*3}, candidate, replay_limit=0)


def initialize(root, master):
    source = SCHOOL/'updates'/('%06d' % BASE)
    parent = parent_identity(source)
    definition = suite(master, parent['model_id']).manifest()
    if parent['checkpoint_sha256'] != PROFILE['parent_manifest_sha256']:
        raise ValueError('declared source checkpoint digest changed')
    manifest_path = root/'study.json'
    existing = read(manifest_path) if manifest_path.exists() else None
    manifest = {'format': 'brass-gae-study/1', 'parent': parent, 'arms': ARMS, 'base_update': BASE,
                'next_game_id': 1376, 'end_update': END, 'games_per_arm': 24, 'suite': definition,
                'global_seconds': GLOBAL_SECONDS, 'arm_seconds': ARM_SECONDS, 'max_bytes': GIB,
                'reserve_bytes': RESERVE_BYTES, 'reserve_seconds': RESERVE_SECONDS,
                'shared_baseline_bytes': existing['shared_baseline_bytes'] if existing else shared_bytes(),
                'archive_path': str(Path(os.environ.get('BRASS_REPLAY_ROOT', ROOT/'replays')).resolve())}
    fixed(manifest_path, manifest)
    for name, lam in ARMS.items():
        dest = root/name
        if not dest.exists():
            used = disk_bytes(root.parent) + max(0, shared_bytes() - manifest['shared_baseline_bytes'])
            if used + RESERVE_BYTES > GIB or shutil.disk_usage(root).free < RESERVE_BYTES:
                raise ValueError('insufficient study disk allowance to initialize arm')
            fork_arm(source, dest, lam)
        meta = read_experiment(dest)
        cfg = TrainConfig.from_json(dest/'config.json')
        parent_cfg = TrainConfig.from_json(SCHOOL/'config.json')
        expected = replace(parent_cfg, ppo=replace(parent_cfg.ppo, gae_lambda=lam),
                           eval=replace(parent_cfg.eval, interval_updates=0),
                           runtime=replace(parent_cfg.runtime, max_wall_time_seconds=ARM_SECONDS,
                                           checkpoint_margin_seconds=RESERVE_SECONDS))
        if (meta['parent'] != parent or cfg != expected or cfg.games_per_update != 2
                or meta['initial_next_game_id'] != 1376 or meta['base_update'] != BASE
                or cfg.runtime.device != 'cpu' or cfg.runtime.torch_threads != 1):
            raise ValueError('study arm configuration or ancestry mismatch')
    return manifest


def value_summary(rows):
    """Average game-level losses within deal, then give each deal equal weight."""
    from collections import defaultdict
    from statistics import mean
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[row['lineup']][str(row['deal_id'])].append(row['terminal_brier'])
    result = {}
    for lineup, deals in grouped.items():
        if any(len(games) != 4 or any(g is None for g in games) for games in deals.values()):
            raise ValueError('study scoring requires four recorded rotations in each deal')
        keys = [k for k in next(iter(deals.values()))[0] if k.endswith('brier')]
        scores = {d: {k: mean(g[k] for g in games) if all(g[k] is not None for g in games) else None
                      for k in keys} for d, games in deals.items()}
        result[lineup] = {'independent_deals': len(deals), 'deal_means': scores,
                          'decisions': sum(g['decisions'] for games in deals.values() for g in games),
                          'mean': {k: mean(d[k] for d in scores.values()) if all(d[k] is not None for d in scores.values())
                                   else None for k in keys}}
    return result


class StudyStopped(Exception):
    """Cooperative evaluation stop before a missing game."""


def run(root, manifest, stop, started):
    from .evaluate import evaluate
    from .train import train
    progress_path = root/'progress.json'
    progress = read(progress_path) if progress_path.exists() else {
        'started_epoch': started, 'deadline_epoch': started + GLOBAL_SECONDS, 'arm_deadlines': {}}
    progress.update(status='running', reason=None)
    write(progress_path, progress)

    def check():
        used = disk_bytes(root.parent) + max(0, shared_bytes() - manifest['shared_baseline_bytes'])
        why = 'manual' if stop() or (root/'stop').exists() else budget_reason(
            progress['deadline_epoch'] - time.time(), used, RESERVE_SECONDS)
        if not why and shutil.disk_usage(root).free < RESERVE_BYTES:
            why = 'free_disk'
        progress['additional_logical_bytes'] = used
        if why:
            raise StudyStopped(why)

    def latest(name):
        return int(max(p.name for p in (root/name/'updates').iterdir() if p.name.isdecimal()))

    try:
        if not (root/'endpoints.json').exists():
            ceiling = END
            for name in ARMS:
                check()
                arm = root/name
                deadline = progress['arm_deadlines'].setdefault(name, time.time() + ARM_SECONDS)
                remaining = deadline - time.time()
                if latest(name) < ceiling and remaining > RESERVE_SECONDS:
                    cfg = TrainConfig.from_json(arm/'config.json')
                    progress.update(phase='training', arm=name)
                    write(progress_path, progress)

                    def before_update():
                        progress['completed_updates'] = {a: latest(a) for a in ARMS}
                        write(progress_path, progress)
                        try:
                            check()
                        except StudyStopped:
                            return True
                        return deadline - time.time() <= RESERVE_SECONDS

                    power = caffeinate(os.getpid())
                    try:
                        report = train(cfg, arm, resume=arm/'latest', max_updates=ceiling, stop_requested=before_update)
                        progress[name+'_stop_reason'] = report.stop_reason
                        del report
                    finally:
                        if power is not None:
                            power.terminate()
                            power.wait()
                    check()
                ceiling = min(ceiling, latest(name))
            if ceiling <= BASE:
                raise StudyStopped('no new matched update')
            paths = {'parent': SCHOOL/'updates'/('%06d' % BASE)}
            paths.update({name: root/name/'updates'/('%06d' % ceiling) for name in ARMS})
            endpoints = {'matched_update': ceiling,
                         'candidates': {name: {'path': str(p), 'id': read(p/'metadata.json')['id']}
                                        for name, p in paths.items()}}
            fixed(root/'endpoints.json', endpoints)

        endpoints = read(root/'endpoints.json')
        result = {'matched_update': endpoints['matched_update'], 'policies': {}}
        for name in ('parent', *ARMS):
            item = endpoints['candidates'][name]
            progress.update(phase='evaluation', arm=name)
            write(progress_path, progress)
            declared = suite(manifest['suite']['master_seed'], item['id'])
            if declared.manifest() != manifest['suite']:
                raise ValueError('shared evaluation suite changed')
            report = evaluate(Path(item['path']), declared, root/'eval', before_game=check)
            result['policies'][name] = {'id': item['id'], 'suite_id': report.suite_id,
                                        'outcomes': {k: asdict(v) for k, v in report.summaries.items()},
                                        'terminal_scores': value_summary(report.results)}
        write(root/'result.json', result)
        progress.update(status='complete', phase='complete')
    except StudyStopped as exc:
        progress.update(status='stopped', reason=str(exc))
    except BaseException as exc:
        progress.update(status='failed', reason=repr(exc))
        raise
    finally:
        progress.update(completed_updates={name: latest(name) for name in ARMS},
                        completed_evaluation_rows=sum(1 for _ in (root/'eval/candidates').glob('*/games/*.json')),
                        additional_logical_bytes=disk_bytes(root.parent) + max(0, shared_bytes() - manifest['shared_baseline_bytes']),
                        updated_epoch=time.time())
        write(progress_path, progress)
    return progress


def main(argv=None):
    started = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('init', 'run'), help='initialize only, or train/evaluate/resume')
    args = parser.parse_args(argv)
    import torch
    from .__main__ import _graceful_signals
    from .train import StopFlag
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    root = STUDY/'study'
    root.mkdir(parents=True, exist_ok=True)
    master = PROFILE['dev_master_seed']
    with writer_lock(root):
        manifest = initialize(root, master)
        if args.action == 'init':
            print(json.dumps({'status': 'initialized', 'manifest': str(root/'study.json')}), flush=True)
        else:
            stop = StopFlag()
            with _graceful_signals(stop):
                print(json.dumps(run(root, manifest, stop, started), sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
