"""Immutable paired frozen-search experiments; only this controller sees deals.

Actual choices alone enter Recorder. Search worlds never leave the player. The
allowance is cumulative and pessimistically exhausted after an unclean crash.
"""
from collections import Counter
from contextlib import nullcontext
from dataclasses import asdict
import hashlib
import gzip
import json
import math
import os
from pathlib import Path
import platform
import resource
import subprocess
import time

import brass
import numpy as np
import torch
from brass_ai.arena import RefereeGame
from brass_ai.replay_archive import Archive, Recorder, digest, outcome
from brass_search import SearchConfig, SearchAgent
from brass_search.identity import player_manifest, search_identity
from .agent import TorchAgent
from .artifacts import capture_rng_states, restore_rng_states
from .evaluate import cluster_summary
from .policy_io import ROOT, inspect_policy, load_policy
from .replay import atomic_json
from .seeding import SeedStreams
from .search_budget import RunInvocation

INVENTORY_ROOT = ROOT/'training/runs'
# Reserve room for a complete bounded 2000-decision native journal and reports.
GAME_STORAGE_RESERVE = 1024*1024


def native_binary_sha256():
    return hashlib.sha256(Path(brass.brass.__file__).read_bytes()).hexdigest()


def runtime_manifest():
    return dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                platform=platform.platform(), machine=platform.machine(), device='cpu',
                threads=1, interop_threads=1,
                native_binary_sha256=native_binary_sha256())


def controller_source():
    names = ['training/brass_rl/'+name+'.py' for name in
             ('search_evaluate', 'search_budget', '__main__', 'seeding', 'replay', 'evaluate', 'artifacts')]
    names += ['engine/tools/brass_ai/'+name+'.py' for name in ('arena', 'replay_archive', 'events', 'seeds')]
    return {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names}


def declared_seeds(root):
    """Inventory explicit local JSON declarations, never deserialize private tensors.

    Includes recorded deal/initial seeds, evaluation suite lists and declared
    training ranges when a master and next_game_id are present in JSON metadata.
    Invalid JSON declarations fail closed rather than silently losing coverage.
    """
    seeds, sources = set(), []
    root = Path(root)
    if not root.exists():
        return seeds, sources
    def visit(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in ('deal_seed', 'seed') and type(item) is int and 0 <= item < 2**64:
                    seeds.add(item)
                elif key == 'deal_seeds' and isinstance(item, list):
                    seeds.update(v for v in item if type(v) is int and 0 <= v < 2**64)
                visit(item)
            master, count = value.get('master_seed'), value.get('next_game_id')
            if type(master) is int and 'games_per_update' in value and 'updates' in value:
                declared = value['games_per_update']*value['updates']
                if type(declared) is not int or not 0 <= declared <= 1000000:
                    raise ValueError('declared training seed inventory range too large')
                count = max(count or 0, declared)
                eval_count = value.get('eval', {}).get('deals', 0)
                if type(eval_count) is not int or not 0 <= eval_count <= 1000000:
                    raise ValueError('declared evaluation seed inventory range too large')
                streams = SeedStreams(master)
                seeds.update(streams.seed(domain, n) for domain in ('dev.deal', 'final.deal') for n in range(eval_count))
            if type(master) is int and type(count) is int and 0 <= count <= 1000000:
                streams = SeedStreams(master)
                seeds.update(streams.seed('train.deal', n) for n in range(count))
        elif isinstance(value, list):
            for item in value:
                visit(item)
    paths = set(root.rglob('*.json')) | set(root.rglob('*.json.gz')) | set(root.rglob('*.jsonl'))
    for path in sorted(paths):
        # GUI exports, durable journals and unified archives are declarations too.
        raw = path.read_bytes()
        if path.suffix == '.jsonl':
            for line in raw.splitlines(keepends=True):
                if line.endswith(b'\n'):
                    visit(json.loads(line))  # ignore only a torn final append
        else:
            visit(json.loads(gzip.decompress(raw) if path.suffix == '.gz' else raw))
        sources.append((str(path.relative_to(root)), hashlib.sha256(raw).hexdigest()))
    return seeds, sources


def prepare_suite(checkpoint, config, *, master_seed, deals=1, wall_seconds=1800,
                  disk_limit_bytes=268435456, frozen_inventory=None, category='evaluation'):
    if not isinstance(config, SearchConfig) or config.wall_time_seconds is not None:
        raise ValueError('fixed-work evaluation requires SearchConfig without per-move wall cutoff')
    if category not in ('evaluation', 'validation'):
        raise ValueError('category must be evaluation or validation')
    streams = SeedStreams(master_seed)
    if type(deals) is not int or not 1 <= deals <= 100000:
        raise ValueError('deals must be positive')
    if type(wall_seconds) not in (int, float) or not math.isfinite(wall_seconds) or wall_seconds <= 0:
        raise ValueError('wall_seconds must be finite and positive')
    if type(disk_limit_bytes) is not int or disk_limit_bytes < 1:
        raise ValueError('disk_limit_bytes must be positive')
    metadata = inspect_policy(Path(checkpoint))
    baseline, search = player_manifest(metadata), player_manifest(metadata, config)
    if frozen_inventory is None:
        excluded, local_sources = declared_seeds(INVENTORY_ROOT)
        archive_seeds, archive_sources = declared_seeds(Path(os.environ.get('BRASS_REPLAY_ROOT', ROOT/'replays'))/'games')
        excluded.update(archive_seeds)
        sources = [('training/'+name, sha) for name, sha in local_sources]
        sources += [('archive/'+name, sha) for name, sha in archive_sources]
        deal_seeds, nonce = [], 0
        while len(deal_seeds) < deals:
            seed = streams.seed('search.dev.deal', nonce)
            nonce += 1
            if seed not in excluded and seed not in deal_seeds:
                deal_seeds.append(seed)
        inventory = {'count': len(excluded), 'source_count': len(sources),
                     'sha256': digest(sources), 'seed_sha256': digest(sorted(excluded)),
                     'last_nonce': nonce, 'scope': 'local training/runs and unified archive games: JSON, gzip JSON and complete JSONL declarations'}
    else:
        inventory, deal_seeds = frozen_inventory
    games = []
    for deal, seed in enumerate(deal_seeds):
        for rotation in range(4):
            sampling = [streams.seed('search.real.sample', deal, rotation, seat) for seat in range(4)]
            for treatment in ('search', 'control'):
                labels = [baseline['id']]*4
                if treatment == 'search':
                    labels[rotation] = search['id']
                games.append(dict(game=len(games), deal_id=deal, deal_seed=seed, rotation=rotation,
                                  candidate_seat=rotation, treatment=treatment, labels=labels, sampling=sampling))
    suite = dict(format='brass-search-eval/1', master_seed=master_seed, deals=deals, category=category,
                 rotations=4, mode='sample', baseline=baseline, search=search,
                 deal_seeds=deal_seeds, seed_inventory=inventory, games=games,
                 domains={'deal': 'search.dev.deal', 'sampling': 'search.real.sample',
                          'search': 'player-private domain-separated streams from supplied sampling seed'},
                 budget={'games': deals*8, 'wall_seconds': float(wall_seconds), 'disk_limit_bytes': disk_limit_bytes},
                 controller_source=controller_source(), runtime=runtime_manifest())
    suite['suite_id'] = digest(suite)
    return suite


def _seats(spec):
    return [{'kind': 'bot', 'name': 'search' if label.startswith('search-') else 'plain',
             'bot': label, 'bot_family': 'ai'} for label in spec['labels']]


def _provenance(suite, spec):
    # Do not set update to the base's number: dashboard must not infer wrapper update.
    return dict(category=suite.get('category', 'evaluation'), interface='search-eval', experiment=suite['suite_id'],
                suite_id=suite['suite_id'], game=spec['game'], deal=spec['deal_id'],
                rotation=spec['rotation'], treatment=spec['treatment'], candidate_seat=spec['candidate_seat'],
                candidate_id=spec['labels'][spec['candidate_seat']],
                search=suite['search'], baseline=suite['baseline'])


def replay_id(suite, spec):
    return digest({'suite_id': suite['suite_id'], 'game': spec['game']})


def _replay_native(record):
    """Native actions/outcomes only; no model or search call and no invented cost."""
    game = brass.Game(record['initial']['seed'])
    for action in record['moves']:
        if game.is_over() or action['player'] != game.current_player():
            raise ValueError('archive actor/outcome mismatch')
        game.apply(json.dumps(action['move']))
    missing_outcome = record.get('result') is None
    if (missing_outcome and record.get('status') not in ('in-progress', 'interrupted')) or (
            not missing_outcome and (not game.is_over() or outcome(game) != record['result'])):
        raise ValueError('archive outcome mismatch')
    return game


def recover_record(record):
    game = _replay_native(record)
    if not game.is_over():
        raise ValueError('archive outcome is nonterminal')
    return dict(vp=list(game.vp()), ranking=list(game.ranking()), plies=len(record['moves']),
                telemetry=None, recovered=True)


def _read_record(archive, suite, spec):
    identity = replay_id(suite, spec)
    try:
        record = archive.read(identity)
    except FileNotFoundError:
        return None
    if record.get('game_id') != identity:
        raise ValueError('immutable archive game ID mismatch')
    if (record['initial'] != {'seed': spec['deal_seed']} or record['seats'] != _seats(spec)
            or record['provenance'] != _provenance(suite, spec)
            or record['engine_id'] != suite['baseline']['engine_id']):
        raise ValueError('immutable archive manifest mismatch')
    return record


def _lineage(suite, spec):
    return dict(spec, suite_id=suite['suite_id'], archive_id=replay_id(suite, spec))


def _load_rows(out, suite, archive):
    rows = []
    for spec in suite['games']:
        path = out/'games'/('%04d.json' % spec['game'])
        record = _read_record(archive, suite, spec)
        if record is not None and record['status'] != 'completed':
            native = _replay_native(record)
            if native.is_over():
                recorder = Recorder(archive, seed=spec['deal_seed'], seats=_seats(spec),
                                    provenance=_provenance(suite, spec), game_id=replay_id(suite, spec))
                try:
                    for move in record['moves']:
                        recorder.append(move['player'], move['move'])
                    recorder.finish(native)
                finally:
                    recorder.interrupt()
                record = _read_record(archive, suite, spec)
        if path.exists():
            row = json.loads(path.read_text())
            if any(row.get(k) != v for k, v in _lineage(suite, spec).items()):
                raise ValueError('immutable completed row manifest mismatch')
            if record is None or record['status'] != 'completed':
                raise ValueError('completed row lacks completed archive')
            verified = recover_record(record)
            if any(row.get(k) != verified[k] for k in ('vp', 'ranking', 'plies')):
                raise ValueError('completed row outcome mismatch')
            rows.append(row)
        elif record is not None and record['status'] == 'completed':
            row = dict(_lineage(suite, spec), **recover_record(record))
            atomic_json(path, row)
            rows.append(row)
    return rows


def storage_bytes(out, archive, suite):
    paths = set(p for p in out.rglob('*') if p.is_file())
    for spec in suite['games']:
        identity = replay_id(suite, spec)
        paths.update(archive.games.glob(identity+'.*'))
        lock = archive.root/'locks'/identity
        if lock.is_file():
            paths.add(lock)
    return sum(path.stat().st_size for path in paths)


def _play(loaded, config, suite, spec, archive, should_stop):
    model, components, metadata = loaded
    agents = [SearchAgent(model, components, spec['sampling'][seat], config, base_policy_id=metadata.id)
              if label == suite['search']['id'] else TorchAgent(model, components, spec['sampling'][seat])
              for seat, label in enumerate(spec['labels'])]
    recorder = Recorder(archive, seed=spec['deal_seed'], seats=_seats(spec),
                        provenance=_provenance(suite, spec), game_id=replay_id(suite, spec))
    latencies, diagnostics = [], []
    applied = 0
    started = time.monotonic()
    existing_plies = len(recorder.record['moves'])
    def telemetry():
        real_memory_calls = applied*(7 if spec['treatment'] == 'search' else 4)
        result = dict(seconds=time.monotonic()-started, decision_latencies=latencies,
                      search_decisions=diagnostics, real_memory_update_calls=real_memory_calls,
                      plain_policy_forward_calls=len(latencies)-len(diagnostics),
                      reconstructed_decisions=min(existing_plies, applied), unapplied_decisions=len(latencies)-applied,
                      peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss *
                      (1 if platform.system() == 'Darwin' else 1024))
        result['whole_agent_inference_calls'] = (real_memory_calls+result['plain_policy_forward_calls']+
                                                sum(d['inference_calls'] for d in diagnostics))
        return result
    try:
        referee = RefereeGame(spec['deal_seed'], agents, spec['labels'], recorder=recorder)
        while not referee.is_over():
            if should_stop():
                recorder.interrupt()
                return {'partial': True, 'telemetry': telemetry()}
            seat, obs = referee.pending()
            before = time.monotonic()
            index = agents[seat].act(obs)
            latencies.append({'seat': seat, 'seconds': time.monotonic()-before})
            if isinstance(agents[seat], SearchAgent):
                diagnostics.append(dict(agents[seat].last_diagnostics))
            # A stop arriving during act cannot publish its pending choice. This
            # boundary is after a complete bounded act (not individual forwards).
            if should_stop():
                recorder.interrupt()
                return {'partial': True, 'telemetry': telemetry()}
            referee.apply_index(index)
            applied += 1
        result = referee.result()
        return dict(vp=list(result.vp), ranking=list(result.ranking), plies=result.moves,
                    telemetry=telemetry(), recovered=False)
    finally:
        recorder.interrupt()


def _metrics(row):
    seat = row['candidate_seat']
    return dict(win=float(row['ranking'][seat] == 0 and row['ranking'].count(0) == 1),
                rank=row['ranking'][seat]+1, vp=row['vp'][seat],
                vp_margin=row['vp'][seat]-max(v for s, v in enumerate(row['vp']) if s != seat))


def build_report(out, suite, rows, archive, reason, *, budget_snapshot=None, inspection=None):
    pairs = []
    indexed = {(r['deal_id'], r['rotation'], r['treatment']): r for r in rows}
    for deal in range(suite['deals']):
        for seat in range(4):
            a, b = indexed.get((deal, seat, 'search')), indexed.get((deal, seat, 'control'))
            if a is not None and b is not None:
                am, bm = _metrics(a), _metrics(b)
                pairs.append(dict(deal_id=deal, rotation=seat, search=am, control=bm,
                                  delta={k: am[k]-bm[k] for k in am}))
    full_deals = [d for d in range(suite['deals']) if sum(p['deal_id'] == d for p in pairs) == 4]
    summaries = {}
    for treatment in ('search', 'control'):
        complete = [dict(r, lineup=treatment) for r in rows if r['deal_id'] in full_deals and r['treatment'] == treatment]
        if complete:
            label = suite['search' if treatment == 'search' else 'baseline']['id']
            summaries[treatment] = asdict(cluster_summary(complete, label, SeedStreams(suite['master_seed']).seed('search.bootstrap')))
            if len(full_deals) < 2:
                for field in ('win_ci', 'rank_ci', 'margin_ci'):
                    summaries[treatment][field] = None
    paired = None
    if full_deals:
        means = np.array([[np.mean([p['delta'][k] for p in pairs if p['deal_id'] == d])
                          for k in ('win','rank','vp','vp_margin')] for d in full_deals])
        import random
        rng = random.Random(SeedStreams(suite['master_seed']).seed('search.paired.bootstrap'))
        draws = np.array([means[[rng.randrange(len(means)) for _ in means]].mean(axis=0) for _ in range(2000)])
        paired = {k: {'mean': float(means[:, i].mean()),
                      'ci95': [float(x) for x in np.percentile(draws[:, i], [2.5,97.5])] if len(full_deals)>1 else None}
                  for i,k in enumerate(('win','rank','vp','vp_margin'))}
    completed_measured = [r['telemetry'] for r in rows if r['telemetry'] is not None]
    attempts = [json.loads(p.read_text()) for p in sorted((out/'attempts').glob('*.json'))]
    if any(a['suite_id'] != suite['suite_id'] for a in attempts):
        raise ValueError('immutable attempt manifest mismatch')
    completed_attempts = {(r['game'], r['attempt_id']) for r in rows
                          if r['telemetry'] is not None and r.get('attempt_id')}
    # Row publication and marker cleanup are separate durable operations. Only
    # the exact measured execution named by a completed row supersedes a marker.
    reconciled_markers = sum((a['game'], a.get('attempt_id')) in completed_attempts for a in attempts)
    attempts = [a for a in attempts if (a['game'], a.get('attempt_id')) not in completed_attempts]
    known_attempts = [a for a in attempts if a['telemetry'] is not None]
    unknown_attempts = [a for a in attempts if a['telemetry'] is None]
    measured = completed_measured+[a['telemetry'] for a in known_attempts]
    partial_games = []
    for spec in suite['games']:
        record = _read_record(archive, suite, spec)
        if record is not None and record['status'] != 'completed':
            partial_games.append(dict(game=spec['game'], archive_id=record['game_id'], plies=len(record['moves']),
                                      status=record['status'], treatment=spec['treatment']))
    latency = [d['seconds'] for t in measured for d in t['decision_latencies']]
    search = [d for t in measured for d in t['search_decisions']]
    budget = json.loads((out/'budget.json').read_text()) if (out/'budget.json').exists() else None
    if budget_snapshot is not None:
        budget = budget_snapshot
    report = dict(format='brass-search-report/1', suite_id=suite['suite_id'], base_id=suite['baseline']['id'],
                  search_id=suite['search']['id'], run_path=str(out.resolve()), replay_root=str(archive.root.resolve()),
                  completed_games=len(rows), remaining_games=len(suite['games'])-len(rows),
                  stop_reason=reason, results=rows, partial_games=partial_games, pairs=pairs, paired_difference=paired,
                  independent_complete_deals=len(full_deals), summaries=summaries,
                  strength_gate='pending: one deal is insufficient; Stage 2 and Stage 3 held-out gates remain pending',
                  budget=budget, disk_bytes=storage_bytes(out, archive, suite),
                  costs={'telemetry_games': len(completed_measured), 'missing_telemetry_games': len(rows)-len(completed_measured),
                         'partial_attempts': len(attempts),
                         'partial_decisions': sum(len(a['telemetry']['decision_latencies']) for a in known_attempts),
                         'reconstructed_decisions': sum(t['reconstructed_decisions'] for t in measured),
                         'unapplied_decisions': sum(t['unapplied_decisions'] for t in measured),
                         'decision_latency_seconds': None if not latency else dict(median=float(np.median(latency)),
                             p95=float(np.percentile(latency,95)), max=max(latency)),
                         'search_decisions': len(search), 'fallbacks': sum(d['fallback'] for d in search),
                         'reasons': dict(Counter(d['reason'] for d in search)),
                         'whole_agent_inference_calls': sum(t['whole_agent_inference_calls'] for t in measured),
                         'real_memory_update_calls': sum(t['real_memory_update_calls'] for t in measured),
                         'native_simulated_applies': sum(d['native_applies'] for d in search),
                         'completed_rounds': sum(d['completed_rounds'] for d in search),
                         'mean_root_fraction': float(np.mean([d['selected_root_count']/d['legal_count'] for d in search])) if search else None,
                         'max_live_worlds': max((d['max_live_worlds'] for d in search), default=None),
                         'search_decision_latency_seconds': None if not search else dict(
                             median=float(np.median([d['elapsed_seconds'] for d in search])),
                             p95=float(np.percentile([d['elapsed_seconds'] for d in search],95)),
                             max=max(d['elapsed_seconds'] for d in search)),
                         'mean_selected_root_prior_mass': float(np.mean([d['selected_root_prior_mass'] for d in search])) if search else None,
                         'peak_rss_bytes': max((t['peak_rss_bytes'] for t in measured), default=None),
                         'games_per_controller_hour': len(rows)*3600/budget['measured_seconds'] if budget and budget['measured_seconds'] else None,
                         'coverage': 'completed and clean interrupted attempts; reconstructed decisions counted as work; unclean-crash work unknown but wall reservation exhausted'},
                  limitations=['uniform observation-consistent worlds are not a strategic posterior',
                               'sampled opponent-perspective value calibration and strategy fusion remain approximate',
                               'controller checks before and after a complete bounded act and between native decisions; an act or apply/broadcast may overrun'])
    costs = report['costs']
    costs['measured_decisions'] = len(latency)
    costs['measured_attempts'] = len(measured)
    costs['reconciled_completed_markers'] = reconciled_markers
    costs['unmeasured_attempts'] = len(unknown_attempts)
    costs['unmeasured_interrupted_games'] = sum(
        not any(a['game'] == p['game'] for a in known_attempts) for p in partial_games)
    unclean = bool(budget and budget.get('reservation') is not None and reason != 'in_progress')
    costs['unclean_attempt_work_unknown'] = unclean
    unknown = bool(costs['missing_telemetry_games'] or costs['unmeasured_interrupted_games'] or unknown_attempts or unclean)
    count_fields = ('search_decisions', 'fallbacks', 'whole_agent_inference_calls', 'real_memory_update_calls',
                    'native_simulated_applies', 'completed_rounds', 'partial_decisions',
                    'reconstructed_decisions', 'unapplied_decisions', 'peak_rss_bytes', 'max_live_worlds')
    costs['measured_work'] = {k: costs[k] for k in count_fields}
    costs['measured_work']['reasons'] = costs['reasons']
    costs['statistics_scope'] = 'latency and candidate coverage describe measured decisions only'
    costs['whole_run_work_complete'] = not unknown
    if unknown:
        costs['reasons'] = None
        for key in count_fields:
            costs[key] = None
    if unknown or not budget or budget.get('reservation') is not None:
        costs['games_per_controller_hour'] = None
    report['budget_authority'] = 'budget.json; report snapshot precedes final durable refund'
    if inspection is not None:
        report['inspection'] = inspection
    atomic_json(out/'report.json', report)
    return report


def _validate_suite(suite):
    payload = dict(suite)
    identity = payload.pop('suite_id')
    if suite.get('format') != 'brass-search-eval/1' or digest(payload) != identity:
        raise ValueError('immutable suite digest mismatch')


def _validate_resume(suite, checkpoint, config, master_seed, deals, wall_seconds, disk_limit_bytes, category):
    expected = {'master_seed': master_seed, 'deals': deals,
                'budget': {'games': deals*8, 'wall_seconds': float(wall_seconds), 'disk_limit_bytes': disk_limit_bytes}}
    if (any(suite[k] != v for k,v in expected.items()) or suite.get('category', 'evaluation') != category or
            suite['search']['config'] != config.canonical_mapping() or
            suite['controller_source'] != controller_source() or suite['runtime'] != runtime_manifest() or
            suite['search']['id'] != search_identity(suite['baseline']['id'], suite['baseline']['schema_id'], config,
                                                    engine_id=suite['baseline']['engine_id'])):
        raise ValueError('immutable suite/config/source/runtime mismatch; cannot resume changed comparison')
    for name, expected_hash in suite['baseline']['policy_content_hashes'].items():
        if hashlib.sha256((Path(checkpoint)/name).read_bytes()).hexdigest() != expected_hash:
            raise ValueError('immutable policy bytes differ from suite')


def _inspection_reason(out, suite, rows):
    path = out/'report.json'
    old = json.loads(path.read_text()) if path.exists() else {}
    if (out/'failure.json').exists():
        return 'correctness_failure'
    budget = json.loads((out/'budget.json').read_text()) if (out/'budget.json').exists() else None
    if budget and budget.get('reservation') is not None:
        return 'unclean_exit'
    reason = old.get('stop_reason')
    if reason not in (None, 'report_only', 'in_progress'):
        return reason
    if len(rows) == len(suite['games']):
        return 'complete'
    return 'wall_limit' if budget and budget['remaining_seconds'] <= 0 else 'interrupted'


def evaluate_search(checkpoint, config, out, *, master_seed, deals=1, wall_seconds=1800,
                    disk_limit_bytes=268435456, stop_requested=None, report_only=False,
                    category='evaluation', announce=False, _invocation_started=None, _invocation=None):
    invocation_started = time.monotonic() if _invocation_started is None else _invocation_started
    out = Path(out)
    if report_only and not (out/'suite.json').exists():
        raise ValueError('report-only requires an existing suite')
    if not report_only:
        if not isinstance(config, SearchConfig) or config.wall_time_seconds is not None:
            raise ValueError('fixed-work evaluation requires SearchConfig without per-move wall cutoff')
        SeedStreams(master_seed)
        if type(deals) is not int or not 1 <= deals <= 100000:
            raise ValueError('deals must be positive')
        if type(disk_limit_bytes) is not int or disk_limit_bytes < 1:
            raise ValueError('disk_limit_bytes must be positive')
        if category not in ('evaluation', 'validation'):
            raise ValueError('category must be evaluation or validation')
    context = (nullcontext(_invocation) if _invocation is not None else
               RunInvocation(out, wall_seconds, report_only=report_only, started=invocation_started))
    with context as invocation:
        path = out/'suite.json'
        old = json.loads(path.read_text()) if path.exists() else None
        if old is not None:
            _validate_suite(old)
        archive = Archive()
        operational = out/'run.json'
        if old is not None and (not operational.exists() or
                json.loads(operational.read_text())['archive_root'] != str(archive.root.resolve())):
            raise ValueError('immutable archive location mismatch')
        if report_only:
            # Inspection deliberately uses the original manifest. No policy file,
            # configuration or gameplay source compatibility is assumed or granted.
            from brass_ai.replay_archive import engine_id
            if old['baseline']['engine_id'] != engine_id():
                raise ValueError('native engine incompatible with recorded suite')
            if old['runtime']['native_binary_sha256'] != native_binary_sha256():
                raise ValueError('native extension binary differs from recorded suite')
            rows = _load_rows(out, old, archive)
            reason = _inspection_reason(out, old, rows)
            return build_report(out, old, rows, archive, reason,
                                inspection={'mode': 'report_only', 'gameplay_compatibility': 'not_granted',
                                            'current_controller_source': controller_source()})
        budget = invocation.budget
        if budget.remaining <= 0:
            # Never clear evidence of an unaccounted crash by reserving zero.
            _validate_resume(old, checkpoint, config, master_seed, deals, wall_seconds, disk_limit_bytes, category)
            rows = _load_rows(out, old, archive)
            return build_report(out, old, rows, archive,
                                'complete' if len(rows) == len(old['games']) else 'wall_limit')
        before_rng = None
        previous_threads = torch.get_num_threads()
        suite, rows, reason, report = old, [], None, None
        try:
            before_rng = capture_rng_states()
            torch.set_num_threads(1)
            if torch.get_num_interop_threads() != 1:
                torch.set_num_interop_threads(1)
            if old is None:
                suite = prepare_suite(checkpoint, config, master_seed=master_seed, deals=deals,
                                      wall_seconds=wall_seconds, disk_limit_bytes=disk_limit_bytes, category=category)
                atomic_json(path, suite)
                revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
                atomic_json(operational, {'archive_root': str(archive.root.resolve()), 'source_revision': revision})
            else:
                _validate_resume(suite, checkpoint, config, master_seed, deals, wall_seconds, disk_limit_bytes, category)
            if announce:
                print(json.dumps({'base_id': suite['baseline']['id'], 'search': suite['search'],
                                  'master_seed': suite['master_seed'], 'run_path': str(out.resolve())}), flush=True)
            try:
                rows = _load_rows(out, suite, archive)
                if len(rows) == len(suite['games']):
                    reason = 'complete'
                elif (out/'failure.json').exists():
                    reason = 'correctness_failure'
                def should_stop():
                    nonlocal reason
                    if stop_requested is not None and stop_requested():
                        reason = 'requested'
                    elif budget.exhausted():
                        reason = 'wall_limit'
                    elif storage_bytes(out, archive, suite) >= disk_limit_bytes-65536:
                        reason = 'disk_limit'
                    return reason is not None
                completed = {r['game'] for r in rows}
                loaded = None
                for spec in suite['games']:
                    if reason is not None:
                        break
                    if spec['game'] in completed:
                        continue
                    if should_stop():
                        break
                    if storage_bytes(out, archive, suite)+GAME_STORAGE_RESERVE >= disk_limit_bytes:
                        reason = 'disk_limit'
                        break
                    if loaded is None:
                        loaded = load_policy(Path(checkpoint))
                        if (player_manifest(loaded[2]) != suite['baseline'] or
                                player_manifest(loaded[2], config) != suite['search']):
                            raise ValueError('loaded policy identity/content hashes differ from immutable suite')
                    count = len(list((out/'attempts').glob('*.json')))
                    attempt_path = out/'attempts'/('%06d.json' % count)
                    attempt_id = digest({'suite_id': suite['suite_id'], 'game': spec['game'], 'attempt': count})
                    # Durable before gameplay: exceptions and write failures cannot
                    # make a later failed retry disappear behind an older known one.
                    atomic_json(attempt_path, dict(suite_id=suite['suite_id'], game=spec['game'],
                                                   attempt_id=attempt_id, partial=True, telemetry=None))
                    result = _play(loaded, config, suite, spec, archive, should_stop)
                    if result.get('partial'):
                        atomic_json(attempt_path,
                                    dict(suite_id=suite['suite_id'], game=spec['game'], attempt_id=attempt_id, **result))
                        break
                    row = dict(_lineage(suite, spec), attempt_id=attempt_id, **result)
                    atomic_json(out/'games'/('%04d.json' % spec['game']), row)
                    rows.append(row)
                    attempt_path.unlink()  # completed row now owns this attempt's measured telemetry
                    build_report(out, suite, rows, archive, 'in_progress')
                if reason is None:
                    reason = 'complete'
            except Exception as exc:
                # Caller input and resume validation are outside this boundary.
                # Only native recovery/gameplay/publication failures poison play.
                atomic_json(out/'failure.json', {'type': type(exc).__name__, 'message': str(exc)})
                reason = 'correctness_failure'
                raise
        finally:
            try:
                torch.set_num_threads(previous_threads)
                if before_rng is not None:
                    restore_rng_states(before_rng)
                if suite is not None and reason is not None:
                    report = build_report(out, suite, rows, archive, reason, budget_snapshot=budget.snapshot())
                # The invocation owner refunds after report/recovery/RNG and
                # (for CLI callers) final stdout bookkeeping has completed.
                invocation.report = report
            except BaseException:
                raise
        return report
