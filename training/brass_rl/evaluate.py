"""Unseen immutable evaluation suites, incremental games and deal-cluster intervals.

Only the controller sees deal seeds. Policy seats get observations/public events.
Directories are single-writer, just like training's completed-update journal.
"""
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
import time
from typing import Mapping, Tuple

import brass
import numpy as np
import torch
from brass_ai.arena import RefereeGame
from brass_ai.replay_archive import Recorder, labeled_seats, recording_context, recording_metadata, digest
from brass_ai.fingerprints import engine_fingerprint
from .agent import TorchAgent
from .artifacts import capture_rng_states, restore_rng_states, export_model_policy
from .policy_io import ROOT, inspect_policy, json_bytes, load_policy
from .replay import atomic_json, retained_ids, select_replays, write_gui_replay
from .seeding import SeedStreams
from .returns import strict_outcomes


def _hash(payload):
    return hashlib.sha256(json_bytes(payload)).hexdigest()


def _opponent_provenance(root: Path):
    """Conservative source provenance, not attestation of a future native binary.

    Fixed native constructors below bypass Stage 1's mutable registry. Include
    their Rust implementations/bindings plus the Python referee and factory;
    paths and bytes are length-framed so additions/removals also change identity.
    """
    root = Path(root)
    names = {'players/baselines/Cargo.toml', 'players/baselines/src/lib.rs', 'players/baselines/src/random.rs', 'players/baselines/src/greedy.rs',
             'engine/bindings/Cargo.toml', 'engine/bindings/src/lib.rs', 'engine/tools/brass_ai/arena.py', 'engine/tools/brass_ai/events.py',
             'engine/tools/brass_ai/seeds.py', 'engine/tools/brass_ai/agents.py', 'engine/tools/brass_ai/fingerprints.py',
             'training/brass_rl/evaluate.py'}
    try:
        for directory in ('players/baselines/src', 'engine/bindings/src', 'engine/core/src'):
            sources = list((root/directory).rglob('*.rs'))
            if not sources:
                raise ValueError('opponent source directory unavailable: '+directory)
            if directory != 'engine/core/src':
                names.update(path.relative_to(root).as_posix() for path in sources)
        engine = engine_fingerprint(root)
        hasher = hashlib.sha256(b'brass-native-baselines-source-v1\0')
        for name in sorted(names):
            for value in (name.encode('utf-8'), (root/name).read_bytes()):
                hasher.update(len(value).to_bytes(8, 'little'))
                hasher.update(value)
        return {'engine_fingerprint': engine, 'baseline_source_fingerprint':
                'native-baselines-sha256-v1:'+hasher.hexdigest(), 'source_files': sorted(names)}
    except OSError as exc:
        raise ValueError('opponent source unavailable for immutable evaluation provenance: '+str(exc)) from exc


def _spec(spec):
    if isinstance(spec, str) and spec in ('candidate', 'random', 'greedy'):
        return spec
    if isinstance(spec, Mapping) and set(spec) == {'id', 'path'} and isinstance(spec['id'], str) and spec['id']:
        return {'id': spec['id'], 'path': str(Path(spec['path']).resolve())}
    raise ValueError('opponents must be random/greedy or immutable {id,path} policy references')


@dataclass(frozen=True)
class EvalSuite:
    stream: str
    master_seed: int
    deals: int
    lineups: Mapping
    candidate_id: str
    replay_limit: int = 10
    outcomes_only: bool = False

    def __post_init__(self):
        if type(self.outcomes_only) is not bool:
            raise ValueError("outcomes_only must be boolean")
        if self.stream not in ('dev', 'final'):
            raise ValueError('evaluation stream must be dev or final')
        SeedStreams(self.master_seed)
        if type(self.deals) is not int or not 1 <= self.deals <= 100000:
            raise ValueError('evaluation deals must be in 1..100000')
        if not isinstance(self.candidate_id, str) or not self.candidate_id:
            raise ValueError('evaluation candidate must be preselected by policy ID')
        if type(self.replay_limit) is not int or not 0 <= self.replay_limit <= 100:
            raise ValueError('evaluation replay_limit must be in 0..100')
        if not isinstance(self.lineups, Mapping) or not self.lineups:
            raise ValueError('evaluation needs named lineups')
        normalized = {}
        for label, entries in self.lineups.items():
            if not isinstance(label, str) or not label or len(entries) != 4:
                raise ValueError('each named lineup must have four seats')
            entries = tuple(_spec(e) for e in entries)
            if sum(e == 'candidate' for e in entries) != 1:
                raise ValueError('each evaluation lineup needs exactly one candidate')
            normalized[label] = entries
        object.__setattr__(self, 'lineups', normalized)

    def manifest(self):
        # Paths are operational; immutable policy identities define comparisons.
        lineups = {name: [e if isinstance(e, str) else {'id': e['id']} for e in entries]
                   for name, entries in self.lineups.items()}
        streams = SeedStreams(self.master_seed)
        payload = {'format': 'brass-eval-suite/1', 'stream': self.stream, 'master_seed': self.master_seed,
                   'deals': self.deals, 'lineups': lineups,
                   'deal_seeds': [streams.seed(self.stream+'.deal', d) for d in range(self.deals)],
                   'mode': 'sample', 'rotations': 4, 'opponent_provenance': _opponent_provenance(ROOT)}
        if self.outcomes_only:
            payload['outcomes_only'] = True
        if self.stream == 'final':
            payload['candidate_id'] = self.candidate_id
        return dict(payload, suite_id=_hash(payload))


@dataclass(frozen=True)
class ClusterSummary:
    win_rate: float
    win_ci: Tuple[float, float]
    mean_rank: float
    rank_ci: Tuple[float, float]
    vp_margin: float
    margin_ci: Tuple[float, float]
    tied_first: int
    seat_games: int
    independent_deals: int


@dataclass(frozen=True)
class EvalReport:
    candidate_id: str
    suite_id: str
    results: Tuple[Mapping, ...]
    summaries: Mapping[str, ClusterSummary]
    exemplars: Mapping[str, int]
    strength_gate: str


def cluster_summary(results, candidate_label: str, bootstrap_seed: int, samples: int = 2000) -> ClusterSummary:
    """Percentile bootstrap over whole independent deal blocks, never rotations."""
    if type(samples) is not int or samples < 1:
        raise ValueError('bootstrap samples must be positive')
    clusters, rotations = defaultdict(list), defaultdict(list)
    for row in results:
        # Explicit focal seat supports repeated baseline labels without merging seats.
        seats = [i for i, label in enumerate(row['labels']) if label == candidate_label]
        seat = row.get('candidate_seat', seats[0] if len(seats) == 1 else None)
        if type(seat) is not int or seat not in seats:
            raise ValueError('each result must identify its labeled candidate seat')
        rank = row['ranking'][seat]
        first = rank == 0
        unique = row['ranking'].count(0) == 1
        values = (float(first and unique), rank+1., row['vp'][seat]-max(v for i,v in enumerate(row['vp']) if i != seat),
                  float(first and not unique))
        clusters[row['deal_id']].append(values)
        rotations[(row['deal_id'],row['lineup'])].append(row['rotation'])
    if not clusters or any(sorted(r) != list(range(4)) for r in rotations.values()):
        raise ValueError('results need all four unique rotations per deal/lineup')
    if len({len(v) for v in clusters.values()}) != 1:
        raise ValueError('deal clusters must contain the same lineup rotations')
    means = np.array([np.mean(clusters[k],axis=0) for k in sorted(clusters)])
    rng = random.Random(bootstrap_seed)
    draws = np.array([means[[rng.randrange(len(means)) for _ in means]].mean(axis=0)[:3] for _ in range(samples)])
    low,high = np.percentile(draws,[2.5,97.5],axis=0)
    point=means.mean(axis=0)
    return ClusterSummary(float(point[0]),(float(low[0]),float(high[0])),float(point[1]),
                          (float(low[1]),float(high[1])),float(point[2]),(float(low[2]),float(high[2])),
                          int(sum(v[3] for rows in clusters.values() for v in rows)),
                          sum(len(v) for v in clusters.values()),len(clusters))


class RecordingCandidate(TorchAgent):
    """Observe the successful sampled Choice, using the usual act/choose code."""
    def reset(self, seat, initial_public_json):
        super().reset(seat, initial_public_json)
        self.samples = []

    def choose(self, output, row):
        choice = super().choose(output, row)
        self.samples.append({'ply': self.ply, 'era': self._public['era'],
                             'round': self._public['round'], 'values': list(choice.values)})
        return choice


def terminal_brier(samples, ranking, seat, plies):
    """Per-game means over candidate decisions and four strict-win indicators."""
    target = np.roll(strict_outcomes(ranking), -seat)
    scores = {'outcome_relative': target.tolist()}
    for prefix, rows in (('', samples), ('early_', [s for s in samples if 2*s['ply'] < plies]),
                         ('canal_opening_', [s for s in samples if s['era'] == 'canal' and s['round'] <= 4]),
                         ('first_', samples[:1])):
        scores[prefix+'decisions'] = len(rows)
        errors = [(np.asarray(row['values'], dtype=float) - target)**2 for row in rows]
        scores[prefix+'brier'] = float(np.mean(errors)) if rows else None
        scores[prefix+'self_brier'] = float(np.mean([e[0] for e in errors])) if rows else None
    return scores


def _play_game(candidate, specs, seed, sampling):
    agents, labels = [], []
    trace_values = not recording_metadata().get("outcomes_only", False)
    for seat, spec in enumerate(specs):
        if spec == 'candidate':
            model, components, metadata = candidate
            agent = RecordingCandidate if trace_values else TorchAgent
            agents.append(agent(model,components,sampling[seat]))
            labels.append(metadata.id)
        elif isinstance(spec,str):
            agents.append(brass.RandomBot(sampling[seat]) if spec == 'random' else brass.GreedyBot(sampling[seat]))
            labels.append(spec)
        else:
            model, components, metadata = spec
            agents.append(TorchAgent(model,components,sampling[seat]))
            labels.append(metadata.id)
    provenance = recording_metadata()
    provenance.setdefault('category', 'evaluation')
    provenance.setdefault('experiment', digest({'labels': labels, 'sampling': sampling}))
    recorder = Recorder(seed=seed, seats=labeled_seats(labels), provenance=provenance)
    referee=RefereeGame(seed,agents,labels, recorder=recorder)
    events=[]
    first_view = None
    candidate_seat = specs.index('candidate') if 'candidate' in specs else None
    try:
        while not referee.is_over():
            seat,obs=referee.pending()
            if seat == candidate_seat and first_view is None:
                first_view = _hash({'observation': obs.to_json(), 'history': [asdict(e) for e in events]})
            events.append(referee.apply_index(agents[seat].act(obs)))
    except BaseException:
        recorder.interrupt()
        raise
    result=referee.result()
    row = {'ranking': list(result.ranking), 'vp': list(result.vp), 'plies': result.moves, 'events': tuple(events)}
    if not trace_values:
        row['replay_id'] = recorder.game_id
    if trace_values and candidate_seat is not None:
        samples = agents[candidate_seat].samples
        row.update(value_predictions=samples, first_view_sha256=first_view,
                   terminal_brier=terminal_brier(samples, result.ranking, candidate_seat, result.moves))
    return row


def evaluate(candidate: Path, suite: EvalSuite, out: Path, *, before_game=None) -> EvalReport:
    """Resume only missing candidate/suite games, preserving all training RNGs."""
    before=capture_rng_states()
    try:
        loaded=load_policy(candidate)
        metadata=loaded[2]
        if metadata.id != suite.candidate_id:
            raise ValueError('checkpoint does not match preselected candidate ID')
        opponents={}
        for entries in suite.lineups.values():
            for spec in entries:
                if not isinstance(spec,str):
                    policy=load_policy(Path(spec['path']))
                    if policy[2].id != spec['id']:
                        raise ValueError('immutable opponent policy identity mismatch')
                    opponents[spec['id']]=policy
        out=Path(out)
        out.mkdir(parents=True,exist_ok=True)
        manifest=suite.manifest()
        manifest_path=out/'suite.json'
        if manifest_path.exists():
            if json.loads(manifest_path.read_text()) != manifest:
                raise ValueError('immutable suite/candidate manifest mismatch; use a new version directory')
        else:
            if any(out.iterdir()):
                raise ValueError('evaluation directory lacks immutable suite manifest')
            atomic_json(manifest_path,manifest)  # Before the first game, including final sample count/hash.
        candidate_out=out/'candidates'/metadata.id
        rows_dir=candidate_out/'games'
        rows_dir.mkdir(parents=True,exist_ok=True)
        replays=candidate_out/'replays'
        replays.mkdir(exist_ok=True)
        streams=SeedStreams(suite.master_seed)
        results=[]
        for lineup_index,(name,entries) in enumerate(sorted(suite.lineups.items())):
            for deal in range(suite.deals):
                for rotation in range(4):
                    game_id=(lineup_index*suite.deals+deal)*4+rotation
                    specs=entries[rotation:]+entries[:rotation]
                    labels=[metadata.id if e == 'candidate' else e if isinstance(e,str) else e['id'] for e in specs]
                    lineage={'game_id': game_id,'deal_id': deal,'rotation': rotation,'lineup': name,
                             'suite_id': manifest['suite_id'],'candidate_id': metadata.id,
                             'checkpoint_sha256': metadata.content_hashes['policy.pt'],
                             'update': metadata.update,'deal_seed': manifest['deal_seeds'][deal],
                             'labels': labels,'candidate_seat': specs.index('candidate')}
                    path=rows_dir/('%08d.json' % game_id)
                    if path.exists():
                        row=json.loads(path.read_text())
                        if any(row.get(k) != v for k,v in lineage.items()):
                            raise ValueError('completed evaluation row lineage mismatch')
                    else:
                        if before_game is not None:
                            before_game()
                        resolved=[e if isinstance(e,str) else opponents[e['id']] for e in specs]
                        sampling=[streams.seed(suite.stream+'.policy',lineup_index,deal,rotation,s) for s in range(4)]
                        started=time.monotonic()
                        with recording_context(**({'outcomes_only': True} if suite.outcomes_only else {}),
                                               run_experiment=recording_metadata().get('run_experiment'),
                                               category='evaluation', experiment=manifest['suite_id'],
                                               game=game_id, update=metadata.update, candidate_id=metadata.id,
                                               lineup=name, rotation=rotation, deal=deal, experiment_path=str(out.resolve())):
                            game=_play_game(loaded,resolved,lineage['deal_seed'],sampling)
                        events=game.pop('events')
                        row=dict(lineage,**game,seconds=time.monotonic()-started)
                        # Validate retained games against the recorded ranks/VP before publication.
                        keep=retained_ids(results+[row],suite.replay_limit)
                        for replay in replays.glob('*.json'):
                            if int(replay.stem) not in keep:
                                replay.unlink()
                        if game_id in keep and events:
                            write_gui_replay(replays/('%08d.json' % game_id),lineage['deal_seed'],labels,events,
                                             expected_ranking=row['ranking'],expected_vp=row['vp'])
                        atomic_json(path,row)  # One completed game is the restart boundary.
                    row.setdefault('terminal_brier', None)  # Legacy rows have no recorded predictions.
                    results.append(row)
        keep=retained_ids(results,suite.replay_limit)
        for replay in replays.glob('*.json'):
            if int(replay.stem) not in keep:
                replay.unlink()
        # Historical outcome rows are not full replays. Never regenerate missing
        # old games. New games already have durable recordings in the common archive.
        summaries={name: cluster_summary([r for r in results if r['lineup'] == name],metadata.id,
                                         streams.seed('bootstrap',i)) for i,name in enumerate(sorted(suite.lineups))}
        greedy=[name for name,entries in suite.lineups.items() if list(entries).count('greedy') == 3]
        passed=(suite.stream == 'final' and suite.deals >= 100 and bool(greedy) and
                all(summaries[n].win_rate >= .6 and summaries[n].win_ci[0] > .5 for n in greedy))
        report=EvalReport(metadata.id,manifest['suite_id'],tuple(results),summaries,select_replays(results),
                          'passed' if passed else 'pending')
        atomic_json(candidate_out/'report.json',asdict(report))
        return report
    finally:
        restore_rng_states(before)


def choose_development_best(previous, current):
    comparable=[r for r in previous if r['suite_id'] == current['suite_id']]+[current]
    return min(comparable,key=lambda r: (-r['win_rate'],r['mean_rank'],r['update'],r['candidate_id']))['candidate_id']


def periodic_development(model, update, league, run_dir, config, journal):
    """Evaluate without publishing best; caller commits returned facts with PPO.

    Initialization is the first fixed comparison. Holdout arrival creates a new
    version. Training-pool churn and fresh earlier checkpoints cannot move it.
    """
    before=capture_rng_states()
    try:
        evaluation=Path(run_dir)/'evaluation'
        from .artifacts import policy_id
        identity=policy_id(model,update)
        candidate=export_model_policy(model,update,evaluation/'policies'/identity)
        frozen={}
        anchors=[s for s in league.snapshots if s.role == 'initial']+list(league.holdouts)
        for snapshot in anchors:
            league.verify(snapshot)
            opponent=deepcopy(model)
            opponent.load_state_dict(torch.load(snapshot.weights_path,map_location=model.device,weights_only=True))
            committed=Path(run_dir)/'updates'/('%06d' % snapshot.update)
            if committed.is_dir():
                from .artifacts import export_policy
                from .policy_io import load_policy
                original, _, original_metadata=load_policy(committed)
                if any(not torch.equal(v.cpu(), original.state_dict()[k]) for k,v in opponent.state_dict().items()):
                    raise ValueError('frozen opponent disagrees with committed checkpoint')
                opponent_id=original_metadata.id
                path=export_policy(committed,evaluation/'policies'/opponent_id)
            else:
                opponent_id=policy_id(opponent,snapshot.update)
                path=export_model_policy(opponent,snapshot.update,evaluation/'policies'/opponent_id)
            frozen[snapshot.role]={'id': opponent_id,'path':str(path)}
            league.verify(snapshot)
        lineups={'random':['candidate']+['random']*3,'greedy':['candidate']+['greedy']*3}
        selection=[]
        for role,ref in sorted(frozen.items()):
            lineups[role]=['candidate']+[ref]*3
            selection.append(role)
        lineups['mixed']=['candidate',frozen['initial'],frozen.get('holdout',frozen['initial']),'greedy']
        selection.append('mixed')
        suite=EvalSuite('dev',league.streams.master,config.eval.deals,lineups,identity,config.eval.replay_limit)
        report=evaluate(candidate,suite,evaluation/'dev'/suite.manifest()['suite_id'])
        score={'candidate_id':identity,'suite_id':report.suite_id,'update':update,
               'win_rate':sum(report.summaries[n].win_rate for n in selection)/len(selection),
               'mean_rank':sum(report.summaries[n].mean_rank for n in selection)/len(selection),
               'selection_lineups':selection,'holdout_available':'holdout' in frozen}
        previous=[entry['metrics'][0]['development']['score'] for entry in journal
                  if entry['metrics'] and 'development' in entry['metrics'][0]]
        result={'score':score,'best_id':choose_development_best(previous,score),
                'summaries':{n:asdict(s) for n,s in report.summaries.items()},'manifest':suite.manifest()}
        # Fresh previous-update comparison is reported separately, never selected.
        earlier=Path(run_dir)/'policies'/('%06d' % (update-1))
        earlier_meta=inspect_policy(earlier)
        earlier_suite=EvalSuite('dev',league.streams.master,config.eval.deals,
            {'earlier':['candidate']+[{'id':earlier_meta.id,'path':str(earlier)}]*3},identity,config.eval.replay_limit)
        earlier_report=evaluate(candidate,earlier_suite,evaluation/'earlier'/earlier_suite.manifest()['suite_id'])
        result['earlier']={'suite_id':earlier_report.suite_id,'opponent_id':earlier_meta.id,
                           'summary':asdict(earlier_report.summaries['earlier'])}
        result['recorded_replay_ids'] = [r['replay_id'] for report in (report, earlier_report)
                                         for r in report.results if 'replay_id' in r]
        return result
    finally:
        restore_rng_states(before)
