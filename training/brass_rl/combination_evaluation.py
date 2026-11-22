# SPDX-License-Identifier: Apache-2.0
"""Recoverable matched outcomes for the combination study's frozen development suite."""
from dataclasses import asdict
import math
from pathlib import Path
import time

import brass
from brass_ai.arena import play_game
from brass_ai.replay_archive import Archive, digest, engine_id, labeled_seats, recording_context, recording_metadata
from .learning_study import fixed, read, write
from .seeding import SeedStreams
from .transfer import validate_replay


def _play_seat(policy, seed):
    if policy is None:
        return brass.GreedyBot(seed)
    from .agent import TorchAgent
    model, components, metadata = policy
    agent = TorchAgent(model, components, seed, mode='sample')
    agent.replay_identity = metadata.id
    return agent


def outcome_id(lineage, origin):
    return digest({'suite': lineage['suite_id'], 'origin': origin['run_experiment'],
                   'game': lineage['game_id'], 'policies': lineage['seat_ids']})


def outcome_row(archive, path, lineage, origin, policies, sampling, before_game):
    seats = labeled_seats(lineage['labels'])
    replay_id = outcome_id(lineage, origin)
    saved = read(path) if path.exists() else None
    report_path = path.parent.parent/'report.json'
    if saved is None and report_path.exists():
        report = read(report_path)
        if report['candidate_id'] != lineage['candidate_id'] or report['suite_id'] != lineage['suite_id']:
            raise ValueError('candidate report does not match the sealed evaluation')
        matches = [r for r in report['results'] if r['game_id'] == lineage['game_id']]
        if len(matches) > 1:
            raise ValueError('duplicate completed game in candidate report')
        saved = matches[0] if matches else None
    seconds = saved['seconds'] if saved is not None else None
    if seconds is not None and (not math.isfinite(seconds) or seconds < 0):
        raise ValueError('invalid matched evaluation duration')
    if not archive.path(replay_id).is_file():
        if saved is not None:
            raise ValueError('completed matched row lacks its archived game')
        if before_game is not None:
            before_game()
        started = time.monotonic()
        agents = [_play_seat(policy, seed) for policy, seed in zip(policies, sampling)]
        with recording_context(**origin):
            play_game(agents, lineage['deal_seed'], labels=lineage['labels'], archive=archive, game_id=replay_id)
        seconds = time.monotonic()-started
    record = archive.read(replay_id)
    if (record['game_id'] != replay_id or record['status'] != 'completed' or record['engine_id'] != engine_id()
            or record['seats'] != seats
            or record['provenance'] != origin or record['initial'] != {'seed': lineage['deal_seed']}):
        raise ValueError('matched replay differs from expected identities, seed or provenance')
    validate_replay(record)
    row = dict(lineage, replay_id=replay_id, seconds=seconds, ranking=record['result']['ranking'],
               vp=record['result']['vp'], plies=len(record['moves']), terminal_brier=None)
    if saved is not None and saved != row:
        raise ValueError('stored matched outcome differs from its audited replay and lineage')
    fixed(path, row)
    return row


def evaluate(candidate, suite, out, *, before_game=None):
    """Use loaded policy metadata and one immutable native outcome per game."""
    from .artifacts import capture_rng_states, restore_rng_states
    from .evaluate import EvalReport, cluster_summary
    from .policy_io import load_policy
    from .replay import select_replays
    if suite.stream != 'dev' or not suite.outcomes_only or suite.replay_limit:
        raise ValueError('expected an outcomes-only development suite with durable native recordings')
    rng = capture_rng_states()
    try:
        candidate = load_policy(candidate)
        metadata = candidate[2]
        if metadata.id != suite.candidate_id:
            raise ValueError('frozen candidate identity mismatch')
        opponents = {}
        for lineup in suite.lineups.values():
            for spec in lineup:
                if isinstance(spec, str):
                    if spec not in ('candidate', 'greedy'):
                        raise ValueError('unexpected combination-study baseline')
                elif spec['id'] not in opponents:
                    opponent = load_policy(Path(spec['path']))
                    if opponent[2].id != spec['id']:
                        raise ValueError('frozen opponent identity mismatch')
                    opponents[spec['id']] = opponent

        manifest = suite.manifest()
        if out.exists() and any(out.iterdir()) and not (out/'suite.json').exists():
            raise ValueError('missing declaration for existing matched evaluation')
        fixed(out/'suite.json', manifest)
        directory = out/'candidates'/metadata.id
        archive, results = Archive(), []
        streams = SeedStreams(suite.master_seed)
        for index, (name, entries) in enumerate(sorted(suite.lineups.items())):
            for deal, seed in enumerate(manifest['deal_seeds']):
                for rotation in range(4):
                    specs = entries[rotation:]+entries[:rotation]
                    policies = [candidate if s == 'candidate' else None if s == 'greedy' else opponents[s['id']]
                                for s in specs]
                    labels = [p[2].id if p else 'greedy' for p in policies]
                    updates = [p[2].update if p else None for p in policies]
                    ids = [seat['bot'] for seat in labeled_seats(labels)]
                    game_id = (index*suite.deals+deal)*4+rotation
                    lineage = dict(game_id=game_id, deal_id=deal, rotation=rotation, lineup=name,
                                   suite_id=manifest['suite_id'], candidate_id=metadata.id, update=metadata.update,
                                   checkpoint_sha256=metadata.content_hashes['policy.pt'], deal_seed=seed, labels=labels,
                                   candidate_seat=specs.index('candidate'), seat_ids=ids, seat_updates=updates)
                    origin = dict(recording_metadata(), category='evaluation', outcomes_only=True, mode='sample',
                                  experiment=manifest['suite_id'], game=game_id, deal=deal, rotation=rotation,
                                  lineup=name, update=metadata.update, candidate_id=metadata.id, seat_ids=ids,
                                  seat_updates=updates, experiment_path=str(out.resolve()))
                    origin.setdefault('run_experiment', manifest['suite_id'])
                    sampling = [streams.seed(suite.stream+'.policy', index, deal, rotation, s) for s in range(4)]
                    row = outcome_row(archive, directory/'games'/('%08d.json'%game_id), lineage, origin,
                                      policies, sampling, before_game)
                    results.append(row)
        summaries = {name: cluster_summary([r for r in results if r['lineup'] == name], metadata.id,
                                            streams.seed('bootstrap', i)) for i, name in enumerate(sorted(suite.lineups))}
        report = EvalReport(metadata.id, manifest['suite_id'], tuple(results), summaries, select_replays(results), 'pending')
        write(directory/'report.json', asdict(report))
        return report
    finally:
        restore_rng_states(rng)
