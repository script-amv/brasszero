# SPDX-License-Identifier: Apache-2.0
"""Three independent 100-game continuations of control 738 with 264 frozen games."""
from dataclasses import replace
from pathlib import Path
import hashlib
import statistics

from . import four_branch_study as prior
from .bounded import disk_bytes
from .config import TrainConfig
from .learning_study import ROOT, fixed, parent_identity, read
from .seeding import SeedStreams

STUDY = ROOT/'training/runs/combination-study-2026-11-20'
SOURCE = prior.STUDY/'control/updates/000738'
SOURCE_ID = 'policy-sha256-v1:223cf6014af1a1e800b914b373f7ed555be3a870525a66879487c9a2f25c4201'
SOURCE_HASH = '749c537d1a0cc6b3102b7f5f4455ec236bac6a93971aaf4e5f2e42db1ba1c983'
OPPONENT = prior.SOURCE
OPPONENT_ID = 'policy-sha256-v1:e0059c09239366d9ab33b1d1e461d19b22ca5651d70c19fa2ee3454e0f90ae0c'
ARMS = ('control', 'BC', 'ABC')
CANDIDATES = ('parent',)+ARMS
BASE, END, FIRST_GAME, GAMES = 738, 788, 1476, 100
MATCHED_MASTER, HEAD_MASTER = 14635809241209871023, 8196954129950821317


def arm_config(parent, name):
    if name == 'control':
        return parent
    if name not in ('BC', 'ABC'):
        raise ValueError('unknown combination arm')
    league = parent.league
    if name == 'ABC':
        league = replace(league, lineup_probabilities=(.8, .2, 0), lineup_schedule='four_of_five')
    return replace(parent, ppo=replace(parent.ppo, reward='score', critic_target='terminal'), league=league)


def suite(candidate):
    from .evaluate import EvalSuite
    frozen = {'id': OPPONENT_ID, 'path': str(OPPONENT)}
    return EvalSuite('dev', MATCHED_MASTER, 8, {'greedy': ['candidate']+['greedy']*3,
                     'frozen688': ['candidate']+[frozen]*3}, candidate, replay_limit=0, outcomes_only=True)


def head_protocol():
    from brass_ai.seeds import game_seeds
    return dict(master_seed=HEAD_MASTER, deal_seeds=game_seeds(HEAD_MASTER, 2),
                lineup=list(CANDIDATES), updates=[BASE, END, END, END], rotations=4,
                mode='sample', device='cpu', torch_threads=1, suite=STUDY.name+'-head-to-head')


def initialize(root, *, checkpoint=lambda arm, declaration=None: None):
    checkpoint(None)
    import torch
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    from .artifacts import load_update
    from .policy_io import load_policy
    parent = parent_identity(SOURCE)
    if parent['checkpoint_sha256'] != SOURCE_HASH or parent['model_id'] != SOURCE_ID:
        raise ValueError('source 738 checkpoint mismatch')
    if load_policy(OPPONENT, 'cpu')[2].id != OPPONENT_ID:
        raise ValueError('original 688 frozen opponent mismatch')
    cfg = TrainConfig.from_json(SOURCE.parents[1]/'config.json')
    state = load_update(SOURCE, config=cfg).state
    if (state.next_game_id != FIRST_GAME or state.completed_updates != BASE or state.master_seed != cfg.master_seed
            or cfg.runtime.device != 'cpu' or cfg.runtime.torch_threads != 1 or cfg.games_per_update != 2
            or cfg.runtime.max_wall_time_seconds is not None or cfg.eval.interval_updates != 0
            or (cfg.ppo.reward, cfg.ppo.critic_target, cfg.ppo.gae_lambda) != ('strict_win', 'gae', .95)
            or (cfg.league.lineup_schedule, cfg.league.lineup_probabilities) != ('sample', (.5, .35, .15))):
        raise ValueError('source 738 training profile mismatch')
    streams = SeedStreams(cfg.master_seed)
    training = [streams.seed('train.deal', gid) for gid in range(FIRST_GAME, FIRST_GAME+GAMES)]
    matched = suite(SOURCE_ID).manifest()
    head = head_protocol()
    eval_deals = matched['deal_seeds']+head['deal_seeds']
    archive, models = prior.shared_roots()
    if root.resolve() == STUDY and archive != ROOT/'replays':
        raise ValueError('scientific run requires the shared GUI replay archive')
    path = root/'study.json'
    prior_def = read(path) if path.exists() else None
    if prior_def:
        freshness = prior_def['freshness']
    else:
        runs = ROOT/'training/runs'
        declarations = list(runs.rglob('protocol.json'))+list(runs.rglob('study.json'))
        checkpoint(None)
        freshness = prior.freshness_evidence(root, eval_deals, training, expected_deals=10,
                                             extra_protocols=declarations)
        checkpoint(None)
        freshness['historical_training'] = prior.historical_training_evidence(
            root, eval_deals, cfg.master_seed, next_game=FIRST_GAME+GAMES, include_ancestry=True)
    definition = dict(format='brass-combination-study/1', parent=parent, arms=list(ARMS),
                      base_update=BASE, end_update=END, first_game_id=FIRST_GAME, games_per_arm=GAMES,
                      games_per_update=cfg.games_per_update, training_master=cfg.master_seed,
                      training_deals=training, evaluation=matched, head_to_head=head,
                      frozen_opponent={'path': str(OPPONENT), 'id': OPPONENT_ID},
                      frozen_lineup_game_ids=[g for g in range(FIRST_GAME, FIRST_GAME+GAMES) if g % 5 == 0],
                      storage_limit_bytes=prior.LIMIT, reserve_bytes=prior.RESERVE, freshness=freshness,
                      archive=str(archive), models=str(models),
                      shared_baseline_bytes=prior_def['shared_baseline_bytes'] if prior_def else disk_bytes(archive, models))
    fixed(path, definition)
    checkpoint(None, definition)
    prior.prepare_branches(root, definition, SOURCE, cfg, arm_config, checkpoint=checkpoint)
    return definition


def verify_training(root, manifest):
    endpoints = prior.verify_training(root, manifest, source=SOURCE, config_for=arm_config)
    # For the shared frozen source, inference uses only its public export.
    endpoints = {k: dict(v) for k, v in endpoints.items()}
    endpoints['parent']['path'] = str(SOURCE.parents[1]/'policies'/SOURCE.name)
    from .policy_io import inspect_policy
    for arm, endpoint in endpoints.items():
        public = Path(endpoint['path'])
        identity = inspect_policy(public)
        if identity.id != endpoint['id'] or identity.update != (BASE if arm == 'parent' else END):
            raise ValueError('frozen policy identity/update mismatch')
        endpoint.update(update=identity.update, content_hashes=dict(identity.content_hashes),
                        manifest_sha256=hashlib.sha256((public/'manifest.json').read_bytes()).hexdigest())
    fixed(root/'frozen-policies.json', endpoints)
    return endpoints


def head_schedule(protocol, endpoints):
    from brass_ai.replay_archive import digest
    names = protocol['lineup']
    if [endpoints[n]['update'] for n in names] != protocol['updates']:
        raise ValueError('mixed-update head-to-head lineup differs from the sealed protocol')
    games = []
    for deal, seed in enumerate(protocol['deal_seeds']):
        for rotation in range(protocol['rotations']):
            arms = names[rotation:]+names[:rotation]
            game = dict(game=len(games), deal=deal, deal_seed=seed, rotation=rotation, arms=arms,
                        seat_updates=[endpoints[n]['update'] for n in arms], seat_ids=[endpoints[n]['id'] for n in arms])
            games.append(dict(game, replay_id=digest({'protocol': protocol, 'game': game})))
    return games


def head_to_head(root, endpoints, protocol, before_game):
    from brass_ai.agents import make_agent
    from brass_ai.arena import play_game
    from brass_ai.replay_archive import Archive, recording_context
    from brass_ai.seeds import match_agent_seed
    from .agent import register_checkpoint
    from .transfer import validate_replay
    archive = Archive(prior.shared_roots()[0])
    directory = root/'head-to-head'
    rows = []
    schedule = head_schedule(protocol, endpoints)
    fixed(directory/'suite.json', dict(protocol, games=schedule))
    for arm, endpoint in endpoints.items():
        register_checkpoint(arm, Path(endpoint['path']), device='cpu', mode='sample')
    for game in schedule:
        replay_id = game['replay_id']
        origin = dict(category='evaluation', experiment=protocol['suite'], suite=protocol['suite'],
                      run_experiment=STUDY.name, game=game['game'], deal=game['deal'], rotation=game['rotation'],
                      seat_updates=game['seat_updates'], seat_ids=game['seat_ids'], mode='sample', interface='arena')
        row_path = directory/'games'/('%02d.json' % game['game'])
        if not archive.path(replay_id).exists():
            if row_path.exists():
                raise ValueError('completed head-to-head row missing its archive')
            before_game('head-to-head')
            agents = [make_agent(arm, match_agent_seed(protocol['master_seed'], game['deal'], game['rotation'], s))
                      for s, arm in enumerate(game['arms'])]
            with recording_context(**origin):
                play_game(agents, game['deal_seed'], labels=game['arms'], archive=archive, game_id=replay_id)
        record = archive.read(replay_id)
        if (record['status'] != 'completed' or record['game_id'] != replay_id
                or record['provenance'] != origin or record['initial']['seed'] != game['deal_seed']
                or [s['name'] for s in record['seats']] != game['arms']
                or [s['bot'] for s in record['seats']] != game['seat_ids']):
            raise ValueError('head-to-head recording differs from the sealed lineup')
        validate_replay(record)
        row = dict(game, vp=record['result']['vp'], ranking=record['result']['ranking'], plies=len(record['moves']))
        fixed(row_path, row)
        rows.append(row)
    summary = {}
    for name in CANDIDATES:
        scores = [row['vp'][row['arms'].index(name)] for row in rows]
        ranks = [row['ranking'][row['arms'].index(name)] for row in rows]
        wins = sum(rank == 0 and row['ranking'].count(0) == 1 for rank, row in zip(ranks, rows))
        summary[name] = dict(mean_vp=statistics.mean(scores), median_vp=statistics.median(scores),
                             lower_quartile_vp=statistics.quantiles(scores, method='inclusive')[0], max_vp=max(scores),
                             strict_wins=wins, mean_rank=1+statistics.mean(ranks))
    result = dict(games=rows, summary=summary, independent_deals=2)
    fixed(directory/'result.json', result)
    return result


def final_evaluation(root, endpoints, manifest, before_game):
    from .combination_evaluation import evaluate
    from .policy_io import inspect_policy
    matched = prior.final_evaluation(root, endpoints, manifest, before_game, suite_for=suite,
                                     candidates=CANDIDATES, study=STUDY, evaluator=evaluate)
    matched['head_to_head'] = head_to_head(root, endpoints, manifest['head_to_head'], before_game)
    for endpoint in endpoints.values():
        public = Path(endpoint['path'])
        current = inspect_policy(public)
        if (current.id != endpoint['id'] or dict(current.content_hashes) != endpoint['content_hashes']
                or hashlib.sha256((public/'manifest.json').read_bytes()).hexdigest() != endpoint['manifest_sha256']):
            raise ValueError('frozen policy bytes changed during evaluation')
    matched['optimizer_steps'] = {}
    for arm in ARMS:
        rows = [read(root/arm/'updates'/('%06d'%n)/'metrics.json')['records'][-1] for n in range(BASE+1, END+1)]
        matched['optimizer_steps'][arm] = sum(row['ppo']['optimizer_steps'] for row in rows)
    return matched


def render(result):
    from .four_branch_report import markdown_report
    report = markdown_report(result, candidates=CANDIDATES, title='Combination study from control 738')
    lines = ['## Training and head-to-head results', '',
             'Adam steps from updates 739 through 788 were '+', '.join(
                 f"{k} {v}" for k, v in result['optimizer_steps'].items())+'.', '',
             'The head-to-head used two additional deals with four seat rotations each. '
             'The parent is frozen source control 738. Control, BC and ABC are frozen update 788.', '',
             '| Game | Deal | Rotation | parent VP | control VP | BC VP | ABC VP |',
             '|---:|---:|---:|---:|---:|---:|---:|']
    for game in result['head_to_head']['games']:
        scores = [game['vp'][game['arms'].index(n)] for n in CANDIDATES]
        lines.append('| %d | %d | %d | %d | %d | %d | %d |' % tuple([game['game']+1, game['deal']+1, game['rotation']]+scores))
    lines += ['', '| Policy | Mean VP | Median VP | Lower quartile VP | Max VP | Strict wins / 8 | Mean rank |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for arm, row in result['head_to_head']['summary'].items():
        lines.append('| %s | %.3f | %.2f | %.2f | %d | %d | %.3f |' % (arm, row['mean_vp'], row['median_vp'],
                     row['lower_quartile_vp'], row['max_vp'], row['strict_wins'], row['mean_rank']))
    if result.get('interrupted_intervals'):
        lines.extend(['', '## Interrupted invocations', '',
                      'Elapsed wall time includes the interruption. Active work between the last observation '
                      'and abrupt exit is unknown. An unfinished training update can repeat collection and '
                      'optimizer work on resume. Only its committed games and Adam steps enter the totals.', ''])
        for gap in result['interrupted_intervals']:
            span = gap['next_started_epoch']-gap['last_observed_epoch']
            lines.append(f"- Last observed in {gap['last_phase']} ({gap['last_arm']}), {span:.2f}s before restart.")
    return report+'\n'+'\n'.join(lines)+'\n'


def run(root, manifest, stop, *, init_only=False):
    return prior.run_study(root, manifest, stop, arms=ARMS, base=BASE, end=END, verify=verify_training,
                           evaluate=final_evaluation, render=render, initialize=initialize if manifest is None else None,
                           init_only=init_only)


def main(argv=None):
    prior.study_cli(argv, STUDY, __spec__.name, initialize, run, __doc__, managed_init=True)


if __name__ == '__main__':
    main()
