# SPDX-License-Identifier: Apache-2.0
"""Predeclared BC793 comparison. Both endpoints freeze before any evaluation."""
import argparse
from collections import Counter
import hashlib
import os
from pathlib import Path
from statistics import mean, median
import time

from .experiments import atomic, encoded
from .seeding import SeedStreams

BASE, END = 793, 843
SOURCE_SHA = '78f49029e569f92d49d565a217621ee9d6a59f3ddfa56ab41283557124145c22'
SOURCE_EXPERIMENT = 'exp-aa925dc7665640e2a4d90c1a6958eac8'
PEERS = (
    'policy-sha256-v1:e0059c09239366d9ab33b1d1e461d19b22ca5651d70c19fa2ee3454e0f90ae0c',
    'policy-sha256-v1:223cf6014af1a1e800b914b373f7ed555be3a870525a66879487c9a2f25c4201',
    'policy-sha256-v1:fc0cd4367c80203134660cdb3397e282370745658b1fbea4f3f1ff4a6ab5bf49')
EVAL_MASTER = 0x94B02EA83F7069A3


def schedule():
    streams = SeedStreams(EVAL_MASTER)
    result = []
    for game in range(50):
        deal, seat = game//2, game%4
        result.append(dict(game_id=game, deal_id=deal, candidate_seat=seat,
                           deal_seed=streams.seed('league-study.final.deal',deal),
                           policy_seeds=[streams.seed('league-study.final.policy',deal,seat,s) for s in range(4)]))
    return result


def declaration():
    return dict(format='brass-league-study/1', source_experiment=SOURCE_EXPERIMENT,
                source_manifest_sha256=SOURCE_SHA, source_policy_id=PEERS[-1], base_update=BASE,
                end_update=END, training_game_ids=list(range(1586,1686)), peers=list(PEERS),
                fixed_schedule=schedule(), study_endpoints_are_experimental=True)


def summarize(arms):
    summary, metrics = {}, {}
    for arm, rows in arms.items():
        if [{k:r[k] for k in ('game_id','deal_id','candidate_seat','deal_seed','policy_seeds')}
                for r in rows] != schedule():
            raise ValueError('evaluation rows disagree with the fixed matched schedule')
        values = []
        for row in rows:
            seat = row['candidate_seat']
            ranks = row['ranking']
            values.append((row['vp'][seat], int(ranks[seat]==0 and ranks.count(0)==1), ranks[seat]))
        metrics[arm] = values
        vp, win, rank = zip(*values)
        ordered = sorted(vp)
        # Linear interpolation at p*(n-1).
        i = (len(vp)-1)/4
        summary[arm] = dict(vp=dict(mean=mean(vp),median=median(vp),
                                     q25=ordered[int(i)]*(1-i%1)+ordered[int(i)+1]*(i%1),max=max(vp)),
                            sole_wins=sum(win), native_rank_counts=dict(sorted(Counter(rank).items())),
                            mean_native_rank=mean(rank),
                            evaluation_seconds=sum(r['seconds'] or 0 for r in rows))
    blocks = []
    for deal in range(25):
        diff = [tuple(n-o for o,n in zip(metrics['old'][i],metrics['new'][i])) for i in (deal*2,deal*2+1)]
        blocks.append(dict(deal_id=deal,vp=mean(d[0] for d in diff),
                           sole_win=mean(d[1] for d in diff),native_rank=mean(d[2] for d in diff)))
    return dict(arms=summary, paired=dict(mean_vp_delta=mean(b['vp'] for b in blocks),
                mean_sole_win_delta=mean(b['sole_win'] for b in blocks),
                mean_native_rank_delta=mean(b['native_rank'] for b in blocks),deal_blocks=blocks))


def rotate(values, seat):
    return values[-seat:]+values[:-seat] if seat else values


def evaluation_identity(plan, candidate, weights_sha, experiment):
    suite = 'league-final-'+hashlib.sha256(encoded(declaration())).hexdigest()
    seat = plan['candidate_seat']
    ids = rotate([candidate]+list(PEERS),seat)
    updates = rotate([END,688,738,793],seat)
    lineage = dict(plan,rotation=seat,lineup='school688-control738-BC793',suite_id=suite,
                   candidate_id=candidate,update=END,checkpoint_sha256=weights_sha,
                   labels=ids,seat_ids=ids,seat_updates=updates)
    origin = dict(category='evaluation',outcomes_only=True,mode='sample',experiment=suite,
                  run_experiment=experiment,game=plan['game_id'],deal=plan['deal_id'],rotation=seat,
                  lineup=lineage['lineup'],update=END,candidate_id=candidate,seat_ids=ids,seat_updates=updates)
    return lineage,origin


def evaluate_endpoint(arm, run, end, root, peers):
    from brass_ai.replay_archive import Archive
    from .combination_evaluation import outcome_row
    from .learning_study import read
    from .policy_io import load_policy
    candidate = load_policy(end)
    identity = candidate[2].id
    experiment = read(run/'experiment.json')['id']
    archive, results = Archive(), []
    for plan in schedule():
        policies = rotate([candidate]+peers,plan['candidate_seat'])
        lineage,origin = evaluation_identity(plan,identity,candidate[2].content_hashes['policy.pt'],experiment)
        results.append(outcome_row(archive,root/'evaluation'/arm/'games'/('%04d.json'%plan['game_id']),
                                   lineage,origin,policies,plan['policy_seeds'],None))
    return results


def check_dynamic_treatment(games, replay_by_game, learner_by_update, canonical_ids, rated_games, master):
    """Reconcile prescribed training lineups, native recording seats and rated IDs."""
    from brass_ai.replay_archive import labeled_seats
    canonical_ids = set(canonical_ids)
    observed = []
    for game in games:
        game_id, lineup = game['game_id'], game['lineup']
        historical = game_id%2 != SeedStreams(master).seed('dynamic.self_play',game_id//2)%2
        current = learner_by_update[game_id//2]
        frozen = [e['snapshot_id'] for e in lineup if e['kind']=='frozen']
        if (len(lineup)!=4 or any(e['kind'] not in ('current','frozen') or
                e['trainable']!=(e['kind']=='current') or (e['snapshot_id'] is None)!=e['trainable'] for e in lineup)
                or len(frozen)!=(3 if historical else 0) or len(set(frozen))!=len(frozen)
                or not set(frozen)<=canonical_ids):
            raise ValueError('unexpected dynamic treatment lineup or foreign opponent')
        seats = labeled_seats([current if e['trainable'] else e['snapshot_id'] for e in lineup])
        if replay_by_game[game_id]['seats']!=seats:
            raise ValueError('archived seats differ from declared learner/frozen lineup')
        if historical:
            observed.append(game_id)
    if observed!=rated_games:
        raise ValueError('rated game IDs differ from observed historical training lineups')


def check_legacy_treatment(run, games, replay_by_game):
    """Reproduce the inherited sampler and its historical digest-based seat labels."""
    from dataclasses import asdict
    from brass_ai.replay_archive import labeled_seats
    from .artifacts import inspect_update_state,restore_league
    from .league import replay_identity
    from .learning_study import read
    previous = None
    for game in games:
        n = game['game_id']//2
        if n!=previous:
            path = run/'updates'/('%06d'%n)
            league = restore_league(inspect_update_state(path))
            current = read(path/'metadata.json')['id']
            previous = n
        lineup = league.sample(game['game_id'])
        labels = [current if e.trainable else replay_identity(e.snapshot_id,league.history[e.snapshot_id])
                  if e.kind=='frozen' else e.kind for e in lineup.entries]
        if ([asdict(e) for e in lineup.entries]!=game['lineup'] or
                labeled_seats(labels)!=replay_by_game[game['game_id']]['seats']):
            raise ValueError('legacy sampler or frozen seat identity differs from archive')


def check_recovery_budget(journals, ids, records, allowed_pending):
    """Match the journals to recordings, permitting only exact next-boundary prefixes."""
    expected, tips = {}, {}
    for name, entries in journals.items():
        tips[name] = entries[-1]['update']
        if (tips[name]>END or [e['update'] for e in entries]!=list(range(BASE,tips[name]+1)) or
                len(set(ids.values()))!=2):
            raise ValueError('unexpected study budget, gaps, or experiment identity')
        for entry in entries:
            n, m = entry['update'], entry['manifests']
            games = m.get('train',[])
            required = [] if n==BASE else [2*(n-1),2*(n-1)+1]
            if ([g['game_id'] for g in games]!=required or len(m.get('replay_ids',[]))!=len(games)
                    or m.get('dev') or m.get('final')):
                raise ValueError('wrong committed training game IDs or allowance')
            for g in games:
                expected[(ids[name],g['game_id'])] = (n,g,set(m['replay_ids']))
    if tips['new']>BASE and tips['old']!=END:
        raise ValueError('new arm precedes declared old-arm completion')
    seen = set()
    for r in records:
        rid, p = r['game_id'],r['provenance']
        if r['category']=='evaluation' and min(tips.values())!=END:
            raise ValueError('early evaluation in partial return')
        key = (p.get('experiment'),p.get('game'))
        if r['category']=='training' and key in expected:
            n,game,refs = expected[key]
            if (key in seen or rid not in refs or r['initial']!={'seed':game['deal_seed']}
                    or r['status']!='completed' or p['update']!=n or
                    any(r['result'][k]!=game['outcome'][k] for k in ('vp','ranking'))):
                raise ValueError('conflicting committed training replay')
            seen.add(key)
        else:
            wanted = allowed_pending.get(rid)
            if wanted is None:
                raise ValueError('uncommitted recording lies outside the next sealed game IDs')
            observed = {k:r.get(k) for k in wanted}
            observed['provenance'] = {k:v for k,v in p.items() if k!='experiment_path'}
            if observed!=wanted:
                raise ValueError('uncommitted recording differs from its declared game')
    if seen!=expected.keys():
        raise ValueError('missing committed training replay')
    return tips


def check_tip(name, state, journal_tip, cfg):
    if (state.config!=cfg or state.master_seed!=cfg.master_seed or state.completed_updates!=journal_tip
            or not BASE<=journal_tip<=END or state.next_game_id!=2*journal_tip
            or (state.dynamic_league is not None)!=(name=='new')):
        raise ValueError('recovered controller cursor, master, config or league differs from the journal')


def check_evaluation_row(archive, path, lineage, origin, replay_id):
    from .combination_evaluation import outcome_row
    if not archive.path(replay_id).is_file():
        raise ValueError('evaluation row lacks its exact durable recording')
    # An existing complete recording takes the validation-only route.
    outcome_row(archive,path,lineage,origin,(),(),None)


def check_dynamic_history(run, journal, members, tip):
    """Replay the scheduled draws and native-rank updates from the declared prior."""
    from dataclasses import asdict
    from .artifacts import inspect_update_state
    from .dynamic_league import DynamicLeague
    rebuilt = DynamicLeague.start(SeedStreams(tip.master_seed),members,tip.holdouts,
                                  history_extent=BASE+1,next_game_id=2*BASE)
    member_ids = tuple(m.id for m in members)
    for entry in journal:
        for game in entry['manifests'].get('train',()):
            lineup = rebuilt.sample(game['game_id'])
            if [asdict(e) for e in lineup.entries]!=game['lineup']:
                raise ValueError('dynamic lineup differs from rated sampling history')
            rebuilt.record_training(game['game_id'],lineup,game['outcome']['ranking'])
        update = entry['update']
        saved = tip if update==journal[-1]['update'] else inspect_update_state(run/'updates'/('%06d'%update))
        if (rebuilt.state!=saved.dynamic_league or rebuilt.history!=saved.league_history
                or member_ids!=saved.league_snapshot_ids):
            raise ValueError('dynamic state diverged from its native training results at update %d'%update)


def check_partial(root, cfg, members):
    from brass_ai.replay_archive import Archive,digest,engine_id,labeled_seats
    from .artifacts import committed_records,load_update,restore_league
    from .league import replay_identity
    from .learning_study import fixed,read
    from .transfer import validate_replay
    journals, ids, tips, pending = {}, {}, {}, {}
    archive = Archive(root/'replays')
    for name in ('old','new'):
        run = root/name
        ids[name] = read(run/'experiment.json')['id']
        journals[name] = committed_records(run)
        tips[name] = load_update(run/'latest')
        check_tip(name,tips[name].state,journals[name][-1]['update'],cfg)
    ids_path = root/'arm-identities.json'
    if not ids_path.exists() and any(len(rows)>1 for rows in journals.values()):
        raise ValueError('committed study prefix lacks its original arm identity seal')
    fixed(ids_path,ids)
    records = archive.records(include_duplicates=True)
    for r in records:
        if r['status']=='completed':
            validate_replay(r)
        elif r['status']!='interrupted':
            raise ValueError('unsupported or active partial recording')
    existing_rows = {p for p in (root/'evaluation').rglob('*') if p.is_file()}
    if all(t.state.completed_updates==END for t in tips.values()):
        for name,tip in tips.items():
            candidate = read(tip.path/'metadata.json')['id']
            weights_sha = read(tip.path/'manifest.json')['content_hashes']['policy.pt']
            for plan in schedule():
                lineage, origin = evaluation_identity(plan,candidate,weights_sha,ids[name])
                rid = digest(dict(suite=lineage['suite_id'],origin=ids[name],game=plan['game_id'],
                                  policies=lineage['seat_ids']))
                pending[rid] = dict(category='evaluation',provenance=origin,initial={'seed':plan['deal_seed']},
                                    seats=labeled_seats(lineage['labels']),engine_id=engine_id())
                row = root/'evaluation'/name/'games'/('%04d.json'%plan['game_id'])
                if row in existing_rows:
                    check_evaluation_row(archive,row,lineage,origin,rid)
                    existing_rows.remove(row)
    else:
        name = 'old' if tips['old'].state.completed_updates<END else 'new'
        loaded = tips[name]
        state = loaded.state
        league = restore_league(state)
        candidate = read(loaded.path/'metadata.json')['id']
        for g in range(state.next_game_id,state.next_game_id+2):
            lineup = league.sample(g)
            labels = [candidate if e.trainable else replay_identity(e.snapshot_id,league.history[e.snapshot_id])
                      if e.kind=='frozen' else e.kind for e in lineup.entries]
            provenance = dict(category='training',experiment=ids[name],update=state.completed_updates+1,game=g)
            replay_id = digest(dict(initial={'seed':league.streams.seed('train.deal',g)},seats=labeled_seats(labels),
                                    provenance=provenance,engine_id=engine_id()))
            pending[replay_id] = dict(category='training',provenance=provenance,engine_id=engine_id(),
                                      seats=labeled_seats(labels),initial={'seed':league.streams.seed('train.deal',g)})
    if existing_rows:
        raise ValueError('unknown or premature evaluation row in recovered study')
    progress = check_recovery_budget(journals,ids,records,pending)
    for name,tip in tips.items():
        run = root/name
        journal = journals[name]
        games = [g for e in journal for g in e['manifests'].get('train',())]
        lookup = {r['provenance']['game']:r for r in records if r['category']=='training' and
                  r['provenance']['experiment']==ids[name]}
        if name=='old':
            check_legacy_treatment(run,games,lookup)
            continue
        current = {e['update']:read(run/'updates'/('%06d'%e['update'])/'metadata.json')['id'] for e in journal}
        check_dynamic_treatment(games,lookup,current,[m.id for m in members],
                                tip.state.dynamic_league['rated_games'],cfg.master_seed)
        check_dynamic_history(run,journal,members,tip.state)
    return progress


def run_study(inputs, root):
    import torch
    import trueskill
    from importlib.metadata import version
    from brass_ai.replay_archive import Archive
    from .artifacts import committed_records, load_update
    from .bounded import disk_bytes
    from .config import TrainConfig
    from .dynamic_league import catalog_members
    from .learning_study import fixed, fork_arm, parent_identity, read
    from .policy_io import load_policy
    from .retention import pin
    from .train import train
    from .transfer import validate_replay

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    os.environ['BRASS_REPLAY_ROOT'] = str(root/'replays')
    root.mkdir(parents=True,exist_ok=True)
    source = inputs/'parent/updates/000793'
    parent = parent_identity(source)
    if (parent['checkpoint_sha256'],parent['model_id'],parent['experiment_id']) != (
            SOURCE_SHA,PEERS[-1],SOURCE_EXPERIMENT):
        raise ValueError('wrong full BC793 boundary')
    protocol = declaration()
    if read(inputs/'protocol.json') != protocol:
        raise ValueError('study schedule differs from precommitted input')
    fixed(root/'protocol.json',protocol)
    cfg = TrainConfig.from_json(inputs/'parent/config.json')
    if (cfg.games_per_update,cfg.concurrent_games,cfg.runtime.device,cfg.runtime.torch_threads,
            cfg.eval.interval_updates,cfg.ppo.reward,cfg.ppo.critic_target,cfg.ppo.gae_lambda) != (
            2,2,'cpu',1,0,'score','terminal',.95):
        raise ValueError('source configuration mismatch')
    members = catalog_members(inputs,BASE)
    if len(members)!=40:
        raise ValueError('expected 40 canonical historical checkpoints')
    peers = [load_policy(inputs/'peers'/str(n)) for n in (688,738,793)]
    if tuple(p[2].id for p in peers)!=PEERS:
        raise ValueError('evaluation peer identity mismatch')
    numerical = {name:version(name) for name in ('trueskill','mpmath')}
    if numerical!={'trueskill':'0.4.5','mpmath':'1.3.0'} or torch.__version__.split('+')[0]!='2.2.2' or torch.cuda.is_available():
        raise ValueError('unexpected numerical runtime for free CPU study')
    fixed(root/'numerical-runtime.json',numerical)
    # Exercise the numerical backend without a game or production ratings.
    env = trueskill.TrueSkill(tau=0., backend='mpmath')
    env.rate([(env.create_rating(),)]*3,ranks=[0,1,1])
    for name in ('old','new'):
        run = root/name
        if not run.exists():
            fork_arm(source, run, cfg.ppo.gae_lambda, study_config=cfg,
                     dynamic_members=members if name=='new' else None)
        meta = read(run/'experiment.json')
        if meta['parent']!=parent or not all(meta['initial_state_verified'].values()):
            raise ValueError('study fork differs from common source')
        boundary = run/'updates/000793'
        initial = load_update(boundary).state
        expected = [s.id for s in members] if name=='new' else [
            r['id'] for r in read(source/'manifest.json')['snapshots'] if r['role']!='holdout']
        if (list(initial.league_snapshot_ids)!=expected or (initial.dynamic_league is not None)!=(name=='new') or
                initial.config!=cfg or initial.completed_updates!=BASE or initial.next_game_id!=1586):
            raise ValueError('starting league, config or counters differ from predeclared arm')
        pin(boundary)
    check_partial(root,cfg,members)
    times = {}
    for name in ('old','new'):
        run = root/name
        start = time.monotonic()
        report = train(cfg,run,run/'latest',max_updates=END)
        if report.completed_updates != END or report.next_game_id!=1686:
            raise ValueError('study stopped before endpoint')
        pin(report.latest_checkpoint)
        timing_path = root/(name+'-training-seconds.json')
        if not timing_path.exists():
            fixed(timing_path,{'seconds':time.monotonic()-start})
        times[name] = read(timing_path)['seconds']
        print(name,'training complete',flush=True)
    candidate_ids = {}
    arms = {}
    for name in ('old','new'):
        run = root/name
        candidate = run/'policies'/('%06d'%END)
        candidate_ids[name] = read(candidate/'metadata.json')['id']
        arms[name] = evaluate_endpoint(name,run,candidate,root,peers)
        print(name,'evaluation complete',flush=True)
    summary = summarize(arms)
    summary.update(endpoint_ids=candidate_ids,final_training_invocation_seconds=times)
    records = Archive().records(include_duplicates=True)
    all_deals = []
    for name in ('old','new'):
        run = root/name
        journal = committed_records(run)
        games = [g for e in journal for g in e['manifests'].get('train',())]
        if ([g['game_id'] for g in games]!=protocol['training_game_ids'] or len(journal)!=51 or
                any(e['manifests'].get('dev') or e['manifests'].get('final') for e in journal)):
            raise ValueError('study training budget or no-interim-evaluation requirement violated')
        all_deals.append([g['deal_seed'] for g in games])
        latest = load_update(run/'latest').state
        if name=='new' and (len(latest.dynamic_league['rated_games'])!=50 or
                            len(latest.league_snapshots)!=40 or latest.dynamic_league['history_extent']!=794):
            raise ValueError('experimental dynamic membership or rating count differs from protocol')
        lookup = {r['provenance']['game']:r for r in records if r['category']=='training' and
                  r['provenance']['experiment']==read(run/'experiment.json')['id']}
        if name=='new':
            current = {e['update']:read(run/'updates'/('%06d'%e['update'])/'metadata.json')['id'] for e in journal}
            check_dynamic_treatment(games,lookup,current,[m.id for m in members],
                                    latest.dynamic_league['rated_games'],latest.master_seed)
        else:
            check_legacy_treatment(run,games,lookup)
        summary['arms'][name]['logical_run_bytes'] = disk_bytes(run)
        training_records = [row for entry in journal for row in entry['metrics']]
        summary['arms'][name]['collection_and_ppo_seconds'] = {
            phase: sum(row['timings_seconds'][phase] for row in training_records)
            for phase in ('collection','ppo')}
        summary['arms'][name]['training_current_seat_counts'] = dict(Counter(
            sum(s['trainable'] for s in g['lineup']) for g in games))
    if all_deals[0]!=all_deals[1] or set(all_deals[0]) & {r['deal_seed'] for r in schedule()}:
        raise ValueError('unpaired training or overlapping evaluation deals')
    if Counter(r['category'] for r in records) != {'training':200,'evaluation':100}:
        raise ValueError('unexpected archive game counts')
    for record in records:
        validate_replay(record)
    summary['archive_counts'] = dict(Counter(r['category'] for r in records))
    summary['logical_study_bytes_before_report'] = disk_bytes(root)
    atomic(root/'results.json',encoded(summary))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs',type=Path)
    parser.add_argument('output',type=Path)
    args = parser.parse_args()
    run_study(args.inputs.resolve(),args.output.resolve())
