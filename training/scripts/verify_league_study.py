# SPDX-License-Identifier: Apache-2.0
"""Audit downloaded, frozen run data without playing or training another game."""
import argparse
from collections import Counter
import json
from pathlib import Path

from brass_ai.replay_archive import Archive
from brass_rl.artifacts import committed_records, load_update
from brass_rl.experiments import encoded
from brass_rl.learning_study import read
from brass_rl.league_study import BASE, END, PEERS, check_dynamic_history, check_dynamic_treatment, check_legacy_treatment, declaration, schedule, summarize
from brass_rl.policy_io import inspect_policy
from brass_rl.retention import file_sha, present_payloads
from brass_rl.transfer import validate_replay


def verify(returned):
    import torch
    torch.set_num_threads(1)
    root = returned/'study'
    protocol = declaration()
    if read(root/'protocol.json')!=protocol or read(returned/'session.json')['status']!='complete':
        raise ValueError('incomplete or unexpected study')
    result = read(root/'results.json')
    archive = Archive(root/'replays')
    records = {r['game_id']:r for r in archive.records(include_duplicates=True)}
    if Counter(r['category'] for r in records.values())!={'training':200,'evaluation':100}:
        raise ValueError('unexpected number of recorded games')
    for r in records.values():
        validate_replay(r)
    arms, proof, accounted, deals = {}, {}, set(), {}
    for name in ('old','new'):
        run = root/name
        base = load_update(run/'updates/000793')
        tip = load_update(run/'latest')
        meta = read(run/'experiment.json')
        if (not all(meta['initial_state_verified'].values()) or
                meta['parent']['checkpoint_sha256']!=protocol['source_manifest_sha256'] or
                base.state.next_game_id!=1586 or tip.state.next_game_id!=1686 or
                base.state.completed_updates!=BASE or tip.state.completed_updates!=END):
            raise ValueError('wrong source, fork or endpoint')
        if (name=='new') != (tip.state.dynamic_league is not None):
            raise ValueError('unexpected final league')
        journal = committed_records(run)
        if [row['update'] for row in journal]!=list(range(BASE,END+1)):
            raise ValueError('study journal has a gap')
        complete = []
        games = []
        for entry in journal:
            update = entry['update']
            boundary = run/'updates'/('%06d'%update)
            manifest = read(boundary/'manifest.json')
            files = present_payloads(boundary,manifest)
            for filename in files:
                if file_sha(boundary/filename)!=manifest['content_hashes'][filename]:
                    raise ValueError('checkpoint hash mismatch')
            if 'optimizer.pt' in files:
                complete.append(update)
            for ref in manifest['snapshots']:
                weight = run/'weights'/(ref['sha256']+'.pt')
                if file_sha(weight)!=ref['sha256']:
                    raise ValueError('wrong frozen weight')
            policy = inspect_policy(run/'policies'/boundary.name)
            if (policy.id!=read(boundary/'metadata.json')['id'] or
                    policy.content_hashes['policy.pt']!=manifest['content_hashes']['policy.pt']):
                raise ValueError('wrong portable policy')
            manifests = entry['manifests']
            played = manifests.get('train',[])
            journal_ids = manifests.get('replay_ids',[])
            if (len(played)!=len(journal_ids) or len(played)!=(0 if update==BASE else 2)
                    or manifests.get('dev') or manifests.get('final')):
                raise ValueError('unexpected game budget or journal')
            by_game = {records[rid]['provenance']['game']:rid for rid in journal_ids}
            for game in played:
                rid = by_game[game['game_id']]
                r = records[rid]
                if (rid in accounted or r['category']!='training' or r['provenance']['experiment']!=meta['id']
                        or r['provenance']['update']!=update or r['initial']['seed']!=game['deal_seed']
                        or r['result']['vp']!=game['outcome']['vp'] or r['result']['ranking']!=game['outcome']['ranking']):
                    raise ValueError('training replay differs from journal')
                accounted.add(rid)
            games.extend(played)
        if complete != [793,800,810,820,830,840,843] or [g['game_id'] for g in games]!=protocol['training_game_ids']:
            raise ValueError('wrong retained full checkpoints or training schedule')
        for n in (BASE, END):
            sha = file_sha(run/'updates'/('%06d'%n)/'manifest.json')
            if read(run/'updates/pins'/(sha+'.json'))!={'update':n,'manifest_sha256':sha}:
                raise ValueError('missing branch base or endpoint pin')
        deals[name] = [g['deal_seed'] for g in games]
        lookup = {r['provenance']['game']:r for r in records.values() if r['category']=='training' and
                  r['provenance']['experiment']==meta['id']}
        if name=='new':
            provenance = read(returned/'input-provenance.json')
            canonical = {'policy-sha256-v1:'+Path(key).parent.name for key in provenance['logical_inputs']
                         if key.startswith('players/models/') and key.endswith('/policy.pt')}
            if len(canonical)!=40 or set(base.state.league_snapshot_ids)!=canonical or set(tip.state.league_snapshot_ids)!=canonical:
                raise ValueError('wrong original canonical history pool')
            current = {e['update']:read(run/'updates'/('%06d'%e['update'])/'metadata.json')['id'] for e in journal}
            check_dynamic_treatment(games,lookup,current,canonical,tip.state.dynamic_league['rated_games'],tip.state.master_seed)
            check_dynamic_history(run,journal,base.state.league_snapshots,tip.state)
        else:
            check_legacy_treatment(run,games,lookup)
        rows = []
        for plan in schedule():
            row = read(root/'evaluation'/name/'games'/('%04d.json'%plan['game_id']))
            ref = row['replay_id']
            r = records[ref]
            seat = plan['candidate_seat']
            candidate = result['endpoint_ids'][name]
            seats = [candidate]+list(PEERS)
            if seat:
                seats = seats[-seat:]+seats[:-seat]
            if (r['category']!='evaluation' or ref in accounted or r['provenance']['run_experiment']!=meta['id']
                    or r['initial']['seed']!=plan['deal_seed'] or [s['bot'] for s in r['seats']]!=seats
                    or r['result']['vp']!=row['vp'] or r['result']['ranking']!=row['ranking']
                    or any(row[key]!=value for key,value in plan.items())):
                raise ValueError('evaluation replay differs from matched declaration')
            accounted.add(ref)
            rows.append(row)
        arms[name] = rows
        proof[name] = dict(boundaries=len(journal), full_checkpoint_updates=complete,
                           final_policy=inspect_policy(run/'policies/000843').id,
                           rated_games=len(tip.state.dynamic_league['rated_games']) if name=='new' else None)
        if proof[name]['final_policy']!=result['endpoint_ids'][name]:
            raise ValueError('evaluation candidate differs from frozen endpoint')
    if accounted!=records.keys() or deals['old']!=deals['new'] or set(deals['old']) & {r['deal_seed'] for r in schedule()}:
        raise ValueError('missing, extra, overlapping or unpaired games')
    recomputed = summarize(arms)
    if encoded(recomputed['paired'])!=encoded(result['paired']):
        raise ValueError('paired results differ from native outcomes')
    for name,expected in recomputed['arms'].items():
        if encoded(expected)!=encoded({key:result['arms'][name][key] for key in expected}):
            raise ValueError('endpoint summary differs from archived outcomes')
    return dict(arms=proof,verified_native_recordings=len(records),paired_deal_blocks=25,
                result_sha256=file_sha(root/'results.json'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('returned',type=Path)
    args=parser.parse_args()
    print(json.dumps(verify(args.returned),indent=2))
