# SPDX-License-Identifier: Apache-2.0
"""Canonical local model and replay library registration. Never restores a trainer."""
from contextlib import ExitStack
import json
import os
from pathlib import Path
import subprocess
import sys

from .experiments import project_for, read_experiment, writer_lock
from .model_inventory import inventory_writer
from .transfer import experiments_under, read


def register_publication(project, run, exports, *, include_evaluation=False):
    """Register newly committed exports under the caller's run writer lock."""
    with inventory_writer(project) as inventory:
        experiment = inventory.register(run)
        for source in exports:
            inventory.publish(source, run=run, experiment=experiment, checkpoint=run/'updates'/source.name)
        if include_evaluation:
            for path in (run/'evaluation/policies').glob('*/metadata.json'):
                inventory.publish(path.parent)


def launch_intent(source, experimental=False):
    project = project_for(source.parent.parent)
    if not experimental and project:
        path = project/'players/models/lineage.json'
        if path.is_file():
            from .experiments import parent_identity
            from .model_inventory import _parent_ref
            if read(path)['mainline_tip'] == _parent_ref(parent_identity(source)):
                return 'mainline'
    return 'experimental'


def adopt_mainline(project, checkpoint, *, expected_tip):
    """Advance the declared tip along exact registered predecessors to a selected boundary."""
    project, checkpoint = Path(project).resolve(), Path(checkpoint).resolve()
    if not checkpoint.name.isdecimal() or checkpoint.parent.name != 'updates' or project_for(checkpoint.parent.parent) != project:
        raise ValueError('adoption requires a numeric checkpoint in the destination project')
    run = checkpoint.parent.parent
    exp = read_experiment(run)['id']
    with writer_lock(run), inventory_writer(project) as inventory:
        chosen = inventory.checkpoints.get((exp, int(checkpoint.name)))
        from .experiments import parent_identity
        from .model_inventory import _parent_ref
        selected = _parent_ref(parent_identity(checkpoint))
        if chosen is None or any(chosen[k] != v for k, v in selected.items()):
            raise ValueError('reconcile the selected checkpoint before adoption')
        tip = inventory.registry['mainline_tip']
        branch = inventory.registry['branches'][exp]
        if selected == tip:
            if not branch.get('adopted_continuation') and expected_tip != tip['policy_id']:
                raise ValueError('declared mainline predecessor changed')
            branch['adopted_continuation'] = True
            return dict(mainline_tip=tip, adopted=0)
        if (tip and tip['policy_id']) != expected_tip:
            raise ValueError('declared mainline predecessor changed')
        chain, visited = [], set()
        cursor = selected
        while cursor != tip:
            key = cursor['experiment_id'], cursor['update']
            occurrence = inventory.checkpoints.get(key)
            if key in visited or occurrence is None or any(occurrence[k] != v for k, v in cursor.items()):
                raise ValueError('missing or conflicting checkpoint on adoption path')
            visited.add(key)
            chain.append(occurrence)
            branch = inventory.registry['branches'][key[0]]
            if branch['parent'] and key[1] == branch['parent']['update']:
                cursor = branch['parent']
            else:
                cursor = inventory.predecessor(*key)
            if cursor is None and (tip is not None or key[1] != 0):
                raise ValueError('no verified predecessor connecting the selected checkpoint to the tip')
        inventory.registry['branches'][exp]['adopted_continuation'] = True
        for occurrence in reversed(chain):
            inventory.accept(occurrence)
        return dict(mainline_tip=inventory.registry['mainline_tip'], adopted=len(chain))


def reconcile_models(project, locations, runs, public_only=()):
    """Run locks precede the catalog lock. Inputs may be sparse historical journals."""
    journals = []
    with inventory_writer(project) as inventory:
        for run in runs:
            experiment = inventory.register(run, locations)
            updates = {p.name: p for p in (run/'updates').glob('[0-9]*') if p.name.isdecimal()}
            policies = {p.name: p for p in (run/'policies').glob('[0-9]*') if p.name.isdecimal()}
            for name in sorted(updates.keys() | policies.keys()):
                boundary = updates.get(name)
                source = policies.get(name) or boundary
                if name != '%06d' % read(source/'metadata.json')['update']:
                    raise ValueError('source policy update mismatch')
                inventory.publish(source, run=run, experiment=experiment, checkpoint=boundary)
                if boundary:
                    from .transfer import file_hash
                    if file_hash(boundary/'manifests.json') != read(boundary/'manifest.json')['content_hashes']['manifests.json']:
                        raise ValueError('journal hash mismatch')
                    journal = read(boundary/'manifests.json')
                    coverage = dict(experiment_id=journal.get('experiment_id', experiment['id']),
                                    replay_ids=journal.get('replay_ids', []),
                                    train=[{k: g[k] for k in ('game_id', 'deal_seed', 'outcome') if k in g}
                                           for g in journal.get('train', [])])
                    journals.append((run, experiment, int(name), coverage))
        for source in public_only:
            inventory.publish(source)
        return dict(models=len(inventory.rows), added_models=inventory.added,
                    reused_models=len(inventory.public)-inventory.added), journals


def reconcile_library(project: Path, *, runs=None) -> dict:
    """Repair registered experiment exports and reconcile available full historical recordings."""
    from .library_replays import reconcile_replays
    project = Path(project).resolve()
    with writer_lock(project/'.experiment-import'):
        locations = experiments_under(project/'training/runs')
        chosen = sorted(locations.values() if runs is None else {Path(r).resolve() for r in runs})
        if not set(chosen) <= set(locations.values()):
            raise ValueError('reconciliation requires registered runs in the destination project')
        with ExitStack() as locks:
            for run in chosen:
                locks.enter_context(writer_lock(run))
            extras = [p.parent for p in (project/'players/models').glob('*/metadata.json')]
            for run in chosen:
                extras.extend(p.parent for p in (run/'evaluation').glob('**/manifest.json')
                              if read(p).get('format') == 'brass-portable-policy-v1')
            models, journals = reconcile_models(project, locations, chosen, extras)
            report = reconcile_replays(project, chosen, journals)
    return dict(report, experiments=len(chosen), **models)


def ingest_kaggle_return(verified: Path, project: Path) -> dict:
    """Reverify the exact sealed archives with the host controller and ingest the fresh extraction."""
    verified = Path(verified).resolve()
    if verified.name != 'verified' or not (verified.parent/'job.json').is_file():
        raise ValueError('expected the verified directory of a prepared local Kaggle job')
    script = Path(__file__).resolve().parents[1]/'scripts/kaggle_control/owner.py'
    env = dict(os.environ)
    root = script.parents[3]
    env['PYTHONPATH'] = os.pathsep.join([str(root/'training'), str(root/'engine/tools'), env.get('PYTHONPATH', '')])
    result = subprocess.run([sys.executable, script, verified.parent, 'ingest',
                             '--project-root', Path(project).resolve()], env=env, text=True, stdout=subprocess.PIPE)
    if result.returncode:
        raise ValueError('Kaggle verification or library ingestion failed; the sealed return can be retried')
    return json.loads(result.stdout)


def _publish_kaggle_state(state, project, summary):
    """Adapt an already owner-verified private extraction to the common immutable-unit publisher."""
    from tempfile import TemporaryDirectory
    from .experiments import link_or_copy
    from .transfer import IDENTITY, _import_tree, _policy, file_hash
    run = state/'run'
    experiment = read_experiment(run)
    identity = experiment['id']
    if not IDENTITY.fullmatch(identity) or identity != summary['experiment']:
        raise ValueError('invalid returned experiment identity')
    with TemporaryDirectory(prefix='library-return-') as temp:
        tree = Path(temp)
        relative = 'training/runs/'+identity
        files = {}
        def add(source, dest):
            if source.is_symlink() or not source.is_file():
                raise ValueError('invalid returned artifact')
            link_or_copy(source, tree/dest)
            files[dest] = dict(sha256=file_hash(source), size=source.stat().st_size)
        for source in sorted(run.rglob('*')):
            rel = source.relative_to(run)
            if (not source.is_file() or any(part.startswith('.') for part in rel.parts)
                    or rel.parts[0] in ('latest', 'best', 'metrics.jsonl') or rel.as_posix() == 'replays/index.json'):
                continue
            add(source, relative+'/'+rel.as_posix())
        for path in (run/'policies').glob('*/metadata.json'):
            meta, _ = _policy(path.parent)
            dest = 'players/models/policy-'+meta['id'].split(':')[-1]
            for name in ('policy.pt', 'metadata.json', 'manifest.json'):
                add(path.parent/name, dest+'/'+name)
        for path in (state/'replays/games').glob('*'):
            add(path, 'replays/games/'+path.name)
        manifest = dict(experiment_id=identity, experiments={identity: relative},
                        files=files, package_id=summary['sha256'])
        return _import_tree(tree, manifest, project, summary, external_parent=experiment['parent'])
