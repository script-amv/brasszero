# SPDX-License-Identifier: Apache-2.0
"""Sparse Adam checkpoints over an immutable, continuous per-update journal."""
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

from .experiments import atomic, encoded, project_for, sync_dir

UPDATE_FILES = frozenset(('policy.pt', 'metadata.json', 'controller.pt', 'optimizer.pt',
                          'metrics.json', 'manifests.json', 'versions.json'))
RECEIPT = 'retention.json'
FORMAT = 'brass-checkpoint-retirement/1'
ROLLING_POLICY = 'brass-checkpoint-retention/1'


def file_sha(path):
    with path.open('rb') as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1048576), b''):
            digest.update(chunk)
        return digest.hexdigest()


def receipt(path, manifest):
    return {'format': FORMAT, 'manifest_sha256': file_sha(path/'manifest.json'),
            'update': manifest['update'], 'retired': {'optimizer.pt': manifest['content_hashes']['optimizer.pt']}}


def present_payloads(path, manifest):
    """Return available payloads only for an exact v1 envelope or verified retirement."""
    if manifest.get('format') != 'brass-complete-update-v1' or set(manifest.get('content_hashes', ())) != UPDATE_FILES:
        raise ValueError('incomplete update manifest: '+str(path))
    if any(not isinstance(h, str) or not re.fullmatch(r'[0-9a-f]{64}', h)
           for h in manifest['content_hashes'].values()):
        raise ValueError('invalid checkpoint payload digest: '+str(path))
    names = {p.name for p in path.iterdir()}
    if RECEIPT in names:
        if (json.loads((path/RECEIPT).read_bytes()) != receipt(path, manifest) or
                path.name != '%06d' % manifest['update'] or manifest['update'] % 10 == 0):
            raise ValueError('invalid checkpoint retirement receipt: '+str(path))
        names.remove(RECEIPT)
        if names == (UPDATE_FILES - {'optimizer.pt'}) | {'manifest.json'}:
            return UPDATE_FILES - {'optimizer.pt'}
    if names != UPDATE_FILES | {'manifest.json'}:
        raise ValueError('incomplete or undeclared update files: '+str(path))
    return UPDATE_FILES


def pin(path):
    """Persist provenance of a full checkpoint even when its child lives elsewhere."""
    sha = file_sha(path/'manifest.json')
    pin_path = path.parent.parent/'updates/pins'/(sha+'.json')
    data = encoded({'update': int(path.name), 'manifest_sha256': sha})
    if pin_path.exists():
        if pin_path.read_bytes() != data:
            raise ValueError('conflicting checkpoint pin: '+str(pin_path))
        return
    atomic(pin_path, data)
    sync_dir(pin_path.parent.parent)


@dataclass
class Plan:
    runs: tuple
    candidates: tuple
    protected: dict
    manifests: dict


def plan_runs(runs, *, verify=True):
    runs = tuple(sorted(set(Path(run).resolve() for run in runs)))
    by_hash = defaultdict(list)
    reasons = defaultdict(set)
    manifests = {}
    model_ids = defaultdict(list)
    best_ids = set()
    experiments = {}
    parent_refs = []
    cache = {}
    def check(path, expected):
        stat = path.stat()
        key = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        if key not in cache:
            cache[key] = file_sha(path)
        if cache[key] != expected:
            raise ValueError('checkpoint hash mismatch: '+str(path))
    for run in runs:
        boundaries = sorted((p for p in (run/'updates').iterdir() if p.name.isdecimal()), key=lambda p: int(p.name))
        if not boundaries:
            raise ValueError('no checkpoint boundaries: '+str(run))
        reasons[boundaries[0]].add('root')
        reasons[boundaries[-1]].add('latest')
        experiment = json.loads((run/'experiment.json').read_bytes()) if (run/'experiment.json').exists() else {}
        if experiment:
            base = run/'updates'/('%06d' % experiment['base_update'])
            if not base.is_dir() or experiment['id'] in experiments:
                raise ValueError('missing experiment base or duplicate experiment ID: '+str(run))
            reasons[base].add('experiment base')
            experiments[experiment['id']] = run
            if experiment.get('parent'):
                parent_refs.append(experiment['parent'])
        if (run/'best').exists():
            best = run/'updates'/json.loads((run/'best').read_bytes())['update']
            if best.parent != run/'updates' or not best.is_dir():
                raise ValueError('invalid best pointer: '+str(run))
            reasons[best].add('best')
        for p in boundaries:
            manifest = json.loads((p/'manifest.json').read_bytes())
            if p.is_symlink() or any(q.is_symlink() for q in p.iterdir()) or p.name != '%06d' % manifest['update']:
                raise ValueError('unsafe checkpoint boundary: '+str(p))
            payloads = present_payloads(p, manifest)
            sha = file_sha(p/'manifest.json')
            by_hash[sha].append(p)
            manifests[p] = sha
            if int(p.name) % 10 == 0:
                reasons[p].add('cadence')
            policy_id = json.loads((p/'metadata.json').read_bytes())['id']
            model_ids[run, policy_id].append(p)
            for row in json.loads((p/'metrics.json').read_bytes())['records']:
                best = (row.get('development') or {}).get('best_id')
                if best:
                    best_ids.add((run, best))
            if verify:
                for name in payloads:
                    check(p/name, manifest['content_hashes'][name])
                for record in manifest['snapshots']:
                    sha = record['sha256']
                    if len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha):
                        raise ValueError('invalid frozen opponent hash')
                    check(run/'weights'/(sha+'.pt'), sha)
        for pp in (run/'updates/pins').glob('*.json'):
            data = json.loads(pp.read_bytes())
            path = run/'updates'/('%06d' % data['update'])
            if (pp.is_symlink() or set(data) != {'update', 'manifest_sha256'} or type(data['update']) is not int or
                    pp.stem != manifests.get(path) or pp.stem != data['manifest_sha256']):
                raise ValueError('invalid checkpoint pin: '+str(pp))
            reasons[path].add('explicit')
    for parent in parent_refs:
        owner = experiments.get(parent['experiment_id'])
        source = owner/'updates'/('%06d' % parent['update']) if owner else None
        if (source is None or manifests.get(source) != parent['checkpoint_sha256'] or
                source not in model_ids.get((owner, parent['model_id']), ())):
            raise ValueError('unresolved declared parent checkpoint: '+str(parent))
        reasons[source].add('fork source')
    for key in best_ids:
        if key not in model_ids:
            raise ValueError('unresolved development selection: '+str(key))
        for p in model_ids[key]:
            reasons[p].add('development selection')
    for p in list(reasons):
        for inherited in by_hash[manifests[p]]:
            if inherited != p:
                reasons[inherited].add('inherited protected boundary')
    for p in reasons:
        if not (p/'optimizer.pt').is_file():
            raise ValueError('protected checkpoint has no optimizer: '+str(p))
    candidates = tuple(p for p in manifests if p not in reasons and (p/'optimizer.pt').exists())
    return Plan(runs, candidates, dict(reasons), manifests)


def retire(path, manifest_sha):
    manifest = json.loads((path/'manifest.json').read_bytes())
    record = receipt(path, manifest)
    if record['manifest_sha256'] != manifest_sha or file_sha(path/'optimizer.pt') != record['retired']['optimizer.pt']:
        raise ValueError('checkpoint changed since inventory: '+str(path))
    # The temporary file belongs outside the immutable numeric directory. A crash
    # before its rename leaves the old full envelope readable by strict readers.
    with tempfile.TemporaryDirectory(prefix='.retire-', dir=path.parent) as temp:
        staged = Path(temp)/RECEIPT
        atomic(staged, encoded(record))
        os.replace(staged, path/RECEIPT)
        sync_dir(path)
    (path/'optimizer.pt').unlink()
    sync_dir(path)


def apply_plan(plan, *, pin_endpoints=False):
    for p, reasons in plan.protected.items():
        if reasons - {'latest', 'cadence', 'root'} or (pin_endpoints and 'latest' in reasons):
            pin(p)
    for p in plan.candidates:
        retire(p, plan.manifests[p])


def _legacy_fork_source(run, parent):
    """Find verified parent tuples in this project; detached v1 children may live elsewhere."""
    project = project_for(run)
    if project is None:
        return False
    try:
        for sibling in inspect_tree(project/'training/runs'):
            declaration = sibling/'experiment.json'
            if declaration.is_file():
                ref = json.loads(declaration.read_bytes()).get('parent') or {}
                if all(ref.get(key) == value for key, value in parent.items()):
                    return True
    except (OSError, ValueError, AttributeError):
        # A partial legacy tree cannot rule out a fork. Its Adam remains available.
        return False
    return False


def maintain_rolling(run, journal, best_id):
    """The caller holds the run writer lock after durable public publication."""
    if len(journal) < 3:
        return
    previous = journal[-2]['update']
    if previous % 10 == 0:
        return
    path = run/'updates'/('%06d' % previous)
    if not (path/'optimizer.pt').exists():
        return
    experiment = json.loads((run/'experiment.json').read_bytes())
    if previous == experiment['base_update']:
        return
    digest = file_sha(path/'manifest.json')
    if (run/'updates/pins'/(digest+'.json')).is_file():
        return
    identity = json.loads((path/'metadata.json').read_bytes())['id']
    if best_id == identity:
        pin(path)
        return
    if journal[-2]['manifests'].get('checkpoint_retention') != ROLLING_POLICY:
        selected = any((row.get('development') or {}).get('best_id') == identity
                       for entry in journal for row in entry['metrics'])
        parent = {'experiment_id': experiment['id'], 'update': previous,
                  'model_id': identity, 'checkpoint_sha256': digest}
        if selected or _legacy_fork_source(run, parent):
            pin(path)
        # Earlier writers did not pin detached children. Even an otherwise
        # readable project cannot prove that this unmarked boundary has none.
        return
    retire(path, digest)


def inspect_tree(root):
    root = Path(root)
    if not root.is_dir():
        raise ValueError('runs root is not a directory: '+str(root))
    def fail_discovery(error):
        raise error
    runs = []
    for directory, children, _ in os.walk(root, onerror=fail_discovery):
        children[:] = [child for child in children if not child.startswith('.')]
        if Path(directory).name == 'updates':
            if any(child.isdecimal() for child in children):
                runs.append(Path(directory).parent)
            children.clear()
    return runs


def summary(plan):
    inodes = defaultdict(list)
    for path in plan.candidates:
        stat = (path/'optimizer.pt').stat()
        inodes[stat.st_dev, stat.st_ino].append(stat)
    return {'runs': len(plan.runs), 'boundaries': len(plan.manifests),
            'protected_boundaries': len(plan.protected), 'retire': len(plan.candidates),
            'optimizer_logical_bytes': sum(st.st_size for group in inodes.values() for st in group),
            'optimizer_allocated_bytes_reclaimable': sum(group[0].st_blocks*512
                for group in inodes.values() if group[0].st_nlink == len(group)),
            'already_retired': len(plan.manifests)-len(plan.protected)-len(plan.candidates)}
