# SPDX-License-Identifier: Apache-2.0
"""Self-contained, content-addressed experiment transfer (standard library only).

Verification/import never unpickle training state. Resume is the explicit trusted
numerical boundary. Package hashes establish integrity, not publisher authenticity.
"""
from contextlib import ExitStack, contextmanager
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import tempfile
from functools import wraps

from .config import BC_LEARNER, TrainConfig
from .retention import pin, plan_runs, present_payloads, UPDATE_FILES
from .experiments import (atomic, encoded, library_intent, link_or_copy, project_for, read_experiment,
                          sync_dir, sync_tree, writer_lock)

FORMAT = 'brass-experiment-package/1'
HEX = re.compile(r'^[0-9a-f]{64}$')
IDENTITY = re.compile(r'^(exp-[0-9a-f]{32}|legacy-[0-9a-f]{64}|bench-20261119-(?:warmup|pair-[1-3])-(?:cpu|cuda))$')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as src:
        for block in iter(lambda: src.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    result = json.loads(Path(path).read_text())
    if not isinstance(result, dict):
        raise ValueError('expected JSON object: '+str(path))
    return result


def safe_path(value):
    path = PurePosixPath(value)
    if not isinstance(value, str) or path.is_absolute() or not path.parts or any(p in ('..', '.') for p in path.parts) or str(path) != value or '\\' in value:
        raise ValueError('unsafe package path')
    return path


def experiments_under(root):
    found = {}
    for file in Path(root).rglob('experiment.json'):
        if any(p.startswith('.') for p in file.relative_to(root).parts):
            continue
        metadata = read_experiment(file.parent)
        identity = metadata['id']
        if not IDENTITY.fullmatch(identity):
            raise ValueError('invalid experiment identity')
        if identity in found and found[identity] != file.parent:
            raise ValueError('conflicting duplicate experiment identity: '+identity)
        found[identity] = file.parent
    return found


def _policy(directory):
    manifest = read(directory/'manifest.json')
    hashes = manifest.get('content_hashes', {})
    if manifest.get('format') == 'brass-complete-update-v1':
        available = present_payloads(directory, manifest)
    elif manifest.get('format') == 'brass-portable-policy-v1':
        available = {'policy.pt', 'metadata.json'}
        if set(hashes) != available or {p.name for p in directory.iterdir()} != available | {'manifest.json'}:
            raise ValueError('incomplete policy/update envelope: '+str(directory))
    else:
        raise ValueError('unknown policy/update envelope: '+str(directory))
    for name, sha in hashes.items():
        if not isinstance(sha, str) or not HEX.fullmatch(sha) or (name in available and file_hash(directory/name) != sha):
            raise ValueError('artifact hash mismatch: '+str(directory/name))
    metadata = read(directory/'metadata.json')
    fields = {k: v for k, v in metadata.items() if k != 'id'}
    expected_id = 'policy-sha256-v1:'+digest(encoded({'metadata': fields, 'weights': hashes['policy.pt']}))
    if metadata.get('id') != expected_id:
        raise ValueError('policy identity mismatch')
    return metadata, manifest


def replay_transition(old, new):
    """Allow only a verified monotonic prefix extension; never hide a conflict."""
    mutable = {'moves', 'status', 'result', 'writer_pid'}
    if {k:v for k,v in old.items() if k not in mutable} != {k:v for k,v in new.items() if k not in mutable}:
        raise ValueError('replay identity/provenance conflict')
    a, b = old['moves'], new['moves']
    if a[:min(len(a),len(b))] != b[:min(len(a),len(b))]:
        raise ValueError('replay action prefix conflict')
    if old['status'] == 'completed':
        if len(b) > len(a) or (new['status'] == 'completed' and (a != b or old['result'] != new['result'])):
            raise ValueError('completed replay result conflict')
        return 'skip'
    if len(b) < len(a):
        if new['status'] == 'completed':
            raise ValueError('completed replay shorter than existing prefix')
        return 'skip'
    return 'advance' if len(b) > len(a) or new['status'] == 'completed' else 'skip'


def validate_replay(record):
    # Native legality/result validation, still without Torch or controller pickle.
    import brass
    from brass_ai.replay_archive import game_file, outcome
    payload = game_file(record)
    game = brass.Game(payload['seed']) if payload.get('seed') is not None else brass.Game.from_json(payload['start_state'])
    for action in payload['moves']:
        if game.is_over() or game.current_player() != action['player']:
            raise ValueError('invalid replay action actor')
        game.apply(json.dumps(action['move']))
    expected = outcome(game) if game.is_over() else None
    if record.get('result') != expected or (record['status'] == 'completed') != game.is_over():
        raise ValueError('replay status/result disagrees with actions')


def _canonical_inventory(inventory):
    """Canonicalize only filesystem-ordered diagnostic collections.

    v1 packages may have APFS/ext4-dependent diagnostic ordering. Their original
    manifest and blob hashes still validate unchanged before this comparison.
    Sorting (not deduplicating) preserves diagnostic multiplicity. Other fields,
    including nested historical provenance and all meaningful sequences, retain
    their exact values/order; this never receives or modifies replay actions.
    """
    result = dict(inventory)
    if 'historical_gaps' in inventory:
        if not isinstance(inventory['historical_gaps'], list):
            raise ValueError('inventory historical_gaps must be a list')
        gaps = []
        for gap in inventory['historical_gaps']:
            if not isinstance(gap, dict):
                raise ValueError('inventory historical_gaps rows must be objects')
            row = dict(gap)
            if 'journal_only_boundaries' in row:
                paths = row['journal_only_boundaries']
                if not isinstance(paths, list) or not all(isinstance(path, str) for path in paths):
                    raise ValueError('inventory journal_only_boundaries must be a list of strings')
                row['journal_only_boundaries'] = sorted(paths)
            gaps.append(row)
        result['historical_gaps'] = sorted(gaps, key=encoded)
    return result


def _inventory(tree, manifest):
    """Recompute semantic closure from payloads; never trust declared counts."""
    runs = manifest['experiments']
    if manifest['experiment_id'] not in runs:
        raise ValueError('selected experiment is missing')
    models = {}
    for path in (tree/'players/models').glob('*/manifest.json'):
        metadata, _ = _policy(path.parent)
        if metadata['id'] in models:
            raise ValueError('duplicate public model identity')
        models[metadata['id']] = path.parent
    from brass_ai.replay_archive import Archive
    archive = Archive(tree/'replays')
    replays = {r['game_id']: r for r in archive.records(include_duplicates=True)}
    for path in archive.games.glob('*'):
        identity = path.name.split('.')[0]
        record = archive.read(identity)
        if record.get('game_id') != identity or record.get('format') != 'brass-replay/1':
            raise ValueError('replay identity/envelope mismatch')
        if record.get('status') not in ('completed', 'interrupted', 'saved'):
            raise ValueError('replay is active or has invalid status')
        validate_replay(record)
    result = {'experiments': len(runs), 'updates': 0, 'models': len(models), 'replays': len(replays), 'historical_gaps': []}
    for identity, relative in sorted(runs.items()):
        if not IDENTITY.fullmatch(identity) or relative != 'training/runs/'+identity:
            raise ValueError('invalid experiment mapping')
        run = tree/relative
        metadata = read_experiment(run)
        library_intent(run, metadata)
        if metadata['id'] != identity:
            raise ValueError('experiment identity mismatch')
        config = read(run/'config.json')
        learner = config.get('learner')
        if 'learner' in config or 'learner' in metadata or metadata['mode'] == 'bc-current':
            if (learner != BC_LEARNER or metadata.get('learner') != learner or
                    metadata['mode'] == 'stability-study' or 'stability_arm' in metadata or 'stability_protocol' in metadata):
                raise ValueError('inconsistent BC learner declarations in config and experiment')
            TrainConfig.from_mapping(config)
        parent = metadata.get('parent')
        if parent:
            if parent['experiment_id'] not in runs or parent['experiment_id'] == identity:
                raise ValueError('missing/invalid ancestor experiment')
            checkpoint = tree/runs[parent['experiment_id']]/'updates'/('%06d' % parent['update'])
            if not (checkpoint/'optimizer.pt').is_file() or file_hash(checkpoint/'manifest.json') != parent['checkpoint_sha256']:
                raise ValueError('ancestor checkpoint identity mismatch')
            if read(checkpoint/'metadata.json')['id'] != parent['model_id']:
                raise ValueError('ancestor model identity mismatch')
        updates = sorted((p for p in (run/'updates').iterdir() if p.name.isdecimal()), key=lambda p: int(p.name))
        if not updates:
            raise ValueError('experiment has no complete updates')
        if not (updates[-1]/'optimizer.pt').is_file():
            raise ValueError('latest checkpoint is not resumable')
        numbers = [int(p.name) for p in updates]
        if numbers[-1] > 10000000:
            raise ValueError('unreasonable update counter')
        gaps = sorted(set(range(numbers[-1]+1))-set(numbers))
        legacy = metadata['mode'] == 'legacy-adopted' or bool(metadata.get('historical_gaps'))
        if gaps and (not legacy or any(n > metadata.get('base_update', 0) for n in gaps)):
            raise ValueError('missing completed update history / gap: '+identity)
        stubs = [p.parent.relative_to(run).as_posix() for p in (run/'ancestry').glob('**/manifest.json')
                 if read(p).get('format') == 'brass-complete-update-v1'
                 and not UPDATE_FILES.issubset({q.name for q in p.parent.iterdir()})]
        if stubs:
            result['historical_gaps'].append({'experiment_id': identity, 'journal_only_boundaries': stubs})
        if metadata.get('historical_parent') and not metadata.get('parent'):
            result['historical_gaps'].append({'experiment_id': identity, 'unresolved_historical_parent': metadata['historical_parent']})
        if gaps:
            result['historical_gaps'].append({'experiment_id': identity, 'missing_updates': gaps})
        for update in updates:
            policy, envelope = _policy(update)
            if update.name != '%06d' % policy['update'] or envelope.get('update') != policy['update']:
                raise ValueError('update counter mismatch')
            if policy['id'] not in models:
                raise ValueError('missing public model for completed update')
            public, _ = _policy(models[policy['id']])
            if public != policy:
                raise ValueError('public model metadata conflict')
            for ref in envelope.get('snapshots', []):
                sha = ref['sha256']
                if not HEX.fullmatch(sha) or file_hash(run/'weights'/(sha+'.pt')) != sha:
                    raise ValueError('missing/corrupt opponent dependency')
            journal = read(update/'manifests.json')
            if (journal.get('learner') != learner or
                    (learner and ('stability_arm' in journal or 'stability_protocol' in journal))):
                raise ValueError('inconsistent learner declaration in committed journal')
            for replay_id in journal.get('replay_ids', []):
                if replay_id not in replays:
                    raise ValueError('missing linked replay: '+replay_id)
            expected = journal.get('train', [])
            owner = journal.get('experiment_id', identity)
            owner_meta = read_experiment(tree/runs[owner]) if owner in runs else metadata
            recording_owner = owner_meta.get('legacy_replay_identity', owner)
            available = {r['provenance'].get('game') for r in replays.values()
                         if r['provenance'].get('experiment') == recording_owner
                         and r['provenance'].get('update') == policy['update'] and r['status'] == 'completed'}
            missing = [g.get('game_id') for g in expected if isinstance(g, dict) and g.get('game_id') not in available]
            inherited_legacy = legacy and policy['update'] <= metadata.get('base_update', 0)
            if missing and not inherited_legacy:
                raise ValueError('missing training replay recordings: '+str(missing))
            if missing:
                result['historical_gaps'].append({'experiment_id': identity, 'update': policy['update'], 'missing_training_replays': missing})
            result['updates'] += 1
        for path in (run/'evaluation').glob('**/suite.json'):
            suite = read(path)
            for entries in suite.get('lineups', {}).values():
                for entry in entries:
                    if isinstance(entry, dict) and entry.get('id') not in models:
                        suite_updates = [read(row).get('update', 0) for row in path.parent.glob('candidates/*/games/*.json')]
                        if legacy and max(suite_updates or [0]) <= metadata.get('base_update', 0):
                            result['historical_gaps'].append({'experiment_id': identity,
                                'missing_historical_evaluation_model': entry.get('id'),
                                'suite': path.relative_to(run).as_posix()})
                        else:
                            raise ValueError('missing evaluation opponent dependency: '+str(entry.get('id')))
        # Outcome rows are not substitutes for complete action histories.
        for path in (run/'evaluation').glob('**/games/*.json'):
            row = read(path)
            matches = [r for r in replays.values() if r['provenance'].get('experiment') == row.get('suite_id')
                       and r['provenance'].get('candidate_id') == row.get('candidate_id')
                       and r['provenance'].get('game') == row.get('game_id') and r['status'] == 'completed']
            if not matches:
                if not legacy or row.get('update', 0) > metadata.get('base_update', 0):
                    raise ValueError('missing evaluation replay: '+str(path))
                result['historical_gaps'].append({'experiment_id': identity, 'missing_evaluation_replay': str(path.relative_to(tree))})
    # Parent cycles cannot be legitimized by circular inclusion.
    for identity in runs:
        seen = set()
        while identity:
            if identity in seen:
                raise ValueError('ancestor cycle')
            seen.add(identity)
            parent = read_experiment(tree/runs[identity]).get('parent')
            identity = parent['experiment_id'] if parent else None
    retention = plan_runs([tree/relative for relative in runs.values()], verify=False)
    retired = len(retention.manifests) - len(retention.protected) - len(retention.candidates)
    if retired:
        result['retired_updates'] = retired
    return _canonical_inventory(result)


def pack(run, destination, *, replay_root=None):
    """Freeze a stopped experiment and all ancestors into a deduplicated archive."""
    run, destination = Path(run).resolve(), Path(destination).resolve()
    if destination.exists():
        raise ValueError('package destination exists')
    if not (run/'experiment.json').exists():
        raise ValueError('historical run needs explicit adopt before packaging')
    primary = read_experiment(run)['id']
    project = project_for(run)
    registry = experiments_under(project/'training/runs' if project else run.parent)
    selected = {}
    def include(identity):
        if identity in selected:
            return
        if identity not in registry:
            raise ValueError('missing ancestor experiment: '+identity)
        selected[identity] = registry[identity]
        parent = read_experiment(registry[identity]).get('parent')
        if parent:
            include(parent['experiment_id'])
    include(primary)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack, tempfile.TemporaryDirectory(prefix='.package-', dir=str(destination.parent)) as temp:
        for path in sorted(selected.values()):
            stack.enter_context(writer_lock(path))
        tree = Path(temp)/'tree'; tree.mkdir()
        files = {}
        def add(source, relative):
            safe_path(relative)
            if source.is_symlink() or not source.is_file():
                raise ValueError('package requires regular files, not symlinks: '+str(source))
            target = tree/relative
            sha = file_hash(source)
            if relative in files:
                if files[relative]['sha256'] != sha:
                    raise ValueError('same ID has conflicting bytes')
                return
            link_or_copy(source, target)
            files[relative] = {'sha256': sha, 'size': source.stat().st_size}
        def add_model(directory):
            metadata, envelope = _policy(directory)
            name = 'players/models/policy-'+metadata['id'].split(':')[-1]
            for filename in ('policy.pt', 'metadata.json'):
                add(directory/filename, name+'/'+filename)
            public_manifest = encoded({'format': 'brass-portable-policy-v1', 'content_hashes':
                                      {n: envelope['content_hashes'][n] for n in ('policy.pt', 'metadata.json')}})
            temporary_manifest = Path(temp)/('public-'+digest(public_manifest))
            temporary_manifest.write_bytes(public_manifest)
            add(temporary_manifest, name+'/manifest.json')
        for identity, source in sorted(selected.items()):
            prefix = 'training/runs/'+identity
            # Immutable payloads and metadata; pointers/logs/exemplar caches are derived.
            for path in sorted(source.rglob('*')):
                relative = path.relative_to(source)
                if any(p.startswith('.') for p in relative.parts) or not path.is_file():
                    continue
                if relative.parts[0] in ('latest', 'best', 'metrics.jsonl') or relative.as_posix() == 'replays/index.json':
                    continue
                add(path, prefix+'/'+relative.as_posix())
            # Every retained policy is carried, including evaluation exports.
            for path in sorted(source.rglob('manifest.json')):
                if any(p.startswith('.') for p in path.relative_to(source).parts):
                    continue
                kind = read(path).get('format')
                if kind == 'brass-complete-update-v1' and path.parent.parent != source/'updates' and not UPDATE_FILES.issubset({p.name for p in path.parent.iterdir()}):
                    continue  # Historical ancestry journal-only evidence, never a full checkpoint.
                if kind in ('brass-complete-update-v1', 'brass-portable-policy-v1'):
                    add_model(path.parent)
        # Evaluation suites store identities, not source paths. Resolve any
        # externally selected opponent from the shared project model catalog.
        required = set()
        for path in tree.glob('training/runs/*/evaluation/**/suite.json'):
            for entries in read(path).get('lineups', {}).values():
                required.update(e['id'] for e in entries if isinstance(e, dict))
        available = {read(p)['id'] for p in (tree/'players/models').glob('*/metadata.json')}
        if project:
            for path in (project/'players/models').glob('*/metadata.json'):
                if read(path)['id'] in required-available:
                    add_model(path.parent)
                    available.add(read(path)['id'])
        from brass_ai.replay_archive import Archive
        archive = Archive(replay_root or os.environ.get('BRASS_REPLAY_ROOT') or (project/'replays' if project else None))
        aliases = set(selected)
        suites, candidates, linked = set(), set(), set()
        for identity, source in selected.items():
            aliases.add(read_experiment(source).get('legacy_replay_identity', identity))
            for path in source.glob('updates/*/manifests.json'):
                linked.update(read(path).get('replay_ids', []))
            for path in source.glob('evaluation/**/suite.json'):
                suites.add(read(path).get('suite_id'))
            for path in source.glob('evaluation/**/games/*.json'):
                row = read(path); suites.add(row.get('suite_id')); candidates.add(row.get('candidate_id'))
        for record in archive.records(include_duplicates=True):
            provenance = record['provenance']
            historical_paths = [p.get('path', '') for p in provenance.get('sources', [])]
            historical_paths.append(provenance.get('experiment', ''))
            historical_match = any(str(Path(p)).startswith(str(source)+os.sep)
                                   for p in historical_paths if isinstance(p, str)
                                   for source in selected.values())
            if (not historical_match and record['game_id'] not in linked and provenance.get('experiment') not in aliases
                    and provenance.get('run_experiment') not in aliases
                    and not (provenance.get('experiment') in suites and provenance.get('candidate_id') in candidates)):
                continue
            if record['status'] == 'in-progress':
                raise ValueError('replay still active; stop writer before package')
            source = archive.path(record['game_id'])
            if not source.exists():
                source = archive.path(record['game_id'], '.jsonl')
            add(source, 'replays/games/'+source.name)
        manifest = {'format': FORMAT, 'experiment_id': primary,
                    'experiments': {i: 'training/runs/'+i for i in selected}, 'files': files}
        inventory = _inventory(tree, manifest)
        manifest['inventory'] = inventory
        manifest['package_id'] = digest(encoded(manifest))
        target = Path(temp)/'package.tar.gz'
        with tarfile.open(target, 'w:gz') as archive_out:
            data = encoded(manifest)
            item = tarfile.TarInfo('manifest.json'); item.size = len(data)
            archive_out.addfile(item, io.BytesIO(data))
            emitted = set()
            for relative, row in sorted(files.items()):
                if row['sha256'] not in emitted:
                    archive_out.add(tree/relative, arcname='blobs/'+row['sha256'], recursive=False)
                    emitted.add(row['sha256'])
        # Verify the actual serialized archive before atomic publication.
        report = verify(target)
        with target.open('rb') as stream:
            os.fsync(stream.fileno())
        os.link(target, destination)  # exclusive: never replace a concurrent destination
        sync_dir(destination.parent)
        report['package'] = str(destination)
        report['package_bytes'] = destination.stat().st_size
        return report


@contextmanager
def _unpack(package):
    with tempfile.TemporaryDirectory(prefix='brass-verify-') as name:
        root = Path(name); blobs = root/'blobs'; blobs.mkdir()
        with tarfile.open(package, 'r:*') as archive:
            members = archive.getmembers()
            if len({m.name for m in members}) != len(members) or any(not m.isfile() for m in members):
                raise ValueError('package requires unique regular members; links forbidden')
            headers = [m for m in members if m.name == 'manifest.json']
            if len(headers) != 1 or headers[0].size > 32*1024**2:
                raise ValueError('missing/oversized package manifest')
            manifest = json.load(archive.extractfile(headers[0]))
            if (not isinstance(manifest, dict) or set(manifest) != {'format','experiment_id','experiments','files','inventory','package_id'}
                    or not all(isinstance(manifest[k], dict) for k in ('experiments','files','inventory'))
                    or manifest.get('format') != FORMAT):
                raise ValueError('unknown experiment package format')
            claimed = manifest.get('package_id')
            if claimed != digest(encoded({k: v for k, v in manifest.items() if k != 'package_id'})):
                raise ValueError('package manifest integrity mismatch')
            rows = manifest['files']
            expected = {}
            for relative, row in rows.items():
                safe_path(relative)
                if (not isinstance(row, dict) or set(row) != {'sha256','size'} or not isinstance(row['sha256'], str)
                        or not HEX.fullmatch(row['sha256']) or type(row['size']) is not int or row['size'] < 0):
                    raise ValueError('invalid blob declaration')
                if row['sha256'] in expected and expected[row['sha256']] != row['size']:
                    raise ValueError('conflicting blob sizes')
                expected[row['sha256']] = row['size']
            if {m.name for m in members} != {'manifest.json'} | {'blobs/'+h for h in expected}:
                raise ValueError('missing or undeclared package blobs')
            for member in members:
                if member.name == 'manifest.json':
                    continue
                sha = member.name.split('/')[1]
                if member.size != expected[sha]:
                    raise ValueError('blob size mismatch')
                with archive.extractfile(member) as source, (blobs/sha).open('wb') as dest:
                    shutil.copyfileobj(source, dest)
                if file_hash(blobs/sha) != sha:
                    raise ValueError('package blob integrity mismatch')
        tree = root/'tree'; tree.mkdir()
        for relative, row in rows.items():
            path = safe_path(relative)
            allowed = ((len(path.parts) >= 5 and path.parts[:2] == ('training', 'runs') and path.parts[2] in manifest['experiments'])
                       or (len(path.parts) == 4 and path.parts[:2] == ('players', 'models') and path.parts[3] in ('manifest.json', 'metadata.json', 'policy.pt'))
                       or (len(path.parts) == 3 and path.parts[:2] == ('replays', 'games')))
            # Run metadata sits directly inside its run directory.
            allowed = allowed or (len(path.parts) == 4 and path.parts[:2] == ('training', 'runs') and path.parts[2] in manifest['experiments'])
            if not allowed:
                raise ValueError('file outside supported experiment layout')
            link_or_copy(blobs/row['sha256'], tree/relative)
        inventory = _inventory(tree, manifest)
        if _canonical_inventory(inventory) != _canonical_inventory(manifest['inventory']):
            raise ValueError('package inventory/completeness mismatch')
        yield tree, manifest


def _summary(package, manifest):
    rows = manifest['files'].values()
    sizes = {r['sha256']: r['size'] for r in rows}
    return dict(manifest['inventory'], package_id=manifest['package_id'], experiment_id=manifest['experiment_id'],
                logical_bytes=sum(r['size'] for r in rows), unique_blob_bytes=sum(sizes.values()),
                package_bytes=Path(package).stat().st_size,
                verification='hashes, identities, native replay legality and dependency/history closure (reported historical gaps excepted); no pickle loaded')


def package_errors(function):
    @wraps(function)
    def checked(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except (KeyError, IndexError, TypeError, tarfile.TarError) as exc:
            raise ValueError('invalid experiment package structure: '+str(exc)) from exc
    return checked


@package_errors
def verify(package):
    with _unpack(package) as (_, manifest):
        return _summary(package, manifest)


def _exchange_directories(left, right):
    """Atomically publish one complete numeric journal on supported Mac/Linux filesystems."""
    import ctypes
    import sys
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == 'darwin' and hasattr(libc, 'renamex_np'):
        result = libc.renamex_np(os.fsencode(left), os.fsencode(right), 2)  # RENAME_SWAP
    elif sys.platform.startswith('linux') and hasattr(libc, 'renameat2'):
        result = libc.renameat2(-100, os.fsencode(left), -100, os.fsencode(right), 2)  # RENAME_EXCHANGE
    else:
        raise ValueError('atomic update directory exchange unavailable for incremental import')
    if result:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    sync_dir(left.parent)
    sync_dir(right.parent)


def _overlay(source, destination):
    """Add missing regular files without modifying existing hard-linked bytes."""
    changed = False
    if source.is_symlink() or destination.is_symlink():
        raise ValueError('symlink in checkpoint import')
    if source.is_dir():
        for path in source.rglob('*'):
            target = destination/path.relative_to(source)
            if path.is_symlink() or target.is_symlink():
                raise ValueError('symlink in checkpoint import')
            if path.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            elif target.exists():
                if file_hash(path) != file_hash(target):
                    raise ValueError('immutable ID/content conflict: '+str(target))
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                os.link(path, target)
                changed = True
    return changed


def _prepare_updates(staged, manifest, destinations, external_parents=()):
    incoming = {i: staged/r for i, r in manifest['experiments'].items()}
    order = []
    def visit(identity):
        if identity not in order:
            parent = read_experiment(incoming[identity]).get('parent')
            if parent and parent not in external_parents:
                visit(parent['experiment_id'])
            order.append(identity)
    visit(manifest['experiment_id'])
    if set(order) != set(incoming):
        raise ValueError('package contains unrelated experiments')
    existing = [target for target in destinations.values() if target.exists()]
    plan_runs(existing, verify=False, external_parents=external_parents)
    images, exchanges = {}, {}
    for identity in order:
        target, source = destinations[identity], incoming[identity]
        if not target.exists():
            images[identity] = source
            continue
        # Validate the independent destination before an incoming receipt can authorize
        # absence, or an incoming Adam can accidentally repair unmarked corruption.
        for update in (target/'updates').iterdir():
            if update.name.isdecimal():
                _, envelope = _policy(update)
                for ref in envelope.get('snapshots', []):
                    sha = ref['sha256']
                    if not HEX.fullmatch(sha) or file_hash(target/'weights'/(sha+'.pt')) != sha:
                        raise ValueError('existing frozen opponent dependency is corrupt')
        image = staged/'.merged'/identity
        image.mkdir(parents=True)
        os.link(target/'experiment.json', image/'experiment.json')
        if (target/'best').exists():
            os.link(target/'best', image/'best')
        shutil.copytree(target/'updates', image/'updates', copy_function=os.link, symlinks=True)
        if _overlay(source/'updates', image/'updates'):
            if target.stat().st_dev != image.stat().st_dev:
                raise ValueError('incremental checkpoint import requires a single filesystem')
            exchanges[identity] = image/'updates'
        images[identity] = image
    context = plan_runs(images.values(), verify=False, external_parents=external_parents)
    permanent = {context.manifests[path] for path, reasons in context.protected.items()
                 if reasons & {'fork source', 'best', 'development selection', 'explicit'}}
    derived = 0
    for identity, image in images.items():
        target = destinations[identity]
        for checkpoint in (image/'updates').iterdir():
            if not checkpoint.name.isdecimal():
                continue
            sha = context.manifests[checkpoint]
            pin_file = Path('updates/pins')/(sha+'.json')
            if sha in permanent and not (image/pin_file).exists():
                pin(checkpoint)
                derived += 1
                if target.exists():
                    exchanges[identity] = image/'updates'
        if identity in exchanges and target.stat().st_dev != image.stat().st_dev:
            raise ValueError('incremental checkpoint import requires a single filesystem')
    return order, exchanges, derived


@package_errors
def import_package(package, project):
    """Preflight all conflicts before publication; immutable units commit atomically.

    New runs appear in one directory rename. Existing runs exchange a complete
    merged update journal, including pins, after dependencies/models/replays.
    A crash leaves the old or new complete journal. Repeat import idempotently. No cross-directory global transaction claimed.
    """
    with _unpack(package) as (tree, manifest):
        return _import_tree(tree, manifest, project, _summary(package, manifest))


def _import_tree(tree, manifest, project, summary, *, external_parent=None):
    from .library import reconcile_models
    from .model_inventory import inventory_writer
    project = Path(project).resolve()
    with writer_lock(project/'.experiment-import'):
        registry = experiments_under(project/'training/runs')
        destinations = {i: registry.get(i, project/'training/runs'/i) for i in manifest['experiments']}
        with ExitStack() as locks:
            sources = set(destinations.values())
            if external_parent:
                owner = registry.get(external_parent['experiment_id'])
                if not owner:
                    raise ValueError('external parent experiment is missing from destination')
                sources.add(owner)
            for run in sorted(sources):
                locks.enter_context(writer_lock(run))
            if external_parent:
                boundary = owner/'updates'/('%06d' % external_parent['update'])
                policy, envelope = _policy(boundary)
                if (not (boundary/'optimizer.pt').is_file() or policy['id'] != external_parent['model_id']
                        or file_hash(boundary/'manifest.json') != external_parent['checkpoint_sha256']
                        or external_parent.get('policy_sha256', envelope['content_hashes']['policy.pt'])
                           != envelope['content_hashes']['policy.pt']):
                    raise ValueError('external parent checkpoint identity mismatch')
            # Resolve public IDs before path conflict checks/counting. Curated
            # names (ppo-300 etc.) remain the canonical existing destination.
            existing_models = {}
            for path in (project/'players/models').glob('*/metadata.json'):
                value = read(path)
                if value['id'] in existing_models:
                    if _policy(existing_models[value['id']]) != _policy(path.parent):
                        raise ValueError('public model identity conflict')
                existing_models[value['id']] = path.parent
            model_destinations = {}
            for path in (tree/'players/models').glob('*/metadata.json'):
                identity = read(path)['id']
                if identity in existing_models and _policy(existing_models[identity]) != _policy(path.parent):
                    raise ValueError('public model identity conflict')
                model_destinations[path.parent.name] = existing_models.get(identity, project/'players/models'/path.parent.name)
            from brass_ai.replay_archive import Archive
            replay_dest = Archive(project/'replays')
            # Lock each logical replay ID including GUI/external writers.
            game_locks = locks.enter_context(ExitStack())
            replay_actions = {}
            for path in (tree/'replays/games').glob('*'):
                replay_id = path.name.split('.')[0]
                game_locks.enter_context(writer_lock(replay_dest.root/'locks'/('transfer-'+replay_id)))
                import fcntl
                (replay_dest.root/'locks').mkdir(exist_ok=True)
                handle = game_locks.enter_context((replay_dest.root/'locks'/replay_id).open('a+b'))
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as exc:
                    raise ValueError('destination replay is active') from exc
                old_paths = list(replay_dest.games.glob(replay_id+'.json*'))
                action = 'add'
                if old_paths:
                    old = replay_dest.read(replay_id)
                    if old['status'] == 'in-progress':
                        # We hold the exclusive recorder lock: an abandoned
                        # header is interrupted, not an active external writer.
                        old['status'] = 'interrupted'
                    validate_replay(old)
                    action = replay_transition(old, Archive(tree/'replays').read(replay_id))
                replay_actions[path.name] = (action, old_paths)
            targets = {}
            for relative, row in manifest['files'].items():
                parts = PurePosixPath(relative).parts
                if parts[:2] == ('training', 'runs'):
                    target = destinations[parts[2]].joinpath(*parts[3:])
                elif parts[:2] == ('players', 'models'):
                    target = model_destinations[parts[2]]/parts[3]
                else:
                    target = project/relative
                if any(p.is_symlink() for p in [target]+list(target.parents) if p != project.parent):
                    raise ValueError('import destination contains symlink')
                if parts[:2] != ('replays', 'games') and target.exists() and (not target.is_file() or file_hash(target) != row['sha256']):
                    raise ValueError('immutable ID/content conflict: '+str(target))
                targets[relative] = target
            added = sum(not p.exists() for r,p in targets.items() if not r.startswith('replays/')) + sum(a != 'skip' for a,_ in replay_actions.values())
            # Stage on destination filesystem. No input content is published before
            # all verification/conflict checks and staging writes have succeeded.
            project.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix='.experiment-stage-', dir=str(project)) as temp:
                staged = Path(temp)
                for relative in targets:
                    link_or_copy(tree/relative, staged/relative)
                order, exchanges, derived = _prepare_updates(staged, manifest, destinations, [external_parent] if external_parent else ())
                added += derived
                catalog_lock = locks.enter_context(ExitStack())
                inventory = catalog_lock.enter_context(inventory_writer(project))
                existing_runs = {dest: dest for dest in destinations.values() if dest.exists()}
                if existing_runs:
                    reconcile_models(project, registry, existing_runs, inventory=inventory, destinations=existing_runs)
                staged_runs = {i: staged/manifest['experiments'][i] for i in order}
                reconcile_models(project, {**registry, **staged_runs}, staged_runs.values(), inventory=inventory,
                                 destinations={staged_runs[i]: destinations[i] for i in order})
                sync_tree(staged)
                # Shared dependencies first: models as complete directories, replays as files.
                added_models = 0
                for source in (staged/'players/models').iterdir():
                    identity = read(source/'metadata.json')['id']
                    if identity in existing_models:
                        continue
                    target = project/'players/models'/source.name
                    if not target.exists():
                        target.parent.mkdir(parents=True, exist_ok=True)
                        os.replace(source, target); sync_dir(target.parent)
                        added_models += 1
                for source in (staged/'replays/games').glob('*'):
                    target = project/'replays/games'/source.name
                    action, old_paths = replay_actions[source.name]
                    if action == 'skip':
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    # Retain original interrupted bytes before advancing the one
                    # logical record. History is outside the library games index.
                    for old in old_paths:
                        history = project/'replays/history'/old.name.split('.')[0]/(file_hash(old)+'-'+old.name)
                        if not history.exists():
                            link_or_copy(old, history); sync_dir(history.parent)
                    os.replace(source, target); sync_dir(target.parent)
                    for old in old_paths:
                        if old != target:
                            old.unlink()
                    sync_dir(target.parent)
                for identity in order:
                    target = destinations[identity]
                    source = staged/manifest['experiments'][identity]
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if not target.exists():
                        os.replace(source, target); sync_dir(target.parent)
                        continue
                    # Make dependencies durable before exposing any extension. The old
                    # and new update directories each have a complete, resumable latest.
                    for path in sorted(source.rglob('*')):
                        relative = path.relative_to(source)
                        if relative.parts[0] == 'updates':
                            continue
                        dest = target/relative
                        if not path.exists() or dest.exists():
                            continue
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        os.replace(path, dest)
                        sync_dir(dest.parent)
                    if identity in exchanges:
                        _exchange_directories(exchanges[identity], target/'updates')
                for target in destinations.values():
                    updates = sorted((target/'updates').glob('[0-9]*'), key=lambda p: int(p.name))
                    atomic(target/'latest', encoded({'update': updates[-1].name}))
                    metrics = b''.join(encoded(row) for u in updates for row in read(u/'metrics.json')['records'])
                    atomic(target/'metrics.jsonl', metrics)
                models, journals = reconcile_models(project, {**registry, **destinations},
                                             [destinations[i] for i in order], model_destinations.values(), inventory=inventory)
                receipt = dict(summary, added_files=added, catalog_models=models['models'], added_models=added_models,
                               reused_models=len(model_destinations)-added_models,
                               ambiguous_continuations=models['ambiguous_continuations'],
                               added_replays=sum(a == 'add' for a, _ in replay_actions.values()),
                               advanced_replays=sum(a == 'advance' for a, _ in replay_actions.values()),
                               deduplicated_replays=sum(a == 'skip' for a, _ in replay_actions.values()),
                               experiment_path=str(destinations[manifest['experiment_id']]))
            # Read interrupted journals in their actual lock state while the run locks still protect exemplars.
            catalog_lock.close()
            game_locks.close()
            from .library_replays import reconcile_replays
            coverage = reconcile_replays(project, destinations.values(), journals)
            receipt['library_coverage'] = coverage
            receipt['indexed_replays'] = coverage['indexed_replays']
            receipt['added_replays'] += coverage['added_replays']
        atomic(project/'training/runs'/('.import-'+manifest['package_id']+'.json'), encoded(receipt))
        return receipt
