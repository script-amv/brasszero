# SPDX-License-Identifier: Apache-2.0
"""Experiment identity, exclusive writers and explicit full-state branching.

Metadata is additive. Historical checkpoint bytes are never rewritten. Numerical
imports are lazy so identity/transfer tooling remains standard-library-only.
"""
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import tempfile
import uuid

FORMAT = 'brass-experiment/1'


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()


def atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.'+path.name+'-', dir=str(path.parent))
    try:
        with os.fdopen(fd, 'wb') as out:
            out.write(data); out.flush(); os.fsync(out.fileno())
        os.replace(name, path)
        sync_dir(path.parent)
    finally:
        Path(name).unlink(missing_ok=True)


def sync_dir(path):
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def sync_tree(root):
    for parent, directories, files in os.walk(root, topdown=False):
        sync_dir(Path(parent))


@contextmanager
def writer_lock(run):
    run = Path(run).resolve()
    run.parent.mkdir(parents=True, exist_ok=True)
    with (run.parent/('.'+run.name+'.writer.lock')).open('a+b') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('experiment is active; stop its writer before transfer/resume') from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def project_for(run):
    for root in Path(run).resolve().parents:
        if root.name == 'runs' and root.parent.name == 'training':
            return root.parent.parent
    return None


def read_experiment(run):
    value = json.loads((Path(run)/'experiment.json').read_text())
    if value.get('format') != FORMAT or not isinstance(value.get('id'), str):
        raise ValueError('invalid experiment manifest')
    return value


def parent_identity(source):
    meta = json.loads((source/'metadata.json').read_bytes())
    manifest = (source/'manifest.json').read_bytes()
    return {'experiment_id': read_experiment(source.parent.parent)['id'], 'update': meta['update'],
            'checkpoint_sha256': hashlib.sha256(manifest).hexdigest(),
            'model_id': meta['id'], 'policy_sha256': json.loads(manifest)['content_hashes']['policy.pt']}


def same_state(a, b):
    """Exact tensor, RNG tuple and Adam slot comparisons across serialization."""
    import numpy as np
    import torch
    if type(a) is not type(b):
        return False
    if isinstance(a, torch.Tensor):
        return a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return a.dtype == b.dtype and np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same_state(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(same_state(x, y) for x, y in zip(a, b))
    return a == b


def ensure_experiment(run, config, *, state=None, parent=None):
    run = Path(run)
    if (run/'experiment.json').exists():
        metadata = read_experiment(run)
        if parent is not None and metadata.get('parent') != parent:
            raise ValueError('experiment provenance already exists with a different parent; never rewrite it')
        return metadata
    legacy = state is not None
    initial = sorted((run/'updates').glob('[0-9]*')) if legacy else []
    identity = ('legacy-'+hashlib.sha256((initial[0]/'manifest.json').read_bytes()).hexdigest()
                if legacy else 'exp-'+uuid.uuid4().hex)
    value = {'format': FORMAT, 'id': identity, 'mode': 'legacy-adopted' if legacy else 'new',
             'parent': parent, 'base_update': state.completed_updates if legacy else 0,
             'created_at': datetime.now(timezone.utc).isoformat(),
             'producer': {'system': platform.system(), 'machine': platform.machine(), 'python': platform.python_version()},
             'historical_gaps': ['Pre-adoption updates/models/replays may be missing; inventory required.'] if legacy else []}
    if legacy:
        lineage = run/'ancestry/continuation-lineage.json'
        if lineage.exists():
            value['historical_parent'] = json.loads(lineage.read_text())
        from .replay import experiment_identity
        value['legacy_replay_identity'] = experiment_identity(state.config, state.master_seed)
    if not (run/'config.json').exists():
        atomic(run/'config.json', encoded(asdict(config)))
    atomic(run/'experiment.json', encoded(value))
    return value


def link_or_copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copyfile(source, target)
    with target.open('rb') as stream:
        os.fsync(stream.fileno())


def branch_experiment(checkpoint, destination):
    """Create a new full-state identity from an explicit trusted boundary.

    Retained checkpoint prefix includes development-best and league dependencies.
    Parent provenance records the exact checkpoint manifest, not an absolute path.
    """
    from .artifacts import load_update, capture_rng_states, restore_rng_states
    source = Path(checkpoint).resolve()
    destination = Path(destination).resolve()
    if source.name in ('latest', 'best'):
        raise ValueError('branch requires an explicit numeric checkpoint, not latest/best')
    parent = source.parent.parent
    if destination.exists():
        raise ValueError('branch destination already exists; choose a new experiment')
    before = capture_rng_states()
    temporary = None
    with writer_lock(parent), writer_lock(destination):
        try:
            loaded = load_update(source)
            metadata = ensure_experiment(parent, loaded.state.config, state=loaded.state)
            temporary = Path(tempfile.mkdtemp(prefix='.branch-', dir=str(destination.parent)))
            for directory in ('updates', 'weights', 'policies'):
                if not (parent/directory).exists():
                    continue
                for path in (parent/directory).rglob('*'):
                    if not path.is_file() or path.is_symlink():
                        continue
                    relative = path.relative_to(parent)
                    if directory in ('updates', 'policies') and (not relative.parts[1].isdecimal() or int(relative.parts[1]) > loaded.state.completed_updates):
                        continue
                    link_or_copy(path, temporary/relative)
            from .retention import pin
            pin(source)
            for pin_file in (parent/'updates/pins').glob('*.json'):
                if json.loads(pin_file.read_bytes())['update'] <= loaded.state.completed_updates:
                    link_or_copy(pin_file, temporary/'updates/pins'/pin_file.name)
            own = {'format': FORMAT, 'id': 'exp-'+uuid.uuid4().hex, 'mode': 'full-state-branch',
                   'parent': {'experiment_id': metadata['id'], 'update': loaded.state.completed_updates,
                              'checkpoint_sha256': hashlib.sha256((source/'manifest.json').read_bytes()).hexdigest(),
                              'model_id': json.loads((source/'metadata.json').read_text())['id']},
                   'base_update': loaded.state.completed_updates,
                   'created_at': datetime.now(timezone.utc).isoformat(),
                   'producer': {'system': platform.system(), 'machine': platform.machine(), 'python': platform.python_version()},
                   'historical_gaps': metadata.get('historical_gaps', [])}
            if loaded.state.config.learner:
                own['learner'] = loaded.state.config.learner
            atomic(temporary/'experiment.json', encoded(own))
            atomic(temporary/'config.json', encoded(asdict(loaded.state.config)))
            sync_tree(temporary)
            os.replace(temporary, destination); temporary = None
            sync_dir(destination.parent)
            return own
        finally:
            if temporary is not None:
                shutil.rmtree(temporary)
            restore_rng_states(before)


def preflight(run, metadata, state, policy, *, resumed, notes):
    profile = ({'learner': state.config.learner, 'critic_optimizer': 'restored',
                'loss_priority_next_game_id': state.loss_priority['next_game_id']} if state.config.learner else {})
    return {**profile, 'event': 'preflight', 'experiment_id': metadata['id'], 'mode': metadata['mode'],
            'checkpoint': str(Path(run)/'updates'/('%06d' % state.completed_updates)),
            'last_completed_update': state.completed_updates, 'completed_training_games': state.next_game_id,
            'model_id': policy, 'restoration': 'full-state' if resumed else 'initialized',
            'optimizer': 'restored' if resumed else 'initialized',
            'rng': 'restored-with-limitations' if notes else 'restored' if resumed else 'initialized',
            'opponents': {'status': 'restored' if resumed else 'initialized',
                          'training': list(state.league_snapshot_ids), 'holdouts': [s.id for s in state.holdouts]},
            'next_update': state.completed_updates+1, 'device': state.config.runtime.device,
            'limitations': list(notes),
            'recovery': 'Only complete numeric update directories count. Any uncommitted collection/PPO is repeated; recorded prefixes are verified.',
            'parent': metadata.get('parent'),
            'historical_gaps': metadata.get('historical_gaps', []),
            'historical_parent': metadata.get('historical_parent')}


def adopt_experiment(run, parent_checkpoint=None):
    from .artifacts import load_update, capture_rng_states, restore_rng_states
    before = capture_rng_states()
    try:
        parent = None
        if parent_checkpoint is not None:
            checkpoint = Path(parent_checkpoint).resolve()
            if checkpoint.name in ('latest', 'best') or checkpoint.parent.parent == Path(run).resolve():
                raise ValueError('adoption parent requires a different experiment and explicit numeric checkpoint')
            with writer_lock(checkpoint.parent.parent):
                original = load_update(checkpoint)
                from .retention import pin
                pin(checkpoint)
                metadata = ensure_experiment(checkpoint.parent.parent, original.state.config, state=original.state)
                parent = {'experiment_id': metadata['id'], 'update': original.state.completed_updates,
                          'checkpoint_sha256': hashlib.sha256((checkpoint/'manifest.json').read_bytes()).hexdigest(),
                          'model_id': json.loads((checkpoint/'metadata.json').read_text())['id']}
        with writer_lock(run):
            loaded = load_update(Path(run)/'latest')
            return ensure_experiment(Path(run), loaded.state.config, state=loaded.state, parent=parent)
    finally:
        restore_rng_states(before)
