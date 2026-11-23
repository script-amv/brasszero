"""Private controller recordings. Standard library only; never passed to agents.

One fsynced JSONL write-ahead action per decision, then atomic gzip publication.
The index is a disposable view; records contain all metadata. No retention policy.
"""
import argparse
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from functools import lru_cache, wraps
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import uuid

from .fingerprints import engine_fingerprint

ROOT = Path(__file__).resolve().parents[3]
FORMAT = 'brass-replay/1'


class ReplayStorageError(RuntimeError):
    """Recording failed; the controller must stop, never evict games."""



def storage_operation(function):
    @wraps(function)
    def checked(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except OSError as exc:
            raise ReplayStorageError('Replay archive storage failed; controller stopped. Repair storage or free space; no recordings were evicted.') from exc
    return checked


_ARCHIVE_ROOT = ContextVar('replay_archive_root', default=None)


@contextmanager
def archive_context(root):
    token = _ARCHIVE_ROOT.set(root)
    try:
        yield
    finally:
        _ARCHIVE_ROOT.reset(token)


_CONTEXT = ContextVar('replay_controller_context', default={})


@contextmanager
def recording_context(**values):
    token = _CONTEXT.set({**_CONTEXT.get(), **values})
    try:
        yield
    finally:
        _CONTEXT.reset(token)


def recording_metadata():
    return dict(_CONTEXT.get())


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


@lru_cache(None)
def engine_id():
    return engine_fingerprint(ROOT)


@lru_cache(None)
def native_id(kind):
    """Native training/evaluation bots differ from GUI adapters; scope their IDs."""
    h = hashlib.sha256()
    for p in sorted(list((ROOT/'players/baselines/src').glob('*.rs')) + list((ROOT/'engine/bindings/src').glob('*.rs'))):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    h.update(engine_id().encode())
    return kind + ':native-v1:' + h.hexdigest()


def labeled_seats(labels):
    return [{'kind': 'bot', 'name': label,
             'bot': native_id(label) if label in ('random', 'greedy') else label,
             'bot_family': label if label in ('random', 'greedy') else 'ai'} for label in labels]


def seat_identities(seats, legacy=False):
    identities = []
    for seat in seats:
        label = seat.get('name', '')
        identity = seat.get('bot')
        family = seat.get('bot_family')
        if legacy and not identity and (label.startswith('policy-sha256-v1:') or ':weights-sha256:' in label):
            identity, family = label, 'ai'
        if legacy and not identity and label in ('random', 'greedy'):
            identities.append({'kind': label, 'version_id': None, 'label': label,
                               'identity_status': 'unresolved-historical-version'})
        elif legacy and not identity and label.startswith(('AI ', 'ppo-', 'torch:', 'current', 'frozen')):
            identities.append({'kind': 'ai', 'version_id': None, 'label': label,
                               'identity_status': 'unresolved-historical-version'})
        elif identity:
            unresolved = identity in ('random', 'greedy')
            identities.append({'kind': family or ('ai' if identity.startswith('policy-') else 'bot'),
                               'version_id': None if unresolved else identity, 'label': label,
                               'identity_status': 'unresolved-historical-version' if unresolved else 'recorded'})
        else:
            identities.append({'kind': 'human', 'version_id': None, 'label': label, 'identity_status': 'human'})
    return identities


@storage_operation
def atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix='.'+path.name, dir=str(path.parent))
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
        fd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        Path(temp).unlink(missing_ok=True)


def game_file(record):
    if record.get('format') == 'brass-gui-game/1':
        return record
    if record.get('format') != FORMAT:
        raise ValueError('unsupported replay format')
    if record['engine_id'] != engine_id():
        raise ValueError('recording requires a different engine: '+record['engine_id'])
    return dict(format='brass-gui-game/1', seed=record['initial'].get('seed'),
                start_state=record['initial'].get('start_state'), seats=record['seats'],
                lineups=record.get('lineups', []), moves=record['moves'])


def decode(data):
    if data.startswith(b'\x1f\x8b'):
        data = gzip.decompress(data)
    return json.loads(data)


def canonical_game(payload):
    file = game_file(payload)
    initial = {'seed': file['seed']} if file.get('seed') is not None else {'start_state': file['start_state']}
    def seats(rows):
        return [{k: v for k, v in row.items() if v is not None} for row in rows]
    lineup = [{'ply': row['ply'], 'seats': seats(row['seats'])} for row in (file.get('lineups') or [])]
    final_seats = seats(file['seats'])
    if len(lineup) == 1 and lineup[0] == {'ply': 0, 'seats': final_seats}:
        lineup = []
    return dict(initial=initial, seats=final_seats,
                moves=[{'player': m['player'], 'move': m['move']} for m in file['moves']], lineups=lineup)


def has_llm_seat(record):
    return any(s.get('bot_family') == 'llm' for s in record['seats']) or any(
        s.get('bot_family') == 'llm' for lineup in record.get('lineups', []) for s in lineup['seats'])


def llm_gamefile_relation(record):
    """Derive the action-only copy's ID from the immutable recording itself."""
    if (record.get('format') != FORMAT or record.get('duplicate_of') or
            not has_llm_seat(record) or record.get('engine_id') != engine_id()):
        return None
    copy_id = digest(canonical_game(record))
    is_copy = (record['game_id'] == copy_id and record['category'] == 'gui-import' and
               record['provenance'].get('source') == 'uploaded GameFile' and not record.get('llm_status'))
    return copy_id, is_copy


def shadowed_gamefiles(relations):
    """Only hide synthetic action-only copies. Distinct controller IDs stay visible."""
    copies, durable = set(), set()
    for relation in relations:
        if relation is not None:
            copy_id, is_copy = relation
            if is_copy:
                copies.add(copy_id)
            else:
                durable.add(copy_id)
    return copies & durable


def origin_matches(record, canonical, sources):
    """Link legacy exemplars only with controller provenance AND verified actions."""
    if record['initial'] != canonical['initial'] or record['moves'] != canonical['moves']:
        return False
    if [s['name'] for s in record['seats']] != [s['name'] for s in canonical['seats']]:
        return False
    metadata = record['provenance']
    for source in sources:
        p = source.get('provenance', {})
        game_id = p.get('game_id', p.get('game'))
        same_game = str(metadata.get('game')) == str(game_id).lstrip('0') or (metadata.get('game') == 0 and str(game_id).strip('0') == '')
        suite = p.get('suite_id')
        experiment_path = metadata.get('experiment_path')
        bound = (suite is not None and suite == metadata.get('experiment')) or (
            experiment_path is not None and Path(p['experiment']).resolve() == Path(experiment_path).resolve())
        if not same_game or not bound:
            continue
        # Exact identities in the incoming file must agree. Legacy unversioned
        # native labels are linked by the authoritative controller origin above.
        incoming = seat_identities(canonical['seats'], legacy=True)
        existing = seat_identities(record['seats'])
        if all(a['version_id'] is None or a['version_id'] == b['version_id'] for a, b in zip(incoming, existing)):
            return True
    return False


def outcome(game):
    ranking = list(game.ranking())
    return {'vp': list(game.vp()), 'ranking': ranking,
            'winner': ranking.index(0) if ranking.count(0) == 1 else None}


def summary(record):
    return {**{k: record.get(k) for k in ('game_id', 'status', 'category', 'seats', 'provenance',
                                          'result', 'created_at', 'engine_id', 'seat_identities')},
            'plies': len(record['moves'])}


class Archive:
    @storage_operation
    def __init__(self, root=None):
        self.root = Path(root or os.environ.get('BRASS_REPLAY_ROOT') or _ARCHIVE_ROOT.get() or ROOT/'replays')
        self.games = self.root/'games'
        self.games.mkdir(parents=True, exist_ok=True)

    def path(self, game_id, suffix='.json.gz'):
        if not re.fullmatch(r'[a-f0-9]{32,64}', game_id):
            raise ValueError('invalid replay ID')
        return self.games/(game_id+suffix)

    def read(self, game_id):
        path = self.path(game_id)
        try:
            return decode(path.read_bytes())
        except FileNotFoundError:
            pass
        journal = self.path(game_id, '.jsonl')
        try:
            data = journal.read_bytes()
        except FileNotFoundError:
            # Completion may publish gzip and remove its journal between reads.
            # One bounded retry of the published target covers that handoff.
            return decode(path.read_bytes())
        lines = data.splitlines(keepends=True)
        if not lines or not lines[0].endswith(b'\n'):
            raise ValueError('incomplete replay header')
        record = json.loads(lines[0])
        moves = []
        for line in lines[1:]:
            if not line.endswith(b'\n'):
                break  # a process died during the last append
            item = json.loads(line)
            if 'status' in item:
                record['status'] = item['status']
                if 'llm_status' in item:
                    record.setdefault('llm_status', {}).update(item['llm_status'])
            else:
                moves.append(item)
        record['moves'] = moves
        if record['status'] == 'in-progress':
            lock_path = self.root/'locks'/game_id
            if not lock_path.exists():
                record['status'] = 'interrupted'
            else:
                with lock_path.open('rb') as lock:
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        pass
                    else:
                        record['status'] = 'interrupted'
        return record

    def records(self, include_duplicates=False):
        ids = {p.name.split('.')[0] for p in self.games.glob('*.json*') if not p.name.startswith('.')}
        records = [self.read(identity) for identity in sorted(ids)]
        if include_duplicates:
            return records
        hidden = shadowed_gamefiles(llm_gamefile_relation(r) for r in records)
        return [r for r in records if not r.get('duplicate_of') and r['game_id'] not in hidden]

    def rebuild(self, changed_ids=None):
        """Rebuild from durable records, or update a known machine-training batch."""
        (self.root/'locks').mkdir(exist_ok=True)
        with (self.root/'locks/.index.writer.lock').open('a+b') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            index = self.root/'index.jsonl'
            rows = {}
            if changed_ids is not None and index.exists():
                for line in index.read_bytes().splitlines():
                    row = json.loads(line)
                    rows[row['game_id']] = row
                ids = set(changed_ids)
            else:
                ids = {p.name.split('.')[0] for p in self.games.glob('*.json*') if not p.name.startswith('.')}
            relations = []
            for identity in sorted(ids):
                record = self.read(identity)
                if record.get('duplicate_of'):
                    rows.pop(identity, None)
                    continue
                relation = llm_gamefile_relation(record)
                if relation:
                    relations.append(relation)
                row = {k: record.get(k) for k in ('game_id', 'status', 'category', 'created_at', 'result')}
                row['plies'] = len(record['moves'])
                row['provenance'] = {k: record['provenance'][k] for k in ('experiment', 'game', 'update', 'candidate_id') if k in record['provenance']}
                row['seat_versions'] = [s['version_id'] for s in record.get('seat_identities', seat_identities(record['seats']))]
                rows[identity] = row
            for identity in shadowed_gamefiles(relations):
                rows.pop(identity, None)
            ordered = [rows[i] for i in sorted(rows)]
            atomic_bytes(index, b''.join(encoded(r)+b'\n' for r in ordered))
            return ordered

    def list(self, query=''):
        # Read source records: index loss/staleness cannot hide recordings.
        rows = [summary(r) for r in self.records()]
        return [r for r in rows if query.casefold() in encoded(r).decode().casefold()]

    @storage_operation
    def annotate(self, record):
        """Add import metadata without freezing a resumable journal into gzip."""
        target = self.path(record['game_id'])
        if target.exists():
            atomic_bytes(target, gzip.compress(encoded(record)+b'\n', compresslevel=6, mtime=0))
            return
        lock_path = self.root/'locks'/record['game_id']
        with lock_path.open('a+b') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ReplayStorageError('Game is still recording; retry legacy import after it stops') from exc
            header = {**record, 'moves': []}
            atomic_bytes(self.path(record['game_id'], '.jsonl'), encoded(header)+b'\n'+b''.join(encoded(m)+b'\n' for m in record['moves']))

    def publish(self, record):
        path = self.path(record['game_id'])
        if path.exists():
            old = self.read(record['game_id'])
            for field in ('initial', 'seats', 'moves', 'result', 'engine_id'):
                if old.get(field) != record.get(field):
                    raise ValueError('replay ID conflict; existing game is preserved')
            return False
        atomic_bytes(path, gzip.compress(encoded(record)+b'\n', compresslevel=6, mtime=0))
        return True


class Recorder:
    @storage_operation
    def __init__(self, archive=None, *, seed=None, start_state=None, seats, provenance,
                 game_id=None, lineups=None):
        self.closed = False
        self._lock_file = None
        try:
            self._initialize(archive, seed=seed, start_state=start_state, seats=seats,
                             provenance=provenance, game_id=game_id, lineups=lineups)
        except BaseException:
            self._close()
            raise

    def _close(self):
        self.closed = True
        if self._lock_file is not None:
            self._lock_file.close()

    def _initialize(self, archive=None, *, seed=None, start_state=None, seats, provenance,
                    game_id=None, lineups=None):
        self.archive = archive or Archive()
        if (seed is None) == (start_state is None):
            raise ValueError('exactly one replay initial condition required')
        if seed is not None and (type(seed) is not int or not 0 <= seed < 2**64):
            raise ValueError('seed must be uint64')
        initial = {'seed': seed} if seed is not None else {'start_state': start_state}
        identity = {'initial': initial, 'seats': seats, 'provenance': provenance, 'engine_id': engine_id()}
        stable = {**identity, 'provenance': {k: v for k, v in provenance.items() if k != 'experiment_path'}}
        self.game_id = game_id or digest(stable)
        self.path = self.archive.path(self.game_id, '.jsonl')
        self.record = dict(identity, format=FORMAT, game_id=self.game_id,
                           category=provenance.get('category', 'unknown'),
                           created_at=datetime.now(timezone.utc).isoformat(),
                           status='in-progress', result=None, moves=[], lineups=lineups or [], writer_pid=os.getpid(),
                           seat_identities=seat_identities(seats))
        lock_dir = self.archive.root/'locks'
        lock_dir.mkdir(exist_ok=True)
        self._lock_file = (lock_dir/self.game_id).open('a+b')
        try:
            fcntl.flock(self._lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._close()
            raise ReplayStorageError('Another controller is recording this game; stop rather than overwrite') from exc
        self.failed = False
        self.position = 0
        self.existing = None
        if self.archive.path(self.game_id).exists() or self.path.exists():
            self.existing = self.archive.read(self.game_id)
            for field in ('initial', 'seats', 'engine_id'):
                if self.existing[field] != self.record[field]:
                    raise ValueError('replay resume identity mismatch')
            self.record = self.existing
            if self.record['status'] != 'completed':
                # Preserve durable prefix and remove only an incomplete final line.
                header = {**self.record, 'moves': [], 'status': 'in-progress', 'writer_pid': os.getpid()}
                atomic_bytes(self.path, encoded(header)+b'\n'+b''.join(encoded(m)+b'\n' for m in self.record['moves']))
                self.record['status'] = 'in-progress'
        else:
            atomic_bytes(self.path, encoded(self.record)+b'\n')

    def _write(self, item):
        if self.closed:
            raise ReplayStorageError('Replay writer is closed; create a new controller to resume')
        try:
            with self.path.open('ab') as f:
                f.write(encoded(item)+b'\n')
                f.flush()
                os.fsync(f.fileno())
        except OSError as exc:
            self.failed = True
            self._close()
            raise ReplayStorageError('Replay archive write failed; game stopped. Free space or repair storage; no recordings were evicted.') from exc

    def set_llm_status(self, seat, status):
        """Journal compact totals from the trusted live seat before applying its action."""
        if (type(seat) is not int or not 0 <= seat < 4 or
                self.record['seats'][seat].get('bot_family') != 'llm'):
            raise ValueError('LLM recording requires an LLM seat')
        if self.closed:
            raise ReplayStorageError('Replay writer is closed')
        if self.position < len(self.record['moves']) or self.record['status'] == 'completed':
            return  # Retried historical actions keep their original provider metadata.
        fields = ('calls', 'max_calls', 'error', 'reported_model', 'reported_provider', 'routes',
                  'usage', 'usage_reports')
        snapshot = {key: status[key] for key in fields if key in status}
        entry = {str(seat): snapshot}
        self._write({'status': self.record['status'], 'llm_status': entry})
        self.record.setdefault('llm_status', {}).update(entry)

    def append(self, actor, move):
        if self.closed:
            raise ReplayStorageError('Replay writer is closed; create a new controller to resume')
        item = {'player': actor, 'move': move}
        if self.position < len(self.record['moves']):
            if self.record['moves'][self.position] != item:
                raise ValueError('replay retry differs from saved action; stop without overwriting')
        else:
            if self.record['status'] == 'completed':
                raise ValueError('replay retry longer than completed game')
            self._write(item)  # Write before game mutation; caller validated this action.
            self.record['moves'].append(item)
        self.position += 1

    def finish(self, game):
        if not game.is_over() or self.position != len(self.record['moves']):
            raise ValueError('cannot complete an unfinished replay')
        result = outcome(game)
        if self.record['status'] == 'completed' and self.record['result'] != result:
            raise ValueError('completed replay outcome mismatch')
        if self.closed:
            if self.record['status'] == 'completed':
                return
            raise ReplayStorageError('Replay writer is closed')
        self.record.update(status='completed', result=result)
        try:
            self.archive.publish(self.record)
            self.path.unlink(missing_ok=True)  # redundant journal after durable publication
        except (OSError, ReplayStorageError) as exc:
            self.failed = True
            raise ReplayStorageError('Replay completion could not be published; durable journal retained. Repair archive storage before continuing.') from exc
        finally:
            self._close()

    def interrupt(self):
        if self.closed:
            return
        try:
            if self.record['status'] != 'completed' and not self.failed:
                self._write({'status': 'interrupted'})
                self.record['status'] = 'interrupted'
        finally:
            self._close()


def receipt_evidence(receipt):
    """Compare sparse journal receipts with their default-filled replay form."""
    row = dict(receipt)
    for field in ('reported_model', 'reported_provider', 'error'):
        row.setdefault(field, None)
    row.setdefault('routes', [])
    row['usage'] = {k: v for k, v in row.get('usage', {}).items() if v is not None}
    row['usage_reports'] = {k: v for k, v in row.get('usage_reports', {}).items() if v != 0}
    return row


def retain_upload(archive, record):
    """Publish under the recorder's per-ID lock. Never replace an existing file."""
    game_id = record['game_id']
    published, journal = archive.path(game_id), archive.path(game_id, '.jsonl')
    locks = archive.root/'locks'
    locks.mkdir(exist_ok=True)
    with (locks/game_id).open('a+b') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            # A recording controller may be creating its header or completing.
            # Matching uploads can be acknowledged, but never written here.
            if not (published.exists() or journal.exists()):
                raise ValueError('Replay writer is busy; retry the upload') from None
        else:
            if not (published.exists() or journal.exists()):
                archive.publish(record)
                return game_id

        existing = archive.read(game_id)
        incoming = canonical_game(record)
        current = canonical_game(existing)
        # An older journal snapshot is already covered by its recorded prefix.
        if journal.exists():
            current['moves'] = current['moves'][:len(incoming['moves'])]
        if current != incoming:
            raise ValueError('Replay ID conflict; existing game is preserved')
        for seat, incoming_receipt in record.get('llm_status', {}).items():
            previous = existing.get('llm_status', {}).get(seat)
            if previous is None or receipt_evidence(incoming_receipt) != receipt_evidence(previous):
                raise ValueError('Replay receipt conflict; existing game is preserved')
    return game_id


def import_payload(archive, payload):
    """Validate and retain a passive upload, deduplicating equivalent GameFiles."""
    import brass
    canonical = canonical_game(payload)
    game = brass.Game(canonical['initial']['seed']) if 'seed' in canonical['initial'] else brass.Game.from_json(canonical['initial']['start_state'])
    for ply, action in enumerate(canonical['moves']):
        if action['player'] != game.current_player():
            raise ValueError('uploaded actor mismatch at ply '+str(ply))
        game.apply(json.dumps(action['move']))
    result = outcome(game) if game.is_over() else None
    if payload.get('format') == FORMAT and payload.get('result') != result:
        raise ValueError('uploaded replay result disagrees with actions')
    durable = payload.get('format') == FORMAT
    # Two independent LLM sessions can have identical actions and different bills.
    # A GameFile has no session identity; its canonical copy is hidden once a
    # durable counterpart exists. Both originals remain reproducible by their IDs.
    if not durable or not has_llm_seat(canonical):
        matches = [r for r in archive.records() if canonical_game(r) == canonical]
        if matches:
            return min(r['game_id'] for r in matches)
    if durable:
        record = dict(payload, status='completed' if game.is_over() else 'interrupted')
    else:
        record = dict(canonical, format=FORMAT, game_id=digest(canonical), engine_id=engine_id(),
                      category='gui-import', provenance={'category':'gui-import','source':'uploaded GameFile'},
                      created_at=datetime.now(timezone.utc).isoformat(), result=result,
                      status='completed' if game.is_over() else 'interrupted',
                      seat_identities=seat_identities(canonical['seats'], legacy=True))
    return retain_upload(archive, record)


def legacy_source_metadata(path):
    """Use available outcome metadata only as provenance, never as actions."""
    details = {'experiment': str(path.parent), 'game': path.stem}
    row_path = path.parent.parent/'games'/path.name
    if path.parent.name == 'replays' and row_path.is_file():
        row = json.loads(row_path.read_text())
        details.update(category='evaluation', **{k: row[k] for k in (
            'game_id', 'suite_id', 'candidate_id', 'update', 'lineup', 'rotation', 'deal_id') if k in row})
    elif path.parent.name == 'replays':
        details['category'] = 'training'
        details['experiment'] = str(path.parent.parent)
    elif path.name.endswith('.brass.json'):
        details['category'] = 'gui'
    else:
        details['category'] = 'legacy'
    return details


def import_legacy(archive, source):
    """Import only existing action-bearing GUI files; never regenerate a game.

    Canonical payload IDs deduplicate copied runs. Every original path/hash is
    retained in the record. Missing historical identities stay explicitly unknown.
    """
    import brass
    grouped = {}
    report = {'full_files': 0, 'imported': 0, 'duplicates': 0, 'invalid': []}
    for path in sorted(Path(source).rglob('*.json')):
        if not path.is_file():
            continue
        try:
            data = path.read_bytes()
            if b'brass-gui-game/1' not in data:
                continue
            payload = json.loads(data)
            if payload.get('format') != 'brass-gui-game/1' or not isinstance(payload.get('moves'), list):
                continue
            canonical = canonical_game(payload)
            identity = digest(canonical)
            item = grouped.setdefault(identity, {'canonical': canonical, 'sources': []})
            item['sources'].append({'path': str(path.resolve()), 'sha256': hashlib.sha256(data).hexdigest(),
                                    'provenance': legacy_source_metadata(path)})
            report['full_files'] += 1
        except (ValueError, KeyError, TypeError) as err:
            report['invalid'].append({'path': str(path), 'error': str(err)})
    existing_records = archive.records()
    for identity, item in grouped.items():
        c = item['canonical']
        exact = [r for r in existing_records if canonical_game(r) == c]
        origins = [r for r in existing_records if origin_matches(r, c, item['sources'])]
        if len({digest(r['seats']) for r in origins}) > 1:
            origins = []  # ambiguous historical baseline version: never collapse known versions
        if len([r for r in exact if not r['provenance'].get('imported')]) > 1 and not origins:
            exact = [r for r in exact if r['provenance'].get('imported')]
        matches = list({r['game_id']: r for r in exact + origins}.values())
        if matches:
            primary = min(matches, key=lambda r: r['game_id'])
            identity = primary['game_id']
            # Retain redundant published records as explicit aliases, never delete
            # them. All source provenance is merged into the primary record.
            for duplicate in matches:
                if duplicate['game_id'] != identity:
                    duplicate['duplicate_of'] = identity
                    atomic_bytes(archive.path(duplicate['game_id']), gzip.compress(encoded(duplicate)+b'\n', compresslevel=6, mtime=0))
                    item['sources'].extend(duplicate['provenance'].get('sources', []))
            if not primary['provenance'].get('imported') or canonical_game(primary) != c:
                # The modern record has stronger pinned identities than the legacy
                # human-labelled exemplar. Preserve all its metadata and payload.
                known = {(x['path'], x['sha256']): x for x in primary['provenance'].get('sources', [])}
                for row in item['sources']:
                    known[(row['path'], row['sha256'])] = row
                primary['provenance']['sources'] = list(known.values())
                archive.annotate(primary)
                continue
        try:
            game = brass.Game(c['initial']['seed']) if 'seed' in c['initial'] else brass.Game.from_json(c['initial']['start_state'])
            for ply, move in enumerate(c['moves']):
                if game.current_player() != move['player']:
                    raise ValueError('actor mismatch at ply '+str(ply))
                game.apply(json.dumps(move['move']))
            # Existing replay labels are preserved verbatim; do not turn a display
            # name into an asserted historical model or native implementation ID.
            provenance = {**item['sources'][0]['provenance'], 'imported': True, 'sources': item['sources'],
                          'identity_note': 'Original seat fields preserved; labels without version IDs are unresolved historical identities.'}
            record = dict(c, format=FORMAT, game_id=identity, engine_id=engine_id(),
                          category=provenance['category'], provenance=provenance,
                          created_at=datetime.now(timezone.utc).isoformat(),
                          status='completed' if game.is_over() else 'interrupted',
                          result=outcome(game) if game.is_over() else None, seat_identities=seat_identities(c['seats'], legacy=True))
            if matches:
                record['lineups'] = primary.get('lineups', [])
            if archive.publish(record):
                report['imported'] += 1
            else:
                old = archive.read(identity)
                old_sources = {(x['path'], x['sha256']): x for x in old['provenance'].get('sources', [])}
                for source_row in item['sources']:
                    old_sources[(source_row['path'], source_row['sha256'])] = source_row
                updated = dict(provenance, sources=list(old_sources.values()))
                if old['provenance'] != updated or old.get('seat_identities') != record['seat_identities']:
                    old['seat_identities'] = record['seat_identities']
                    old['provenance'] = updated
                    old['category'] = updated['category']
                    atomic_bytes(archive.path(identity), gzip.compress(encoded(old)+b'\n', compresslevel=6, mtime=0))
        except (ValueError, KeyError, TypeError) as err:
            report['invalid'].append({'sources': item['sources'], 'error': str(err)})
    report['duplicates'] = report['full_files'] - report['imported'] - len(report['invalid'])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['rebuild', 'import'])
    parser.add_argument('--root', type=Path)
    parser.add_argument('--source', type=Path, default=ROOT/'training/runs')
    args = parser.parse_args()
    archive = Archive(args.root)
    if args.command == 'import':
        print(json.dumps(import_legacy(archive, args.source), indent=2))
    print(json.dumps({'indexed': len(archive.rebuild())}))


if __name__ == '__main__':
    main()
