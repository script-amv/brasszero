# SPDX-License-Identifier: Apache-2.0
"""Locked projections of exact actor occurrences and locally accepted ancestry."""
from contextlib import contextmanager
from datetime import date
import fcntl
import json
from pathlib import Path
import os
import tempfile

from .experiments import atomic, encoded, library_intent, link_or_copy, read_experiment, sync_dir
from .transfer import _policy, digest, file_hash, read

SECTIONS = ('models', 'other_preserved_snapshots', 'unavailable_preserved_snapshots')
FORMAT = 'brass-library-lineage/1'


def _save(path, value):
    data = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)+'\n').encode()
    if not path.exists() or path.read_bytes() != data:
        atomic(path, data)


def _ref(experiment, update, policy, checkpoint):
    return dict(experiment_id=experiment, update=update, policy_id=policy, checkpoint_sha256=checkpoint)


def _parent_ref(parent):
    return _ref(parent['experiment_id'], parent['update'], parent['model_id'], parent['checkpoint_sha256'])


def _seed(project, catalog):
    registry = dict(format=FORMAT, mainline_tip=None, branches={}, occurrences={})
    declarations = {}
    for section in SECTIONS:
        for row in catalog.get(section, []):
            provenance = row.get('provenance', {})
            location = provenance.get('run')
            source = Path(row.get('source_export') or '.')
            if not location and source.parent.name == 'policies':
                location = source.parent.parent.as_posix()
            exp = provenance.get('experiment_id')
            if location and location not in declarations:
                declarations[location] = (read_experiment(project/location)['id']
                                          if (project/location/'experiment.json').is_file() else None)
            if declarations.get(location):
                actual = declarations[location]
                if exp and exp != actual:
                    raise ValueError('catalog experiment provenance mismatch')
                exp = actual
            # Stable unresolved source groups allow historical catalog data to remain visible.
            branch = exp or 'historical-'+digest(encoded(location or row['policy_id']))[:16]
            registry['branches'].setdefault(branch, dict(label=row.get('branch') or 'Unknown history',
                    experiment_id=exp, run=location, parent=None))
            ref = _ref(exp, row['update'], row['policy_id'], None)
            occurrence = dict(ref, branch_id=branch, lineage=row.get('lineage', 'experimental'))
            registry['occurrences'][row['policy_id']] = [occurrence]
            if row['name'] == catalog.get('latest') and occurrence['lineage'] == 'mainline':
                registry['mainline_tip'] = ref
    return registry


@contextmanager
def inventory_writer(project):
    lock = project/'training/runs/.model-catalog.writer.lock'
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open('a+b') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        models = project/'players/models'
        catalog = (read(models/'catalog.json') if (models/'catalog.json').exists()
                   else {'format': 'brass-playable-model-catalog/1'})
        for section in SECTIONS:
            catalog.setdefault(section, [])
        registry = (read(models/'lineage.json') if (models/'lineage.json').exists()
                    else _seed(project, catalog))
        if registry.get('format') != FORMAT:
            raise ValueError('unknown library lineage format')
        inventory = Inventory(project, catalog, registry)
        yield inventory
        inventory.save()


class Inventory:
    def __init__(self, project, catalog, registry):
        self.project, self.catalog, self.registry = project, catalog, registry
        self.rows = {r['policy_id']: (s, r) for s in SECTIONS for r in catalog[s]}
        self.public = {}
        self.added = 0
        self.locations_by_policy = None
        self.checkpoints = {}
        for policy, occurrences in registry['occurrences'].items():
            for occurrence in occurrences:
                self.checkpoints[occurrence['branch_id'], occurrence['update']] = occurrence
        self.locations = None

    def register(self, run, locations=None, *, destination=None):
        exp = read_experiment(run)
        identity = exp['id']
        parent = exp.get('parent')
        intent = library_intent(run, exp)
        verified = None
        if parent:
            if locations is not None:
                owner = locations.get(parent['experiment_id'])
            else:
                known = self.registry['branches'].get(parent['experiment_id'], {}).get('run')
                if not known and self.locations is None:
                    from .transfer import experiments_under
                    self.locations = experiments_under(self.project/'training/runs')
                owner = self.project/known if known else self.locations.get(parent['experiment_id'])
            if owner:
                boundary = owner/'updates'/('%06d' % parent['update'])
                if (boundary/'manifest.json').is_file():
                    meta, envelope = _policy(boundary)
                    if (read_experiment(owner)['id'] != parent['experiment_id']
                            or meta['id'] != parent['model_id']
                            or file_hash(boundary/'manifest.json') != parent['checkpoint_sha256']
                            or parent.get('policy_sha256', envelope['content_hashes']['policy.pt'])
                               != envelope['content_hashes']['policy.pt']):
                        raise ValueError('parent checkpoint provenance conflict')
                    base = run/'updates'/boundary.name
                    if exp['base_update'] == parent['update'] and (base/'manifest.json').exists():
                        actor, _ = _policy(base)
                        if actor['id'] == meta['id'] and actor['update'] == parent['update']:
                            verified = _parent_ref(parent)
        branch = self.registry['branches'].setdefault(identity, {})
        if branch.get('parent') and verified and branch['parent'] != verified:
            raise ValueError('branch parent provenance conflict')
        if branch.get('declaration_sha256') not in (None, file_hash(run/'experiment.json')):
            raise ValueError('experiment declaration conflict')
        branch.update(experiment_id=identity, label=exp.get('stability_arm') or ('BC continuation' if exp['mode'] == 'bc-current' else run.name),
                      run=(destination or run).relative_to(self.project).as_posix(), parent=verified,
                      library_intent=intent,
                      declaration_sha256=file_hash(run/'experiment.json'), base_update=exp['base_update'])
        return exp

    def publish(self, source, *, run=None, experiment=None, checkpoint=None, preflight=False):
        meta, manifest = _policy(source)
        identity = meta['id']
        update = meta['update']
        sha = None
        if checkpoint is not None:
            actor, envelope = _policy(checkpoint) if checkpoint != source else (meta, manifest)
            if (actor['id'] != identity or envelope.get('update') != update or checkpoint.name != '%06d' % update
                    or any(envelope['content_hashes'][n] != manifest['content_hashes'][n] for n in ('metadata.json', 'policy.pt'))):
                raise ValueError('checkpoint and playable policy disagree')
            sha = file_hash(checkpoint/'manifest.json')
        if experiment:
            self.record(experiment, update, identity, sha)
        if preflight:
            return
        prior = self.rows.get(identity)
        directory = self.project/'players/models'
        public = directory/('policy-'+identity.split(':')[-1])
        if prior:
            cited = self.project/prior[1]['path'] if prior[1].get('path') else None
            if cited and cited.parent == directory and cited.is_dir():
                public = cited
        if identity not in self.public:
            # Older curated names might precede the catalog. Resolve them once per operation.
            if not public.exists():
                if self.locations_by_policy is None:
                    self.locations_by_policy = {read(p/'metadata.json')['id']: p for p in directory.iterdir()
                                                if p.is_dir() and not p.name.startswith(('policy-', '.'))
                                                and (p/'metadata.json').exists()} if directory.exists() else {}
                public = self.locations_by_policy.get(identity, public)
            if public.exists():
                other, envelope = _policy(public)
                if other != meta or envelope['format'] != 'brass-portable-policy-v1':
                    raise ValueError('public policy identity/content conflict')
            else:
                directory.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory(prefix='.policy-', dir=directory) as temp:
                    staged = Path(temp)/'actor'
                    for name in ('policy.pt', 'metadata.json'):
                        link_or_copy(source/name, staged/name)
                    if manifest['format'] == 'brass-portable-policy-v1':
                        link_or_copy(source/'manifest.json', staged/'manifest.json')
                    else:
                        atomic(staged/'manifest.json', encoded(dict(format='brass-portable-policy-v1',
                            content_hashes={n: manifest['content_hashes'][n] for n in ('policy.pt', 'metadata.json')})))
                    sync_dir(staged)
                    os.rename(staged, public)
                    self.added += 1
                    sync_dir(directory)
            self.public[identity] = public
        public = self.public[identity]
        relative = source.relative_to(self.project).as_posix()
        if prior is None:
            branch = run.name if run else 'unknown'
            row = dict(name='ppo-%d-%s-%s' % (update, branch, identity.split(':')[-1][:12]),
                       update=update, policy_id=identity, path=public.relative_to(self.project).as_posix(),
                       source_export=relative, other_preserved_locations=[], role=branch,
                       branch=branch, lineage='experimental', schema_id=meta['schema_id'],
                       engine_fingerprint=meta['engine_fingerprint'],
                       files={n: file_hash(public/n) for n in ('metadata.json', 'manifest.json', 'policy.pt')})
            if experiment:
                row['provenance'] = dict(run=run.relative_to(self.project).as_posix(),
                                         experiment_id=experiment['id'], producer=experiment.get('producer', {}))
            section = 'other_preserved_snapshots'
            self.catalog[section].append(row)
            self.rows[identity] = section, row
        else:
            _, row = prior
            old_path = row.get('path')
            row['path'] = public.relative_to(self.project).as_posix()
            if old_path and old_path != row['path'] and old_path not in row.setdefault('other_preserved_locations', []):
                row['other_preserved_locations'].append(old_path)
            if relative not in (row.get('source_export'), row['path']) and relative not in row.setdefault('other_preserved_locations', []):
                row['other_preserved_locations'].append(relative)
        if not experiment:
            if identity not in self.registry['occurrences']:
                branch = 'unknown-'+identity.split(':')[-1]
                self.registry['branches'][branch] = dict(experiment_id=None, label='Unknown history', run=None, parent=None)
                self.registry['occurrences'][identity] = [dict(_ref(None, update, identity, None),
                                                              branch_id=branch, lineage='experimental')]
            return

    def record(self, experiment, update, identity, sha):
        if experiment.get('parent') and update < experiment['parent']['update']:
            return  # Copied pre-fork history belongs to its original experiment.
        exp = experiment['id']
        occurrence = self.checkpoints.get((exp, update))
        if occurrence is not None:
            if occurrence['policy_id'] != identity or sha and occurrence['checkpoint_sha256'] not in (None, sha):
                raise ValueError('conflicting policy occurrence')
            if sha:
                occurrence['checkpoint_sha256'] = sha
        else:
            occurrence = dict(_ref(exp, update, identity, sha), branch_id=exp, lineage='experimental')
            self.registry['occurrences'].setdefault(identity, []).append(occurrence)
            self.checkpoints[exp, update] = occurrence
        tip = self.registry['mainline_tip']
        if tip and (tip['policy_id'], tip['experiment_id']) == (identity, exp) and sha:
            tip['checkpoint_sha256'] = sha

    def continuation_candidates(self):
        tip = self.registry['mainline_tip']
        if not tip or not tip['checkpoint_sha256']:
            return []
        update = tip['update']+1
        candidates = []
        for exp, branch in self.registry['branches'].items():
            occurrence = self.checkpoints.get((exp, update))
            if not occurrence or not occurrence['checkpoint_sha256']:
                continue
            local_resume = branch.get('adopted_continuation') and tip['experiment_id'] == exp
            portable_intent = branch.get('library_intent') == 'mainline'
            if local_resume or portable_intent:
                previous = self.predecessor(exp, update)
                fork_from_tip = (branch['parent'] == tip and previous is not None
                                 and previous['policy_id'] == tip['policy_id'])
                if previous == tip or fork_from_tip:
                    candidates.append(occurrence)
        return sorted(candidates, key=lambda o: o['experiment_id'])

    def advance(self):
        """Advance after discovery so neither directory ordering nor duplicate weight IDs break ties."""
        candidates = self.continuation_candidates()
        while len(candidates) == 1:
            self.accept(candidates[0])
            candidates = self.continuation_candidates()
        unresolved = [_ref(o['experiment_id'], o['update'], o['policy_id'], o['checkpoint_sha256']) for o in candidates]
        self.registry['ambiguous_continuations'] = unresolved
        return unresolved

    def predecessor(self, exp, update):
        old = self.checkpoints.get((exp, update-1))
        if old and old['checkpoint_sha256']:
            return _ref(exp, update-1, old['policy_id'], old['checkpoint_sha256'])
        return None

    def accept(self, occurrence):
        occurrence['lineage'] = 'mainline'
        exp = occurrence['experiment_id']
        parent = self.registry['branches'][exp].get('parent')
        if parent and occurrence['update'] == parent['update']+1:
            base = self.checkpoints.get((exp, parent['update']))
            if base and base['policy_id'] == parent['policy_id']:
                base['lineage'] = 'mainline'
        self.registry['mainline_tip'] = {k: occurrence[k] for k in ('experiment_id', 'policy_id', 'update', 'checkpoint_sha256')}

    def save(self):
        for identity, occurrences in self.registry['occurrences'].items():
            occurrences.sort(key=lambda o: o['branch_id'])
            if identity not in self.rows or not any(o['lineage'] == 'mainline' for o in occurrences):
                continue
            section, row = self.rows[identity]
            if section == 'other_preserved_snapshots':
                self.catalog[section].remove(row)
                self.catalog['models'].append(row)
                self.rows[identity] = 'models', row
                row.update(lineage='mainline', branch='BC')
        tip = self.registry['mainline_tip']
        if tip and tip['policy_id'] in self.rows:
            self.catalog['latest'] = self.rows[tip['policy_id']][1]['name']
        self.catalog['model_count'] = len(self.catalog['models'])
        directory = self.project/'players/models'
        _save(directory/'lineage.json', self.registry)
        path = directory/'catalog.json'
        if not path.exists() or read(path) != self.catalog:
            self.catalog['updated'] = date.today().isoformat()
            _save(path, self.catalog)


def update_inventory(project, run, exports):
    """Register the publisher's historical (identity, source, public) tuples."""
    from .library import register_publication
    sources = []
    for identity, source, public in exports:
        if _policy(public)[0]['id'] != identity or _policy(source)[0]['id'] != identity:
            raise ValueError('published model identity mismatch')
        sources.append(source)
    register_publication(project, run, sources)
