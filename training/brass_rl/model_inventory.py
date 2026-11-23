"""Add published policies to the project inventory without changing artifact bytes."""
import fcntl
import hashlib
import json
from datetime import date

from .experiments import atomic, read_experiment
from .config import BC_LEARNER
from .main_bc import CURRENT_DESTINATION, DESTINATION, source_parent

SECTIONS = ('models', 'other_preserved_snapshots', 'unavailable_preserved_snapshots')


def policy_entry(project, source, public, run, experiment, lineage, branch):
    metadata = json.loads((public/'metadata.json').read_bytes())
    identity = metadata['id']
    digest = identity.split(':')[-1]
    return {'name': 'ppo-%d-%s-%s' % (metadata['update'], branch, digest[:12]),
            'update': metadata['update'], 'policy_id': identity,
            'path': public.relative_to(project).as_posix(),
            'source_export': source.relative_to(project).as_posix(),
            'other_preserved_locations': [], 'role': branch, 'lineage': lineage, 'branch': branch,
            'engine_fingerprint': metadata['engine_fingerprint'], 'schema_id': metadata['schema_id'],
            'files': {name: hashlib.sha256((public/name).read_bytes()).hexdigest()
                      for name in ('policy.pt', 'metadata.json', 'manifest.json')},
            'provenance': {'run': run.relative_to(project).as_posix(), 'experiment_id': experiment['id'],
                           'producer': experiment.get('producer', {})}}


def update_inventory(project, run, exports):
    catalog = project/'players/models/catalog.json'
    if not catalog.is_file():
        return
    experiment = read_experiment(run)
    relative = run.relative_to(project)
    is_main = ((relative == DESTINATION and experiment.get('parent') == source_parent()) or
               (relative == CURRENT_DESTINATION and experiment.get('learner') == BC_LEARNER))
    lock = project/'training/runs/.model-catalog.writer.lock'
    with lock.open('a+b') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        inventory = json.loads(catalog.read_bytes())
        known = {row['policy_id']: (section, row)
                 for section in SECTIONS for row in inventory.get(section, [])}
        changed = False
        tip = None
        for identity, source, public in exports:
            main = is_main and int(source.name) > experiment['base_update']
            prior = known.get(identity)
            if prior is None:
                section = 'models' if main else 'other_preserved_snapshots'
                entry = policy_entry(project, source, public, run, experiment,
                                     'mainline' if main else 'experimental', 'BC' if main else run.name)
                inventory[section].append(entry)
                known[identity] = section, entry
                changed = True
            else:
                section, entry = prior
                # Independent deterministic runs can produce exactly the same
                # policy. Preserve its first publication and record the BC path.
                if main and entry['lineage'] != 'mainline':
                    inventory[section].remove(entry)
                    inventory['models'].append(entry)
                    entry.update(lineage='mainline', branch='BC')
                    location = source.relative_to(project).as_posix()
                    if location != entry['source_export'] and location not in entry['other_preserved_locations']:
                        entry['other_preserved_locations'].append(location)
                    known[identity] = 'models', entry
                    changed = True
            if main:
                tip = entry['name']
        if tip is not None and inventory.get('latest') != tip:
            inventory['latest'] = tip
            changed = True
        if changed:
            inventory['updated'] = date.today().isoformat()
            inventory['model_count'] = len(inventory['models'])
            atomic(catalog, (json.dumps(inventory, indent=2, ensure_ascii=False)+'\n').encode())
