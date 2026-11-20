"""Path/seed-independent, length-framed identity for the inference closure."""
import hashlib
import json
from pathlib import Path

from brass_ai.fingerprints import engine_fingerprint
from .config import SearchConfig

ROOT = Path(__file__).resolve().parents[3]
ALGORITHM = 'root-gumbel-policy-rollout-halving-v1'


def source_manifest():
    """Logical paths and hashes only; excludes tests, machine paths and runtime."""
    paths = list((ROOT / 'players/search/brass_search').glob('*.py'))
    paths += [ROOT / 'training/brass_rl' / (name + '.py') for name in
              ('agent', 'config', 'distributions', 'features', 'model', 'schema', 'policy_io')]
    paths += [ROOT / 'engine/tools/brass_ai' / (name + '.py') for name in ('events', 'fingerprints')]
    paths += list((ROOT / 'engine/bindings/src').rglob('*.rs'))
    paths += [ROOT / 'engine/bindings/Cargo.toml']
    return {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths)}


def search_identity(base_policy_id, schema_id, config, *, engine_id=None):
    """Bind complete config, base/schema/engine and length-framed inference sources.

    Source digests are framed alongside logical names. Runtime/device and sampling
    seeds belong in separate run provenance, never this portable player identity.
    """
    if not isinstance(config, SearchConfig):
        raise ValueError('config must be SearchConfig')
    if not isinstance(base_policy_id, str) or not base_policy_id or not isinstance(schema_id, str) or not schema_id:
        raise ValueError('base policy and schema identity are required')
    fields = {'algorithm': ALGORITHM, 'config': config.canonical_mapping(),
              'base_policy_id': base_policy_id, 'schema_id': schema_id,
              'engine_id': engine_id or engine_fingerprint(ROOT)}
    hasher = hashlib.sha256(b'brass-search-player-v1\0')
    chunks = [json.dumps(fields, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()]
    for name in sorted(source_manifest()):
        chunks.extend((name.encode(), (ROOT / name).read_bytes()))
    for chunk in chunks:
        hasher.update(len(chunk).to_bytes(8, 'little'))
        hasher.update(chunk)
    return 'search-sha256-v1:' + hasher.hexdigest()


def player_manifest(metadata, config=None):
    """Public normalized replay/run manifest; config=None means plain sample policy.

    metadata is validated PolicyMetadata from load_policy. No path or seed is emitted.
    """
    result = {'id': metadata.id, 'family': 'ai', 'mode': 'sample',
              'base_policy_id': metadata.id, 'base_update': metadata.update,
              'schema_id': metadata.schema_id, 'engine_id': metadata.engine_fingerprint,
              'model_id': metadata.model_id, 'component_hash': metadata.component_hash,
              'policy_content_hashes': dict(metadata.content_hashes)}
    if config is not None:
        result.update(id=search_identity(metadata.id, metadata.schema_id, config,
                                        engine_id=metadata.engine_fingerprint),
                      algorithm=ALGORITHM, config=config.canonical_mapping(),
                      source_hashes=source_manifest())
    return result
