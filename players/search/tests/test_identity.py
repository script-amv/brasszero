# SPDX-License-Identifier: Apache-2.0
"""Source content and optional-dependency boundaries, independent of checkout path."""
from dataclasses import replace
import os
from pathlib import Path
import shutil
import subprocess
import sys

from brass_search.config import SearchConfig
from brass_search import identity


def test_source_bytes_engine_schema_and_config_bind_identity(tmp_path, monkeypatch):
    cfg = SearchConfig()
    original = identity.search_identity('policy', 'schema', cfg, engine_id='engine')
    for logical in identity.source_manifest():
        path = tmp_path / logical
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(identity.ROOT / logical, path)
    monkeypatch.setattr(identity, 'ROOT', tmp_path)
    assert identity.search_identity('policy', 'schema', cfg, engine_id='engine') == original
    for args in (('policy', 'other-schema', cfg, 'engine'),
                 ('policy', 'schema', cfg, 'other-engine'),
                 ('policy', 'schema', replace(cfg, value_scale=9), 'engine')):
        assert identity.search_identity(*args[:3], engine_id=args[3]) != original
    file = tmp_path / 'training/brass_rl/features.py'
    file.write_bytes(file.read_bytes() + b'\n# source identity mutation\n')
    assert identity.search_identity('policy', 'schema', cfg, engine_id='engine') != original


def test_configuration_imports_without_torch():
    code = '''
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'numpy', 'brass'):
            raise ModuleNotFoundError('blocked optional module')
sys.meta_path.insert(0, Block())
from brass_search import SearchConfig
assert SearchConfig(max_rollouts=0).max_rollouts == 0
'''
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]))
    result = subprocess.run([sys.executable, '-c', code], env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
