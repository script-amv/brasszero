# SPDX-License-Identifier: Apache-2.0
"""Trusted provenance is rooted at the source tree, never the launch directory."""
import subprocess


def test_source_checkout_ignores_cwd_and_declared_archive_revision(tmp_path, monkeypatch):
    from brass_rl import source
    root = tmp_path/'checkout'
    root.mkdir()
    subprocess.run(['git', 'init', '-q', str(root)], check=True)
    subprocess.run(['git', '-C', str(root), '-c', 'user.name=Test', '-c', 'user.email=test' + chr(64) + 'example.invalid',
                    'commit', '--allow-empty', '-qm', 'fixture'], check=True)
    expected = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    monkeypatch.setattr(source, 'SOURCE_ROOT', root)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('BRASS_SOURCE_REVISION', 'f'*40)
    assert source.source_provenance() == {'kind': 'git', 'revision': expected, 'dirty': False}


def test_archive_provenance_is_explicit_and_never_inherits_parent_git(tmp_path, monkeypatch):
    from brass_rl import source
    root = tmp_path/'parent'
    root.mkdir()
    subprocess.run(['git', 'init', '-q', str(root)], check=True)
    extracted = root/'extracted'
    extracted.mkdir()
    monkeypatch.setattr(source, 'SOURCE_ROOT', extracted)
    monkeypatch.delenv('BRASS_SOURCE_REVISION', raising=False)
    assert source.source_provenance() == {'kind': 'unavailable', 'revision': None}
    monkeypatch.setenv('BRASS_SOURCE_REVISION', 'a'*40)
    assert source.source_provenance() == {'kind': 'archive-declared', 'revision': 'a'*40}
    monkeypatch.setenv('BRASS_SOURCE_REVISION', 'not-a-revision')
    import pytest
    with pytest.raises(ValueError, match='BRASS_SOURCE_REVISION'):
        source.source_provenance()
