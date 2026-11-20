# SPDX-License-Identifier: Apache-2.0
import hashlib
import subprocess
import sys
from pathlib import Path


def test_source_fingerprint_framing_and_scope(tmp_path):
    from brass_ai.fingerprints import engine_fingerprint
    files = {"Cargo.lock": b"lock", "engine/Cargo.toml": b"cargo",
             "engine/src/z.rs": b"z", "engine/src/nested/a.rs": b"a"}
    for name, data in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    h = hashlib.sha256(b"brass-engine-source-v1\0")
    for name in sorted(files):
        raw = name.encode()
        h.update(len(raw).to_bytes(8, "little") + raw)
        h.update(len(files[name]).to_bytes(8, "little") + files[name])
    expected = "engine-sha256-v1:" + h.hexdigest()
    assert engine_fingerprint(tmp_path) == expected
    (tmp_path / "README.md").write_text("documentation")
    assert engine_fingerprint(tmp_path) == expected
    (tmp_path / "engine/src/z.rs").write_text("new engine")
    assert engine_fingerprint(tmp_path) != expected


def test_fingerprint_import_is_stdlib_only():
    root = Path(__file__).resolve().parents[3]
    result = subprocess.run([sys.executable, "-c",
        "import sys; sys.path.insert(0, %r); import brass_ai.fingerprints; "
        "assert 'torch' not in sys.modules; assert 'numpy' not in sys.modules; "
        "assert 'brass' not in sys.modules" % str(root / "engine/tools")], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_organized_layout_preserves_legacy_engine_identity(tmp_path):
    """Moving physical files must not retire every saved policy and replay."""
    from brass_ai.fingerprints import engine_fingerprint
    core = tmp_path / 'engine'
    (core / 'src').mkdir(parents=True)
    (core / 'Cargo.toml').write_text('manifest')
    (core / 'src/lib.rs').write_text('rules')
    (tmp_path / 'Cargo.lock').write_text('lock')
    expected = engine_fingerprint(tmp_path)
    core.rename(tmp_path / 'staging')
    core.mkdir()
    (tmp_path / 'staging').rename(core / 'core')
    assert engine_fingerprint(tmp_path) == expected
    (core / 'core/src/lib.rs').write_text('changed rules')
    assert engine_fingerprint(tmp_path) != expected
