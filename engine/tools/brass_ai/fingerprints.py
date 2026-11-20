"""Versioned source identity shared by training and GUI (standard library only)."""
import hashlib
from pathlib import Path


def engine_fingerprint(root: Path) -> str:
    """Hash logical v1 names, independently of the checkout's physical layout.

    Saved policies use ``engine/*`` names. The organized checkout stores these
    exact source bytes in ``engine/core/*``; moving them must not change their
    identity. Legacy source archives remain readable with the same algorithm.
    """
    root = Path(root)
    core = root / "engine/core"
    if not core.is_dir():
        core = root / "engine"
    files = {"Cargo.lock": root / "Cargo.lock", "engine/Cargo.toml": core / "Cargo.toml"}
    files.update(("engine/" + path.relative_to(core).as_posix(), path)
                 for path in (core / "src").rglob("*.rs"))
    hasher = hashlib.sha256(b"brass-engine-source-v1\0")
    for name in sorted(files):
        encoded, data = name.encode("utf-8"), files[name].read_bytes()
        for value in (encoded, data):
            hasher.update(len(value).to_bytes(8, "little", signed=False))
            hasher.update(value)
    return "engine-sha256-v1:" + hasher.hexdigest()
