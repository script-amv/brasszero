# SPDX-License-Identifier: Apache-2.0
"""Shared RL test setup; importing tests does not require torch."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "engine/tools"), str(ROOT / "training")]


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: full Brass training, run explicitly with -m slow")


def pytest_collection_modifyitems(config, items):
    import pytest
    if "slow" in (config.getoption("-m") or ""):
        return
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(pytest.mark.skip(reason="full Brass training; run with -m slow"))


"""Keep automatic game recordings from test fixtures out of the user's library."""
import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_replay_archive(tmp_path, monkeypatch, shared_fixture_replay_archive):
    monkeypatch.setenv('BRASS_REPLAY_ROOT', str(tmp_path/'replay-archive'))


@pytest.fixture(scope="session", autouse=True)
def shared_fixture_replay_archive(tmp_path_factory):
    # Module/session-scoped fixtures run before function-scoped autouse fixtures.
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("BRASS_REPLAY_ROOT", str(tmp_path_factory.mktemp("shared-replay-archive")))
        yield
