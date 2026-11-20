# SPDX-License-Identifier: Apache-2.0
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: long strength runs, skipped unless -m slow")


def pytest_collection_modifyitems(config, items):
    if "slow" in (config.getoption("-m") or ""):
        return
    skip = pytest.mark.skip(reason="slow test; run with -m slow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)


"""Keep automatic game recordings from test fixtures out of the user's library."""


@pytest.fixture(autouse=True)
def isolated_replay_archive(tmp_path, monkeypatch, shared_fixture_replay_archive):
    monkeypatch.setenv('BRASS_REPLAY_ROOT', str(tmp_path/'replay-archive'))


@pytest.fixture(scope="session", autouse=True)
def shared_fixture_replay_archive(tmp_path_factory):
    # Module/session-scoped fixtures run before function-scoped autouse fixtures.
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("BRASS_REPLAY_ROOT", str(tmp_path_factory.mktemp("shared-replay-archive")))
        yield
