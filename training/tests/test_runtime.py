# SPDX-License-Identifier: Apache-2.0
"""Exercise real local numerical operations and optional Stage 1 imports."""
import json
import math
import platform

import pytest

from brass_rl.runtime import runtime_check
from helpers import run_without_torch


def test_optional_import():
    result = run_without_torch("""
import json
import brass_ai
import brass_rl
import brass_rl.config
import brass_rl.runtime
from brass_ai.agents import known_agents
assert 'torch' not in sys.modules
print(json.dumps(known_agents()))
""")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == ["greedy", "random"]


def test_runtime_check_exercises_optimizer_and_checkpoint():
    report = runtime_check("cpu")
    assert report["device"] == "cpu"
    assert report["versions"]["torch"].startswith("2.2.2")
    assert report["versions"]["numpy"] == "1.26.4"
    profiles = {"Darwin": "3.9.", "Linux": "3.10."}
    assert platform.system() in profiles, "record/test a supported dependency profile for this platform"
    assert report["versions"]["python"].startswith(profiles[platform.system()])
    assert report["platform"]
    assert report["checkpoint_roundtrip"] is True
    assert report["parameters_changed"] is True
    assert report["loss_after"] < report["loss_before"]
    assert report["grad_norm"] > 0
    assert all(math.isfinite(report[key]) for key in ("loss_before", "loss_after", "grad_norm"))
    json.dumps(report, allow_nan=False)


def test_runtime_check_reports_missing_optional_dependency():
    result = run_without_torch("""
from brass_rl.runtime import runtime_check
try:
    runtime_check('cpu')
except RuntimeError as exc:
    assert 'optional' in str(exc).lower()
else:
    raise AssertionError('missing torch must fail clearly')
""")
    assert result.returncode == 0, result.stderr


def test_runtime_check_rejects_unknown_device():
    with pytest.raises(ValueError, match="device"):
        runtime_check("tpu")
