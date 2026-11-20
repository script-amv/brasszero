# SPDX-License-Identifier: Apache-2.0
from dataclasses import FrozenInstanceError, replace

import pytest


def test_version_pins_all_limits_and_the_registered_adapter():
    from brass_llm import Profile, ProviderRegistry

    p = Profile("Example", "lab/model", provider="fixture")
    assert Profile.from_dict(p.to_dict()) == p
    with pytest.raises(FrozenInstanceError):
        p.name = "Changed"
    registry = ProviderRegistry()
    registry.register("fixture", "v1", lambda key: None)
    r2 = ProviderRegistry()
    r2.register("fixture", "v2", lambda key: None)
    assert registry.identity(p) != r2.identity(p)
    assert registry.identity(p) != registry.identity(
        replace(p, max_bytes=p.max_bytes - 1)
    )
    assert registry.provenance(p)["profile"] == p.to_dict()
    assert registry.make_agent(p, "opaque-secret").replay_identity == registry.identity(
        p
    )
    assert "opaque-secret" not in repr(registry.make_agent(p, "opaque-secret"))
    assert "opaque-secret" not in str(registry.provenance(p))
    assert p.timeout_seconds == 120 and p.max_output_tokens == 1024
    assert p.max_calls == 100 and p.max_bytes == 512 * 1024


@pytest.mark.parametrize(
    "extra",
    [
        {"extra-secret": "opaque-secret"},
        {"model": "sk-or-v1-verysecret"},
        {"provider": "https://host"},
        {"name": "sk-or-v1-verysecret"},
        {"response_mode": "sk-verysecret"},
        {"max_calls": True},
        {"max_bytes": 0},
        {"timeout_seconds": float("inf")},
        {"max_output_tokens": 999999999},
    ],
)
def test_invalid_profiles_have_static_errors(extra):
    from brass_llm import LLMError, Profile

    with pytest.raises(LLMError) as failure:
        Profile.from_dict({"name": "Example", "model": "lab/model", **extra})
    assert failure.value.code == "profile"
    assert "secret" not in str(failure.value)


def test_reflected_credential_cannot_enter_metadata():
    from brass_llm import LLMError, Profile, ProviderRegistry

    r = ProviderRegistry()
    p = Profile("Local Test", "vendor/opaque-secret")
    with pytest.raises(LLMError) as failure:
        r.make_agent(p, "opaque-secret")
    assert "opaque-secret" not in str(failure.value)


def test_profile_provenance_pins_native_component_and_action_renderer():
    from brass_ai.replay_archive import engine_id
    from brass_llm import Profile, ProviderRegistry

    provenance = ProviderRegistry().provenance(Profile("Example", "lab/model"))
    assert provenance["engine_id"] == engine_id()
    assert all(len(provenance[key]) == 64 for key in ("prompt_sha256", "source_sha256"))


def test_ordinary_task_labels_and_credential_boundaries():
    from brass_llm import Credential, Profile
    from brass_llm.protocol import metadata_label

    key = Credential("opaque-exact")
    assert Profile("Task-1", "vendor/task-2").name == "Task-1"
    assert metadata_label("Task-1", key) == "Task-1"
    for label in ("vendor/sk-test", "Label (SK-test)", "prefixopaque-exacttail"):
        assert metadata_label(label, key) is None
