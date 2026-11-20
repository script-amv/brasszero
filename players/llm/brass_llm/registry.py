"""Pinned profile/adapter construction. Factories are called only for live seats."""

import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path

from brass_ai.replay_archive import engine_id

from .agent import LLMAgent
from .openrouter import OpenRouter
from .prompt import PROMPT_VERSION, PROTOCOL_VERSION, instructions
from .protocol import Credential, LLMError, Profile, compact, looks_secret


@lru_cache(maxsize=1)
def _source_id():
    h = hashlib.sha256()
    for name in (
        "agent.py",
        "history.py",
        "openrouter.py",
        "prompt.py",
        "protocol.py",
        "registry.py",
    ):
        h.update(
            name.encode() + b"\0" + Path(__file__).with_name(name).read_bytes() + b"\0"
        )
    return h.hexdigest()


class ProviderRegistry:
    def __init__(self):
        self._entries = {}
        self.register("openrouter", "openrouter-chat/1", OpenRouter)

    def register(self, name, version, factory):
        if (
            type(name) is not str
            or not re.fullmatch("[a-z][a-z0-9_-]{0,39}", name)
            or type(version) is not str
            or not re.fullmatch("[a-zA-Z0-9_.:/-]{1,128}", version)
            or looks_secret(name)
            or looks_secret(version)
            or name in self._entries
            or not callable(factory)
        ):
            raise LLMError("profile")
        self._entries[name] = version, factory

    def providers(self):
        return [
            {"id": name, "version": entry[0]} for name, entry in self._entries.items()
        ]

    def provenance(self, profile):
        if not isinstance(profile, Profile) or profile.provider not in self._entries:
            raise LLMError("profile")
        return {
            "profile": profile.to_dict(),
            "adapter_version": self._entries[profile.provider][0],
            "protocol_version": PROTOCOL_VERSION,
            "prompt_version": PROMPT_VERSION,
            "prompt_sha256": hashlib.sha256(instructions().encode()).hexdigest(),
            "source_sha256": _source_id(),
            "engine_id": engine_id(),
        }

    def identity(self, profile):
        content = json.dumps(
            self.provenance(profile), sort_keys=True, separators=(",", ":")
        ).encode()
        return "llm:sha256-v1:" + hashlib.sha256(content).hexdigest()

    def make_agent(self, profile, credential):
        provenance = self.provenance(profile)
        lease = (
            credential if isinstance(credential, Credential) else Credential(credential)
        )
        if any(
            lease.occurs_in(v)
            for v in (
                profile.name,
                profile.model,
                profile.provider,
                profile.response_mode,
            )
        ):
            raise LLMError("credential")
        try:
            adapter = self._entries[profile.provider][1](lease)
        except Exception:  # noqa: BLE001 -- never expose an adapter exception
            error = LLMError("provider")
        else:
            return LLMAgent(
                profile, self.identity(profile), compact(provenance), lease, adapter
            )
        raise error
