"""Non-secret configuration and the provider boundary (stdlib only)."""

# ruff: noqa: UP045 -- get_type_hints on Python 3.9 needs typing.Optional.
from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
from typing import Optional, Protocol

_ERRORS = {
    "profile": "Invalid LLM profile or unsupported setting.",
    "credential": "A valid credential is required.",
    "provider": "Provider rejected the request or returned an error.",
    "authentication": "Provider rejected the credential or its permissions.",
    "credit": "Provider reports insufficient credits.",
    "rate_limit": "Provider rate limit reached.",
    "http": "Provider HTTP error or redirect.",
    "transport": "Could not complete the provider connection.",
    "timeout": "Provider request timed out.",
    "request_size": "Complete request exceeds the configured byte limit.",
    "response_size": "Provider response exceeds the configured byte limit.",
    "truncated": "Provider truncated the reply.",
    "refusal": "Provider refused the request.",
    "reply": "Invalid or stale provider reply.",
    "busy": "A decision is already pending.",
    "cancelled": "Decision cancelled.",
    "closed": "LLM seat is closed.",
    "history": "Public event or observation is inconsistent with this seat.",
    "call_limit": "LLM seat reached its call limit.",
}


class LLMError(ValueError):
    """A fixed public error code and message. Never forwards untrusted text."""

    def __init__(self, code):
        self.code = code if type(code) is str and code in _ERRORS else "provider"
        super().__init__(_ERRORS[self.code])


def compact(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def looks_secret(text):
    return bool(re.search(r"(?:(?<![a-z0-9])sk-|bearer\s|://|[?&]key=)", text, re.IGNORECASE))


@dataclass(frozen=True)
class Profile:
    name: str
    model: str
    provider: str = "openrouter"
    response_mode: str = "json"
    timeout_seconds: float = 120
    max_output_tokens: int = 1024
    max_calls: int = 100
    max_bytes: int = 512 * 1024

    def __post_init__(self):
        patterns = (
            (self.name, r"[\w .+():/-]{1,80}"),
            (self.model, r"[A-Za-z0-9][A-Za-z0-9_./:+-]{0,199}"),
            (self.provider, r"[a-z][a-z0-9_-]{0,39}"),
        )
        for value, pattern in patterns:
            if (
                type(value) is not str
                or not re.fullmatch(pattern, value)
                or not value.strip()
                or looks_secret(value)
            ):
                raise LLMError("profile")
        if self.response_mode not in ("json", "strict"):
            raise LLMError("profile")
        if (
            type(self.timeout_seconds) not in (float, int)
            or not 0.05 <= self.timeout_seconds <= 600
        ):
            raise LLMError("profile")
        object.__setattr__(self, "timeout_seconds", float(self.timeout_seconds))
        for value, lower, upper in (
            (self.max_calls, 1, 1000),
            (self.max_bytes, 1024, 4 * 1024 * 1024),
            (self.max_output_tokens, 1, 16384),
        ):
            if type(value) is not int or not lower <= value <= upper:
                raise LLMError("profile")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        allowed = {item.name for item in fields(cls)}
        if type(data) is not dict or not {"model", "name"} <= data.keys() <= allowed:
            raise LLMError("profile")
        return cls(**data)


@dataclass(frozen=True)
class TurnRequest:
    turn_id: str
    instructions: str = field(repr=False)
    turn_json: str = field(repr=False)


@dataclass(frozen=True)
class ProviderReply:
    content: Optional[str] = field(repr=False)
    reported_model: Optional[str] = field(default=None, repr=False)
    reported_provider: Optional[str] = field(default=None, repr=False)
    usage: Mapping = field(default_factory=dict, repr=False)
    error_code: Optional[str] = field(default=None, repr=False)


class Adapter(Protocol):
    def complete(self, profile: Profile, request: TurnRequest) -> ProviderReply: ...


class Credential:
    """Memory-only credential lease. Default repr contains no value."""

    __slots__ = ("_value",)

    def __init__(self, value):
        if type(value) is not str or not re.fullmatch(r"[!-~]{1,4096}", value):
            raise LLMError("credential")
        self._value = value

    def __reduce__(self):
        raise TypeError("Credential cannot be serialized")

    def occurs_in(self, text):
        return self._value in text

    def bearer(self):
        return "Bearer " + self._value


def metadata_label(value, credential):
    if (
        type(value) is str
        and re.fullmatch(r"[\w .()/+:\-]{1,200}", value)
        and not (looks_secret(value) or credential.occurs_in(value))
    ):
        return value
    return None


USAGE_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens", "cost")


def usage_fields(value):
    try:
        if not isinstance(value, Mapping):
            return {}
        fields = {key: value.get(key) for key in USAGE_FIELDS}
    except Exception:  # noqa: BLE001 -- provider accessors can raise with private data
        return {}
    clean = {}
    for key, number in fields.items():
        if (
            type(number) is int
            and 0 <= number <= 2**53
            or key == "cost"
            and type(number) is float
            and math.isfinite(number)
            and 0 <= number <= 2**53
        ):
            clean[key] = number
    return clean
