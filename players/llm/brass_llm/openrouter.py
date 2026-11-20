"""One non-streaming Chat Completions POST to the fixed OpenRouter origin."""

import http.client
import json
import socket
import threading
import time
from queue import Empty, Queue

from .protocol import (
    Credential,
    LLMError,
    ProviderReply,
    compact,
    metadata_label,
    usage_fields,
)


def _http_code(status):
    return {
        401: "authentication",
        402: "credit",
        403: "authentication",
        408: "timeout",
        429: "rate_limit",
    }.get(status, "http")


def _parse(data, credential):
    if type(data) is not dict:
        raise LLMError("reply")
    report = {
        "reported_model": metadata_label(data.get("model"), credential),
        "reported_provider": metadata_label(data.get("provider"), credential),
        "usage": usage_fields(data.get("usage")),
    }
    if "error" in data:
        return ProviderReply(None, **report, error_code="provider")
    choices = data.get("choices")
    if type(choices) is not list or len(choices) != 1 or type(choices[0]) is not dict:
        return ProviderReply(None, **report, error_code="reply")
    choice = choices[0]
    message = choice.get("message")
    finish = choice.get("finish_reason")
    if "error" in choice:
        error = "provider"
    elif finish == "length":
        error = "truncated"
    elif finish == "content_filter" or (
        isinstance(message, dict) and message.get("refusal")
    ):
        error = "refusal"
    elif (
        finish != "stop"
        or not isinstance(message, dict)
        or type(message.get("content")) is not str
    ):
        error = "reply"
    else:
        return ProviderReply(message["content"], **report)
    return ProviderReply(None, **report, error_code=error)


class OpenRouter:
    def __init__(self, credential):
        self._credential = (
            credential if isinstance(credential, Credential) else Credential(credential)
        )

    @staticmethod
    def _connection(timeout):
        return http.client.HTTPSConnection("openrouter.ai", timeout=timeout)

    def complete(self, profile, request):
        body = {
            "model": profile.model,
            "stream": False,
            "max_tokens": profile.max_output_tokens,
            "plugins": [{"id": "context-compression", "enabled": False}],
            "messages": [
                {"role": "system", "content": request.instructions},
                {"role": "user", "content": request.turn_json},
            ],
        }
        if profile.response_mode == "strict":
            schema = {
                "type": "object",
                "properties": {
                    "turn_id": {"type": "string"},
                    "action_id": {"type": "integer"},
                },
                "required": ["turn_id", "action_id"],
                "additionalProperties": False,
            }
            body.update(
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "brass_action",
                        "strict": True,
                        "schema": schema,
                    },
                },
                provider={"require_parameters": True},
            )
        wire = compact(body).encode()
        if len(wire) > profile.max_bytes:
            raise LLMError("request_size")

        # Socket inactivity timeouts alone do not bound DNS resolution or a trickling response.
        # Keep the absolute deadline even when the OS resolver has not returned. A late
        # connection checks the deadline before it can send the credential or paid request.
        deadline = time.monotonic() + profile.timeout_seconds
        result = Queue(maxsize=1)
        connection = self._connection(profile.timeout_seconds)
        connected = []

        def exchange():
            try:
                connection.connect()
                connected.append(connection.sock)
                if time.monotonic() >= deadline:
                    raise LLMError("timeout")
                connection.request(
                    "POST",
                    "/api/v1/chat/completions",
                    wire,
                    headers={
                        "Authorization": self._credential.bearer(),
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                    },
                )
                with connection.getresponse() as response:
                    if response.status != 200:
                        raise LLMError(_http_code(response.status))
                    declared = response.length
                    if declared is not None and declared > profile.max_bytes:
                        raise LLMError("response_size")
                    raw = response.read(profile.max_bytes + 1)
                    if declared is not None and len(raw) != declared:
                        raise LLMError("truncated")
                    if len(raw) > profile.max_bytes:
                        raise LLMError("response_size")
                try:
                    data = json.loads(raw)
                except (ValueError, UnicodeError, RecursionError):
                    raise LLMError("reply") from None
                value = _parse(data, self._credential)
            except LLMError as exc:
                value = LLMError(exc.code)
            except (socket.timeout, TimeoutError):
                value = LLMError("timeout")
            except http.client.IncompleteRead:
                value = LLMError("truncated")
            except (OSError, http.client.HTTPException):
                value = LLMError("transport")
            finally:
                connection.close()
            result.put(value)

        threading.Thread(target=exchange, daemon=True, name="brass-openrouter").start()
        try:
            value = result.get(timeout=max(0, deadline - time.monotonic()))
        except Empty:
            if connected:
                try:
                    connected[0].shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
            value = LLMError("timeout")
        if isinstance(value, LLMError):
            raise value
        return value
