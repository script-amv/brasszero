# SPDX-License-Identifier: Apache-2.0
import http.client
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import brass
import pytest

KEY = "fixture-sensitive-key"


@pytest.fixture
def endpoint(monkeypatch):
    from brass_llm import OpenRouter

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self.server.received.append(
                (self.path, self.headers["Authorization"], body)
            )
            turn = json.loads(body["messages"][1]["content"])
            response = {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(
                                {"turn_id": turn["turn_id"], "action_id": 0}
                            )
                        },
                    }
                ],
                "model": "a/b-2026",
                "provider": "Example Provider",
                "usage": {"total_tokens": 9, "cost": 0.01},
            }
            code, response = self.server.reply(response)
            self.send_response(code)
            self.send_header("Location", self.server.redirect)
            if hasattr(self.server, "declared_length"):
                self.send_header("Content-Length", str(self.server.declared_length))
            self.end_headers()
            try:
                self.wfile.write(json.dumps(response).encode())
            except OSError:
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.received = []
    server.reply = lambda body: (200, body)
    server.redirect = f"http://localhost:{server.server_port}/redirect"
    task = threading.Thread(target=server.serve_forever, daemon=True)
    task.start()
    monkeypatch.setattr(
        OpenRouter,
        "_connection",
        staticmethod(
            lambda timeout: http.client.HTTPConnection(
                "127.0.0.1", server.server_port, timeout=timeout
            )
        ),
    )
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        task.join()


def call(mode="json", **limits):
    from brass_llm import Profile, ProviderRegistry

    p = Profile("Fixture", "a/b", response_mode=mode, **limits)
    agent = ProviderRegistry().make_agent(p, KEY)
    game = brass.Game(42)
    obs = game.observation(game.current_player())
    agent.reset(obs.me, obs.public_json())
    index = agent.act(obs)
    return index, agent.status()


@pytest.mark.parametrize("mode", ["json", "strict"])
def test_wire_protocol_and_reported_accounting(endpoint, mode):
    result, status = call(mode)
    assert (
        result == 0
        and status["usage"]["total_tokens"] == 9
        and status["usage"]["cost"] == 0.01
    )
    assert (
        status["reported_model"] == "a/b-2026"
        and status["reported_provider"] == "Example Provider"
    )
    path, authorization, body = endpoint.received[0]
    assert path == "/api/v1/chat/completions" and authorization == "Bearer " + KEY
    assert (
        body["model"] == "a/b"
        and body["stream"] is False
        and body["max_tokens"] == 1024
    )
    assert body["plugins"] == [{"id": "context-compression", "enabled": False}]
    assert KEY not in str(body)
    if mode == "strict":
        schema = body["response_format"]["json_schema"]
        assert (
            schema["strict"] is True
            and schema["schema"]["additionalProperties"] is False
        )
        assert set(schema["schema"]["required"]) == {"turn_id", "action_id"}
        assert body["provider"]["require_parameters"] is True
    else:
        assert "response_format" not in body and "provider" not in body


@pytest.mark.parametrize(
    "case,code",
    [
        ("unauthorized", "authentication"),
        ("rate", "rate_limit"),
        ("redirect", "http"),
        ("body_error", "provider"),
        ("refusal", "refusal"),
        ("length", "truncated"),
        ("empty", "reply"),
        ("large", "response_size"),
        ("timeout", "timeout"),
        ("missing_finish", "reply"),
    ],
)
def test_no_retries_or_reflected_errors(endpoint, case, code):
    from brass_llm import LLMError

    def reply(data):
        choice = data["choices"][0]
        if case == "timeout":
            time.sleep(0.25)
        elif case in ("unauthorized", "rate", "redirect", "body_error"):
            return {
                "unauthorized": 401,
                "rate": 429,
                "redirect": 307,
                "body_error": 200,
            }[case], {"error": {"message": KEY}}
        elif case == "refusal":
            choice["message"]["refusal"] = KEY
        elif case == "length":
            choice["finish_reason"] = "length"
        elif case == "empty":
            choice["message"]["content"] = None
        elif case == "large":
            data["padding"] = "x" * (512 * 1024)
        elif case == "missing_finish":
            del choice["finish_reason"]
        return 200, data

    endpoint.reply = reply
    with pytest.raises(LLMError) as error:
        call(timeout_seconds=0.12 if case == "timeout" else 3)
    assert error.value.code == code
    assert KEY not in str(error.value)
    assert len(endpoint.received) == 1


def test_reflected_metadata_and_unknown_usage_are_unavailable(endpoint):
    endpoint.reply = lambda data: (
        200,
        dict(
            data,
            model="alias-" + KEY,
            provider=KEY,
            usage={"total_tokens": True, "cost": -4},
        ),
    )
    index, status = call()
    assert index == 0 and KEY not in str(status)
    assert status["reported_model"] is None and status["reported_provider"] is None
    assert all(v is None for v in status["usage"].values())


def test_incomplete_http_body_cannot_be_accepted_as_a_finished_reply(endpoint):
    # The connection closes after a valid JSON object but before its declared length.
    from brass_llm import LLMError

    endpoint.declared_length = 400000
    with pytest.raises(LLMError) as error:
        call()
    assert error.value.code == "truncated"
