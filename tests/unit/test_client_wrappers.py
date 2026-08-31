"""SafeLlamafileClient thin-wrapper tests (no network).

Pins the wrapper behavior added v0.19.1/the 2026-08-31 reliability session:
- outbound role normalization (trailing system hoisted to the front)
- assistant tool_calls arguments coerced string -> object (llama-server jinja
  raises on string args in multi-turn history)
- forge 0.8.1+ malformed-500 rescue passes through the wrapper
- non-malformed 500 cascades as BackendError (no raw body leak)
- last_thinking captured from ToolCall.reasoning
- max_tokens default applied
"""

from __future__ import annotations

import json

import httpx
import pytest

from forge.core.workflow import TextResponse, ToolCall
from forge.errors import BackendError

from coding_guardrails.proxy.client import SafeLlamafileClient

MSGS = [{"role": "user", "content": "u"}, {"role": "system", "content": "sys"}]
TOOL_MSG = {
    "role": "assistant",
    "content": "",
    "tool_calls": [{
        "id": "c1", "type": "function",
        "function": {"name": "bash", "arguments": "{\"command\": \"ls\"}"},
    }],
}


async def _make(handler) -> SafeLlamafileClient:
    c = SafeLlamafileClient(
        gguf_path="/tmp/M.gguf", base_url="http://x/v1",
        mode="native", default_max_tokens=777,
    )
    c._http = httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=5.0)
    return c


def _capture(captured: list[dict]):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured.append({
            "roles": [m["role"] for m in body["messages"]],
            "max_tokens": body.get("max_tokens"),
        })
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
            request=request,
        )
    return handler


def _tok(messages=MSGS):
    return [{"role": "user", "content": "hi"}] if messages is MSGS else messages


class TestRoleNormalization:
    async def test_hoists_trailing_system(self) -> None:
        captured: list[dict] = []
        c = await _make(_capture(captured))
        await c.send(MSGS)
        assert captured[0]["roles"][0] == "system"


class TestWireToolArgCoercion:
    async def test_string_args_coerced_to_objects_on_the_wire(self) -> None:
        captured: list[dict] = []
        handler = _capture(captured)
        # Re-capture the tool_calls too
        def _h(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            captured.append(body)
            return httpx.Response(
                200,
                json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
                request=request,
            )
        c = await _make(_h)
        await c.send([{"role": "user", "content": "hi"}, TOOL_MSG,
                      {"role": "tool", "tool_call_id": "c1", "content": "ls"}])
        tcs = captured[0]["messages"][1]["tool_calls"]
        assert isinstance(tcs[0]["function"]["arguments"], dict)
        assert tcs[0]["function"]["arguments"] == {"command": "ls"}

    async def test_unparseable_args_left_untouched(self) -> None:
        handler = _capture([])
        c = await _make(handler)
        bad = {"role": "assistant", "content": "", "tool_calls": [{
            "id": "c1", "type": "function",
            "function": {"name": "bash", "arguments": "{oops"},
        }]}
        # Should not raise; coercion leaves the bad string as-is.
        await c.send([{"role": "user", "content": "hi"}, bad])


class TestRescueInheritance:
    async def test_malformed_500_rescue_returns_toolcall(self) -> None:
        from forge.core.workflow import ToolSpec

        tool = ToolSpec.from_json_schema(
            name="bash", description="Run a shell command",
            schema={"type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"]},
        )

        async def handler(request: httpx.Request) -> httpx.Response:
            # llama.cpp malformed <tool_call> XML echoed inside the 500 body
            # (Qwen-coder format — the shape the model was pretrained on).
            return httpx.Response(
                500,
                text='{"error": {"message": "Failed to parse input: "}} '
                     '<tool_call><function=bash><parameter=command>ls</parameter></function></tool_call>',
                request=request,
            )
        c = await _make(handler)
        res = await c.send(_tok(), tools=[tool])
        assert isinstance(res, list) and res[0].tool == "bash"
        assert res[0].args == {"command": "ls"}

    async def test_non_malformed_500_raises_backend_error(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="boom", request=request)
        c = await _make(handler)
        with pytest.raises(BackendError):
            await c.send(_tok())


class TestThinkingCapture:
    async def test_last_thinking_from_toolcall(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={
                "choices": [{"message": {
                    "role": "assistant", "content": "",
                    "reasoning_content": "my thinking",
                    "tool_calls": [{
                        "id": "c1", "type": "function",
                        "function": {"name": "bash", "arguments": "{\"command\": \"ls\"}"},
                    }],
                }}],
            }, request=request)
        c = await _make(handler)
        await c.send(_tok())
        assert c.last_thinking == "my thinking"


class TestMaxTokens:
    async def test_default_max_tokens_applied(self) -> None:
        captured: list[dict] = []
        c = await _make(_capture(captured))
        await c.send(_tok())
        assert captured[0]["max_tokens"] == 777