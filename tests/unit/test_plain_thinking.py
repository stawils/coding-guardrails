"""Thinking-mode reliability tests for the plain (no-tool) path.

Covers the 2026-09 thinking work:
- auto-no-thinking default still forces enable_thinking=false on plain requests
- explicit opt-in (chat_template_kwargs.enable_thinking / reasoning_effort)
  keeps thinking on and the captured reasoning reaches the agent per
  reasoning_replay (keep-last → reasoning_content, full → content)
- thinking_budget_tokens is applied server-side unless the request sets its own
- empty-content-after-thinking (reasoning ate the budget) retries ONCE with
  thinking off so the agent always gets a real answer
- the client capture wrapper recovers reasoning_content from raw text
  responses (Forge drops it on the text parse path)
"""

from __future__ import annotations

import json

import httpx

from forge.core.workflow import TextResponse

from coding_guardrails.proxy.client import SafeLlamafileClient
from coding_guardrails.proxy.handler import handle_chat_completions

from tests.unit.test_handler_respond import _FakeClient


class _RecordingClient(_FakeClient):
    """_FakeClient that also records sampling and exposes last_thinking."""

    def __init__(self, responses: list, last_thinking: str = "") -> None:
        super().__init__(responses)
        self.samplings: list[dict | None] = []
        self.last_thinking = last_thinking

    async def send(self, messages, tools=None, sampling=None):
        self.samplings.append(sampling)
        return await super().send(messages, tools=tools, sampling=sampling)


_PLAIN = {
    "model": "m",
    "stream": False,
    "messages": [{"role": "user", "content": "hi"}],
}


async def _handle(body, client, **kw):
    from forge.context.manager import ContextManager
    from forge.context.strategies import TieredCompact
    from coding_guardrails.middleware import CodingGuardrails

    return await handle_chat_completions(
        body, client, ContextManager(strategy=TieredCompact(), budget_tokens=32000),
        CodingGuardrails(), max_retries=2, **kw,
    )


class TestPlainThinkingPolicy:
    async def test_auto_on_defaults_thinking_off(self) -> None:
        client = _RecordingClient([TextResponse(content="direct answer")])
        result = await _handle(_PLAIN, client)
        msg = result["choices"][0]["message"]
        assert msg["content"] == "direct answer"
        assert "reasoning_content" not in msg
        assert client.samplings[0]["chat_template_kwargs"]["enable_thinking"] is False

    async def test_auto_on_request_enable_thinking_true_wins_and_relays(self) -> None:
        """Explicit per-request opt-in overrides auto-no-thinking and the captured
        reasoning reaches the agent in reasoning_content (keep-last default)."""
        body = {**_PLAIN, "chat_template_kwargs": {"enable_thinking": True}}
        client = _RecordingClient(
            [TextResponse(content="sum is 5050")], last_thinking="compute 100*101/2.",
        )
        result = await _handle(body, client)
        msg = result["choices"][0]["message"]
        assert msg["content"] == "sum is 5050"
        assert msg["reasoning_content"] == "compute 100*101/2."
        ckw = client.samplings[0]["chat_template_kwargs"]
        assert ckw["enable_thinking"] is True
        # Budget applied automatically and eagerly (request set no effort/budget)
        assert client.samplings[0]["thinking_budget_tokens"] == 4096

    async def test_auto_on_request_effort_level_enables_thinking(self) -> None:
        body = {**_PLAIN, "reasoning_effort": "high"}
        client = _RecordingClient(
            [TextResponse(content="answer")], last_thinking="thinking hard",
        )
        result = await _handle(body, client)
        assert result["choices"][0]["message"]["reasoning_content"] == "thinking hard"
        assert client.samplings[0]["reasoning_effort"] == "high"
        # Explicit effort → the proxy does not impose its own budget
        assert "thinking_budget_tokens" not in client.samplings[0]

    async def test_effort_none_forces_thinking_off(self) -> None:
        body = {**_PLAIN, "reasoning_effort": "none"}
        client = _RecordingClient(
            [TextResponse(content="answer")], last_thinking="",
        )
        result = await _handle(body, client)
        msg = result["choices"][0]["message"]
        assert msg["content"] == "answer"
        assert "reasoning_content" not in msg

    async def test_auto_off_leaves_template_default_and_relays(self) -> None:
        body = {**_PLAIN}
        client = _RecordingClient(
            [TextResponse(content="answer")], last_thinking="why I think so",
        )
        result = await _handle(body, client, auto_no_thinking=False)
        assert result["choices"][0]["message"]["reasoning_content"] == "why I think so"
        # No forced enable_thinking on the wire — the template default applies
        assert "enable_thinking" not in client.samplings[0].get("chat_template_kwargs", {})


class TestPlainThinkingRelayModes:
    async def test_replay_full_merges_reasoning_into_content(self) -> None:
        body = {**_PLAIN, "chat_template_kwargs": {"enable_thinking": True}}
        client = _RecordingClient(
            [TextResponse(content="answer")], last_thinking="thinking",
        )
        result = await _handle(body, client, reasoning_replay="full")
        msg = result["choices"][0]["message"]
        assert msg["content"] == "thinking\n\nanswer"
        assert "reasoning_content" not in msg

    async def test_replay_none_drops_reasoning(self) -> None:
        body = {**_PLAIN, "chat_template_kwargs": {"enable_thinking": True}}
        client = _RecordingClient(
            [TextResponse(content="answer")], last_thinking="thinking",
        )
        result = await _handle(body, client, reasoning_replay="none")
        msg = result["choices"][0]["message"]
        assert msg["content"] == "answer"
        assert "reasoning_content" not in msg

    async def test_stream_relays_reasoning_delta_first(self) -> None:
        body = {**{k: v for k, v in _PLAIN.items() if k != "stream"},
                "stream": True, "chat_template_kwargs": {"enable_thinking": True}}
        client = _RecordingClient(
            [TextResponse(content="answer")], last_thinking="thinking",
        )
        events = await _handle(body, client)
        assert events[0]["choices"][0]["delta"]["reasoning_content"] == "thinking"
        assert events[0]["choices"][0]["delta"].get("role") == "assistant"
        bodies = "".join(
            e["choices"][0]["delta"].get("content", "") for e in events
        )
        assert "answer" in bodies


class TestPlainThinkingReliability:
    async def test_empty_after_thinking_retries_once_without_thinking(self) -> None:
        """Thinking ate the budget (reasoning present, empty answer): retry once
        with enable_thinking=false; the answer comes from the retry and the stale
        reasoning is NOT leaked onto the wire."""
        body = {**_PLAIN, "chat_template_kwargs": {"enable_thinking": True}}
        client = _RecordingClient([TextResponse(content=""), TextResponse(content="real answer")])
        client.last_thinking = "long reasoning that ran out of budget"

        result = await _handle(body, client)
        msg = result["choices"][0]["message"]
        assert msg["content"] == "real answer"
        assert "reasoning_content" not in msg
        assert len(client.samplings) == 2
        assert client.samplings[1]["chat_template_kwargs"]["enable_thinking"] is False
        assert "thinking_budget_tokens" not in client.samplings[1]
        assert "reasoning_effort" not in client.samplings[1]

    async def test_retry_still_empty_returns_empty(self) -> None:
        body = {**_PLAIN, "chat_template_kwargs": {"enable_thinking": True}}
        client = _RecordingClient([TextResponse(content=""), TextResponse(content="")])
        client.last_thinking = "thinking"
        result = await _handle(body, client)
        assert result["choices"][0]["message"]["content"] == ""

    async def test_request_owned_budget_not_overridden(self) -> None:
        body = {**_PLAIN, "chat_template_kwargs": {"enable_thinking": True},
                "thinking_budget_tokens": 512}
        client = _RecordingClient([TextResponse(content="answer")], last_thinking="t")
        await _handle(body, client)
        assert client.samplings[0]["thinking_budget_tokens"] == 512


class TestClientTextReasoningCapture:
    async def test_text_response_reasoning_captured_into_last_thinking(self) -> None:
        """The raw envelope's reasoning_content survives Forge's text parse via
        the capture wrapper and lands in client.last_thinking."""

        def handler(request: httpx.Request) -> httpx.Response:
            data = json.loads(request.content)
            assert data.get("chat_template_kwargs", {}).get("enable_thinking") is True
            return httpx.Response(
                200,
                json={"choices": [{"message": {
                    "role": "assistant",
                    "content": "sum is 5050",
                    "reasoning_content": "compute 100*101/2",
                }}]},
                request=request,
            )

        c = SafeLlamafileClient(
            gguf_path="/tmp/M.gguf", base_url="http://x/v1",
            mode="native", default_max_tokens=777,
        )
        c._http = httpx.AsyncClient(
            transport=httpx.MockTransport(handler), timeout=5.0,
        )
        # Re-attach the capture wrapper (normal init applied it; tests replace _http)
        from coding_guardrails.proxy.client import _ReasoningCaptureHttp
        c._http = _ReasoningCaptureHttp(c._http)

        result = await c.send(
            [{"role": "user", "content": "sum 1..100"}],
            sampling={"chat_template_kwargs": {"enable_thinking": True}},
        )
        assert isinstance(result, TextResponse)
        assert result.content == "sum is 5050"
        assert c.last_thinking == "compute 100*101/2"

    async def test_reasoning_effort_and_budget_forwarded_on_wire(self) -> None:
        captured: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(json.loads(request.content))
            return httpx.Response(
                200,
                json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
                request=request,
            )

        c = SafeLlamafileClient(
            gguf_path="/tmp/M.gguf", base_url="http://x/v1",
            mode="native", default_max_tokens=777,
        )
        c._http = httpx.AsyncClient(
            transport=httpx.MockTransport(handler), timeout=5.0,
        )
        from coding_guardrails.proxy.client import _ReasoningCaptureHttp
        c._http = _ReasoningCaptureHttp(c._http)

        await c.send(
            [{"role": "user", "content": "hi"}],
            sampling={"reasoning_effort": "high", "thinking_budget_tokens": 2048},
        )
        body = captured[0]
        assert body["reasoning_effort"] == "high"
        assert body["thinking_budget_tokens"] == 2048
        assert body["max_tokens"] == 777