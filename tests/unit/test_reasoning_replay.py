"""Reasoning replay wiring tests.

forge-guardrails >= 0.7.5 defaults its response converters to
``reasoning_replay="none"``: the client still *captures* thinking (observability,
logs) but the wire response to the agent carries neither the thinking in
``content`` nor a ``reasoning_content`` field. This silently looks like
"thinking is disabled" to the agent — the exact complaint that motivated this
test. Our proxy must explicitly opt back in so captured reasoning is delivered.

Validated end-to-end on 2026-08-31: backend emits reasoning_content, client
captures it (``last_thinking`` + ``ToolCall.reasoning``), and the final OpenAI
response only contains it when ``reasoning_replay`` is passed to forge's
converters. See plans/2026-08-31_handoff.md.
"""

from __future__ import annotations

from forge.context.manager import ContextManager
from forge.context.strategies import TieredCompact
from forge.core.workflow import TextResponse, ToolCall

from coding_guardrails.middleware import CodingGuardrails
from coding_guardrails.proxy.handler import handle_chat_completions

from tests.unit.test_handler_respond import _FakeClient, _tool

BASH = _tool("bash", {"command": {"type": "string"}})
THINKING = "The user wants the sum. Let me compute 100*101/2 = 5050."


def _ctx() -> ContextManager:
    return ContextManager(strategy=TieredCompact(), budget_tokens=32000)


async def _run(body: dict, responses: list, reasoning_replay: str = "keep-last"):
    client = _FakeClient(responses)
    guardrails = CodingGuardrails()
    body = {**body, "messages": [{"role": "user", "content": "hi"}]}
    return await handle_chat_completions(
        body, client, _ctx(), guardrails, max_retries=2,
        reasoning_replay=reasoning_replay,
    )


class TestReasoningReplay:
    async def test_tool_calls_default_replay_keep_last(self) -> None:
        """Tool-call responses carry thinking in reasoning_content by default."""
        client = _FakeClient([[ToolCall(tool="bash", args={"command": "echo 5050"},
                                       reasoning=THINKING)]])
        body = {
            "model": "m", "stream": False,
            "messages": [{"role": "user", "content": "sum 1..100"}],
            "tools": [BASH],
        }
        result = await handle_chat_completions(
            body, client, _ctx(), CodingGuardrails(), max_retries=2,
        )
        msg = result["choices"][0]["message"]
        assert msg["reasoning_content"] == THINKING
        assert msg["content"] is None
        assert [tc["function"]["name"] for tc in msg["tool_calls"]] == ["bash"]

    async def test_replay_full_puts_thinking_in_content(self) -> None:
        """reasoning_replay='full' restores the historical content channel."""
        result = await _run(
            {"model": "m", "stream": False, "tools": [BASH]},
            [[ToolCall(tool="bash", args={"command": "echo 5050"}, reasoning=THINKING)]],
            reasoning_replay="full",
        )
        msg = result["choices"][0]["message"]
        assert msg["content"] == THINKING
        assert "reasoning_content" not in msg

    async def test_replay_none_drops_thinking(self) -> None:
        """reasoning_replay='none' matches forge's default (observability only)."""
        result = await _run(
            {"model": "m", "stream": False, "tools": [BASH]},
            [[ToolCall(tool="bash", args={"command": "echo 5050"}, reasoning=THINKING)]],
            reasoning_replay="none",
        )
        msg = result["choices"][0]["message"]
        assert msg["content"] is None
        assert "reasoning_content" not in msg

    async def test_streaming_sse_carries_reasoning_content(self) -> None:
        """SSE event stream includes the reasoning_content delta (keep-last)."""
        result = await _run(
            {"model": "m", "stream": True, "tools": [BASH]},
            [[ToolCall(tool="bash", args={"command": "echo 5050"}, reasoning=THINKING)]],
        )
        assert isinstance(result, list)
        deltas = []
        for event in result:
            if not isinstance(event, dict):
                continue
            delta = event["choices"][0].get("delta", {})
            deltas.append(delta)
        rc = [d for d in deltas if d.get("reasoning_content")]
        assert rc, "no reasoning_content delta in SSE stream"
        assert rc[0]["reasoning_content"] == THINKING

    async def test_text_response_untouched(self) -> None:
        """Text responses are unaffected by the replay option."""
        result = await _run(
            {"model": "m", "stream": False},
            [TextResponse(content="direct answer")],
        )
        assert result["choices"][0]["message"]["content"] == "direct answer"