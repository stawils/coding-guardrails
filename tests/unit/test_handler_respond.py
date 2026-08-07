"""Tests for the proxy handler's respond-tool handling.

Regression tests for v0.16.1: the proxy converted respond() to text even when
the agent explicitly declared a respond tool in the request. That made every
terminal-tool workflow (Forge evals, custom respond() agents) look like a
prose failure — tool_selection went from 5/5 (pre-v0.7.4) to 0/5 for every
model after the conversion was added (v0.7.4, 2026-06-01).

Fixed behavior:
- Request declares a respond tool  → respond() passes through as a tool call.
- Request has no respond tool      → respond() converts to text (Pi/Cline etc.
  don't declare it; text is their terminal signal).
- Respond-declared requests get terminal enforcement + a terminal-aware retry
  nudge so the model calls respond() first try instead of prose.
"""

from forge.context.manager import ContextManager
from forge.context.strategies import TieredCompact
from forge.core.workflow import TextResponse, ToolCall

from coding_guardrails.middleware import CodingGuardrails
from coding_guardrails.proxy.handler import (
    _terminal_retry_nudge,
    _text_retry_nudge,
    handle_chat_completions,
)


def _tool(name: str, params: dict | None = None) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"Tool {name}",
            "parameters": {
                "type": "object",
                "properties": params or {"message": {"type": "string"}},
                "required": list((params or {"message": {"type": "string"}}).keys()),
            },
        },
    }


RESPOND = _tool("respond")


class _FakeClient:
    """Minimal LLMClient stand-in: returns canned responses in order."""

    api_format = "openai"

    def __init__(self, responses: list) -> None:
        self._responses = list(responses)
        self.sent: list[list[dict]] = []

    async def send(self, messages, tools=None, sampling=None):
        self.sent.append(messages)
        return self._responses.pop(0)


def _ctx() -> ContextManager:
    return ContextManager(strategy=TieredCompact(), budget_tokens=32000)


async def _run(body: dict, responses: list, request_tools=None):
    client = _FakeClient(responses)
    guardrails = CodingGuardrails()
    body = {**body, "messages": [{"role": "user", "content": "hi"}]}
    if request_tools is not None:
        body["tools"] = request_tools
    await handle_chat_completions(
        body, client, _ctx(), guardrails, max_retries=2,
    )
    return client


def _sent_text(client: _FakeClient) -> str:
    """Concatenated text content of everything sent to the fake client."""
    parts = []
    for batch in client.sent:
        for m in batch:
            content = m.get("content")
            if content:
                parts.append(content)
    return "\n".join(parts)


class TestRespondPassThrough:
    """respond() passes through when the request declared a respond tool."""

    async def test_respond_declared_returns_tool_call_response(self) -> None:
        """End-to-end: the openai response object carries the respond tool call."""
        client = _FakeClient([[ToolCall(tool="respond", args={"message": "all done"})]])
        guardrails = CodingGuardrails()
        body = {
            "model": "test-model",
            "stream": False,
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [RESPOND],
        }
        result = await handle_chat_completions(body, client, _ctx(), guardrails, max_retries=2)
        msg = result["choices"][0]["message"]
        assert msg["tool_calls"], "respond() must pass through as a tool call"
        tc = msg["tool_calls"][0]
        assert tc["function"]["name"] == "respond"
        assert "all done" in tc["function"]["arguments"]

    async def test_undeclared_respond_rejected_by_validator(self) -> None:
        """respond() without a declared respond tool → unknown-tool rejection, retry.

        The handler's respond()→text conversion is a safety net; the normal path
        for undisclosed respond() is Forge's validator rejecting it as an unknown
        tool, so the model retries with a declared tool.
        """
        client = _FakeClient([
            [ToolCall(tool="respond", args={"message": "all done"})],
            [ToolCall(tool="inspect", args={"target": "x"})],
        ])
        guardrails = CodingGuardrails()
        body = {
            "model": "test-model",
            "stream": False,
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [_tool("inspect", {"target": {"type": "string"}})],
        }
        result = await handle_chat_completions(body, client, _ctx(), guardrails, max_retries=2)
        msg = result["choices"][0]["message"]
        names = [tc["function"]["name"] for tc in (msg.get("tool_calls") or [])]
        assert names == ["inspect"]  # retried with the declared tool

    async def test_respond_with_other_tools_runs_layer2(self) -> None:
        """respond() alongside a benign tool routes through guardrails (unchanged)."""
        client = _FakeClient([[
            ToolCall(tool="respond", args={"message": "done"}),
            ToolCall(tool="inspect", args={"target": "x"}),
        ]])
        guardrails = CodingGuardrails()
        body = {
            "model": "test-model",
            "stream": False,
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [RESPOND, _tool("inspect", {"target": {"type": "string"}})],
        }
        result = await handle_chat_completions(body, client, _ctx(), guardrails, max_retries=2)
        msg = result["choices"][0]["message"]
        names = [tc["function"]["name"] for tc in (msg.get("tool_calls") or [])]
        assert "inspect" in names  # benign tool passed guardrails
        assert "respond" not in names  # respond not emitted alongside (existing behavior)

    async def test_prose_then_respond_terminates(self) -> None:
        """Prose first try → terminal nudge → model calls respond() → passes through."""
        client = _FakeClient([
            TextResponse(content="Here is the answer."),
            [ToolCall(tool="respond", args={"message": "done"})],
        ])
        guardrails = CodingGuardrails()
        body = {
            "model": "test-model",
            "stream": False,
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [RESPOND],
        }
        result = await handle_chat_completions(body, client, _ctx(), guardrails, max_retries=2)
        msg = result["choices"][0]["message"]
        tc = (msg.get("tool_calls") or [None])[0]
        assert tc is not None and tc["function"]["name"] == "respond"


class TestTerminalEnforcement:
    """Respond-declared requests get terminal enforcement injected."""

    async def test_enforcement_injected_when_respond_declared(self) -> None:
        client = await _run(
            {"model": "test-model", "stream": False},
            [[ToolCall(tool="respond", args={"message": "done"})]],
            request_tools=[RESPOND],
        )
        text = _sent_text(client)
        assert "call the respond tool with your final answer" in text
        assert "plain text summary" not in text

    async def test_no_respond_enforcement_without_respond_tool(self) -> None:
        client = _FakeClient([[ToolCall(tool="inspect", args={"target": "x"})]])
        guardrails = CodingGuardrails()
        body = {
            "model": "test-model",
            "stream": False,
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [_tool("inspect", {"target": {"type": "string"}})],
        }
        await handle_chat_completions(body, client, _ctx(), guardrails, max_retries=2)
        text = _sent_text(client)
        # No respond tool declared → no respond() enforcement text
        assert "call the respond tool" not in text


class TestRetryNudges:
    """Retry nudges: terminal-aware vs plain-text, selected per request."""

    def test_terminal_nudge_names_the_respond_tool(self) -> None:
        nudge = _terminal_retry_nudge("anything")
        assert "call the respond tool with your final answer" in nudge
        assert "plain text" not in nudge

    def test_text_nudge_unchanged(self) -> None:
        nudge = _text_retry_nudge("anything")
        assert "plain text summary" in nudge

    async def test_terminal_nudge_selected_for_respond_requests(self) -> None:
        """A prose response in a respond-declared request gets the terminal nudge."""
        client = _FakeClient([
            TextResponse(content="first prose"),
            [ToolCall(tool="respond", args={"message": "done"})],
        ])
        guardrails = CodingGuardrails()
        body = {
            "model": "test-model",
            "stream": False,
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [RESPOND],
        }
        await handle_chat_completions(body, client, _ctx(), guardrails, max_retries=2)
        text = _sent_text(client)
        assert "call the respond tool with your final answer" in text
