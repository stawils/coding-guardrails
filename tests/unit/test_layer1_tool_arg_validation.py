"""Layer 1 retry-path tests for forge 0.9.x nudge kinds.

Found 2026-08-31 via cg-worker assessment: forge 0.9.x emits
nudge.kind='tool_arg_validation' when a model emits a tool call with
non-dict arguments (the 0.7.4 malformed-args tool-error channel). Our
layer1 wrapper's _NUDGE_KIND_TO_TYPE mapping lacked the kind → KeyError
crashed the whole request, and it charged the retry budget instead of the
tool-error budget. This file pins both regressions.
"""

from __future__ import annotations

import pytest

from forge.context.manager import ContextManager
from forge.context.strategies import TieredCompact
from forge.core.messages import Message, MessageMeta, MessageRole, MessageType
from forge.core.workflow import ToolCall
from forge.errors import ToolCallError
from forge.guardrails import ErrorTracker, ResponseValidator

from coding_guardrails.proxy.layer1 import run_inference_instrumented

TOOLS = ["bash"]


class _SeqClient:
    """Returns canned LLMResponses in order, then repeats the last."""

    api_format = "openai"

    def __init__(self, responses: list) -> None:
        self._responses = list(responses)
        self._last = None

    async def send(self, messages, tools=None, sampling=None):
        if self._responses:
            self._last = self._responses.pop(0)
        return self._last


def _ctx() -> ContextManager:
    return ContextManager(strategy=TieredCompact(), budget_tokens=32000)


def _msgs() -> list[Message]:
    return [Message(MessageRole.USER, "hi", MessageMeta(MessageType.USER_INPUT))]


async def _run(client, *, max_attempts: int = 4, max_retries: int = 3,
               max_tool_errors: int = 2):
    validator = ResponseValidator(TOOLS, rescue_enabled=True)
    tracker = ErrorTracker(max_retries=max_retries, max_tool_errors=max_tool_errors)
    return await run_inference_instrumented(
        _msgs(), client, _ctx(), validator, tracker, [],
        max_attempts=max_attempts,
    )


class TestToolArgValidation:
    async def test_malformed_args_do_not_crash_and_emit_tool_error(self) -> None:
        """One malformed-args call then a valid call: no KeyError, recovers,
        and the corrective signal rides the role=tool channel."""
        client = _SeqClient([
            [ToolCall(tool="bash", args="not-a-dict")],     # malformed → nudge
            [ToolCall(tool="bash", args={"command": "ls"})],  # valid → success
        ])
        result = await _run(client)
        assert result is not None
        assert result.response[0].args == {"command": "ls"}
        tool_msgs = [m.content for m in result.new_messages if m.role == MessageRole.TOOL]
        assert any("ToolArgValidationError" in t for t in tool_msgs), tool_msgs

    async def test_persistent_malformed_args_exhaust_tool_error_budget(self) -> None:
        """Repeated malformed args exhaust max_tool_errors (not max_retries)
        and raise ToolCallError naming the tool-error budget."""
        client = _SeqClient([[ToolCall(tool="bash", args="bad")]])
        with pytest.raises(ToolCallError) as exc_info:
            await _run(client, max_tool_errors=1)
        assert "max_tool_errors=1" in str(exc_info.value)
        assert "tool_arg_validation" in str(exc_info.value)

    async def test_unknown_tool_still_uses_unknown_tool_prefix(self) -> None:
        """Unknown-tool nudges keep the [UnknownTool] tool-error prefix."""
        client = _SeqClient([
            [ToolCall(tool="nope", args={"a": 1})],
            [ToolCall(tool="bash", args={"command": "ls"})],
        ])
        result = await _run(client)
        assert result is not None
        tool_msgs = [m.content for m in result.new_messages if m.role == MessageRole.TOOL]
        assert any("UnknownTool" in t for t in tool_msgs), tool_msgs