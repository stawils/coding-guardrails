"""Integration tests for the full proxy pipeline.

Tests the end-to-end request flow:
1. OpenAI-compatible request → Layer 1 (Forge) → Layer 2 (guardrails) → response
2. Mock the LLM backend at the http level
3. Test tool calls, text responses, and blocked commands
"""

from __future__ import annotations

import json

import pytest
from unittest.mock import AsyncMock

from forge.core.workflow import ToolCall, TextResponse

from coding_guardrails.middleware import CodingGuardrails
from coding_guardrails.proxy.handler import handle_chat_completions


# ─────────────────────────────────────────────────────────────────────────────
# Test fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_context_manager():
    """Create a mock ContextManager."""
    from unittest.mock import MagicMock
    return MagicMock()


@pytest.fixture
def mock_guardrails():
    """Create a default CodingGuardrails instance."""
    return CodingGuardrails.defaults()


@pytest.fixture
def mock_llm_client():
    """Create a mock LLMClient that simulates LLM responses."""
    from unittest.mock import MagicMock
    client = MagicMock()
    client.api_format = "ollama"
    client.last_thinking = ""
    client.last_usage = {}
    client.base_url = "http://localhost:8080"
    return client


# ─────────────────────────────────────────────────────────────────────────────
# Test: Full pipeline — tool call passes through
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_pipeline_tool_call_passthrough(
    mock_llm_client,
    mock_context_manager,
    mock_guardrails,
):
    """Test that a tool call from LLM passes through Layer 1 and Layer 2."""
    # Mock LLM response: returns a list with a tool call
    mock_llm_client.send = AsyncMock(return_value=[
        ToolCall(
            tool="bash",
            args={"command": "ls -la"},
        ),
    ])

    body = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "List directory contents"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "bash",
                    "description": "Execute a shell command",
                    "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
                },
            },
        ],
    }

    result = await handle_chat_completions(
        body=body,
        client=mock_llm_client,
        context_manager=mock_context_manager,
        guardrails=mock_guardrails,
    )

    # Verify the tool call was returned in OpenAI format
    assert result is not None
    assert "choices" in result
    assert len(result["choices"]) == 1
    assert result["choices"][0]["message"]["tool_calls"]
    assert result["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "bash"
    assert json.loads(result["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])["command"] == "ls -la"


# ─────────────────────────────────────────────────────────────────────────────
# Test: Full pipeline — text response (attempt 1, no tool history)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_pipeline_text_response_attempt1_no_history(
    mock_llm_client,
    mock_context_manager,
    mock_guardrails,
):
    """Test that a text response on attempt 1 with no tool history passes through."""
    # Mock LLM response: returns a text response
    mock_llm_client.send = AsyncMock(return_value=TextResponse(
        content="Task is complete. Here's the summary.",
    ))

    body = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "What is 2+2?"}],
    }

    result = await handle_chat_completions(
        body=body,
        client=mock_llm_client,
        context_manager=mock_context_manager,
        guardrails=mock_guardrails,
    )

    # Verify text response was returned in OpenAI format
    assert result is not None
    assert result["choices"][0]["message"]["content"] == "Task is complete. Here's the summary."


# ─────────────────────────────────────────────────────────────────────────────
# Test: Full pipeline — text response (attempt 1, with tool history)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_pipeline_text_response_attempt1_with_history(
    mock_llm_client,
    mock_context_manager,
    mock_guardrails,
):
    """Test that a text response on attempt 1 with tool history passes through.

    This is the fix we just made: if the model returns text on attempt 1
    but there's tool history in the conversation, we should pass it through
    instead of retrying.
    """
    # Mock LLM response: returns a text response
    mock_llm_client.send = AsyncMock(return_value=TextResponse(
        content="Task is complete. I've finished the work.",
    ))

    # Messages include assistant tool calls in history
    body = {
        "model": "test-model",
        "messages": [
            {"role": "user", "content": "Do something"},
            {"role": "assistant", "tool_calls": [
                {"id": "call_1", "type": "function", "function": {
                    "name": "bash",
                    "arguments": json.dumps({"command": "ls"})
                }},
            ]},
            {"role": "assistant", "content": "Task is complete. I've finished the work."},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "bash",
                    "description": "Execute a shell command",
                    "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
                },
            },
        ],
    }

    result = await handle_chat_completions(
        body=body,
        client=mock_llm_client,
        context_manager=mock_context_manager,
        guardrails=mock_guardrails,
    )

    # Verify text response was returned (passthrough)
    assert result is not None
    assert result["choices"][0]["message"]["content"] == "Task is complete. I've finished the work."


# ─────────────────────────────────────────────────────────────────────────────
# Test: Full pipeline — destructive command blocked by Layer 2
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_pipeline_destructive_command_blocked(
    mock_llm_client,
    mock_context_manager,
    mock_guardrails,
):
    """Test that a destructive command (rm -rf /) is blocked by Layer 2."""
    # Mock LLM response: returns a destructive tool call
    mock_llm_client.send = AsyncMock(return_value=[
        ToolCall(
            tool="bash",
            args={"command": "rm -rf /"},
        ),
    ])

    body = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Delete everything"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "bash",
                    "description": "Execute a shell command",
                    "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
                },
            },
        ],
    }

    result = await handle_chat_completions(
        body=body,
        client=mock_llm_client,
        context_manager=mock_context_manager,
        guardrails=mock_guardrails,
    )

    # Verify a text response was returned (blocked calls return text, not tool_calls)
    assert result is not None
    message_content = result["choices"][0]["message"]["content"]
    assert "blocked" in message_content.lower() or "safety" in message_content.lower()


# ─────────────────────────────────────────────────────────────────────────────
# Test: Full pipeline — path traversal blocked by Layer 2
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_pipeline_path_traversal_blocked(
    mock_llm_client,
    mock_context_manager,
    mock_guardrails,
):
    """Test that a path traversal attempt is blocked by Layer 2."""
    # Mock LLM response: returns a file read with path traversal
    mock_llm_client.send = AsyncMock(return_value=[
        ToolCall(
            tool="read",
            args={"path": "../../../etc/passwd"},
        ),
    ])

    body = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Read /etc/passwd"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "read",
                    "description": "Read a file",
                    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
                },
            },
        ],
    }

    result = await handle_chat_completions(
        body=body,
        client=mock_llm_client,
        context_manager=mock_context_manager,
        guardrails=mock_guardrails,
    )

    # Verify a text response was returned (blocked calls return text, not tool_calls)
    assert result is not None
    message_content = result["choices"][0]["message"]["content"]
    assert "outside" in message_content.lower() or "workspace" in message_content.lower() or "blocked" in message_content.lower()


# ─────────────────────────────────────────────────────────────────────────────
# Test: Layer 2 rule composition — check all rules
# ─────────────────────────────────────────────────────────────────────────────

def test_layer2_check_composition(mock_guardrails):
    """Test that Layer 2 properly composes and checks all rules."""
    # Create a valid tool call
    call = ToolCall(
        tool="bash",
        args={"command": "echo hello"},
    )

    result = mock_guardrails.check([call])

    # Should be allowed
    assert len(result.allowed) == 1
    assert len(result.blocked) == 0
    assert len(result.nudges) == 0


# ─────────────────────────────────────────────────────────────────────────────
# Test: Server integration — plain chat passthrough
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_pipeline_no_tools_plain_chat(
    mock_llm_client,
    mock_context_manager,
    mock_guardrails,
):
    """Test that a request without tools is passed through directly."""
    # Mock LLM response: returns plain text
    mock_llm_client.send = AsyncMock(return_value=TextResponse(
        content="This is a plain text response.",
    ))

    body = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello"}],
    }

    result = await handle_chat_completions(
        body=body,
        client=mock_llm_client,
        context_manager=mock_context_manager,
        guardrails=mock_guardrails,
    )

    # Verify text response was returned
    assert result is not None
    assert result["choices"][0]["message"]["content"] == "This is a plain text response."
