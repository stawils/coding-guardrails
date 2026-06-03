"""Tests for text passthrough logic in layer1.py.

Tests the logic at lines 238-270 of src/coding_guardrails/proxy/layer1.py
that determines when a TextResponse should be passed through instead of retried.

Passthrough conditions (from layer1.py):
- If isinstance(response, TextResponse):
    content = response.content.strip()
    has_tool_history = any(
        m.metadata and m.metadata.type in (
            MessageType.TOOL_CALL,
            MessageType.TOOL_RESULT,
        )
        for m in messages
    )
    first_attempt_text_ok = attempts == 1 and not has_tool_history
    retry_text_ok = attempts > 1

    if content and len(content) > 30 and (len(content) > 100 or thinking) and (first_attempt_text_ok or retry_text_ok):
        # Pass through
"""

import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from forge.core.messages import Message, MessageMeta, MessageRole, MessageType
from forge.core.workflow import TextResponse, ToolCall
from forge.guardrails import ResponseValidator
from forge.context.manager import ContextManager
from forge.clients.base import LLMClient


class MockValidator:
    """Mock ResponseValidator that always returns a nudge for retry."""

    def __init__(self):
        self.rescue_enabled = False

    def validate(self, response):
        """Return a validation that needs retry."""
        class MockValidation:
            needs_retry = True
            nudge = MagicMock()
            nudge.kind = "unknown_tool"
            nudge.content = "Please call tools instead"
            tool_calls = None
            rescue_enabled = False
        return MockValidation()


class MockContextManager:
    """Mock ContextManager."""

    def maybe_compact(self, messages, step_index=0, step_hint=""):
        """Return messages unchanged."""
        return messages

    def check_thresholds(self, messages):
        """Return empty string - no warning."""
        return ""


def _make_tool_call(tool_name, args, reasoning=None):
    """Helper to create a ToolCall with name property."""
    tc = MagicMock(spec=ToolCall)
    tc.tool = tool_name
    tc.args = args
    tc.reasoning = reasoning
    tc.name = tool_name
    return tc


def _make_message(role, content, metadata=None, tool_calls=None, tool_name=None, tool_call_id=None):
    """Helper to create a Message."""
    meta_dict = {}
    if metadata:
        meta_dict["type"] = metadata.get("type", MessageType.USER_INPUT)
        meta_dict["step_index"] = metadata.get("step_index")
    else:
        meta_dict["type"] = MessageType.USER_INPUT
    return Message(
        role=role,
        content=content,
        metadata=MessageMeta(**meta_dict),
        tool_calls=tool_calls,
        tool_name=tool_name,
        tool_call_id=tool_call_id,
    )


@pytest.fixture
def mock_client():
    """Create a mock LLMClient."""
    client = MagicMock(spec=LLMClient)
    client.last_thinking = ""
    client.api_format = "ollama"
    client.last_usage = {}
    return client


@pytest.fixture
def mock_context_manager():
    """Create a mock ContextManager."""
    return MockContextManager()


@pytest.fixture
def mock_validator():
    """Create a mock ResponseValidator."""
    return MockValidator()


@pytest.fixture
def mock_error_tracker():
    """Create a mock ErrorTracker."""
    tracker = MagicMock()
    tracker.max_retries = 3
    tracker._consecutive_retries = 0
    tracker.retries_exhausted = False
    tracker.record_retry = MagicMock()
    return tracker


@pytest.fixture
def mock_tool_specs():
    """Create empty tool specs."""
    return []





@pytest.mark.asyncio
async def test_first_attempt_no_history_passes(
    mock_client, mock_context_manager, mock_validator, mock_error_tracker,
    mock_tool_specs, caplog
):
    """
    Test 1: First attempt, no tool history, substantive text → passthrough.

    Conditions:
    - attempt=1
    - no TOOL_CALL or TOOL_RESULT messages in history
    - text > 30 chars, > 100 chars OR has thinking
    """
    mock_client.last_thinking = ""

    messages = [
        _make_message(MessageRole.USER, "Describe the function"),
    ]

    substantive_text = "This is a detailed explanation of what the function does and how it works in the system and its purpose in the workflow."

    async def mock_send(*args, **kwargs):
        return TextResponse(content=substantive_text)

    with patch.object(mock_client, "send", mock_send):
        from coding_guardrails.proxy.layer1 import run_inference_instrumented

        result = await run_inference_instrumented(
            messages=messages,
            client=mock_client,
            context_manager=mock_context_manager,
            validator=mock_validator,
            error_tracker=mock_error_tracker,
            tool_specs=mock_tool_specs,
            tool_call_counter=0,
            step_index=0,
        )

        assert result is not None
        assert isinstance(result.response, TextResponse)
        assert len(result.response.content) > 30
        assert result.attempts == 1


@pytest.mark.asyncio
async def test_first_attempt_short_text_no_passthrough(
    mock_client, mock_context_manager, mock_validator, mock_error_tracker,
    mock_tool_specs, caplog
):
    """
    Test 2: First attempt, no history, short text (<30 chars) → no passthrough.

    Conditions:
    - attempt=1
    - no TOOL_CALL or TOOL_RESULT messages in history
    - text < 30 chars
    """
    mock_client.last_thinking = ""

    messages = [
        _make_message(MessageRole.USER, "What?"),
    ]

    short_text = "What?"

    async def mock_send(*args, **kwargs):
        return TextResponse(content=short_text)

    with patch.object(mock_client, "send", mock_send):
        from coding_guardrails.proxy.layer1 import run_inference_instrumented

        result = await run_inference_instrumented(
            messages=messages,
            client=mock_client,
            context_manager=mock_context_manager,
            validator=mock_validator,
            error_tracker=mock_error_tracker,
            tool_specs=mock_tool_specs,
            tool_call_counter=0,
            step_index=0,
        )

        # Should return None (retry path) because text is too short
        assert result is None


@pytest.mark.asyncio
async def test_first_attempt_with_tool_history_blocked(
    mock_client, mock_context_manager, mock_validator, mock_error_tracker,
    mock_tool_specs, caplog
):
    """
    Test 3: First attempt, has tool history, substantive text → passthrough.

    Conditions:
    - attempt=1
    - messages include TOOL_CALL or TOOL_RESULT in metadata
    - substantive text > 100 chars to ensure passthrough

    This test verifies the fix for the infinite loop bug where text was
    blocked on first attempt even after the model finished a task.
    """
    mock_client.last_thinking = ""

    messages = [
        _make_message(
            MessageRole.USER,
            "Calculate 2+2",
            metadata={"type": MessageType.USER_INPUT},
        ),
        _make_message(
            MessageRole.ASSISTANT,
            "",
            metadata={"type": MessageType.TOOL_CALL},
            tool_calls=[_make_tool_call("calculator", {"operation": "add", "a": 2, "b": 2})],
        ),
        _make_message(
            MessageRole.TOOL,
            "Result: 4",
            metadata={"type": MessageType.TOOL_RESULT},
            tool_name="calculator",
            tool_call_id="call_000000001",
        ),
        _make_message(
            MessageRole.ASSISTANT,
            "",
            metadata={"type": MessageType.TOOL_CALL},
            tool_calls=[_make_tool_call("calculator", {"operation": "add", "a": 2, "b": 2})],
        ),
    ]

    # Text > 100 chars to ensure passthrough
    substantive_text = "The result is 4. The calculation was performed successfully using the calculator tool." * 3

    async def mock_send(*args, **kwargs):
        return TextResponse(content=substantive_text)

    with patch.object(mock_client, "send", mock_send):
        from coding_guardrails.proxy.layer1 import run_inference_instrumented

        result = await run_inference_instrumented(
            messages=messages,
            client=mock_client,
            context_manager=mock_context_manager,
            validator=mock_validator,
            error_tracker=mock_error_tracker,
            tool_specs=mock_tool_specs,
            tool_call_counter=0,
            step_index=0,
        )

        # Should return passthrough result even with tool history
        assert result is not None
        assert isinstance(result.response, TextResponse)
        assert len(result.response.content) > 100
        assert result.attempts == 1


@pytest.mark.asyncio
async def test_first_attempt_with_tool_call_in_history(
    mock_client, mock_context_manager, mock_validator, mock_error_tracker,
    mock_tool_specs, caplog
):
    """
    Test 4: First attempt, one message with TOOL_CALL metadata → passthrough.

    Conditions:
    - attempt=1
    - at least one message has metadata.type=TOOL_CALL
    - text > 100 chars to ensure passthrough

    This test verifies the fix for the infinite loop bug where text was
    blocked on first attempt even after the model finished a task.
    """
    mock_client.last_thinking = ""

    messages = [
        _make_message(
            MessageRole.ASSISTANT,
            "",
            metadata={"type": MessageType.TOOL_CALL},
            tool_calls=[_make_tool_call("read_file", {"path": "test.py"})],
        ),
    ]

    # Text > 100 chars to ensure passthrough
    async def mock_send(*args, **kwargs):
        return TextResponse(content="File read successfully. The test file contains the expected test cases and assertions. All tests passed. The implementation is correct and follows best practices. The code is clean and maintainable." * 2)

    with patch.object(mock_client, "send", mock_send):
        from coding_guardrails.proxy.layer1 import run_inference_instrumented

        result = await run_inference_instrumented(
            messages=messages,
            client=mock_client,
            context_manager=mock_context_manager,
            validator=mock_validator,
            error_tracker=mock_error_tracker,
            tool_specs=mock_tool_specs,
            tool_call_counter=0,
            step_index=0,
        )

        # Should return passthrough result even with tool call in history
        assert result is not None
        assert isinstance(result.response, TextResponse)
        assert len(result.response.content) > 100
        assert result.attempts == 1


@pytest.mark.asyncio
async def test_attempt_2_with_history_passes(
    mock_client, mock_context_manager, mock_validator, mock_error_tracker,
    mock_tool_specs, caplog
):
    """
    Test 5: Attempt 2, has tool history, substantive text → passthrough.

    Conditions:
    - attempt=2 (or >1)
    - has tool history
    - substantive text
    """
    mock_client.last_thinking = ""

    messages = [
        _make_message(
            MessageRole.USER,
            "Calculate 2+2",
            metadata={"type": MessageType.USER_INPUT},
        ),
    ]

    substantive_text = "After being nudged by the system multiple times, the model now provides the final result and detailed explanation."

    call_count = 0

    async def mock_send_wrapper(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        # First call returns empty text to trigger retry
        # Second call returns substantive text
        if call_count == 1:
            return TextResponse(content="")
        return TextResponse(content=substantive_text)

    with patch.object(mock_client, "send", mock_send_wrapper):
        from coding_guardrails.proxy.layer1 import run_inference_instrumented

        # First call will fail and retry
        result = await run_inference_instrumented(
            messages=messages,
            client=mock_client,
            context_manager=mock_context_manager,
            validator=mock_validator,
            error_tracker=mock_error_tracker,
            tool_specs=mock_tool_specs,
            tool_call_counter=0,
            step_index=0,
        )

        # Should return passthrough result
        assert result is not None
        assert isinstance(result.response, TextResponse)
        assert len(result.response.content) > 30
        assert result.attempts > 1


@pytest.mark.asyncio
async def test_attempt_3_passes(
    mock_client, mock_context_manager, mock_validator, mock_error_tracker,
    mock_tool_specs, caplog
):
    """
    Test 6: Attempt 3, has tool history → passthrough.

    Conditions:
    - attempt=3 (or >1)
    - has tool history
    - substantive text
    """
    mock_client.last_thinking = ""

    messages = [
        _make_message(MessageRole.USER, "Do something"),
    ]

    substantive_text = "This is the final attempt after multiple retries and nudges. The model now provides a complete answer and explanation."

    call_count = 0

    async def mock_send_wrapper(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        # First call returns empty text to trigger retry
        # Second call returns substantive text
        if call_count == 1:
            return TextResponse(content="")
        return TextResponse(content=substantive_text)

    with patch.object(mock_client, "send", mock_send_wrapper):
        from coding_guardrails.proxy.layer1 import run_inference_instrumented

        result = await run_inference_instrumented(
            messages=messages,
            client=mock_client,
            context_manager=mock_context_manager,
            validator=mock_validator,
            error_tracker=mock_error_tracker,
            tool_specs=mock_tool_specs,
            tool_call_counter=0,
            step_index=0,
        )

        # Should return passthrough result
        assert result is not None
        assert isinstance(result.response, TextResponse)
        assert result.attempts > 1


@pytest.mark.asyncio
async def test_empty_text_no_passthrough(
    mock_client, mock_context_manager, mock_validator, mock_error_tracker,
    mock_tool_specs, caplog
):
    """
    Test 7: Empty/whitespace text → no passthrough.

    Conditions:
    - attempt=1
    - no tool history
    - empty or whitespace-only text
    """
    mock_client.last_thinking = ""

    messages = [
        _make_message(MessageRole.USER, "Hello"),
    ]

    empty_text = ""

    async def mock_send(*args, **kwargs):
        return TextResponse(content=empty_text)

    with patch.object(mock_client, "send", mock_send):
        from coding_guardrails.proxy.layer1 import run_inference_instrumented

        result = await run_inference_instrumented(
            messages=messages,
            client=mock_client,
            context_manager=mock_context_manager,
            validator=mock_validator,
            error_tracker=mock_error_tracker,
            tool_specs=mock_tool_specs,
            tool_call_counter=0,
            step_index=0,
        )

        # Should return None (retry path)
        assert result is None


@pytest.mark.asyncio
async def test_short_text_no_passthrough(
    mock_client, mock_context_manager, mock_validator, mock_error_tracker,
    mock_tool_specs, caplog
):
    """
    Test 8: Text between 31-99 chars, no thinking → no passthrough.

    Conditions:
    - attempt=1
    - no tool history
    - text > 30 chars but <= 99 chars, no thinking
    """
    mock_client.last_thinking = ""

    messages = [
        _make_message(MessageRole.USER, "Explain this"),
    ]

    # Text between 31-99 chars
    short_text = "This is a moderately long explanation that is not quite long enough."

    async def mock_send(*args, **kwargs):
        return TextResponse(content=short_text)

    with patch.object(mock_client, "send", mock_send):
        from coding_guardrails.proxy.layer1 import run_inference_instrumented

        result = await run_inference_instrumented(
            messages=messages,
            client=mock_client,
            context_manager=mock_context_manager,
            validator=mock_validator,
            error_tracker=mock_error_tracker,
            tool_specs=mock_tool_specs,
            tool_call_counter=0,
            step_index=0,
        )

        # Should return None (retry path) because text is too short
        assert result is None


@pytest.mark.asyncio
async def test_long_text_passthrough(
    mock_client, mock_context_manager, mock_validator, mock_error_tracker,
    mock_tool_specs, caplog
):
    """
    Test 9: Text >100 chars → passthrough (if other conditions met).

    Conditions:
    - attempt=1
    - no tool history
    - text > 100 chars
    """
    mock_client.last_thinking = ""

    messages = [
        _make_message(MessageRole.USER, "Explain this concept in detail"),
    ]

    # Text > 100 chars
    long_text = "This is a very long explanation that exceeds one hundred characters. " \
               "It continues with more text to ensure we have enough length. " \
               "This should definitely trigger the passthrough condition."

    async def mock_send(*args, **kwargs):
        return TextResponse(content=long_text)

    with patch.object(mock_client, "send", mock_send):
        from coding_guardrails.proxy.layer1 import run_inference_instrumented

        result = await run_inference_instrumented(
            messages=messages,
            client=mock_client,
            context_manager=mock_context_manager,
            validator=mock_validator,
            error_tracker=mock_error_tracker,
            tool_specs=mock_tool_specs,
            tool_call_counter=0,
            step_index=0,
        )

        # Should return passthrough result
        assert result is not None
        assert isinstance(result.response, TextResponse)
        assert len(result.response.content) > 100
        assert result.attempts == 1


@pytest.mark.asyncio
async def test_short_text_with_thinking_passthrough(
    mock_client, mock_context_manager, mock_validator, mock_error_tracker,
    mock_tool_specs, caplog
):
    """
    Test 10: Text >30 chars with thinking tokens → passthrough.

    Conditions:
    - attempt=1
    - no tool history
    - text > 30 chars AND has thinking
    """
    mock_client.last_thinking = "Let me think about this carefully. The answer is yes."

    messages = [
        _make_message(MessageRole.USER, "Yes or no?"),
    ]

    # Text > 30 chars
    short_text = "Yes, I agree with your statement."

    async def mock_send(*args, **kwargs):
        return TextResponse(content=short_text)

    with patch.object(mock_client, "send", mock_send):
        from coding_guardrails.proxy.layer1 import run_inference_instrumented

        result = await run_inference_instrumented(
            messages=messages,
            client=mock_client,
            context_manager=mock_context_manager,
            validator=mock_validator,
            error_tracker=mock_error_tracker,
            tool_specs=mock_tool_specs,
            tool_call_counter=0,
            step_index=0,
        )

        # Should return passthrough result because of thinking
        assert result is not None
        assert isinstance(result.response, TextResponse)
        assert result.attempts == 1
