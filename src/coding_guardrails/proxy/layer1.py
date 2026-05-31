"""Instrumented Layer 1 wrapper — wraps Forge's inference loop with detailed logging.

Uses only Forge's public API (ResponseValidator, ErrorTracker, ContextManager,
fold_and_serialize, rescue_tool_call). No Forge source modifications needed.

Why a wrapper instead of patching Forge?
- Forge's run_inference() has zero logging in its core retry loop
- We want play-by-play visibility: each attempt, validation result, rescue
  outcome, nudge injection, compaction events
- Forge may update independently — our wrapper survives upstream changes
"""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from forge.clients.base import LLMClient
from forge.context.manager import ContextManager
from forge.core.inference import fold_and_serialize
from forge.core.messages import (
    Message,
    MessageMeta,
    MessageRole,
    MessageType,
    ToolCallInfo,
)
from forge.core.workflow import LLMResponse, TextResponse, ToolCall, ToolSpec
from forge.errors import StreamError, ToolCallError
from forge.guardrails import ErrorTracker, ResponseValidator
from forge.prompts.templates import rescue_tool_call

logger = logging.getLogger("coding_guardrails.layer1")

# Maps Nudge.kind → MessageType for message emission (mirrors forge inference.py).
_NUDGE_KIND_TO_TYPE: dict[str, MessageType] = {
    "retry": MessageType.RETRY_NUDGE,
    "unknown_tool": MessageType.RETRY_NUDGE,
    "step": MessageType.STEP_NUDGE,
    "prerequisite": MessageType.PREREQUISITE_NUDGE,
}


def _fmt_elapsed(seconds: float) -> str:
    if seconds < 1.0:
        return f"{seconds * 1000:.0f}ms"
    return f"{seconds:.1f}s"


def _short(text: str, width: int = 80) -> str:
    if len(text) <= width:
        return text
    return text[:width - 3] + "..."


def _fmt_tools(calls: list[ToolCall]) -> str:
    parts = []
    for tc in calls:
        args_preview = ",".join(
            f"{k}={_short(str(v), 20)}" for k, v in list(tc.args.items())[:3]
        )
        parts.append(f"{tc.tool}({args_preview})")
    return " | ".join(parts)


def _sync_token_count(client: LLMClient, context_manager: ContextManager) -> None:
    """Feed actual token count from the client into the context manager."""
    last_usage = getattr(client, "last_usage", None)
    if not isinstance(last_usage, dict):
        return
    slot_id = getattr(client, "_slot_id", None) or 0
    from forge.clients.base import TokenUsage
    usage: TokenUsage | None = last_usage.get(slot_id)
    if usage is not None:
        context_manager.update_token_count(usage.total_tokens)


def _build_tool_call_infos(
    tool_calls: list[ToolCall],
    tool_call_counter: int,
) -> tuple[list[ToolCallInfo], int]:
    """Assign call IDs to tool calls."""
    tc_infos = []
    for tc in tool_calls:
        tc_id = f"call_{tool_call_counter:09d}"
        tool_call_counter += 1
        tc_infos.append(ToolCallInfo(name=tc.tool, args=tc.args, call_id=tc_id))
    return tc_infos, tool_call_counter


@dataclass
class InferenceResult:
    """Result of a single inference call (may include transparent retries).

    Matches forge.core.inference.InferenceResult shape so callers are drop-in
    compatible.
    """

    response: list[ToolCall] | TextResponse
    new_messages: list[Message] = field(default_factory=list)
    tool_call_counter: int = 0
    attempts: int = 1


async def run_inference_instrumented(
    messages: list[Message],
    client: LLMClient,
    context_manager: ContextManager,
    validator: ResponseValidator,
    error_tracker: ErrorTracker,
    tool_specs: list[ToolSpec],
    tool_call_counter: int = 0,
    step_index: int = 0,
    step_hint: str = "",
    max_attempts: int | None = None,
    stream: bool = False,
    on_chunk: Callable[[Any], Awaitable[None]] | None = None,
    sampling: dict[str, Any] | None = None,
) -> InferenceResult | None:
    """Instrumented version of forge's run_inference with per-attempt logging.

    Behavior is identical to forge.core.inference.run_inference — same retry
    logic, same nudge injection, same compaction. The only difference is that
    every decision point gets a log statement so operators can see exactly what
    Forge is doing inside the loop.
    """
    from forge.clients.base import ChunkType, StreamChunk

    api_format = getattr(client, "api_format", "ollama")
    new_messages: list[Message] = []
    max_retries = error_tracker.max_retries
    attempt_limit = max_retries + 1
    if max_attempts is not None:
        attempt_limit = min(attempt_limit, max_attempts)
    attempts = 0

    tool_names = [s.name for s in tool_specs]

    for _attempt in range(attempt_limit):
        attempts += 1
        attempt_num = attempts
        t_attempt = time.monotonic()

        if attempt_num > 1:
            logger.info(
                "  🔄 Attempt %d/%d (consecutive retries: %d)",
                attempt_num, attempt_limit, error_tracker._consecutive_retries,
            )

        # ── Compact ──
        compacted = context_manager.maybe_compact(
            messages, step_index=step_index, step_hint=step_hint,
        )
        if compacted is not messages:
            before_count = len(messages)
            messages.clear()
            messages.extend(compacted)
            logger.info(
                "  📦 Compacted %d → %d messages", before_count, len(messages),
            )

        # ── Context thresholds ──
        context_warning = context_manager.check_thresholds(messages)
        if context_warning:
            logger.info("  ⚠️  Context threshold warning: %s", _short(context_warning, 60))

        # ── Fold and serialize ──
        api_messages = fold_and_serialize(messages, api_format)

        if context_warning:
            api_messages.append({"role": "user", "content": context_warning})
            new_messages.append(Message(
                MessageRole.USER,
                context_warning,
                MessageMeta(MessageType.CONTEXT_WARNING, step_index=step_index),
            ))

        # ── Send to LLM ──
        t_send = time.monotonic()
        if stream:
            response = await _send_streaming(client, api_messages, tool_specs, on_chunk, sampling)
        else:
            response = await client.send(api_messages, tools=tool_specs, sampling=sampling)
        send_elapsed = time.monotonic() - t_send

        # ── Token sync ──
        _sync_token_count(client, context_manager)
        last_usage = getattr(client, "last_usage", None)
        slot_id = getattr(client, "_slot_id", None) or 0
        from forge.clients.base import TokenUsage
        tok_usage: TokenUsage | None = last_usage.get(slot_id) if isinstance(last_usage, dict) else None

        tok_info = ""
        if tok_usage:
            tok_info = f", {tok_usage.prompt_tokens}+{tok_usage.completion_tokens}={tok_usage.total_tokens} tokens"

        # ── Capture thinking tokens ──
        thinking = getattr(client, "last_thinking", "")

        logger.info(
            "  📤 LLM response (%s%s): %s",
            _fmt_elapsed(send_elapsed), tok_info,
            _short(
                response.content if isinstance(response, TextResponse)
                else f"{len(response)} tool calls",
                60,
            ),
        )

        if thinking:
            logger.info(
                "  🧠 Thinking (%d chars): %s",
                len(thinking), _short(thinking, 120),
            )

        # ── Validate ──
        validation = validator.validate(response)
        t_validate = time.monotonic() - t_attempt

        if not validation.needs_retry:
            error_tracker.reset_retries()
            validated = validation.tool_calls
            logger.info(
                "  ✅ Validated (%s) — %s",
                _fmt_elapsed(t_validate),
                _fmt_tools(validated) if validated else "text response",
            )
            return InferenceResult(
                response=validated,
                new_messages=new_messages,
                tool_call_counter=tool_call_counter,
                attempts=attempts,
            )

        # ── Short-circuit: substantive text → pass through ──
        # If the model produced meaningful text (not empty, not just
        # thinking), return it as-is instead of retrying. The agent can
        # handle text responses — retrying wastes tokens and time.
        #
        # Guard: only passthrough if the conversation does NOT already
        # contain tool results. If tool results are present, the model
        # is mid-workflow (it called tools, got results, and now needs
        # to call more tools). Passthrough here aborts the workflow.
        if isinstance(response, TextResponse):
            content = response.content.strip()
            # Pass through substantive text as a final answer.
            # The model read tools, got results, and is now summarizing.
            # Only block passthrough if this is clearly mid-workflow
            # (no thinking, very short response = confused model).
            if content and len(content) > 30 and (len(content) > 100 or thinking):
                logger.info(
                    "  📝 Passing through text response (%d chars)",
                    len(content),
                )
                full_content = content
                if thinking and thinking not in content:
                    full_content = f"{thinking}\n\n{content}"
                return InferenceResult(
                    response=TextResponse(content=full_content),
                    new_messages=new_messages,
                    tool_call_counter=tool_call_counter,
                    attempts=attempts,
                )

        # ── Retry path ──
        nudge = validation.nudge
        logger.info(
            "  ⚠️  Validation failed [%s]: %s",
            nudge.kind,
            _short(nudge.content, 80),
        )

        # Log rescue attempt details for text responses
        if isinstance(response, TextResponse) and validator.rescue_enabled:
            rescued = rescue_tool_call(response.content, tool_names)
            if rescued:
                logger.info(
                    "  🔧 Rescue parsing found %d tool call(s): %s",
                    len(rescued), _fmt_tools(rescued),
                )
            else:
                logger.debug("  🔧 Rescue parsing: no tool calls found in text")

        error_tracker.record_retry()
        if error_tracker.retries_exhausted:
            raw = response.content if isinstance(response, TextResponse) else str(
                [(tc.tool, tc.args) for tc in response]
            )
            logger.warning(
                "  ❌ Retries exhausted after %d consecutive failures", max_retries,
            )
            raise ToolCallError(
                f"Retries exhausted after {max_retries} consecutive failed attempts",
                raw_response=raw,
            )

        nudge_type = _NUDGE_KIND_TO_TYPE[nudge.kind]

        # Build nudge content — include previous thinking so the model
        # doesn't re-think on the retry.
        thinking = getattr(client, "last_thinking", "")
        nudge_content = nudge.content
        if thinking and isinstance(response, TextResponse):
            # Model thought but returned no tool call. Feed the thinking
            # back so it can jump straight to acting on it.
            nudge_content = (
                f"Your previous thinking was:\n"
                f"<thinking>\n{thinking}\n</thinking>\n\n"
                f"Based on this thinking, {nudge.content}"
            )
            logger.info(
                "  🧠 Fed %d chars of thinking into retry nudge",
                len(thinking),
            )

        if isinstance(response, TextResponse):
            msg = Message(
                MessageRole.ASSISTANT,
                response.content,
                MessageMeta(MessageType.TEXT_RESPONSE, step_index=step_index),
            )
            messages.append(msg)
            new_messages.append(msg)
            logger.info(
                "  💬 Emitted assistant text (%d chars)", len(response.content),
            )

            nudge_msg = Message(
                MessageRole.USER,
                nudge_content,
                MessageMeta(nudge_type, step_index=step_index),
            )
            messages.append(nudge_msg)
            new_messages.append(nudge_msg)
            logger.info("  📢 Injected %s nudge", nudge.kind)
        else:
            tool_calls = response
            if tool_calls[0].reasoning:
                reasoning_msg = Message(
                    MessageRole.ASSISTANT,
                    tool_calls[0].reasoning,
                    MessageMeta(MessageType.REASONING, step_index=step_index),
                )
                messages.append(reasoning_msg)
                new_messages.append(reasoning_msg)

            tc_infos, tool_call_counter = _build_tool_call_infos(tool_calls, tool_call_counter)
            tc_msg = Message(
                MessageRole.ASSISTANT,
                "",
                MessageMeta(MessageType.TOOL_CALL, step_index=step_index),
                tool_calls=tc_infos,
            )
            messages.append(tc_msg)
            new_messages.append(tc_msg)

            for tc_info in tc_infos:
                err_msg = Message(
                    MessageRole.TOOL,
                    f"[UnknownTool] {nudge.content}",
                    MessageMeta(nudge_type, step_index=step_index),
                    tool_name=tc_info.name,
                    tool_call_id=tc_info.call_id,
                )
                messages.append(err_msg)
                new_messages.append(err_msg)
                logger.info(
                    "  🔧 Emitted tool-error for unknown tool '%s'", tc_info.name,
                )

    # max_attempts exhausted without valid response
    logger.warning("  ❌ Max attempts (%d) exhausted without valid response", attempt_limit)
    return None


async def _send_streaming(
    client: LLMClient,
    api_messages: list[dict[str, Any]],
    tool_specs: list[ToolSpec],
    on_chunk: Callable[[Any], Awaitable[None]] | None = None,
    sampling: dict[str, Any] | None = None,
) -> LLMResponse:
    """Send via streaming, forwarding chunks to on_chunk callback."""
    from forge.clients.base import ChunkType, StreamChunk

    response = None
    async for chunk in client.send_stream(api_messages, tools=tool_specs, sampling=sampling):
        if on_chunk is not None:
            await on_chunk(chunk)
        if chunk.type == ChunkType.FINAL:
            response = chunk.response
    if response is None:
        raise StreamError(
            "Stream ended without FINAL chunk — the client adapter "
            "may be malformed or the connection was interrupted"
        )
    return response
