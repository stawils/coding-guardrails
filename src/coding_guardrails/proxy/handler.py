"""Request handler — bridges HTTP, Forge Layer 1, and our Layer 2 guardrails.

Chain of responsibility:
1. Agent → OpenAI request → our proxy
2. Layer 1 (Forge): run_inference → rescue, validate, retry
3. Layer 2 (our rules): check tool calls against coding guardrails
4. If blocked → return tool-error response
5. If allowed → return OpenAI-compatible response to agent
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from forge.clients.base import LLMClient
from forge.context.manager import ContextManager
from forge.core.inference import fold_and_serialize, run_inference
from forge.core.workflow import ToolCall, ToolSpec, TextResponse
from forge.errors import ToolCallError
from forge.guardrails import ErrorTracker, ResponseValidator
from forge.proxy.convert import (
    openai_to_messages,
    tool_calls_to_openai,
    tool_calls_to_sse_events,
    text_response_to_openai,
    text_to_sse_events,
)
from forge.tools.respond import RESPOND_TOOL_NAME, respond_spec

from coding_guardrails.middleware import CodingGuardrails
from coding_guardrails.rules.base import ToolCall as GuardrailToolCall

logger = logging.getLogger("coding_guardrails.proxy")

# ── Banner helpers ──────────────────────────────────────────────────────────

_BANNER_WIDTH = 60


def _banner(label: str, char: str = "─") -> str:
    pad = _BANNER_WIDTH - len(label) - 4
    left = pad // 2
    right = pad - left
    return f"{char * left} ▸ {label} ◂ {char * right}"


def _short(msg: str, width: int = 80) -> str:
    if len(msg) <= width:
        return msg
    return msg[:width - 3] + "..."


def _fmt_tools(calls: list[ToolCall]) -> str:
    parts = [f"{tc.tool}({','.join(f'{k}={_short(str(v),20)}' for k, v in list(tc.args.items())[:3])})" for tc in calls]
    return " | ".join(parts)


def _fmt_elapsed(seconds: float) -> str:
    if seconds < 1.0:
        return f"{seconds * 1000:.0f}ms"
    return f"{seconds:.1f}s"

# OpenAI-compatible top-level body fields plumbed from inbound to client.
# Note: max_tokens / n_predict are handled by SafeLlamafileClient, not here.
_SAMPLING_FIELDS = (
    "temperature", "top_p", "top_k", "min_p",
    "repeat_penalty", "presence_penalty", "seed",
    "chat_template_kwargs",
)


def _extract_sampling(body: dict[str, Any]) -> dict[str, Any] | None:
    """Pull recognized sampling fields from the inbound request body."""
    extracted = {f: body[f] for f in _SAMPLING_FIELDS if f in body}
    # Also forward max_tokens variants — SafeLlamafileClient handles them
    for field in ("max_tokens", "max_completion_tokens", "n_predict"):
        if field in body:
            extracted[field] = body[field]
    # Normalize: max_completion_tokens → max_tokens
    if "max_completion_tokens" in extracted and "max_tokens" not in extracted:
        extracted["max_tokens"] = extracted.pop("max_completion_tokens")
    else:
        extracted.pop("max_completion_tokens", None)
    return extracted or None


def _extract_tool_specs(request_tools: list[dict[str, Any]] | None) -> list[ToolSpec]:
    """Extract ToolSpec objects from OpenAI tools array."""
    if not request_tools:
        return []
    specs = []
    for tool in request_tools:
        if tool.get("type") != "function":
            continue
        func = tool.get("function", {})
        specs.append(ToolSpec.from_json_schema(
            name=func.get("name", ""),
            description=func.get("description", ""),
            schema=func.get("parameters", {}),
        ))
    return specs


def _forge_call_to_guardrail_call(tc: ToolCall) -> GuardrailToolCall:
    """Convert a Forge ToolCall to our guardrail ToolCall."""
    return GuardrailToolCall(tool=tc.tool, args=dict(tc.args))


def _make_block_response(
    blocked_tool: str,
    nudge: str,
    model: str = "coding-guardrails",
) -> dict[str, Any]:
    """Create an OpenAI tool-error response for a blocked call.

    Returns the nudge as a tool result with an error indicator,
    so the agent sees it as a failed tool call and can adapt.
    """
    import uuid
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "model": model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {
                        "name": blocked_tool,
                        "arguments": json.dumps({}),
                    },
                }],
            },
            "finish_reason": "tool_calls",
        }],
        "guardrail": {
            "blocked": True,
            "nudge": nudge,
        },
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


async def handle_chat_completions(
    body: dict[str, Any],
    client: LLMClient,
    context_manager: ContextManager,
    guardrails: CodingGuardrails,
    max_retries: int = 3,
    rescue_enabled: bool = True,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Handle /v1/chat/completions with Forge Layer 1 + our Layer 2.

    1. Forge validates, rescues, retries (Layer 1)
    2. Our guardrails check tool calls (Layer 2)
    3. Return OpenAI-compatible response
    """
    openai_messages = body.get("messages", [])
    request_tools = body.get("tools")
    is_stream = body.get("stream", False)
    model_name = body.get("model", "coding-guardrails")
    sampling = _extract_sampling(body)

    # Preprocess messages to fix patterns that confuse local models.
    # Pi sends empty user messages ("\n") as "continue" signals and includes
    # assistant text responses in history. Both cause Qwen3.5-9B to return
    # text instead of tool calls. Fix: replace empty users, strip assistant text.
    if request_tools:
        cleaned = []
        for m in openai_messages:
            role = m.get("role", "")
            content = m.get("content", "")
            tc = m.get("tool_calls")

            # Replace empty/trivial user messages
            if role == "user" and (not content or (isinstance(content, str) and content.strip() == "")):
                m = {**m, "content": "Continue working."}

            # Remove assistant text-only responses (no tool_calls)
            # These teach the model to respond with text
            if role == "assistant" and not tc and isinstance(content, str) and content.strip():
                continue

            cleaned.append(m)
        openai_messages = cleaned

    # Inject tool-call enforcement into system prompt.
    _TOOL_ENFORCEMENT = (
        "CRITICAL: You MUST always respond by calling bash, read, edit, or write. "
        "Never respond with plain text. "
        "If unsure what to do, call bash with 'echo ready'."
    )
    if openai_messages and request_tools:
        first = openai_messages[0]
        if first.get("role") == "system":
            content = first.get("content", "")
            if _TOOL_ENFORCEMENT not in content:
                openai_messages[0] = {**first, "content": content + "\n\n" + _TOOL_ENFORCEMENT}
        else:
            openai_messages.insert(0, {"role": "system", "content": _TOOL_ENFORCEMENT})

    # Convert inbound
    messages = openai_to_messages(openai_messages)
    tool_specs = _extract_tool_specs(request_tools)

    # Note: we do NOT inject Forge's respond() tool. With local models like
    # Qwen3.5-9B, respond() becomes an escape hatch — the model calls respond()
    # instead of action tools (bash, read, edit), causing high retry rates.
    # Forge handles this gracefully: text responses pass through Layer 1 as-is.

    tool_names = [s.name for s in tool_specs]

    # No tools → plain chat completion, pass through
    if not tool_specs:
        logger.info("💬 Plain text (no tools)")
        t0 = time.monotonic()
        api_format = getattr(client, "api_format", "ollama")
        api_messages = fold_and_serialize(messages, api_format)
        response = await client.send(api_messages, tools=None, sampling=sampling)
        elapsed = time.monotonic() - t0
        text = response.content if isinstance(response, TextResponse) else ""
        logger.info("✅ Text response (%s, %d chars)", _fmt_elapsed(elapsed), len(text))
        if is_stream:
            return text_to_sse_events(text, model=model_name)
        return text_response_to_openai(text, model=model_name)

    # ── Layer 1: Forge (rescue, validate, retry) ──
    logger.info(_banner("LAYER 1 · Forge"))
    logger.info("🔧 %d tools, %d msgs", len(tool_names), len(messages))
    t0 = time.monotonic()

    validator = ResponseValidator(tool_names, rescue_enabled=rescue_enabled)
    error_tracker = ErrorTracker(max_retries=max_retries)

    try:
        result = await run_inference(
            messages=messages,
            client=client,
            context_manager=context_manager,
            validator=validator,
            error_tracker=error_tracker,
            tool_specs=tool_specs,
            sampling=sampling,
        )
    except ToolCallError as exc:
        raw = exc.raw_response or ""
        logger.warning("❌ Layer 1 failed after %d retries (%s)", max_retries, _short(raw, 80))
        if is_stream:
            return text_to_sse_events(raw, model=model_name)
        return text_response_to_openai(raw, model=model_name)

    elapsed_l1 = time.monotonic() - t0

    if result is None:
        logger.info("⚠️  Model returned empty")
        if is_stream:
            return text_to_sse_events("", model=model_name)
        return text_response_to_openai("", model=model_name)

    # Log Layer 1 activity from result metadata
    attempts = result.attempts
    new_msgs = result.new_messages
    if attempts > 1 or new_msgs:
        logger.info("  🔄 %d attempt%s, %d retry msgs",
                    attempts, "s" if attempts != 1 else "", len(new_msgs))
        for nm in new_msgs:
            mt = nm.metadata.type.value if hasattr(nm.metadata.type, 'value') else str(nm.metadata.type)
            logger.info("     ↳ %s: %s", mt, _short(nm.content, 60))

    tool_calls = result.response

    # Strip respond() calls
    respond_calls = [tc for tc in tool_calls if tc.tool == RESPOND_TOOL_NAME]
    other_calls = [tc for tc in tool_calls if tc.tool != RESPOND_TOOL_NAME]

    if respond_calls and not other_calls:
        text = respond_calls[0].args.get("message", "")
        logger.info("📝 Model responded with text (%s)", _fmt_elapsed(elapsed_l1))
        if is_stream:
            return text_to_sse_events(text, model=model_name)
        return text_response_to_openai(text, model=model_name)

    if not other_calls:
        logger.info("⚠️  No actionable tool calls")
        if is_stream:
            return text_to_sse_events("", model=model_name)
        return text_response_to_openai("", model=model_name)

    attempts_tag = f"[%d attempt%s]" % (attempts, "s" if attempts != 1 else "") if attempts > 1 else ""
    logger.info("✅ Layer 1 done %s (%s, %d tool calls: %s)",
                attempts_tag, _fmt_elapsed(elapsed_l1), len(other_calls), _fmt_tools(other_calls))

    # ── Layer 2: Coding guardrails ──
    logger.info(_banner("LAYER 2 · Guardrails"))
    t1 = time.monotonic()

    guardrail_calls = [_forge_call_to_guardrail_call(tc) for tc in other_calls]
    guardrail_result = guardrails.check(guardrail_calls)

    # Record executed calls (for stateful rules like prerequisites)
    if guardrail_result.allowed:
        guardrails.record(guardrail_result.allowed)
        for call in guardrail_result.allowed:
            logger.info("  ✅ %s — allowed", call.tool)

    # Log blocks
    if guardrail_result.has_blocks:
        for block in guardrail_result.blocked:
            logger.info("  🚫 %s — BLOCKED [%s]", block.tool, block.reason or "policy violation")
            logger.info("     ↳ %s", _short(block.nudge or "", 60))

    # Log nudges
    if guardrail_result.has_nudges:
        for nudge in guardrail_result.nudges:
            logger.info("  ⚠️  %s — nudged [%s]", nudge.tool, nudge.reason or "advisory")
            logger.info("     ↳ %s", _short(nudge.nudge or "", 60))

    elapsed_l2 = time.monotonic() - t1

    # If any call was hard-blocked, return block responses
    if guardrail_result.has_blocks:
        logger.info("⛔ Request BLOCKED by Layer 2 (%s)", _fmt_elapsed(elapsed_l2))
        # Return the first block as the response with all nudges appended
        block = guardrail_result.blocked[0]
        nudge_text = block.nudge or "Action blocked by guardrails."

        # Append any additional nudges
        if guardrail_result.has_nudges:
            extra = " ".join(n.nudge for n in guardrail_result.nudges if n.nudge)
            if extra:
                nudge_text = f"{nudge_text} {extra}"

        if is_stream:
            return text_to_sse_events(nudge_text, model=model_name)
        # Return a block response — the agent sees this as guidance
        return _make_block_response(block.tool, nudge_text, model=model_name)

    # All clear
    logger.info("✅ Request PASSED (%s)", _fmt_elapsed(elapsed_l2))

    if is_stream:
        return tool_calls_to_sse_events(other_calls, model=model_name)
    return tool_calls_to_openai(other_calls, model=model_name)
