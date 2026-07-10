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
from forge.core.inference import fold_and_serialize
from coding_guardrails.proxy.layer1 import run_inference_instrumented as run_inference
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
from coding_guardrails.proxy.acceptance import wrap_bare_acceptance_report
from coding_guardrails.rules.base import ToolCall as GuardrailToolCall

logger = logging.getLogger("coding_guardrails.proxy")

# ── Banner helpers ──────────────────────────────────────────────────────────

_BANNER_WIDTH = 60


def _banner(label: str, char: str = "-") -> str:
    pad = _BANNER_WIDTH - len(label) - 4
    left = pad // 2
    right = pad - left
    return f"{char * left} >> {label} << {char * right}"


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


def _text_retry_nudge(raw_response: str) -> str:
    """Softer retry nudge that acknowledges the task might be done."""
    return (
        "Your previous response was text instead of a tool call. "
        "If the task is complete, respond with a plain text summary. "
        "Otherwise, make a tool call to continue working."
    )


async def handle_chat_completions(
    body: dict[str, Any],
    client: LLMClient,
    context_manager: ContextManager,
    guardrails: CodingGuardrails,
    max_retries: int = 3,
    rescue_enabled: bool = True,
    auto_no_thinking: bool = True,
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

    # Reset stateful rules on new conversations.
    # Detect new conversation: no assistant messages in history (no prior
    # tool calls or text responses). This handles /new, /resume to fresh
    # sessions, and first requests from eval runners. More robust than
    # checking message count, which varies with system prompt length.
    has_assistant = any(m.get("role") == "assistant" for m in openai_messages)
    if not has_assistant:
        if guardrails.loop_detection:
            guardrails.loop_detection.reset()
        if guardrails.thoroughness:
            guardrails.thoroughness.reset()

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

    # Inject tool-call enforcement for real coding agents.
    # Only when the request includes coding tools (bash, read, edit, write).
    _REAL_AGENT_TOOLS = {"bash", "read", "write", "edit"}
    tool_names_lower = set()
    if request_tools:
        for t in request_tools:
            fname = t.get("function", {}).get("name", "")
            if fname:
                tool_names_lower.add(fname.lower())

    if _REAL_AGENT_TOOLS & tool_names_lower:
        enforcement = (
            "When working on a task, respond by calling tools (bash, read, edit, write). "
            "When the task is COMPLETE and you have nothing left to do, respond with plain text summarizing what was done. "
            "Do NOT call unnecessary tools just to have a tool call. If unsure, call bash with 'echo ready'."
        )
        if openai_messages:
            first = openai_messages[0]
            if first.get("role") == "system":
                content = first.get("content", "")
                if enforcement not in content:
                    openai_messages[0] = {**first, "content": content + "\n\n" + enforcement}
            else:
                openai_messages.insert(0, {"role": "system", "content": enforcement})

    # Convert inbound
    messages = openai_to_messages(openai_messages)
    tool_specs = _extract_tool_specs(request_tools)

    # Note: we do NOT inject Forge's respond() tool. With local models like
    # Qwen3.5-9B, respond() becomes an escape hatch — the model calls respond()
    # instead of action tools (bash, read, edit), causing high retry rates.
    # Forge handles this gracefully: text responses pass through Layer 1 as-is.

    tool_names = [s.name for s in tool_specs]

    # No tools → plain chat completion (generation). Auto-disable thinking so
    # Qwen3.5 emits a clean direct answer instead of reasoning_content that eats
    # the token budget. Overridable per-request via chat_template_kwargs.enable_thinking.
    if not tool_specs:
        if auto_no_thinking:
            if sampling is None:
                sampling = {}
            sampling.setdefault("chat_template_kwargs", {}).setdefault("enable_thinking", False)
        logger.info("Plain text (no tools)%s", " [auto enable_thinking=false]" if auto_no_thinking else "")
        t0 = time.monotonic()
        api_format = getattr(client, "api_format", "ollama")
        api_messages = fold_and_serialize(messages, api_format)
        response = await client.send(api_messages, tools=None, sampling=sampling)
        elapsed = time.monotonic() - t0
        text = response.content if isinstance(response, TextResponse) else ""
        logger.info("Text response (%s, %d chars)", _fmt_elapsed(elapsed), len(text))
        if is_stream:
            return text_to_sse_events(text, model=model_name)
        return text_response_to_openai(text, model=model_name)

    # ── Layer 1: Forge (rescue, validate, retry) ──
    logger.info(_banner("LAYER 1 - Forge"))
    logger.info("Tools: %d, msgs: %d", len(tool_names), len(messages))
    t0 = time.monotonic()

    validator = ResponseValidator(
        tool_names,
        rescue_enabled=rescue_enabled,
        retry_nudge_fn=_text_retry_nudge,
    )
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
        logger.warning("Layer 1 failed after %d retries (%s)", max_retries, _short(raw, 80))
        raw = wrap_bare_acceptance_report(raw)
        if is_stream:
            return text_to_sse_events(raw, model=model_name)
        return text_response_to_openai(raw, model=model_name)

    elapsed_l1 = time.monotonic() - t0

    if result is None:
        logger.info("WARN: Model returned empty (attempts exhausted)")
        if is_stream:
            return text_to_sse_events("", model=model_name)
        return text_response_to_openai("", model=model_name)

    attempts = result.attempts
    response = result.response

    # If the model returned text (not tool calls), pass it through to the agent.
    if isinstance(response, TextResponse):
        text = wrap_bare_acceptance_report(response.content)
        logger.info("Model responded with text (%d chars, %s, %d attempt%s)",
                    len(text), _fmt_elapsed(elapsed_l1),
                    attempts, "s" if attempts != 1 else "")
        if is_stream:
            return text_to_sse_events(text, model=model_name)
        return text_response_to_openai(text, model=model_name)

    tool_calls = response

    # If all calls are respond(), pass them through as tool calls.
    # Forge's eval runner needs to see the terminal tool executed.
    respond_calls = [tc for tc in tool_calls if tc.tool == RESPOND_TOOL_NAME]
    other_calls = [tc for tc in tool_calls if tc.tool != RESPOND_TOOL_NAME]

    if respond_calls and not other_calls:
        # Convert respond() to text — most agents (Pi, Cline, etc.)
        # don't have a respond tool. The model is saying "I'm done."
        msg = respond_calls[0].args.get("message", respond_calls[0].args.get("answer", ""))
        attempts_tag = f"[%d attempt%s]" % (attempts, "s" if attempts != 1 else "") if attempts > 1 else ""
        logger.info("L1 done %s (%s, respond -> text: %s)",
                    attempts_tag, _fmt_elapsed(elapsed_l1), _short(msg, 60))
        if is_stream:
            return text_to_sse_events(msg, model=model_name)
        return text_response_to_openai(msg, model=model_name)

    if not other_calls:
        logger.info("WARN: No actionable tool calls")
        if is_stream:
            return text_to_sse_events("", model=model_name)
        return text_response_to_openai("", model=model_name)

    attempts_tag = f"[%d attempt%s]" % (attempts, "s" if attempts != 1 else "") if attempts > 1 else ""
    logger.info("L1 done %s (%s, %d tool calls: %s)",
                attempts_tag, _fmt_elapsed(elapsed_l1), len(other_calls), _fmt_tools(other_calls))

    # ── Layer 2: Coding guardrails ──
    logger.info(_banner("LAYER 2 - Guardrails"))
    t1 = time.monotonic()

    # Feed conversation context to thoroughness rule
    if guardrails.thoroughness:
        available = {t.get("function", {}).get("name", "") for t in (request_tools or [])}
        available.discard("")
        guardrails.thoroughness.set_context(openai_messages, available)

    guardrail_calls = [_forge_call_to_guardrail_call(tc) for tc in other_calls]
    guardrail_result = guardrails.check(guardrail_calls)

    # Record executed calls (for stateful rules like prerequisites)
    if guardrail_result.allowed:
        guardrails.record(guardrail_result.allowed)

    elapsed_l2 = time.monotonic() - t1

    # If any call was hard-blocked, return block responses
    if guardrail_result.has_blocks:
        logger.info("BLOCKED (%s)", _fmt_elapsed(elapsed_l2))
        block = guardrail_result.blocked[0]
        nudge_text = block.nudge or "Action blocked by guardrails."

        if guardrail_result.has_nudges:
            extra = " ".join(n.nudge for n in guardrail_result.nudges if n.nudge)
            if extra:
                nudge_text = f"{nudge_text} {extra}"

        if is_stream:
            return text_to_sse_events(nudge_text, model=model_name)
        # Return as text so the agent sees the nudge directly.
        # Empty-args tool-call blocks confuse agents into retrying.
        return text_response_to_openai(nudge_text, model=model_name)

    # All clear
    logger.info("PASSED (%s)", _fmt_elapsed(elapsed_l2))

    if is_stream:
        return tool_calls_to_sse_events(other_calls, model=model_name)
    return tool_calls_to_openai(other_calls, model=model_name)
