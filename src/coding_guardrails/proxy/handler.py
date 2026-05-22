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

# OpenAI-compatible top-level body fields plumbed from inbound to client.
_SAMPLING_FIELDS = (
    "temperature", "top_p", "top_k", "min_p",
    "repeat_penalty", "presence_penalty", "seed",
    "chat_template_kwargs",
)


def _extract_sampling(body: dict[str, Any]) -> dict[str, Any] | None:
    """Pull recognized sampling fields from the inbound request body."""
    extracted = {f: body[f] for f in _SAMPLING_FIELDS if f in body}
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

    # Convert inbound
    messages = openai_to_messages(openai_messages)
    tool_specs = _extract_tool_specs(request_tools)

    # Inject respond tool
    if tool_specs and not any(s.name == RESPOND_TOOL_NAME for s in tool_specs):
        tool_specs.append(respond_spec())

    tool_names = [s.name for s in tool_specs]

    # No tools → plain chat completion, pass through
    if not tool_specs:
        logger.info("No tools, passing through to backend")
        api_format = getattr(client, "api_format", "ollama")
        api_messages = fold_and_serialize(messages, api_format)
        response = await client.send(api_messages, tools=None, sampling=sampling)
        text = response.content if isinstance(response, TextResponse) else ""
        if is_stream:
            return text_to_sse_events(text, model=model_name)
        return text_response_to_openai(text, model=model_name)

    # ── Layer 1: Forge guardrails (rescue, validate, retry) ──
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
        logger.warning("Layer 1 retries exhausted: %.120s", raw)
        if is_stream:
            return text_to_sse_events(raw, model=model_name)
        return text_response_to_openai(raw, model=model_name)

    if result is None:
        if is_stream:
            return text_to_sse_events("", model=model_name)
        return text_response_to_openai("", model=model_name)

    tool_calls = result.response

    # Strip respond() calls
    respond_calls = [tc for tc in tool_calls if tc.tool == RESPOND_TOOL_NAME]
    other_calls = [tc for tc in tool_calls if tc.tool != RESPOND_TOOL_NAME]

    if respond_calls and not other_calls:
        text = respond_calls[0].args.get("message", "")
        logger.info("Stripping respond(), returning as text")
        if is_stream:
            return text_to_sse_events(text, model=model_name)
        return text_response_to_openai(text, model=model_name)

    if not other_calls:
        if is_stream:
            return text_to_sse_events("", model=model_name)
        return text_response_to_openai("", model=model_name)

    # ── Layer 2: Coding guardrails ──
    guardrail_calls = [_forge_call_to_guardrail_call(tc) for tc in other_calls]
    guardrail_result = guardrails.check(guardrail_calls)

    # Record executed calls (for stateful rules like prerequisites)
    if guardrail_result.allowed:
        guardrails.record(guardrail_result.allowed)

    # Log what happened
    if guardrail_result.has_blocks:
        for block in guardrail_result.blocked:
            logger.info(
                "LAYER 2 BLOCK: tool=%s reason=%s",
                block.tool, block.reason or block.nudge,
            )
    if guardrail_result.has_nudges:
        for nudge in guardrail_result.nudges:
            logger.info("LAYER 2 NUDGE: tool=%s", nudge.tool)

    # If any call was hard-blocked, return block responses
    if guardrail_result.has_blocks:
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

    # All clear — return validated tool calls
    # If there are nudges, log them (agent doesn't see them unless we inject)
    # For now, nudges are advisory — we could inject them as system messages
    # in a future iteration. For v0.1, they're logged only.

    if is_stream:
        return tool_calls_to_sse_events(other_calls, model=model_name)
    return tool_calls_to_openai(other_calls, model=model_name)
