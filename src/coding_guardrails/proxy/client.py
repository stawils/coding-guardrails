"""Extended Forge client that preserves thinking tokens.

Two extensions over Forge's LlamafileClient:
1. max_tokens / n_predict forwarding to prevent runaway generation
2. Thinking token capture — when the model thinks but returns no tool
   call, Forge strips the thinking and returns empty TextResponse.
   This client preserves thinking as TextResponse content so Layer 1
   can log it and decide what to do (retry with context, feed to agent).

Does NOT modify Forge source. Overrides only the two send paths that
strip thinking (_send_native, _send_prompt) to preserve reasoning.
"""

from __future__ import annotations

import json
from typing import Any

from forge.clients.llamafile import (
    LlamafileClient,
    _extract_think_tags,
    _merge_consecutive,
    _downgrade_messages,
    extract_tool_call,
    format_tool,
    build_tool_prompt,
)
from forge.core.workflow import LLMResponse, TextResponse, ToolCall, ToolSpec
from forge.errors import BackendError


class SafeLlamafileClient(LlamafileClient):
    """LlamafileClient that forwards max_tokens and preserves thinking."""

    _EXTRA_SAMPLING_FIELDS = ("max_tokens", "n_predict")

    def __init__(self, *args: Any, default_max_tokens: int = 8192, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._default_max_tokens = default_max_tokens
        # Thinking tokens from the most recent response.
        # Populated regardless of whether the response was tool calls or text.
        self.last_thinking: str = ""

    def _apply_sampling(
        self, body: dict[str, Any], sampling: dict[str, Any] | None = None,
    ) -> None:
        super()._apply_sampling(body, sampling)

        for field in self._EXTRA_SAMPLING_FIELDS:
            override = (sampling or {}).get(field)
            if override is not None:
                body[field] = override
                return

        body.setdefault("max_tokens", self._default_max_tokens)

    # ── Send overrides that preserve thinking ──────────────────────────

    async def send(
        self,
        messages: list[dict[str, str]],
        tools: list[ToolSpec] | None = None,
        sampling: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Send and capture thinking tokens."""
        self.last_thinking = ""

        # Resolve mode on first call (same as parent)
        if self.resolved_mode is None:
            return await self._resolve_and_send(messages, tools, sampling)
        elif self.resolved_mode == "native":
            return await self._send_native(messages, tools, sampling)
        else:
            return await self._send_prompt(messages, tools, sampling)

    async def _send_native(
        self,
        messages: list[dict[str, str]],
        tools: list[ToolSpec] | None,
        sampling: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Native FC send that preserves reasoning in empty-text responses."""
        merged = _merge_consecutive(messages)
        body: dict[str, Any] = {
            "model": self.model,
            "messages": merged,
            "cache_prompt": self._cache_prompt,
        }
        self._apply_slot_id(body)
        self._apply_sampling(body, sampling)
        if tools:
            body["tools"] = [format_tool(t) for t in tools]

        resp = await self._http.post(
            f"{self.base_url}/chat/completions", json=body
        )
        if resp.status_code == 500:
            return TextResponse(content=resp.text)
        if resp.status_code != 200:
            raise BackendError(resp.status_code, resp.text)
        data = resp.json()
        self._record_usage(data)

        top_choice = data["choices"][0]
        choice = top_choice["message"]
        raw_reasoning = choice.get("reasoning_content", "") or ""
        raw_content = choice.get("content", "") or ""

        # Store thinking for Layer 1 to read
        resolved_reasoning = self._resolve_reasoning(raw_reasoning, raw_content)
        if resolved_reasoning:
            self.last_thinking = resolved_reasoning

        raw_tool_calls = choice.get("tool_calls")
        if raw_tool_calls:
            result_calls: list[ToolCall] = []
            for i, tc_entry in enumerate(raw_tool_calls):
                tc_func = tc_entry["function"]
                args = tc_func.get("arguments", "{}")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        return TextResponse(content=choice.get("content", args))
                result_calls.append(ToolCall(
                    tool=tc_func["name"],
                    args=args,
                    reasoning=resolved_reasoning if i == 0 else None,
                ))
            return result_calls

        # TextResponse — if content is empty but we have reasoning,
        # preserve it so Layer 1 sees non-empty text and can log/think.
        _, cleaned = _extract_think_tags(raw_content)
        if not cleaned.strip() and raw_reasoning:
            return TextResponse(content=raw_reasoning)
        return TextResponse(content=cleaned)

    async def _send_prompt(
        self,
        messages: list[dict[str, str]],
        tools: list[ToolSpec] | None,
        sampling: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Prompt-injected send that preserves reasoning in empty-text responses."""
        prepared = _merge_consecutive(_downgrade_messages(messages))
        if tools:
            tool_prompt = build_tool_prompt(tools)
            prepared[0] = {
                **prepared[0],
                "content": tool_prompt + "\n\n" + prepared[0]["content"],
            }

        body: dict[str, Any] = {
            "model": self.model,
            "messages": prepared,
            "cache_prompt": self._cache_prompt,
        }
        self._apply_slot_id(body)
        self._apply_sampling(body, sampling)

        resp = await self._http.post(
            f"{self.base_url}/chat/completions", json=body
        )
        resp.raise_for_status()
        data = resp.json()
        self._record_usage(data)

        top_choice = data["choices"][0]
        content = top_choice["message"].get("content", "")
        reasoning_content = top_choice["message"].get("reasoning_content", "")

        if tools:
            think_text, cleaned = _extract_think_tags(content)
            tool_names = [t.name for t in tools]
            tc_list = extract_tool_call(cleaned, tool_names)

            # Store thinking
            resolved = self._resolve_reasoning(reasoning_content, think_text)
            if resolved:
                self.last_thinking = resolved

            if tc_list:
                tc_list[0].reasoning = resolved
                return tc_list

            # No tool call found — preserve thinking if content is empty
            if not cleaned.strip() and (reasoning_content or think_text):
                preserved = reasoning_content or think_text
                return TextResponse(content=preserved)

        # Strip think tags from TextResponse — keep clean content
        if content:
            _, content = _extract_think_tags(content)
        return TextResponse(content=content)
