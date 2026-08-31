"""Extended Forge client: outbound role normalization + thinking retention.

Three extensions over Forge's LlamafileClient, all as thin wrappers around the
parent's send paths (no Forge source re-implementation):
1. Outbound role normalization — hoist system/developer messages to the front
   so the qwen3_5 jinja template's "System message must be at the beginning"
   guard never fires (pi project-instructions arrive as trailing system msgs).
2. Acceptance-finalization prefill (F9 fix) — append a JSON prefill as a
   trailing assistant message so pi acceptance reports come back structured.
3. Thinking retention — captured reasoning (ToolCall.reasoning) is surfaced as
   ``client.last_thinking`` for Layer 1's logging + retry-nudge injection.

Thin wrappers mean Forge's own improvements (malformed-500 tool-call rescue,
credential forwarding, envelope guards, argument decoding) are inherited on
upgrade instead of being shadowed by a copy of stale internals.
"""

from __future__ import annotations

import logging
from typing import Any

from forge.clients.llamafile import LlamafileClient
from forge.core.workflow import LLMResponse, ToolSpec
from forge.errors import BackendError

from coding_guardrails.proxy.handler import _normalize_message_roles


class SafeLlamafileClient(LlamafileClient):
    """LlamafileClient that forwards max_tokens and preserves thinking."""

    _EXTRA_SAMPLING_FIELDS = ("max_tokens", "n_predict")

    def __init__(self, *args: Any, default_max_tokens: int = 8192, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._default_max_tokens = default_max_tokens
        # Ensure resolved_mode exists even if parent init didn't set it
        if not hasattr(self, "resolved_mode"):
            self.resolved_mode = None
        # Thinking tokens from the most recent response.
        # Populated regardless of whether the response was tool calls or text.
        self.last_thinking: str = ""

    # ── Acceptance finalization prefill (F9 fix) ───────────────────────────────
    # When pi-subagents runs the acceptance-finalization turn, the model often
    # does correct work but refuses to emit the structured acceptance-report
    # JSON (it narrates in prose instead). We detect the finalization turn by
    # its stable marker and APPEND a trailing assistant message containing the
    # opening of the report JSON. llama-server treats a trailing assistant
    # message as a prefix to continue, so the model is forced to complete the
    # JSON object — it cannot switch back to prose mid-object.
    #
    # This is format priming, not content fabrication: the model still
    # generates every field value (ids, evidence, summaries) itself.
    _ACCEPTANCE_MARKER = "Acceptance Finalization"
    _ACCEPTANCE_PREFILL = '{"criteriaSatisfied": [{"id": "'

    def _resolve_acceptance_prefill(self, user_texts: list[str]) -> str:
        """Pick a prefill that seeds the contract's first criterion id.

        The finalization nudge's *example* block uses a generic 'criterion-1'
        id, which the model copies verbatim into its report. Pi then rejects
        with 'Required criterion <id> was not reported' because the model's
        id never matches the contract's id. The contract criteria are listed
        in the nudge as markdown 'Criteria:\n- <id>: <must>'; seeding that id
        makes the model's criteriaSatisfied entry line up with the contract.

        Seeds only the id label; the model still generates status and
        evidence itself (no result fabrication). Falls back to the generic
        prefill when no criterion id can be parsed.
        """
        import re
        for text in user_texts:
            crit_idx = text.find("Criteria:")
            if crit_idx < 0:
                continue
            m = re.search(r"(?m)^[ \t]*-[ \t]+([A-Za-z0-9][A-Za-z0-9_-]*)[ \t]*:", text[crit_idx:])
            if m:
                return '{"criteriaSatisfied": [{"id": "%s", "status": "' % m.group(1)
        return self._ACCEPTANCE_PREFILL

    def _inject_acceptance_prefill(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        """If the conversation contains an acceptance-finalization prompt AND
        the trailing turn is the model's chance to respond, append a JSON
        prefill so the model continues in report-JSON mode.

        Returns the (possibly extended) messages list. Detection scans ALL
        user messages (Forge may interleave tool results or trailing system
        turns after the finalization prompt). The prefill is only valid when
        the last message is a user/tool turn (the model is about to speak);
        a trailing assistant turn means Forge is mid-exchange and priming
        would corrupt it.
        """
        if not messages:
            return messages

        def _extract_text(content: Any) -> str:
            """Flatten OpenAI content (str OR list-of-parts) to plain text."""
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = []
                for p in content:
                    if isinstance(p, dict) and p.get("type") == "text" and isinstance(p.get("text"), str):
                        parts.append(p["text"])
                return "\n".join(parts)
            return ""

        try:
            user_texts = [
                _extract_text(m.get("content"))
                for m in messages
                if isinstance(m, dict) and m.get("role") == "user"
            ]
        except Exception:
            return messages
        is_finalization = any(self._ACCEPTANCE_MARKER in t for t in user_texts)
        if not is_finalization:
            return messages
        # Only prime when the model is about to generate. If the last message
        # is already an assistant turn, the request is a tool-result follow-up
        # or a Forge retry — a prefill here would be concatenated wrongly.
        last = messages[-1]
        if isinstance(last, dict) and last.get("role") == "assistant":
            return messages
        result = list(messages)
        prefill = self._resolve_acceptance_prefill(user_texts)
        result.append({"role": "assistant", "content": prefill})
        logging.getLogger("coding_guardrails.client").info(
            "acceptance-finalization prefill injected (msgs=%d, last_role=%s)",
            len(messages), last.get("role") if isinstance(last, dict) else "?",
        )
        return result

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
        passthrough: dict[str, Any] | None = None,
        inbound_anthropic_body: dict[str, Any] | None = None,
        raw_openai_tools: Any = None,
        extra_headers: dict[str, str] | None = None,
    ) -> LLMResponse:
        """Dispatch through Forge's send path, then retain captured reasoning."""
        self.last_thinking = ""

        result = await super().send(
            messages, tools=tools, sampling=sampling, passthrough=passthrough,
            inbound_anthropic_body=inbound_anthropic_body,
            raw_openai_tools=raw_openai_tools, extra_headers=extra_headers,
        )
        # Retain thinking for Layer 1 (logging + retry-nudge injection).
        if isinstance(result, list) and result and getattr(result[0], "reasoning", None):
            self.last_thinking = result[0].reasoning
        return result

    async def _send_native(
        self,
        messages: list[dict[str, str]],
        tools: list[ToolSpec] | None,
        sampling: dict[str, Any] | None = None,
        passthrough: dict[str, Any] | None = None,
        raw_openai_tools: Any = None,
        extra_headers: dict[str, str] | None = None,
    ) -> LLMResponse:
        """Thin wrapper: role-normalize + acceptance-prefill, then Forge's native path.

        Forge 0.8.1+ owns the malformed-tool-call 500 rescue, argument decoding
        and envelope guards — delegate instead of re-implementing (see module
        docstring). Only the outbound role invariant and the acceptance prefill
        are applied here; everything else is inherited.
        """
        prepared = self._inject_acceptance_prefill(_normalize_message_roles(messages))
        try:
            return await super()._send_native(
                prepared, tools=tools, sampling=sampling, passthrough=passthrough,
                raw_openai_tools=raw_openai_tools, extra_headers=extra_headers,
            )
        except BackendError as exc:
            logging.getLogger("coding_guardrails.client").warning(
                "outbound roles=%s msgs=%d (resp %s)",
                [m.get("role") for m in prepared], len(prepared), exc.status_code,
            )
            raise

    async def _send_prompt(
        self,
        messages: list[dict[str, str]],
        tools: list[ToolSpec] | None,
        sampling: dict[str, Any] | None = None,
        passthrough: dict[str, Any] | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> LLMResponse:
        """Thin wrapper: role-normalize + acceptance-prefill, then Forge's prompt path."""
        prepared = self._inject_acceptance_prefill(_normalize_message_roles(messages))
        try:
            return await super()._send_prompt(
                prepared, tools=tools, sampling=sampling, passthrough=passthrough,
                extra_headers=extra_headers,
            )
        except BackendError as exc:
            logging.getLogger("coding_guardrails.client").warning(
                "outbound roles=%s msgs=%d (resp %s)",
                [m.get("role") for m in prepared], len(prepared), exc.status_code,
            )
            raise
