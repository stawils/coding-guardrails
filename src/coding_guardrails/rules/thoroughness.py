"""Thoroughness rule — nudge when the model submits before exploring tools.

When a model calls a terminal tool (submit, report, respond) after using
only a few of the available tools, this rule fires an advisory nudge
suggesting it consider whether it has gathered all necessary information.

This is a general-purpose guardrail — no hardcoded tool names or value
patterns. It simply compares tools used vs tools available and nudges
when exploration is low.

Common failure modes this addresses:
- Submitting reports with unresolved placeholder values
- Flagging items without investigating aliases/edge cases
- Returning partial answers when more data was available
"""

from __future__ import annotations

from dataclasses import dataclass, field

from coding_guardrails.rules.base import RuleResult, ToolCall


# Tool names that indicate a final submission
_TERMINAL_PATTERNS = (
    "submit", "report", "respond", "answer", "present",
    "summarize", "diagnose", "recommend", "complete",
)


def _is_terminal(name: str) -> bool:
    return any(p in name.lower() for p in _TERMINAL_PATTERNS)


def _tools_used_in_messages(messages: list[dict]) -> set[str]:
    """Find unique tool names from assistant tool_calls in conversation."""
    used: set[str] = set()
    for m in messages:
        if m.get("role") == "assistant":
            for tc in m.get("tool_calls", []):
                name = tc.get("function", {}).get("name", "")
                if name:
                    used.add(name)
    return used


@dataclass
class ThoroughnessRule:
    """Nudge when terminal calls are made with low tool exploration.

    The handler calls set_context() before each check() to provide
    the conversation messages and available tool names.

    Attributes:
        min_tools: Minimum unique non-terminal tools expected before
            submitting without a nudge.
        cooldown: Max nudges per conversation before going silent.
    """

    min_tools: int = 3
    min_ratio: float = 0.4
    cooldown: int = 2
    _fire_count: int = 0
    _messages: list[dict] = field(default_factory=list, repr=False)
    _available_tools: set[str] = field(default_factory=set, repr=False)

    @property
    def name(self) -> str:
        return "thoroughness"

    def set_context(self, messages: list[dict], available_tools: set[str]) -> None:
        """Feed conversation state before checking.

        Called by the handler before each check() batch.
        """
        self._messages = messages
        self._available_tools = available_tools

    def check(self, call: ToolCall) -> RuleResult:
        if not _is_terminal(call.tool):
            return RuleResult.allow(call.tool)

        if self._fire_count >= self.cooldown:
            return RuleResult.allow(call.tool)

        if not self._messages or not self._available_tools:
            return RuleResult.allow(call.tool)

        tools_used = _tools_used_in_messages(self._messages)
        non_terminal_used = {t for t in tools_used if not _is_terminal(t)}
        non_terminal_available = {t for t in self._available_tools if not _is_terminal(t)}

        # Don't nudge if there are very few tools available
        if len(non_terminal_available) <= self.min_tools:
            return RuleResult.allow(call.tool)

        ratio_used = len(non_terminal_used) / len(non_terminal_available)
        if len(non_terminal_used) < self.min_tools or ratio_used < self.min_ratio:
            unused = sorted(non_terminal_available - non_terminal_used)[:5]
            self._fire_count += 1
            return RuleResult.nudge(
                call.tool,
                message=(
                    f"You've only used {len(non_terminal_used)} of "
                    f"{len(non_terminal_available)} available tools. "
                    f"Consider gathering more information before submitting. "
                    f"Tools not yet tried: {', '.join(unused)}."
                ),
            )

        return RuleResult.allow(call.tool)

    def record(self, calls: list[ToolCall]) -> None:
        pass

    def reset(self) -> None:
        """Reset for a new conversation."""
        self._fire_count = 0
