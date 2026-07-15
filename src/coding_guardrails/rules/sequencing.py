"""Workflow sequencing — soft nudges for test-after-change.

Suggests running tests after code edits. Soft by default (nudge, not block).
Uses prefix matching so it works with any agent's tool naming convention.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from coding_guardrails.rules.base import RuleResult, ToolCall
from coding_guardrails.rules.prerequisites import _tool_matches

# Default trigger/suggest prefixes covering all major agents.
_DEFAULT_EDIT_PREFIXES = ("edit", "write", "create")
_DEFAULT_SUGGEST_PREFIXES = ("bash", "shell", "run", "exec", "uv")


@dataclass
class SequenceRule:
    """Suggest workflow steps after certain tool calls.

    Uses prefix matching: trigger_prefixes="edit" matches 'edit',
    'edit_file', 'Edit', etc.

    Attributes:
        trigger_prefixes: Tool name prefixes that trigger the suggestion.
        suggest_prefixes: Tool name prefixes that satisfy the suggestion.
        strength: "soft" (nudge) or "hard" (block until done).
        nudge: Nudge message shown to the agent.
        cooldown: Minimum number of calls between repeated nudges.
    """

    trigger_prefixes: tuple[str, ...] = _DEFAULT_EDIT_PREFIXES
    suggest_prefixes: tuple[str, ...] = _DEFAULT_SUGGEST_PREFIXES
    strength: str = "soft"
    nudge: str = "Consider running tests to verify your changes."
    cooldown: int = 3

    _calls_since_nudge: int = field(default=0, repr=False)
    _pending: bool = field(default=False, repr=False)

    @property
    def has_pending(self) -> bool:
        """Whether a pending test nudge is active."""
        return self._pending

    @property
    def name(self) -> str:
        return "sequencing"

    def check(self, call: ToolCall) -> RuleResult:
        # Trigger: agent just edited/wrote a file
        if _tool_matches(call.tool, self.trigger_prefixes):
            self._pending = True
            self._calls_since_nudge = 0

            if self.strength == "hard":
                return RuleResult.block(
                    call.tool,
                    nudge=self.nudge,
                    reason=f"hard sequence: {call.tool} -> test",
                )

            return RuleResult.allow(call.tool)

        # Satisfaction: agent is running a command (might be tests)
        if self._pending and _tool_matches(call.tool, self.suggest_prefixes):
            self._pending = False
            self._calls_since_nudge = 0
            return RuleResult.allow(call.tool)

        # Cooldown nudge: agent hasn't run tests after edits
        if self._pending:
            self._calls_since_nudge += 1
            if self._calls_since_nudge >= self.cooldown:
                self._calls_since_nudge = 0
                return RuleResult.nudge(
                    call.tool,
                    message=self.nudge,
                )

        return RuleResult.allow(call.tool)

    def record(self, calls: list[ToolCall]) -> None:
        """Track if suggested follow-up was executed."""
        for call in calls:
            if self._pending and _tool_matches(call.tool, self.suggest_prefixes):
                self._pending = False
                self._calls_since_nudge = 0
