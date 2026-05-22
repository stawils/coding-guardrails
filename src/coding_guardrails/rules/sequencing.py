"""Workflow sequencing — soft nudges for test-after-change.

Suggests running tests after code edits. Soft by default (nudge, not block).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from coding_guardrails.rules.base import Action, RuleResult, ToolCall


@dataclass
class SequenceRule:
    """Suggest workflow steps after certain tool calls.

    When the agent uses a trigger tool (e.g. edit_file), suggest
    a follow-up tool (e.g. run_test) with configurable strength.

    Attributes:
        rules: List of (trigger, suggest, strength, nudge) dicts.
            trigger: Tool name that triggers the suggestion.
            suggest: Suggested follow-up tool name.
            strength: "soft" (nudge) or "hard" (block until done).
            nudge: Custom nudge message.
        cooldown: Minimum number of calls between repeated nudges.
    """

    rules: list[dict] = field(default_factory=lambda: [
        {
            "trigger": "edit_file",
            "suggest": "bash",
            "strength": "soft",
            "nudge": "Consider running tests to verify your changes.",
        },
        {
            "trigger": "write_file",
            "suggest": "bash",
            "strength": "soft",
            "nudge": "Consider running tests to verify the new file.",
        },
    ])
    cooldown: int = 3

    _DEFAULTS_RULES: ClassVar[list[dict]] = [
        {
            "trigger": "edit_file",
            "suggest": "bash",
            "strength": "soft",
            "nudge": "Consider running tests to verify your changes.",
        },
        {
            "trigger": "write_file",
            "suggest": "bash",
            "strength": "soft",
            "nudge": "Consider running tests to verify the new file.",
        },
    ]

    def __post_init__(self) -> None:
        if self.rules is None:
            object.__setattr__(self, 'rules', list(self._DEFAULTS_RULES))

    _calls_since_nudge: int = field(default=0, repr=False)
    _pending_suggestion: str | None = field(default=None, repr=False)

    @property
    def name(self) -> str:
        return "sequencing"

    def check(self, call: ToolCall) -> RuleResult:
        for rule in self.rules:
            if call.tool == rule["trigger"]:
                self._pending_suggestion = rule["suggest"]
                self._calls_since_nudge = 0

                if rule["strength"] == "hard":
                    return RuleResult.block(
                        call.tool,
                        nudge=rule["nudge"],
                        reason=f"hard sequence: {rule['trigger']} → {rule['suggest']}",
                    )

                # Soft nudge — allow the call but suggest follow-up
                return RuleResult.allow(call.tool)

            # Check if the agent is already doing the suggested action
            if self._pending_suggestion and call.tool == self._pending_suggestion:
                self._pending_suggestion = None
                self._calls_since_nudge = 0
                return RuleResult.allow(call.tool)

        # If we have a pending suggestion and cooldown has passed, nudge
        if self._pending_suggestion:
            self._calls_since_nudge += 1
            if self._calls_since_nudge >= self.cooldown:
                self._calls_since_nudge = 0
                # Find the nudge text for the pending suggestion
                for rule in self.rules:
                    if rule["suggest"] == self._pending_suggestion:
                        return RuleResult.nudge(
                            call.tool,
                            message=rule["nudge"],
                        )

        return RuleResult.allow(call.tool)

    def record(self, calls: list[ToolCall]) -> None:
        """Track if suggested follow-up was executed."""
        for call in calls:
            if self._pending_suggestion and call.tool == self._pending_suggestion:
                self._pending_suggestion = None
                self._calls_since_nudge = 0
