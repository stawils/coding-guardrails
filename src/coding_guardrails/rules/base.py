"""Rule protocol and core types for coding guardrails.

Each rule inspects tool calls and returns an action:
- allow: tool call is safe to execute
- block: tool call is blocked, inject nudge message
- nudge: tool call is allowed but with a soft suggestion
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol


class Action(str, Enum):
    """Result of checking a tool call against a rule."""

    ALLOW = "allow"
    BLOCK = "block"
    NUDGE = "nudge"


@dataclass(frozen=True)
class ToolCall:
    """A single tool invocation to check.

    Attributes:
        tool: Tool name (e.g. "edit_file", "bash").
        args: Tool arguments as a dict.
    """

    tool: str
    args: dict[str, Any]


@dataclass(frozen=True)
class RuleResult:
    """Result from checking one tool call against one rule.

    Attributes:
        action: What to do (allow, block, or nudge).
        tool: The tool name that was checked.
        nudge: Corrective message (set when action is block or nudge).
        reason: Human-readable explanation for logging.
        rule_name: Name of the rule that produced this result.
    """

    action: Action
    tool: str
    nudge: str | None = None
    reason: str | None = None
    rule_name: str = ""

    @staticmethod
    def allow(tool: str) -> RuleResult:
        return RuleResult(action=Action.ALLOW, tool=tool)

    @staticmethod
    def block(tool: str, nudge: str, *, reason: str | None = None) -> RuleResult:
        return RuleResult(action=Action.BLOCK, tool=tool, nudge=nudge, reason=reason)

    @staticmethod
    def nudge(tool: str, message: str) -> RuleResult:
        return RuleResult(action=Action.NUDGE, tool=tool, nudge=message)


@dataclass
class CheckResult:
    """Aggregated result from checking all tool calls against all rules.

    Attributes:
        blocked: List of blocked tool calls (hard blocks).
        nudges: List of soft nudge messages.
        allowed: List of tool calls that passed all rules.
    """

    blocked: list[RuleResult] = field(default_factory=list)
    nudges: list[RuleResult] = field(default_factory=list)
    allowed: list[ToolCall] = field(default_factory=list)

    @property
    def has_blocks(self) -> bool:
        return len(self.blocked) > 0

    @property
    def has_nudges(self) -> bool:
        return len(self.nudges) > 0

    def nudge_messages(self) -> list[str]:
        """Return all nudge messages as strings."""
        return [r.nudge for r in self.nudges if r.nudge]

    def block_messages(self) -> list[str]:
        """Return all block messages as strings."""
        return [r.nudge for r in self.blocked if r.nudge]

    def summary(self) -> str:
        """One-line summary of the check result."""
        parts = []
        if self.blocked:
            parts.append(f"{len(self.blocked)} blocked")
        if self.nudges:
            parts.append(f"{len(self.nudges)} nudged")
        if self.allowed:
            parts.append(f"{len(self.allowed)} allowed")
        return " | ".join(parts) if parts else "no calls"


class Rule(Protocol):
    """Protocol for coding guardrail rules.

    Each rule implements check() to inspect tool calls and record()
    to update internal state after tools are executed.
    """

    @property
    def name(self) -> str: ...

    def check(self, call: ToolCall) -> RuleResult:
        """Check a single tool call against this rule.

        Args:
            call: The tool call to check.

        Returns:
            RuleResult with action allow, block, or nudge.
        """
        ...

    def record(self, calls: list[ToolCall]) -> None:
        """Record executed tool calls to update internal state.

        Called after tools are executed to track what the agent has done
        (e.g. which files have been read, which commands have run).

        Args:
            calls: List of tool calls that were executed.
        """
        ...
