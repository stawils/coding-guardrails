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


# Judge band thresholds, shared by rules, middleware and tests so there is one
# definition of "how sure is sure enough". Only consulted when a judge fired;
# deterministic results never carry a probability.
JUDGE_BLOCK_THRESHOLD = 0.8   # p > this  -> BLOCK
JUDGE_NUDGE_THRESHOLD = 0.5   # p > this  -> NUDGE, else ALLOW


def judge_action(p: float, thresholds: tuple[float, float] = (
    JUDGE_NUDGE_THRESHOLD, JUDGE_BLOCK_THRESHOLD,
)) -> Action:
    """Map a judge probability to an Action using the shared band thresholds."""
    nudge_at, block_at = thresholds
    if p > block_at:
        return Action.BLOCK
    if p > nudge_at:
        return Action.NUDGE
    return Action.ALLOW


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
        confidence: Judge probability for the fired action, or None when the
            result came from a deterministic rule.
        judge_consulted: True when a semantic judge produced this result.
        judge_engine: Engine name that answered (e.g. "decider-0.8b").
        judge_outcome: Band/outcome label ("block" | "nudge" | "allow" |
            "timeout" | "error"), for audit.
    """

    action: Action
    tool: str
    nudge: str | None = None
    reason: str | None = None
    rule_name: str = ""
    confidence: float | None = None
    judge_consulted: bool = False
    judge_engine: str | None = None
    judge_outcome: str | None = None

    @staticmethod
    def allow(tool: str) -> RuleResult:
        return RuleResult(action=Action.ALLOW, tool=tool)

    @staticmethod
    def block(tool: str, nudge: str, *, reason: str | None = None) -> RuleResult:
        return RuleResult(action=Action.BLOCK, tool=tool, nudge=nudge, reason=reason)

    # The nudge(...) factory is defined below, outside the class body: a dataclass
    # field and a same-named method cannot coexist here — the method would shadow
    # the field default, making allow(x).nudge return the function instead of None.


# Factory for NUDGE results. Assigned outside RuleResult's body to avoid the
# field/method name collision noted above; keeps RuleResult.nudge(tool, message)
# working while the field default stays None.
def _rule_result_nudge(tool: str, message: str) -> RuleResult:
    return RuleResult(action=Action.NUDGE, tool=tool, nudge=message)


RuleResult.nudge = staticmethod(_rule_result_nudge)  # type: ignore[method-assign]


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

    def judge_results(self) -> list[RuleResult]:
        """Return every result (block or nudge) that a judge produced."""
        return [r for r in (*self.blocked, *self.nudges) if r.judge_consulted]

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
