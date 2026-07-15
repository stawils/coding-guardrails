"""Tool resolution — handle empty results and errors.

Provides helpful nudges when tool results are empty or contain errors,
guiding the agent to adapt its approach.
"""

from __future__ import annotations

from dataclasses import dataclass

from coding_guardrails.rules.base import RuleResult, ToolCall


@dataclass
class ToolResolutionRule:
    """Nudge on empty or error tool results.

    This rule inspects tool call arguments that might indicate
    the agent is struggling with empty results or errors.

    Note: This rule is typically applied AFTER tool execution,
    by checking tool results. Here we provide nudges when the
    agent's next call suggests it didn't handle a previous
    empty/error result well.

    Attributes:
        empty_result_nudge: Nudge when search/query returns nothing.
        error_output_nudge: Nudge when command produces errors.
    """

    empty_result_nudge: str = "Query returned no results. Try broadening your search or checking the path."
    error_output_nudge: str = "Command produced errors. Read the error output carefully before proceeding."

    @property
    def name(self) -> str:
        return "tool_resolution"

    def check(self, call: ToolCall) -> RuleResult:
        # This rule operates on context that comes from tool results,
        # which arrive between calls. The middleware passes that context.
        # For now, we always allow — nudges are injected by the middleware
        # when it detects empty/error results.
        return RuleResult.allow(call.tool)

    def check_result(self, tool: str, result: str) -> RuleResult | None:
        """Check a tool result for empty or error patterns.

        Called by middleware after tool execution with the result text.

        Args:
            tool: Tool name that produced the result.
            result: The tool result text.

        Returns:
            RuleResult with nudge if pattern detected, None otherwise.
        """
        if not result or not result.strip():
            return RuleResult.nudge(
                tool,
                message=self.empty_result_nudge,
            )

        # Check for common error patterns in tool output
        result_lower = result.lower()
        error_indicators = [
            "error:",
            "not found",
            "no such file",
            "permission denied",
            "command not found",
            "fatal:",
            "traceback (most recent call last)",
            "exception",
            "failed",
        ]
        for indicator in error_indicators:
            if indicator in result_lower:
                return RuleResult.nudge(
                    tool,
                    message=self.error_output_nudge,
                )

        return None

    def record(self, calls: list[ToolCall]) -> None:
        """Tool resolution is stateless — nothing to record."""
        pass
