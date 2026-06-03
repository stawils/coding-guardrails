"""Session budget — cap total operations per session.

Prevents runaway agents from performing unlimited operations.
Warns at 80%, blocks at 100% of configured budgets.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from coding_guardrails.rules.base import Action, RuleResult, ToolCall
from coding_guardrails.rules.prerequisites import _tool_matches

_WRITE_TOOLS = ("edit", "write", "create")
_SHELL_TOOLS = ("bash", "shell", "exec", "run", "command")
_READ_TOOLS = ("read", "cat", "head", "tail")


@dataclass
class SessionBudgetRule:
    """Cap total operations per session.

    Attributes:
        max_file_ops: Maximum file edit/write operations.
        max_commands: Maximum shell command executions.
        max_reads: Maximum file read operations (0 = unlimited).
        warn_at: Fraction (0-1) at which to warn.
    """

    max_file_ops: int = 100
    max_commands: int = 200
    max_reads: int = 0  # unlimited by default
    warn_at: float = 0.8

    _file_ops: int = field(default=0, repr=False)
    _commands: int = field(default=0, repr=False)
    _reads: int = field(default=0, repr=False)
    _warned_files: bool = field(default=False, repr=False)
    _warned_commands: bool = field(default=False, repr=False)
    _warned_reads: bool = field(default=False, repr=False)

    @property
    def name(self) -> str:
        return "session_budget"

    def check(self, call: ToolCall) -> RuleResult:
        # Track reads (unlimited by default, just counting)
        if _tool_matches(call.tool, _READ_TOOLS) and self.max_reads > 0:
            if self._reads >= self.max_reads:
                return RuleResult.block(
                    call.tool,
                    nudge=f"Budget exhausted: read limit reached ({self._reads}/{self.max_reads}). "
                    "Stop reading and use what you have.",
                    reason=f"read budget: {self._reads}/{self.max_reads}",
                )
            if self._reads >= int(self.max_reads * self.warn_at) and not self._warned_reads:
                return RuleResult.nudge(
                    call.tool,
                    message=f"Advisory: Read budget: {self._reads}/{self.max_reads} "
                    f"({self._reads * 100 // self.max_reads}%).",
                )

        # Track file operations
        if _tool_matches(call.tool, _WRITE_TOOLS):
            if self._file_ops >= self.max_file_ops:
                return RuleResult.block(
                    call.tool,
                    nudge=f"Budget exhausted: file operation limit reached ({self._file_ops}/{self.max_file_ops}). "
                    "Stop editing files.",
                    reason=f"file budget: {self._file_ops}/{self.max_file_ops}",
                )
            if (self._file_ops >= int(self.max_file_ops * self.warn_at)
                    and not self._warned_files):
                self._warned_files = True
                return RuleResult.nudge(
                    call.tool,
                    message=f"Advisory: File operations: {self._file_ops}/{self.max_file_ops} "
                    f"({self._file_ops * 100 // self.max_file_ops}%). "
                    "You're approaching the session limit.",
                )

        # Track commands
        if _tool_matches(call.tool, _SHELL_TOOLS):
            if self._commands >= self.max_commands:
                return RuleResult.block(
                    call.tool,
                    nudge=f"Budget exhausted: command limit reached ({self._commands}/{self.max_commands}). "
                    "Stop executing commands.",
                    reason=f"command budget: {self._commands}/{self.max_commands}",
                )
            if (self._commands >= int(self.max_commands * self.warn_at)
                    and not self._warned_commands):
                self._warned_commands = True
                return RuleResult.nudge(
                    call.tool,
                    message=f"Advisory: Commands: {self._commands}/{self.max_commands} "
                    f"({self._commands * 100 // self.max_commands}%). "
                    "You're approaching the session limit.",
                )

        return RuleResult.allow(call.tool)

    def record(self, calls: list[ToolCall]) -> None:
        """Update counters after execution."""
        for call in calls:
            if _tool_matches(call.tool, _WRITE_TOOLS):
                self._file_ops += 1
            elif _tool_matches(call.tool, _SHELL_TOOLS):
                self._commands += 1
            elif _tool_matches(call.tool, _READ_TOOLS):
                self._reads += 1



    def reset(self) -> None:
        """Reset all counters to zero."""
        self._file_ops = 0
        self._commands = 0
        self._reads = 0
        self._warned_files = False
        self._warned_commands = False
        self._warned_reads = False
