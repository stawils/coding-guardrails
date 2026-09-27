"""Session budget — cap total operations per conversation.

Prevents runaway agents from performing unlimited operations. The budget is a
per-conversation safety net, not a task limit: the proxy resets it whenever a
new conversation starts (see ``proxy/handler.py``), so long-horizon work spread
across many conversations is never blocked by operations accumulated earlier.

Warns at ``warn_at`` of a budget and blocks at 100%. ``check()`` reserves
budget for calls it allows, so a single multi-call batch can never overshoot
the cap; ``record()`` reconciles those reservations against what actually
executed. This keeps the reported count exact (no more ``102/100``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from coding_guardrails.rules.base import RuleResult, ToolCall
from coding_guardrails.rules.prerequisites import _tool_matches

_WRITE_TOOLS = ("edit", "write", "create")
_SHELL_TOOLS = ("bash", "shell", "exec", "run", "command")
_READ_TOOLS = ("read", "cat", "head", "tail")

# Long-horizon-friendly defaults. This rule is a runaway backstop, not a task
# budget: loop/duplicate detection catch actual thrashing, so the session cap
# only needs to stop pathological sessions. Low defaults (the old 100/200)
# turned "keep working" into a hard stop mid-task.
DEFAULT_MAX_FILE_OPS = 1000
DEFAULT_MAX_COMMANDS = 2000
DEFAULT_MAX_READS = 0  # unlimited
DEFAULT_WARN_AT = 0.8

# Appended to exhaustion blocks so the agent has a way out instead of retrying
# the blocked call forever. Mirrors the loop_detection block convention.
_WRAP_UP = (
    "This is a proxy safety cap, not a task result. Stop starting new calls of "
    "this kind and wrap up: if the task is complete, call respond() with your "
    "final answer; otherwise report what remains and what blocked you."
)


@dataclass
class SessionBudgetRule:
    """Cap total operations per conversation.

    Attributes:
        max_file_ops: Maximum file edit/write operations.
        max_commands: Maximum shell command executions.
        max_reads: Maximum file read operations (0 = unlimited).
        warn_at: Fraction (0-1) at which to warn.
    """

    max_file_ops: int = DEFAULT_MAX_FILE_OPS
    max_commands: int = DEFAULT_MAX_COMMANDS
    max_reads: int = DEFAULT_MAX_READS
    warn_at: float = DEFAULT_WARN_AT

    _file_ops: int = field(default=0, repr=False)
    _commands: int = field(default=0, repr=False)
    _reads: int = field(default=0, repr=False)
    # Reservations made by check() for calls that passed but have not been
    # recorded yet. Counted toward the effective total so a batch cannot
    # overshoot the cap; cleared by record()/begin_batch()/reset().
    _pending_file_ops: int = field(default=0, repr=False)
    _pending_commands: int = field(default=0, repr=False)
    _pending_reads: int = field(default=0, repr=False)
    _warned_files: bool = field(default=False, repr=False)
    _warned_commands: bool = field(default=False, repr=False)
    _warned_reads: bool = field(default=False, repr=False)

    @property
    def file_op_count(self) -> int:
        """Number of file operations recorded."""
        return self._file_ops

    @property
    def command_count(self) -> int:
        """Number of commands recorded."""
        return self._commands

    @property
    def read_count(self) -> int:
        """Number of reads recorded."""
        return self._reads

    @property
    def name(self) -> str:
        return "session_budget"

    def begin_batch(self) -> None:
        """Drop reservations left over from a previous batch.

        Called by the middleware before checking a new batch so a call that was
        allowed by this rule but blocked by another rule cannot leak a phantom
        reservation into later batches.
        """
        self._pending_file_ops = 0
        self._pending_commands = 0
        self._pending_reads = 0

    def check(self, call: ToolCall) -> RuleResult:
        # Track reads (unlimited by default, just counting)
        if _tool_matches(call.tool, _READ_TOOLS) and self.max_reads > 0:
            effective = self._reads + self._pending_reads
            if effective >= self.max_reads:
                return RuleResult.block(
                    call.tool,
                    nudge=f"Budget exhausted: read limit reached ({effective}/{self.max_reads}). "
                    "Stop reading and use what you have. " + _WRAP_UP,
                    reason=f"read budget: {effective}/{self.max_reads}",
                )
            self._pending_reads += 1
            if effective >= int(self.max_reads * self.warn_at) and not self._warned_reads:
                self._warned_reads = True
                return RuleResult.nudge(
                    call.tool,
                    message=f"Advisory: Read budget: {effective}/{self.max_reads} "
                    f"({effective * 100 // self.max_reads}%).",
                )
            return RuleResult.allow(call.tool)

        # Track file operations
        if _tool_matches(call.tool, _WRITE_TOOLS):
            effective = self._file_ops + self._pending_file_ops
            if effective >= self.max_file_ops:
                return RuleResult.block(
                    call.tool,
                    nudge=f"Budget exhausted: file operation limit reached "
                    f"({effective}/{self.max_file_ops}). Stop editing files. " + _WRAP_UP,
                    reason=f"file budget: {effective}/{self.max_file_ops}",
                )
            self._pending_file_ops += 1
            if (effective >= int(self.max_file_ops * self.warn_at)
                    and not self._warned_files):
                self._warned_files = True
                return RuleResult.nudge(
                    call.tool,
                    message=f"Advisory: File operations: {effective}/{self.max_file_ops} "
                    f"({effective * 100 // self.max_file_ops}%). "
                    "You're approaching the session limit.",
                )
            return RuleResult.allow(call.tool)

        # Track commands
        if _tool_matches(call.tool, _SHELL_TOOLS):
            effective = self._commands + self._pending_commands
            if effective >= self.max_commands:
                return RuleResult.block(
                    call.tool,
                    nudge=f"Budget exhausted: command limit reached "
                    f"({effective}/{self.max_commands}). Stop executing commands. " + _WRAP_UP,
                    reason=f"command budget: {effective}/{self.max_commands}",
                )
            self._pending_commands += 1
            if (effective >= int(self.max_commands * self.warn_at)
                    and not self._warned_commands):
                self._warned_commands = True
                return RuleResult.nudge(
                    call.tool,
                    message=f"Advisory: Commands: {effective}/{self.max_commands} "
                    f"({effective * 100 // self.max_commands}%). "
                    "You're approaching the session limit.",
                )
            return RuleResult.allow(call.tool)

        return RuleResult.allow(call.tool)

    def record(self, calls: list[ToolCall]) -> None:
        """Update counters after execution.

        Reconciles the reservations made by ``check()``: reservations are
        cleared and every call in ``calls`` is counted exactly once.
        """
        self._pending_file_ops = 0
        self._pending_commands = 0
        self._pending_reads = 0
        for call in calls:
            if _tool_matches(call.tool, _WRITE_TOOLS):
                self._file_ops += 1
            elif _tool_matches(call.tool, _SHELL_TOOLS):
                self._commands += 1
            elif _tool_matches(call.tool, _READ_TOOLS):
                self._reads += 1

    def reset(self) -> None:
        """Reset all counters and warnings to zero."""
        self._file_ops = 0
        self._commands = 0
        self._reads = 0
        self._pending_file_ops = 0
        self._pending_commands = 0
        self._pending_reads = 0
        self._warned_files = False
        self._warned_commands = False
        self._warned_reads = False
