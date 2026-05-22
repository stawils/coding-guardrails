"""Destructive command blocking.

Blocks shell commands that could cause irreversible damage:
rm -rf /, fork bombs, pipe-to-shell, format disks, etc.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import ClassVar

from coding_guardrails.rules.base import Action, RuleResult, ToolCall


@dataclass
class CommandSafetyRule:
    """Block destructive shell commands.

    Checks the "command" argument of bash/shell/exec tools against
    a list of blocked commands and patterns.

    Attributes:
        command_args: Argument names that contain shell commands.
        blocked: Exact command prefixes that are always blocked.
        blocked_patterns: Regex patterns for dangerous commands.
        require_confirmation: Commands that should trigger a confirmation nudge.
    """

    command_args: list[str] = field(default_factory=lambda: ["command", "cmd", "script"])

    blocked: list[str] = field(default_factory=lambda: [
        "rm -rf / ",
        "rm -rf /*",
        "rm -rf ~",
        "rm -rf /*",
        "rm -rf ~/*",
        "dd if=",
        "mkfs.",
        ":(){ :|:& };:",
    ])

    blocked_patterns: list[str] = field(default_factory=lambda: [
        r"chmod\s+777\s+/",
        r"curl\s+.*\|\s*(ba)?sh",
        r"wget\s+.*\|\s*(ba)?sh",
        r">\s*/dev/sd[a-z]",
        r"rm\s+-rf\s+/\s*$",
    ])

    require_confirmation: list[str] = field(default_factory=lambda: [
        "rm -rf",
        "git push --force",
        "DROP TABLE",
    ])

    _DEFAULTS_BLOCKED: ClassVar[list[str]] = [
        "rm -rf / ",
        "rm -rf /*",
        "rm -rf ~",
        "rm -rf /*",
        "rm -rf ~/*",
        "dd if=",
        "mkfs.",
        ":(){ :|:& };:",
    ]
    _DEFAULTS_PATTERNS: ClassVar[list[str]] = [
        r"chmod\s+777\s+/",
        r"curl\s+.*\|\s*(ba)?sh",
        r"wget\s+.*\|\s*(ba)?sh",
        r">\s*/dev/sd[a-z]",
        r"rm\s+-rf\s+/\s*$",
    ]
    _DEFAULTS_CONFIRM: ClassVar[list[str]] = [
        "rm -rf",
        "git push --force",
        "DROP TABLE",
    ]

    def __post_init__(self) -> None:
        if self.blocked is None:
            object.__setattr__(self, 'blocked', list(self._DEFAULTS_BLOCKED))
        if self.blocked_patterns is None:
            object.__setattr__(self, 'blocked_patterns', list(self._DEFAULTS_PATTERNS))
        if self.require_confirmation is None:
            object.__setattr__(self, 'require_confirmation', list(self._DEFAULTS_CONFIRM))

    @property
    def name(self) -> str:
        return "command_safety"

    def check(self, call: ToolCall) -> RuleResult:
        # Only check tools that look like shell execution
        tool_lower = call.tool.lower()
        if not any(kw in tool_lower for kw in ["bash", "shell", "exec", "run", "command"]):
            return RuleResult.allow(call.tool)

        for arg_name in self.command_args:
            command = call.args.get(arg_name)
            if not command or not isinstance(command, str):
                continue

            result = self._check_command(command, call.tool)
            if result is not None:
                return result

        return RuleResult.allow(call.tool)

    def _check_command(self, command: str, tool: str) -> RuleResult | None:
        """Check a single command string. Returns None if safe."""

        # Hard blocks — exact matches (prefix-based)
        for blocked in self.blocked:
            if command.strip().startswith(blocked):
                return RuleResult.block(
                    tool,
                    nudge=f"Command blocked for safety: '{blocked}...'",
                    reason=f"blocked command: {command[:100]}",
                )

        # Hard blocks — pattern matches
        for pattern in self.blocked_patterns:
            if re.search(pattern, command, re.IGNORECASE):
                return RuleResult.block(
                    tool,
                    nudge="Command blocked: contains a dangerous pattern.",
                    reason=f"blocked pattern: {command[:100]}",
                )

        # Confirmation nudges
        for confirm_cmd in self.require_confirmation:
            if confirm_cmd.lower() in command.lower():
                return RuleResult.nudge(
                    tool,
                    message=f"⚠️ Potentially destructive command detected: '{confirm_cmd}'. "
                    "Consider whether this is intended and add a confirmation step.",
                )

        return None

    def record(self, calls: list[ToolCall]) -> None:
        """Command safety is stateless — nothing to record."""
        pass
