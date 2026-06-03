"""Destructive command blocking.

Blocks shell commands that could cause irreversible damage:
rm -rf /, fork bombs, pipe-to-shell, format disks, sudo, etc.
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
        require_confirmation: Commands that trigger a confirmation nudge.
    """

    command_args: list[str] = field(default_factory=lambda: ["command", "cmd", "script"])

    blocked: list[str] = field(default_factory=lambda: [
        # Filesystem destruction
        "rm -rf / ",
        "rm -rf /*",
        "rm -rf ~",
        "rm -rf ~/*",
        "dd if=",
        "mkfs.",
        ":(){ :|:& };:",
        # Privilege escalation
        "sudo ",
        "sudo(",
        "su -",
        "su root",
        # Service manipulation
        "systemctl stop",
        "systemctl disable",
        "systemctl restart",
        "systemctl mask",
        "service stop",
        "shutdown",
        "reboot",
        "init 0",
        "init 6",
        # Disk/device access
        "> /dev/sd",
    ])

    blocked_patterns: list[str] = field(default_factory=lambda: [
        # Permission escalation
        r"chmod\s+777\s+/",
        r"chmod\s+666\s+/",
        # Download + execute (pipe to shell)
        r"curl\s+.*\|\s*(ba)?sh",
        r"wget\s+.*\|\s*(ba)?sh",
        # Download + execute (two-step)
        r"curl\s+.*-o\s+\S+.*&&\s*(ba)?sh\s",
        r"wget\s+.*-O\s+\S+.*&&\s*(ba)?sh\s",
        # Eval/execute fetched content
        r"eval\s+['\"]?\$?\(",
        r"bash\s+-c\s+['\"]?\$?\(",
        r"source\s+<\(",
        r"\.\s+<\(",                            # dot-source via process substitution
        r"exec\s+<\(",
        # Disk/device redirect
        r">\s*/dev/sd[a-z]",
        # Root filesystem removal (exact end)
        r"rm\s+-rf\s+/\s*$",
        # Git destructive operations
        r"git\s+clean\s+-fdx?",
        r"git\s+reset\s+--hard",
        r"git\s+checkout\s+--\s+\.",
        r"git\s+branch\s+-[dD]\s+(main|master)",
        r"git\s+push\s+.*--force",
        # Credential theft
        r"cat\s+/etc/shadow",
        r"cat\s+/root/.ssh",
        r"cp\s+/etc/shadow",
        # Bypass prevention patterns
        r"rm\s*\\\s*-rf",          # backslash-escaped rm
        r"bash\s+-c\s+.*\$\(",    # command substitution in bash -c
        r"\x60[^%].*\x60",         # backtick execution
    ])

    require_confirmation: list[str] = field(default_factory=lambda: [
        "rm -rf",
        "DROP TABLE",
        "DELETE FROM",
        "TRUNCATE",
    ])

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

        # Strip backslash escapes between chars (r\m → rm, su\do → sudo)
        cleaned = re.sub(r"\\(?=[a-zA-Z])", "", command)

        # Hard blocks — exact matches (prefix-based) - check both original and cleaned
        for blocked in (self.blocked or []):
            if command.strip().startswith(blocked) or cleaned.strip().startswith(blocked):
                return RuleResult.block(
                    tool,
                    nudge=f"Command blocked for safety: '{blocked}...'",
                    reason=f"blocked command: {command[:100]}",
                )

        # Hard blocks — pattern matches - check both original and cleaned
        for pattern in (self.blocked_patterns or []):
            if re.search(pattern, command, re.IGNORECASE) or re.search(pattern, cleaned, re.IGNORECASE):
                return RuleResult.block(
                    tool,
                    nudge="Command blocked: contains a dangerous pattern.",
                    reason=f"blocked pattern: {command[:100]}",
                )

        # Confirmation nudges
        for confirm_cmd in (self.require_confirmation or []):
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
