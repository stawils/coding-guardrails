"""Destructive command blocking.

Blocks shell commands that could cause irreversible damage:
rm -rf /, fork bombs, pipe-to-shell, format disks, sudo, etc.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from coding_guardrails.rules.base import RuleResult, ToolCall


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
        # Encoded payload execution — decode then pipe to a shell
        r"base64\s+(?:-d|--decode)[^|]*\|\s*(?:ba)?sh",
        r"\bxxd\s+-r[^|]*\|\s*(?:ba)?sh",
        r"openssl\s+(?:enc|aes|des)[^|]*\s-d[^|]*\|\s*(?:ba)?sh",
        r"\$\(\s*echo\s+[A-Za-z0-9+/=]{16,}\s*\|\s*base64",
        r"(?:exec|eval)\s*\(\s*(?:base64\.b64decode|b64decode)",
        # Remote content into execution context
        r"[A-Za-z_]\w*=\$\(\s*(?:curl|wget)\b",          # fetch-into-var then exec (WS gateway hijack shape)
        r"\b(?:node|deno|python3?|perl|ruby|php)\s+(?:-e|-c)\s+['\"]?\$\(",  # interp eval of command substitution
        # Environment-variable injection — env overrides preceding a command
        r"\bLD_PRELOAD\s*=",
        r"\bLD_AUDIT\s*=",
        r"\bLD_LIBRARY_PATH\s*=\S*\.so",
        r"\bBASH_ENV\s*=\S",
        r"(?:^|&&|\|\||;|\||\s)\s*PATH=[^\s$\"']\S*\s+\S",   # inline PATH= override before cmd
        # Argument injection inside benign tools
        r"ssh\s+[^|]*?;\s*(?:curl|wget|bash|sh|nc|ncat|python|chmod|rm)\b",
        r"ssh\s+[^|]*?&&\s*(?:curl|wget|bash|sh|nc|ncat|python)\b",
        r"\bProxyCommand\s*[= ]",
        r"ssh\s+[^|]*\$\(",   # local command substitution on ssh line
        r"docker\s+(?:run|exec)[^|]*\s(?:-e|--env)[=\s]\s*(?:PATH|LD_PRELOAD|LD_LIBRARY_PATH|BASH_ENV)\s*=",
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

        result = self._match_patterns(command, cleaned, tool)
        if result is not None:
            return result

        # Encoded-payload defense: find base64-looking blobs, decode them,
        # and re-check the decoded text. Catches `echo cm0gLXJmIC8= ... `
        # even when the execution pipe is constructed to dodge the
        # decode-to-shell patterns above. One level deep — no recursion into
        # re-encoded blobs (diminishing returns, fp risk).
        for source in (command, cleaned):
            for blob in re.findall(r"[A-Za-z0-9+/]{16,}={0,2}", source):
                try:
                    import base64 as _b64
                    decoded = _b64.b64decode(blob, validate=True).decode("utf-8", errors="ignore")
                except Exception:  # noqa: BLE001 — not valid base64, skip
                    continue
                if decoded != blob and len(decoded) >= 4:
                    result = self._match_patterns(decoded, decoded, tool, decoded=True)
                    if result is not None:
                        return result

        # Confirmation nudges
        for confirm_cmd in (self.require_confirmation or []):
            if confirm_cmd.lower() in command.lower():
                return RuleResult.nudge(
                    tool,
                    message=f"Advisory: Consider using caution with '{confirm_cmd}'. "
                    "This command may cause data loss. Add a confirmation step before executing.",
                )

        return None

    def _match_patterns(
        self, command: str, cleaned: str, tool: str, *, decoded: bool = False,
    ) -> RuleResult | None:
        """Check hard blocks against original + cleaned forms. None if safe."""

        # Hard blocks — exact matches (prefix-based) - check both original and cleaned
        for blocked in (self.blocked or []):
            if command.strip().startswith(blocked) or cleaned.strip().startswith(blocked):
                return RuleResult.block(
                    tool,
                    nudge=f"Command blocked: '{blocked}...' is destructive.",
                    reason=f"blocked command: {command[:100]}",
                )

        # Hard blocks — pattern matches - check both original and cleaned
        for pattern in (self.blocked_patterns or []):
            if re.search(pattern, command, re.IGNORECASE) or re.search(pattern, cleaned, re.IGNORECASE):
                tag = " (base64-decoded payload)" if decoded else ""
                return RuleResult.block(
                    tool,
                    nudge=f"Command blocked: contains a dangerous pattern{tag}.",
                    reason=f"blocked pattern{tag}: {command[:100]}",
                )

        return None

    def record(self, calls: list[ToolCall]) -> None:
        """Command safety is stateless — nothing to record."""
        pass
