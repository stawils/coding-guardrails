"""Network egress control — block data exfiltration and SSRF.

Prevents agents from:
- Uploading files to external endpoints (curl -d @file, wget --post-file)
- Accessing cloud metadata endpoints (169.254.169.254)
- Making outbound requests to private/internal IPs (configurable)
"""

from __future__ import annotations

import re
import urllib.parse
from dataclasses import dataclass, field
from typing import ClassVar

from coding_guardrails.rules.base import Action, RuleResult, ToolCall
from coding_guardrails.rules.prerequisites import _tool_matches

# Tool prefixes that execute network commands.
_NETWORK_TOOLS = ("bash", "shell", "exec", "run", "command")

# Patterns that indicate file upload / data exfiltration.
_UPLOAD_PATTERNS: list[tuple[str, str]] = [
    (r"curl\s+.*-d\s+@", "curl file upload"),
    (r"curl\s+.*--data\s+@", "curl file upload"),
    (r"curl\s+.*--data-binary\s+@", "curl file upload"),
    (r"curl\s+.*-F\s+.*@", "curl form file upload"),
    (r"curl\s+.*-T\s+", "curl file transfer"),
    (r"wget\s+.*--post-file[=\s]+", "wget file upload"),
    (r"rsync\s+.*@\w", "rsync to remote"),
    (r"scp\s+", "scp file transfer"),
    (r"sftp\s+", "sftp file transfer"),
]

# Cloud metadata / internal endpoints.
_METADATA_PATTERNS: list[tuple[str, str]] = [
    (r"169\.254\.169\.254", "AWS/GCP/Azure metadata endpoint"),
    (r"169\.254\.170\.2", "AWS ECS task metadata"),
    (r"metadata\.google\.internal", "GCP metadata endpoint"),
    (r"metadata\.azure\.com", "Azure metadata endpoint"),
    (r"instance-data\.ec2\.", "EC2 instance metadata"),
]

# Python network patterns.
_PYTHON_NETWORK_PATTERNS: list[tuple[str, str]] = [
    (r"subprocess\.(run|call|Popen|check_output|check_call)\s*\(", "subprocess network call"),
    (r"os\.system\s*\(", "os.system call"),
    (r"requests\.(get|post|put|delete|patch|head)\s*\(", "requests library call"),
    (r"httpx\.(get|post|put|delete|patch|head)\s*\(", "httpx library call"),
    (r"urllib\.request\.urlopen\s*\(", "urllib request"),
    (r"socket\.socket\s*\(", "raw socket creation"),
]


@dataclass
class NetworkRule:
    """Block network exfiltration and SSRF in shell commands.

    Attributes:
        block_uploads: Block commands that upload files externally.
        block_metadata: Block access to cloud metadata endpoints.
        block_private_ips: Block outbound requests to private IPs.
        allowed_hosts: Hostnames/IPs that are always allowed.
        tool_prefixes: Tool name prefixes to check.
    """

    block_uploads: bool = True
    block_metadata: bool = True
    block_private_ips: bool = False
    allowed_hosts: list[str] = field(default_factory=lambda: [
        "localhost",
        "127.0.0.1",
        "0.0.0.0",
        "::1",
    ])
    tool_prefixes: tuple[str, ...] = _NETWORK_TOOLS

    @property
    def name(self) -> str:
        return "network"

    def check(self, call: ToolCall) -> RuleResult:
        if not _tool_matches(call.tool, self.tool_prefixes):
            return RuleResult.allow(call.tool)

        command = ""
        for arg in ("command", "cmd", "script"):
            command = call.args.get(arg, "")
            if command and isinstance(command, str):
                break

        if not command:
            return RuleResult.allow(call.tool)

        return self._check_command(command, call.tool)

    def _check_command(self, command: str, tool: str) -> RuleResult:
        # Check for allowed hosts — skip if command only targets allowed
        if self._only_targets_allowed(command):
            return RuleResult.allow(call=tool) if not isinstance(tool, str) else RuleResult.allow(tool)

        # URL/hex decode command to catch encoded bypass attempts
        decoded_command = urllib.parse.unquote(command)

        if self.block_uploads:
            for pattern, label in _UPLOAD_PATTERNS:
                if re.search(pattern, command) or re.search(pattern, decoded_command):
                    return RuleResult.block(
                        tool,
                        nudge=f"Network upload blocked: {label} detected. "
                        "File uploads to external endpoints are not allowed.",
                        reason=f"network upload: {label}",
                    )

        if self.block_metadata:
            for pattern, label in _METADATA_PATTERNS:
                if re.search(pattern, command) or re.search(pattern, decoded_command):
                    return RuleResult.block(
                        tool,
                        nudge=f"Network blocked: access to {label} is not allowed.",
                        reason=f"metadata endpoint: {label}",
                    )

        # Python network patterns
        for pattern, label in _PYTHON_NETWORK_PATTERNS:
            if re.search(pattern, command) or re.search(pattern, decoded_command):
                return RuleResult.block(
                    tool,
                    nudge=f"Network blocked: Python {label} is not allowed.",
                    reason=f"python network: {label}",
                )

        if self.block_private_ips:
            # Match IPs in URLs or direct in both original and decoded
            def find_ips(text: str) -> list[str]:
                return re.findall(r"(?:https?://)?(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})", text)

            for source_cmd in (command, decoded_command):
                for ip in find_ips(source_cmd):
                    if self._is_private_ip(ip):
                        return RuleResult.block(
                            tool,
                            nudge=f"Network blocked: request to private IP {ip} is not allowed.",
                            reason=f"private IP access: {ip}",
                        )

        return RuleResult.allow(tool)

    def _only_targets_allowed(self, command: str) -> bool:
        """Check if command only targets allowed hosts.

        Returns True only if we find an allowed host as the actual target
        (in a URL or after @), not just mentioned anywhere in the string.
        """
        # Check URLs: http(s)://allowed-host or host:port
        for host in self.allowed_hosts:
            # URL pattern
            if re.search(rf'https?://{re.escape(host)}(?:[:/]|$)', command):
                return True
            # Direct host:port pattern
            if re.search(rf'{re.escape(host)}:\d+', command):
                return True
        return False

    def _is_private_ip(self, ip: str) -> bool:
        """Check if an IP address is in private ranges."""
        parts = ip.split(".")
        if len(parts) != 4:
            return False
        try:
            octets = [int(p) for p in parts]
        except ValueError:
            return False
        # 10.0.0.0/8
        if octets[0] == 10:
            return True
        # 172.16.0.0/12
        if octets[0] == 172 and 16 <= octets[1] <= 31:
            return True
        # 192.168.0.0/16
        if octets[0] == 192 and octets[1] == 168:
            return True
        # 169.254.0.0/16 (link-local)
        if octets[0] == 169 and octets[1] == 254:
            return True
        return False

    def record(self, calls: list[ToolCall]) -> None:
        """Stateless — nothing to record."""
        pass
