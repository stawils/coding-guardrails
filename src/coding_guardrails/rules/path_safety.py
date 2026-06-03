"""Path safety — block access outside allowed workspace.

Prevents path traversal attacks and access to system directories.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import ClassVar
from pathlib import PurePosixPath

from coding_guardrails.rules.base import Action, RuleResult, ToolCall


@dataclass
class PathSafetyRule:
    """Block file access outside allowed workspace.

    Checks tool call arguments that look like file paths against
    a set of blocked prefixes, patterns, and an allowlist.

    Attributes:
        allowlist: Path prefixes that are allowed (e.g. ["/home/user/"]).
        blocked_prefixes: Path prefixes that are always blocked.
        blocked_patterns: Regex patterns that are always blocked.
        path_args: Argument names to check for paths.
    """

    allowlist: list[str] = field(default_factory=lambda: [])
    blocked_prefixes: list[str] = field(default_factory=lambda: [
        "/etc/",
        "/usr/",
        "/boot/",
        "/sys/",
        "/proc/",
        "/root/.ssh/",
        "/root/.gnupg/",
    ])
    blocked_patterns: list[str] = field(default_factory=lambda: [
        r"^[A-Za-z]:/",  # Windows absolute path (C:/, D:/, etc.)
        r"^//",  # UNC path (//server/share)
        r"\.\./",
        r"\.\.\\",
    ])
    path_args: list[str] = field(default_factory=lambda: ["path", "filename", "file", "directory", "dir"])

    _DEFAULTS_PREFIXES: ClassVar[list[str]] = [
        "/etc/",
        "/usr/",
        "/boot/",
        "/sys/",
        "/proc/",
        "/root/.ssh/",
        "/root/.gnupg/",
    ]
    _DEFAULTS_PATTERNS: ClassVar[list[str]] = [
        r"^[A-Za-z]:/",  # Windows absolute path
        r"^//",  # UNC path
        r"\.{2}/",
        r"\.{2}\\\\",
    ]

    def __post_init__(self) -> None:
        if self.blocked_prefixes is None:
            object.__setattr__(self, 'blocked_prefixes', list(self._DEFAULTS_PREFIXES))
        if self.blocked_patterns is None:
            object.__setattr__(self, 'blocked_patterns', list(self._DEFAULTS_PATTERNS))

    @property
    def name(self) -> str:
        return "path_safety"

    def check(self, call: ToolCall) -> RuleResult:
        for arg_name in self.path_args:
            path = call.args.get(arg_name)
            if not path or not isinstance(path, str):
                continue

            result = self._check_path(path, call.tool)
            if result is not None:
                return result

        return RuleResult.allow(call.tool)

    def _check_path(self, path: str, tool: str) -> RuleResult | None:
        """Check a single path. Returns None if path is safe."""

        # Expand environment variables and user home
        expanded = os.path.expandvars(os.path.expanduser(path))
        normalized = os.path.normpath(expanded)
        # Resolve symlinks to prevent symlink-based escapes
        resolved = os.path.realpath(normalized)

        # Normalize backslashes to forward slashes for consistent pattern matching
        normalized_for_check = path.replace("\\", "/")

        # Check blocked patterns (path traversal)
        for pattern in self.blocked_patterns:
            if re.search(pattern, normalized_for_check):
                return RuleResult.block(
                    tool,
                    nudge=f"Path '{path}' contains a blocked traversal pattern.",
                    reason=f"path traversal: {path}",
                )

        # Check blocked prefixes (both normalized and symlink-resolved paths)
        for prefix in self.blocked_prefixes:
            norm_prefix = os.path.normpath(prefix)
            if normalized.startswith(norm_prefix) or resolved.startswith(norm_prefix):
                return RuleResult.block(
                    tool,
                    nudge=f"Path '{path}' is outside the allowed workspace.",
                    reason=f"blocked prefix: {path} matches {prefix}",
                )

        # Check allowlist (if configured, path must match at least one)
        if self.allowlist:
            allowed = False
            for allowed_prefix in self.allowlist:
                expanded_prefix = os.path.expandvars(os.path.expanduser(allowed_prefix))
                norm_prefix = os.path.normpath(expanded_prefix)
                if normalized.startswith(norm_prefix) and resolved.startswith(norm_prefix):
                    allowed = True
                    break
            if not allowed:
                return RuleResult.block(
                    tool,
                    nudge=f"Path '{path}' is outside the allowed workspace.",
                    reason=f"not in allowlist: {path}",
                )

        return None

    def record(self, calls: list[ToolCall]) -> None:
        """Path safety is stateless — nothing to record."""
        pass
