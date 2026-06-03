"""Read-before-edit prerequisite enforcement.

Tracks which files the agent has read. Blocks edit/write operations
on files that haven't been read first. Supports directory-level reads
satisfying file-level edits.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from coding_guardrails.rules.base import Action, RuleResult, ToolCall

# Tool name matching: prefix-based so 'edit' matches 'edit', 'edit_file',
# 'Edit', etc. Covers Pi (edit/read/bash), Claude Code (Edit/Read),
# Aider, OpenCode, and generic agents.
_DEFAULT_EDIT_TOOLS = ("edit", "write", "create")
_DEFAULT_READ_TOOLS = ("read", "cat", "head", "tail", "less")


def _tool_matches(tool: str, prefixes: tuple[str, ...]) -> bool:
    """Check if a tool name matches any prefix (case-insensitive)."""
    tool_lower = tool.lower()
    return any(tool_lower.startswith(p) for p in prefixes)


@dataclass
class PrerequisiteRule:
    """Enforce read-before-edit for file operations.

    Uses prefix matching so tool names like 'edit', 'edit_file', 'Edit'
    all match. Covers Pi, Claude Code, Aider, OpenCode, and generic agents.

    Smart matching:
    - Directory reads satisfy all files under that directory.
    - read(src/main.py) satisfies edit(src/main.py.bak) (same directory).

    Attributes:
        edit_tools: Tool name prefixes that require a prior read.
        read_tools: Tool name prefixes that satisfy the read requirement.
        match_arg: Argument name containing the file path.
        max_violations: Block after this many consecutive violations.
    """

    edit_tools: tuple[str, ...] = _DEFAULT_EDIT_TOOLS
    read_tools: tuple[str, ...] = _DEFAULT_READ_TOOLS
    match_arg: str = "path"
    max_violations: int = 2

    _read_paths: set[str] = field(default_factory=set, repr=False)
    _read_dirs: set[str] = field(default_factory=set, repr=False)
    _violation_count: int = field(default=0, repr=False)

    @property
    def name(self) -> str:
        return "prerequisites"

    def check(self, call: ToolCall) -> RuleResult:
        if not _tool_matches(call.tool, self.edit_tools):
            return RuleResult.allow(call.tool)

        path = call.args.get(self.match_arg, "")
        if not path:
            return RuleResult.allow(call.tool)

        # Normalize: expand user, strip trailing slashes
        normalized = os.path.normpath(os.path.expanduser(path))

        if not self._has_been_read(normalized):
            self._violation_count += 1
            if self._violation_count >= self.max_violations:
                return RuleResult.block(
                    call.tool,
                    nudge=f"Edit blocked: read {path} first before editing.",
                    reason=f"edit without read: {path}",
                )
            return RuleResult.nudge(
                call.tool,
                message=f"Advisory: Read {path} before editing to avoid errors.",
            )

        # No prerequisite violated — reset counter
        self._violation_count = 0
        return RuleResult.allow(call.tool)

    def _has_been_read(self, path: str) -> bool:
        """Check if a path has been read (exact, directory, or parent)."""
        # Exact match
        if path in self._read_paths:
            return True
        # Directory read satisfies all children
        parent = os.path.dirname(path)
        while parent and parent != "/":
            if parent in self._read_dirs:
                return True
            parent = os.path.dirname(parent)
        # Check for "." (project root) — satisfies everything
        if "." in self._read_dirs:
            return True
        return False

    def record(self, calls: list[ToolCall]) -> None:
        """Record which files have been read."""
        for call in calls:
            if _tool_matches(call.tool, self.read_tools):
                path = call.args.get(self.match_arg, "")
                if path:
                    normalized = os.path.normpath(os.path.expanduser(path))
                    self._read_paths.add(normalized)
                    # Track directories too
                    if os.path.isdir(normalized) or normalized.endswith("/"):
                        self._read_dirs.add(normalized.rstrip("/"))

        # Reset violation counter on successful execution
        self._violation_count = 0

    def mark_directory_read(self, path: str) -> None:
        """Mark a directory as having been read (for testing)."""
        normalized = os.path.normpath(os.path.expanduser(path))
        self._read_dirs.add(normalized.rstrip("/"))
