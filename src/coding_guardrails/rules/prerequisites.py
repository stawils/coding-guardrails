"""Read-before-edit prerequisite enforcement.

Tracks which files the agent has read. Blocks edit/write operations
on files that haven't been read first.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from coding_guardrails.rules.base import Action, RuleResult, ToolCall


@dataclass
class PrerequisiteRule:
    """Enforce read-before-edit for file operations.

    Each rule entry maps a tool to a prerequisite tool, matching on a
    shared argument (typically "path" or "filename").

    Attributes:
        rules: List of (tool, requires, match_arg) tuples.
            tool: The tool that requires a prerequisite.
            requires: The prerequisite tool name.
            match_arg: The argument name to match (e.g. "path").
        max_violations: Block after this many consecutive violations.
    """

    rules: list[dict[str, str]] = field(default_factory=lambda: [
        {"tool": "edit_file", "requires": "read_file", "match_arg": "path"},
        {"tool": "write_file", "requires": "read_file", "match_arg": "path"},
    ])
    max_violations: int = 2

    _read_paths: set[str] = field(default_factory=set, repr=False)
    _violation_count: int = field(default=0, repr=False)

    _DEFAULT_RULES: ClassVar[list[dict[str, str]]] = [
        {"tool": "edit_file", "requires": "read_file", "match_arg": "path"},
        {"tool": "write_file", "requires": "read_file", "match_arg": "path"},
    ]

    def __post_init__(self) -> None:
        if self.rules is None:
            object.__setattr__(self, 'rules', list(self._DEFAULT_RULES))

    @property
    def name(self) -> str:
        return "prerequisites"

    def check(self, call: ToolCall) -> RuleResult:
        for rule in self.rules:
            if call.tool != rule["tool"]:
                continue

            match_arg = rule["match_arg"]
            path = call.args.get(match_arg, "")
            if not path:
                continue

            # Normalize: strip trailing slashes for consistent matching
            normalized = path.rstrip("/")

            if normalized not in self._read_paths:
                self._violation_count += 1
                if self._violation_count >= self.max_violations:
                    return RuleResult.block(
                        call.tool,
                        nudge=f"You must read {path} before editing it. "
                        f"Call {rule['requires']} first.",
                        reason=f"edit without read: {path}",
                    )
                return RuleResult.nudge(
                    call.tool,
                    message=f"Consider reading {path} before editing. "
                    f"Call {rule['requires']} first.",
                )

        # No prerequisite violated — reset counter
        self._violation_count = 0
        return RuleResult.allow(call.tool)

    def record(self, calls: list[ToolCall]) -> None:
        """Record which files have been read."""
        for call in calls:
            for rule in self.rules:
                if call.tool == rule["requires"]:
                    match_arg = rule["match_arg"]
                    path = call.args.get(match_arg, "")
                    if path:
                        self._read_paths.add(path.rstrip("/"))

        # Reset violation counter on successful execution
        self._violation_count = 0
