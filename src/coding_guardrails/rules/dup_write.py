"""Duplicate write detection — prevent identical content written to the same file.

Detects when an agent writes the same file with identical content repeatedly.
Escalates from nudge to block, configurable via thresholds. Edit operations
are intentionally ignored since legitimate edits legitimately re-touch the
same file with different content.
"""

from __future__ import annotations

import hashlib
import os.path
from dataclasses import dataclass, field

from coding_guardrails.rules.base import RuleResult, ToolCall


def _tool_matches(tool: str, prefixes: tuple[str, ...]) -> bool:
    """Check if a tool name matches any prefix (case-insensitive)."""
    tool_lower = tool.lower()
    return any(tool_lower.startswith(p) for p in prefixes)


def _content_hash(content: str) -> str:
    """SHA-256 hash of content, truncated for compact storage."""
    return hashlib.sha256(content.encode()).hexdigest()[:16]


@dataclass
class DuplicateWriteRule:
    """Detect and break duplicate identical writes to the same file.

    Tracks, per path, the hash of the last written content. When the same
    content is written again to the same path, it's flagged as a duplicate
    write. Escalates from nudge to block as duplicates accumulate.

    Edit tools are intentionally ignored — legitimate edits re-touch the
    same file with different content and should not be flagged.

    Attributes:
        write_tools: Tool name prefixes that are considered write/create
            operations. Edit is intentionally excluded.
        path_arg: Argument name containing the target file path.
        content_arg: Argument name containing the file content.
        nudge_threshold: Number of identical writes before nudging.
        block_threshold: Number of identical writes before blocking.
        max_tracked: Maximum number of unique paths to track. Oldest
            paths are evicted when this limit is reached.
    """

    write_tools: tuple[str, ...] = ("write", "create")
    path_arg: str = "path"
    content_arg: str = "content"
    nudge_threshold: int = 2
    block_threshold: int = 3
    max_tracked: int = 64

    _path_state: dict[str, dict] = field(
        default_factory=dict,
        repr=False,
    )

    @property
    def name(self) -> str:
        return "dup_write"

    def check(self, call: ToolCall) -> RuleResult:
        if not _tool_matches(call.tool, self.write_tools):
            return RuleResult.allow(call.tool)

        path = call.args.get(self.path_arg, "")
        content = call.args.get(self.content_arg, "")
        if not path or not content:
            return RuleResult.allow(call.tool)

        normalized = os.path.normpath(path)

        # Lazily initialize path state on first sight. check() is READ-ONLY:
        # it must not persist the increment — only record() mutates state,
        # mirroring LoopDetectionRule (check reads history, record appends).
        # Otherwise the check→record cycle double-counts every call.
        state = self._path_state.setdefault(
            normalized, {"hash": "", "count": 0}
        )
        current_hash = _content_hash(content)

        # Not a duplicate of the last write → this call is fine; record()
        # will reset the counter for this path. check() returns allow
        # without mutating.
        if state["hash"] != current_hash:
            return RuleResult.allow(call.tool)

        # Duplicate of the last write. state["count"] is the number of PRIOR
        # identical writes already recorded; this call is the (count+1)th.
        # Escalate using threshold-1 so the Nth identical write triggers
        # (matches LoopDetectionRule's `count >= threshold - 1`).
        # Block checked FIRST (the higher bar) so it isn't shadowed by nudge.
        repeat = state["count"] + 1
        if repeat >= self.block_threshold:
            return RuleResult.block(
                call.tool,
                nudge=f"You've written the same content to {path} "
                f"{repeat} times. This is a duplicate write. "
                f"If the task is done, respond with a short text summary. "
                f"Otherwise, try a completely different approach.",
                reason=f"duplicate write: {path} repeated {repeat}x",
            )

        if repeat >= self.nudge_threshold:
            return RuleResult.nudge(
                call.tool,
                message=f"You've written the same content to {path} "
                f"{repeat} times. If the task is complete, "
                "respond with a short text summary. Otherwise try a "
                "different approach.",
            )

        return RuleResult.allow(call.tool)

    def record(self, calls: list[ToolCall]) -> None:
        """Record executed writes to advance duplicate-write state.

        record() is the SOLE mutator of _path_state (check() is read-only).
        On a duplicate of the last write it advances the per-path counter;
        on new content it resets the counter and stores the new hash.
        """
        for call in calls:
            if not _tool_matches(call.tool, self.write_tools):
                continue

            path = call.args.get(self.path_arg, "")
            content = call.args.get(self.content_arg, "")
            if not path or not content:
                continue

            normalized = os.path.normpath(path)
            current_hash = _content_hash(content)

            state = self._path_state.setdefault(
                normalized, {"hash": "", "count": 0}
            )
            if state["hash"] == current_hash:
                # Duplicate of the last recorded write — advance counter.
                state["count"] += 1
            else:
                # New (or first) content — store it, reset counter.
                state["hash"] = current_hash
                state["count"] = 1

            # Enforce max_tracked — evict oldest if over limit
            if len(self._path_state) > self.max_tracked:
                oldest = next(iter(self._path_state))
                del self._path_state[oldest]
