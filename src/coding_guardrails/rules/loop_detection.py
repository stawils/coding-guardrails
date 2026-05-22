"""Loop detection — detect when an agent is stuck repeating the same call.

Tracks recent tool calls and detects when the agent retries the same
operation multiple times without progress. Escalates from nudge to block.
"""

from __future__ import annotations

import hashlib
import json
from collections import deque
from dataclasses import dataclass, field

from coding_guardrails.rules.base import Action, RuleResult, ToolCall


def _call_fingerprint(call: ToolCall) -> str:
    """Stable hash of tool name + args for duplicate detection."""
    payload = json.dumps({"tool": call.tool, "args": call.args}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


@dataclass
class LoopDetectionRule:
    """Detect and break agent loops.

    Tracks the last N tool call fingerprints. If the same fingerprint
    appears repeatedly, nudges then blocks to break the loop.

    Attributes:
        window: Number of recent calls to track.
        nudge_threshold: Calls before nudging.
        block_threshold: Calls before blocking.
    """

    window: int = 10
    nudge_threshold: int = 3
    block_threshold: int = 5

    _history: deque = field(default_factory=lambda: deque(maxlen=10), repr=False)

    def __post_init__(self) -> None:
        # Sync deque maxlen with window
        object.__setattr__(self, "_history", deque(maxlen=self.window))

    @property
    def name(self) -> str:
        return "loop_detection"

    def check(self, call: ToolCall) -> RuleResult:
        fp = _call_fingerprint(call)

        # Count occurrences of this fingerprint in recent history
        count = sum(1 for h in self._history if h == fp)

        # Don't count the current call — just history
        # count=2 means "this would be the 3rd identical call"
        if count >= self.block_threshold - 1:
            return RuleResult.block(
                call.tool,
                nudge=f"You've called {call.tool} with the same arguments "
                f"{count + 1} times. This isn't working — try a different approach.",
                reason=f"loop detected: {call.tool} repeated {count + 1}x",
            )

        if count >= self.nudge_threshold - 1:
            return RuleResult.nudge(
                call.tool,
                message=f"You've tried {call.tool} {count + 1} times with "
                "the same arguments. Consider trying a different approach.",
            )

        return RuleResult.allow(call.tool)

    def record(self, calls: list[ToolCall]) -> None:
        """Record executed calls for loop tracking."""
        for call in calls:
            self._history.append(_call_fingerprint(call))
