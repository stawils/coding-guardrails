"""Loop detection — detect when an agent is stuck repeating the same call.

Tracks recent tool calls and detects when the agent retries the same
operation multiple times without progress. Escalates from nudge to block.

Two detection modes:
1. **Exact match**: Same tool + same args fingerprint repeated. Nudges at
   nudge_threshold, blocks at block_threshold.
2. **Stagnation**: Recent window has too few unique tool names relative to
   total calls — the agent is cycling through the same few tools without
   making real progress, even if args differ slightly. Blocks at
   stagnation_threshold.
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

    Also detects stagnation — when the agent cycles through a small set
    of tools with different args but no real progress.

    Attributes:
        window: Number of recent calls to track.
        nudge_threshold: Identical calls before nudging.
        block_threshold: Identical calls before blocking.
        stagnation_threshold: Total calls in window before checking
            stagnation. If the window has >= this many calls but <=
            stagnation_unique_tools unique tool names, it's a stagnation
            loop. Default 14 — allows normal exploration.
        stagnation_unique_tools: Maximum unique tool names to consider
            a stagnation loop (default 2 — e.g. alternating between
            bash and telegram_attach).
    """

    window: int = 10
    nudge_threshold: int = 3
    block_threshold: int = 5
    stagnation_threshold: int = 14
    stagnation_unique_tools: int = 2

    _history: deque = field(default_factory=lambda: deque(maxlen=10), repr=False)
    _tool_history: deque = field(default_factory=lambda: deque(maxlen=10), repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_history", deque(maxlen=self.window))
        object.__setattr__(self, "_tool_history", deque(maxlen=self.window))

    @property
    def name(self) -> str:
        return "loop_detection"

    def check(self, call: ToolCall) -> RuleResult:
        fp = _call_fingerprint(call)

        # ── Check 1: Exact match (identical tool + args) ──
        count = sum(1 for h in self._history if h == fp)

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

        # ── Check 2: Stagnation (cycling same few tools, different args) ──
        # Look at what the history would be after this call is recorded.
        # We need the tool names in recent history + this call.
        recent_tools = list(self._tool_history) + [call.tool]
        if len(recent_tools) >= self.stagnation_threshold:
            unique_tools = len(set(recent_tools))
            if unique_tools <= self.stagnation_unique_tools:
                tool_names = ", ".join(sorted(set(recent_tools)))
                return RuleResult.block(
                    call.tool,
                    nudge=(
                        f"You're stuck in a loop cycling between "
                        f"[{tool_names}] with no progress. "
                        f"Step back, review what you've done, and try a "
                        f"completely different approach."
                    ),
                    reason=(
                        f"stagnation: {len(recent_tools)} calls with only "
                        f"{unique_tools} unique tools [{tool_names}]"
                    ),
                )

        return RuleResult.allow(call.tool)

    def record(self, calls: list[ToolCall]) -> None:
        """Record executed calls for loop tracking."""
        for call in calls:
            self._history.append(_call_fingerprint(call))
            self._tool_history.append(call.tool)

    def reset(self) -> None:
        """Clear loop detection history.

        Called when a new conversation starts to avoid cross-contamination
        between independent requests.
        """
        self._history.clear()
        self._tool_history.clear()
