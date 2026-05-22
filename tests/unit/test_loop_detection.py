"""Tests for the loop detection rule."""

import pytest

from coding_guardrails.rules.base import Action, ToolCall
from coding_guardrails.rules.loop_detection import LoopDetectionRule


@pytest.fixture
def rule():
    return LoopDetectionRule(window=10, nudge_threshold=3, block_threshold=5)


def _call(tool="bash", command="pytest"):
    return ToolCall(tool=tool, args={"command": command})


class TestNoLoop:

    def test_first_call_allowed(self, rule):
        result = rule.check(_call())
        assert result.action == Action.ALLOW

    def test_different_calls_allowed(self, rule):
        for cmd in ["pytest", "ls", "cat file.py", "echo hi"]:
            call = ToolCall(tool="bash", args={"command": cmd})
            result = rule.check(call)
            assert result.action == Action.ALLOW
            rule.record([call])


class TestLoopDetection:

    def test_nudge_on_repeat(self, rule):
        call = _call()
        # Record 2 prior calls, check 3rd
        rule.record([call])
        rule.record([call])
        result = rule.check(call)
        assert result.action == Action.NUDGE
        assert "3" in result.nudge or "same" in result.nudge.lower()

    def test_block_on_deep_loop(self, rule):
        call = _call()
        # Record 4 prior calls, check 5th
        for _ in range(4):
            rule.record([call])
        result = rule.check(call)
        assert result.action == Action.BLOCK

    def test_loop_clears_on_different_call(self, rule):
        call = _call()
        different = ToolCall(tool="bash", args={"command": "ls -la"})
        # Record 3 same calls
        for _ in range(3):
            rule.record([call])
        # Different call pushes old ones out (window=10, so they stay)
        # But the different call isn't the same fingerprint
        result = rule.check(different)
        assert result.action == Action.ALLOW

    def test_window_expiry(self):
        rule = LoopDetectionRule(window=3, nudge_threshold=3, block_threshold=5)
        call = _call()
        # Record 3 calls — fills window
        for _ in range(3):
            rule.record([call])
        # Add 2 different calls — old same-calls should roll off
        for cmd in ["ls", "pwd"]:
            different = ToolCall(tool="bash", args={"command": cmd})
            rule.record([different])
        # Now the window only has 2 different calls, original is gone
        result = rule.check(call)
        assert result.action == Action.ALLOW


class TestToolNames:

    def test_any_tool_tracked(self):
        rule = LoopDetectionRule(nudge_threshold=2)
        call = ToolCall(tool="read", args={"path": "src/main.py"})
        rule.record([call])
        result = rule.check(call)
        assert result.action == Action.NUDGE
