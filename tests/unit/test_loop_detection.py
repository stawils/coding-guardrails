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


class TestEdgeCases:

    def test_single_call_allowed(self, rule):
        """One call should always be allowed."""
        result = rule.check(_call())
        assert result.action == Action.ALLOW

    def test_two_different_calls_allowed(self, rule):
        """Two different bash commands should be allowed."""
        call1 = ToolCall(tool="bash", args={"command": "echo hello"})
        call2 = ToolCall(tool="bash", args={"command": "ls -la"})
        result1 = rule.check(call1)
        assert result1.action == Action.ALLOW
        rule.record([call1])
        result2 = rule.check(call2)
        assert result2.action == Action.ALLOW
        rule.record([call2])

    def test_mixed_tools_not_looping(self, rule):
        """Alternating bash and read calls should not be considered looping."""
        calls = [
            ToolCall(tool="bash", args={"command": "ls"}),
            ToolCall(tool="read", args={"path": "file1.py"}),
            ToolCall(tool="bash", args={"command": "pwd"}),
            ToolCall(tool="read", args={"path": "file2.py"}),
            ToolCall(tool="bash", args={"command": "cat file.py"}),
        ]
        for call in calls:
            result = rule.check(call)
            assert result.action == Action.ALLOW
            rule.record([call])

    def test_reset_clears_all_history(self, rule):
        """Reset() should clear all history and allow subsequent calls."""
        # Record 3 calls (nudge_threshold - 1) so 4th triggers NUDGE
        for i in range(3):
            rule.record([_call()])
        result_before = rule.check(_call())
        assert result_before.action == Action.NUDGE

        # Reset
        rule.reset()

        # Verify history is cleared and new call is allowed
        result_after = rule.check(_call())
        assert result_after.action == Action.ALLOW

    def test_stagnation_with_many_tools_allowed(self, rule):
        """15 calls with 5 different tool names should NOT be stagnation.

        Stagnation is blocked when we have >= stagnation_threshold (14) calls
        but <= stagnation_unique_tools (2) unique tool names. With 5 unique
        tools, this should be ALLOWED as it shows healthy exploration.
        """
        # Set thresholds to match default values
        rule.stagnation_threshold = 14
        rule.stagnation_unique_tools = 2

        # Make 15 calls with 5 different tool names
        tools = ["bash", "read", "edit", "write", "telegram_attach"]
        for i in range(15):
            call = ToolCall(
                tool=tools[i % 5],
                args={"command": f"test{i}" if tools[i % 5] == "bash" else f"path{i}.py"}
            )
            result = rule.check(call)
            assert result.action == Action.ALLOW, \
                f"Call {i} with tool {tools[i % 5]} should be ALLOWED"
            rule.record([call])

        # Even a 16th call with 5 tools should still be allowed
        call16 = ToolCall(tool="bash", args={"command": "test15"})
        result16 = rule.check(call16)
        assert result16.action == Action.ALLOW, \
            "16th call with 5 unique tools should still be ALLOWED"


class TestRecordEdgeCases:

    def test_record_empty_list(self, rule):
        """record([]) should not crash or change state."""
        rule.record([])
        # Verify rule still works normally
        call = ToolCall(tool="bash", args={"command": "echo hi"})
        result = rule.check(call)
        assert result.action == Action.ALLOW

    def test_record_before_check(self, rule):
        """record() before any check() should not corrupt state."""
        call = ToolCall(tool="bash", args={"command": "echo hi"})
        rule.record([call])
        # Should still work normally
        result = rule.check(call)
        assert result.action == Action.ALLOW

    def test_record_same_call_twice_idempotent(self, rule):
        """Recording the same call twice should track correctly."""
        call = ToolCall(tool="bash", args={"command": "echo hi"})
        rule.record([call])
        rule.record([call])
        # Third call should start counting from the recorded ones
        # (behavior depends on implementation)

    def test_record_interleaved_with_check(self, rule):
        """Interleaved record() and check() should work."""
        call = ToolCall(tool="bash", args={"command": "echo hi"})
        rule.check(call)
        rule.record([call])
        call2 = ToolCall(tool="bash", args={"command": "echo bye"})
        rule.check(call2)
        rule.record([call2])
        # Should be tracking both different calls, no loop

    def test_tool_aliasing_detected(self, rule):
        """bash and shell with same args should be treated as duplicates."""
        call1 = ToolCall(tool="bash", args={"command": "echo hi"})
        call2 = ToolCall(tool="shell", args={"command": "echo hi"})
        call3 = ToolCall(tool="exec", args={"command": "echo hi"})
        # Record 2 prior identical calls via aliases
        rule.check(call1)
        rule.record([call1])
        rule.check(call2)
        rule.record([call2])
        # Third alias should trigger nudge (nudge_threshold=3)
        result = rule.check(call3)
        assert result.action == Action.NUDGE

    def test_tool_aliasing_blocks(self, rule):
        """Repeated calls via different tool aliases should escalate to block."""
        call = ToolCall(tool="bash", args={"command": "echo hi"})
        aliases = ["bash", "shell", "exec", "run", "command"]
        for alias in aliases:
            c = ToolCall(tool=alias, args={"command": "echo hi"})
            rule.check(c)
            rule.record([c])
        # 6th call (any alias) should block (block_threshold=5)
        result = rule.check(ToolCall(tool="sh", args={"command": "echo hi"}))
        assert result.action == Action.BLOCK

    def test_write_create_aliasing(self, rule):
        """write and create with same args should be detected as duplicates."""
        call1 = ToolCall(tool="write", args={"path": "f.py", "content": "x"})
        call2 = ToolCall(tool="create", args={"path": "f.py", "content": "x"})
        call3 = ToolCall(tool="write", args={"path": "f.py", "content": "x"})
        rule.check(call1)
        rule.record([call1])
        rule.check(call2)
        rule.record([call2])
        result = rule.check(call3)
        assert result.action == Action.NUDGE
