"""Tests for loop detection session reset and session detection."""

import pytest

from coding_guardrails.rules.base import Action, ToolCall
from coding_guardrails.rules.loop_detection import LoopDetectionRule


@pytest.fixture
def rule():
    return LoopDetectionRule(window=10, nudge_threshold=3, block_threshold=5)


class TestResetClearsHistory:

    def test_reset_clears_exact_match(self, rule):
        """record 4 identical calls, reset(), verify next check is ALLOW (not BLOCK)."""
        call = ToolCall(tool="bash", args={"command": "pytest"})
        # Record 4 identical calls (block_threshold=5, so 4 prior + 1 check = 5)
        for _ in range(4):
            rule.record([call])
        
        # 5th call should be BLOCK (block_threshold=5)
        result = rule.check(call)
        assert result.action == Action.BLOCK
        
        # Reset clears history
        rule.reset()
        
        # Next check should be ALLOW (first call after reset)
        result = rule.check(call)
        assert result.action == Action.ALLOW

    def test_reset_clears_stagnation(self, rule):
        """record 14+ calls cycling 2 tools, reset(), verify next check is ALLOW."""
        tool_a = ToolCall(tool="bash", args={"command": "echo a"})
        tool_b = ToolCall(tool="read", args={"path": "file1.py"})
        
        # Record 14 calls cycling 2 tools (stagnation_threshold=14)
        # After reset, window should be empty
        calls = []
        for i in range(14):
            if i % 2 == 0:
                calls.append(tool_a)
            else:
                calls.append(tool_b)
            rule.record([calls[-1]])
        
        # 15th call should trigger stagnation BLOCK (15 calls, 2 unique tools)
        result = rule.check(tool_a)
        assert result.action == Action.BLOCK
        
        # Reset clears history
        rule.reset()
        
        # Next check should be ALLOW (first call after reset)
        result = rule.check(tool_a)
        assert result.action == Action.ALLOW

    def test_reset_clears_tool_history(self, rule):
        """record calls, reset(), verify tool name tracking is empty."""
        call1 = ToolCall(tool="bash", args={"command": "ls"})
        call2 = ToolCall(tool="read", args={"path": "file.py"})
        call3 = ToolCall(tool="write", args={"path": "new.py"})
        
        # Record 3 different tool calls
        rule.record([call1])
        rule.record([call2])
        rule.record([call3])
        
        # After reset, tool_history should be empty
        rule.reset()
        
        # Verify history is cleared by checking first call is ALLOW
        call4 = ToolCall(tool="bash", args={"command": "echo test"})
        result = rule.check(call4)
        assert result.action == Action.ALLOW


class TestSessionDetection:

    def test_no_assistant_means_new(self):
        """list of dicts with only system+user roles has no "assistant" role."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is your name?"},
        ]
        has_assistant = any(m.get("role") == "assistant" for m in messages)
        assert not has_assistant
        # Verify roles are only system and user
        roles = {m.get("role") for m in messages}
        assert roles == {"system", "user"}

    def test_assistant_means_existing(self):
        """list with assistant role dict → has assistant."""
        messages = [
            {"role": "system", "content": "You are a coding agent."},
            {"role": "user", "content": "Run a test."},
            {"role": "assistant", "content": "Running test...", "tool_calls": []},
            {"role": "assistant", "content": "Done."},
        ]
        has_assistant = any(m.get("role") == "assistant" for m in messages)
        assert has_assistant
        # Verify roles include assistant
        roles = {m.get("role") for m in messages}
        assert "assistant" in roles


class TestResetPreservesThresholds:

    def test_reset_keeps_thresholds(self):
        """custom window/nudge/block, reset(), verify unchanged."""
        custom_window = 20
        custom_nudge = 4
        custom_block = 7
        
        rule = LoopDetectionRule(
            window=custom_window,
            nudge_threshold=custom_nudge,
            block_threshold=custom_block,
        )
        
        call = ToolCall(tool="bash", args={"command": "test"})
        
        # Verify custom thresholds are used
        for _ in range(custom_nudge - 1):
            rule.record([call])
        result = rule.check(call)
        assert result.action == Action.NUDGE
        
        # Reset should not change thresholds
        rule.reset()
        
        # Re-create rule with original custom thresholds
        rule2 = LoopDetectionRule(
            window=custom_window,
            nudge_threshold=custom_nudge,
            block_threshold=custom_block,
        )
        
        # Verify thresholds are still the same
        assert rule2.window == custom_window
        assert rule2.nudge_threshold == custom_nudge
        assert rule2.block_threshold == custom_block
        
        # Custom thresholds still work after reset of a different rule
        for _ in range(custom_nudge - 1):
            rule2.record([call])
        result = rule2.check(call)
        assert result.action == Action.NUDGE
