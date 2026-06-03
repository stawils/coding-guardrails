"""Tests for the session budget rule."""

import pytest

from coding_guardrails.rules.base import Action, ToolCall
from coding_guardrails.rules.session_budget import SessionBudgetRule


@pytest.fixture
def rule():
    return SessionBudgetRule(max_file_ops=10, max_commands=20, warn_at=0.8)


class TestFileOps:

    def test_file_ops_allowed_under_limit(self, rule):
        for i in range(8):  # 0-7 = below 80% threshold
            call = ToolCall(tool="edit", args={"path": "f.py"})
            result = rule.check(call)
            assert result.action == Action.ALLOW, f"call {i}: expected ALLOW"
            rule.record([call])
        # 8th = 80% threshold, should nudge
        call = ToolCall(tool="edit", args={"path": "f.py"})
        result = rule.check(call)
        assert result.action == Action.NUDGE
        rule.record([call])
        # 9th = still under limit, but already warned
        call = ToolCall(tool="edit", args={"path": "f.py"})
        result = rule.check(call)
        assert result.action == Action.ALLOW
        rule.record([call])

    def test_file_ops_warn_at_threshold(self, rule):
        for _ in range(8):
            call = ToolCall(tool="edit", args={"path": "f.py"})
            rule.check(call)
            rule.record([call])
        # 9th call = 80%+ of 10
        call = ToolCall(tool="edit", args={"path": "f.py"})
        result = rule.check(call)
        assert result.action == Action.NUDGE

    def test_file_ops_block_at_limit(self, rule):
        for _ in range(10):
            call = ToolCall(tool="edit", args={"path": "f.py"})
            rule.check(call)
            rule.record([call])
        # 11th call should be blocked
        call = ToolCall(tool="edit", args={"path": "f.py"})
        result = rule.check(call)
        assert result.action == Action.BLOCK

    def test_different_edit_tools_count(self, rule):
        for tool in ["edit", "write", "create", "edit_file", "Write"]:
            call = ToolCall(tool=tool, args={"path": "f.py"})
            rule.check(call)
            rule.record([call])
        assert rule._file_ops == 5


class TestCommands:

    def test_commands_allowed_under_limit(self, rule):
        for _ in range(15):
            call = ToolCall(tool="bash", args={"command": "ls"})
            result = rule.check(call)
            assert result.action == Action.ALLOW
            rule.record([call])

    def test_commands_block_at_limit(self, rule):
        for _ in range(20):
            call = ToolCall(tool="bash", args={"command": "ls"})
            rule.check(call)
            rule.record([call])
        call = ToolCall(tool="bash", args={"command": "ls"})
        result = rule.check(call)
        assert result.action == Action.BLOCK


class TestReads:

    def test_reads_unlimited_by_default(self, rule):
        for _ in range(200):
            call = ToolCall(tool="read", args={"path": "f.py"})
            result = rule.check(call)
            assert result.action == Action.ALLOW
            rule.record([call])

    def test_reads_limited_when_configured(self):
        r = SessionBudgetRule(max_reads=5, warn_at=0.8)
        for _ in range(5):
            call = ToolCall(tool="read", args={"path": "f.py"})
            r.check(call)
            r.record([call])
        call = ToolCall(tool="read", args={"path": "f.py"})
        result = r.check(call)
        assert result.action == Action.BLOCK


class TestUnrelatedTools:

    def test_unrelated_tools_always_pass(self, rule):
        for tool in ["grep", "find", "respond", "unknown"]:
            call = ToolCall(tool=tool, args={})
            assert rule.check(call).action == Action.ALLOW


class TestEdgeCases:

    def test_first_call_allowed(self, rule):
        """Very first call should be allowed."""
        # Create a fresh rule instance with higher limits to avoid integer truncation
        r = SessionBudgetRule(max_file_ops=10, max_commands=10, max_reads=10, warn_at=0.8)
        call = ToolCall(tool="edit", args={"path": "f.py"})
        result = r.check(call)
        # First call (0 ops) should be allowed
        assert result.action == Action.ALLOW
        r.record([call])
        # Still under limit
        result = r.check(call)
        assert result.action == Action.ALLOW
        r.record([call])

    def test_non_tracked_tool_ignored(self, rule):
        """Tool name not in tracked list should be allowed."""
        r = SessionBudgetRule(max_file_ops=1, max_commands=1, max_reads=1)
        # These tools aren't tracked by session_budget
        for tool in ["grep", "find", "ls", "unknown_tool"]:
            call = ToolCall(tool=tool, args={})
            result = r.check(call)
            assert result.action == Action.ALLOW, f"Tool {tool} should be allowed"
            r.record([call])

    def test_exactly_at_limit(self, rule):
        """Calls right at the limit should check behavior."""
        r = SessionBudgetRule(max_file_ops=2, max_commands=2, warn_at=0.5)
        # First call (0 ops, 0%)
        call = ToolCall(tool="edit", args={"path": "f.py"})
        result = r.check(call)
        # Should allow
        assert result.action == Action.ALLOW
        r.record([call])
        # Second call (1 op, 50% = at warn threshold)
        result = r.check(call)
        # Should nudge at exactly 50% threshold
        assert result.action == Action.NUDGE
        r.record([call])
        # Third call (2 ops, 100% = at limit)
        result = r.check(call)
        # Should block at exactly 100%
        assert result.action == Action.BLOCK
        r.record([call])
        # Fourth call should still be blocked
        result = r.check(call)
        assert result.action == Action.BLOCK

    def test_reset_clears_count(self, rule):
        """Record calls, reset(), verify count is 0."""
        r = SessionBudgetRule(max_file_ops=10, max_commands=20, max_reads=15)
        # Make some calls
        for _ in range(5):
            call = ToolCall(tool="edit", args={"path": "f.py"})
            r.check(call)
            r.record([call])
        assert r._file_ops == 5
        # Reset counters
        r.reset()
        # Verify all counters are cleared
        assert r._file_ops == 0
        assert r._commands == 0
        assert r._reads == 0
