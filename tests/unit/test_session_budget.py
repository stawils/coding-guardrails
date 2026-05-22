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
