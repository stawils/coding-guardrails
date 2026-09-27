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
        assert rule.file_op_count == 5


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
        assert r.file_op_count == 5
        # Reset counters
        r.reset()
        # Verify all counters are cleared
        assert r.file_op_count == 0
        assert r.command_count == 0
        assert r.read_count == 0


class TestRecordEdgeCases:

    def test_record_empty_list(self, rule):
        """record([]) should not change any counters."""
        rule.record([])
        assert rule.file_op_count == 0
        assert rule.command_count == 0

    def test_record_before_check(self, rule):
        """record() before check() should increment counters."""
        call = ToolCall(tool="bash", args={"command": "echo hi"})
        rule.record([call])
        assert rule.command_count == 1

    def test_record_same_call_twice(self, rule):
        """Recording same call twice should count twice."""
        call = ToolCall(tool="edit", args={"path": "f.py"})
        rule.record([call])
        rule.record([call])
        assert rule.file_op_count == 2

    def test_record_interleaved_with_check(self, rule):
        """Interleaved record() and check() should maintain correct counts."""
        for i in range(5):
            call = ToolCall(tool="bash", args={"command": f"echo {i}"})
            rule.check(call)
            rule.record([call])
        assert rule.command_count == 5


class TestLongHorizonDefaults:

    def test_defaults_are_generous(self):
        """Defaults are a runaway backstop, not a task budget."""
        r = SessionBudgetRule()
        assert r.max_file_ops >= 500
        assert r.max_commands >= 1000
        assert r.max_reads == 0  # unlimited

    def test_heavy_session_survives_past_old_cap(self):
        """Regression for the reported symptom: ~100 ops used to hard-block
        the agent mid-task with \"Budget exhausted\"."""
        r = SessionBudgetRule()
        for i in range(150):
            call = ToolCall(tool="edit", args={"path": f"f{i}.py"})
            assert r.check(call).action != Action.BLOCK
            r.record([call])
        assert r.file_op_count == 150


class TestBatchOvershoot:

    def test_batch_does_not_overshoot_cap(self):
        """A batch of edits at cap-1 must yield exactly one allowed call.

        Regression: check() did not reserve budget, so N calls in one batch
        all passed and record() pushed the counter past the cap (102/100).
        """
        r = SessionBudgetRule(max_file_ops=100, max_commands=200)
        for _ in range(99):
            r.record([ToolCall(tool="edit", args={"path": "f.py"})])
        assert r.file_op_count == 99

        calls = [ToolCall(tool="edit", args={"path": f"f{i}.py"}) for i in range(3)]
        allowed = [c for c in calls if r.check(c).action != Action.BLOCK]
        assert len(allowed) == 1  # only the 100th op fits
        r.record(allowed)
        assert r.file_op_count == 100

        # Every further edit blocks with an exact count, never 101/100.
        blocked = r.check(ToolCall(tool="edit", args={"path": "next.py"}))
        assert blocked.action == Action.BLOCK
        assert "100/100" in blocked.nudge

    def test_begin_batch_drops_leaked_reservations(self):
        r = SessionBudgetRule(max_file_ops=2, warn_at=1.0)
        assert r.check(ToolCall(tool="edit", args={"path": "a.py"})).action == Action.ALLOW
        assert r.check(ToolCall(tool="edit", args={"path": "b.py"})).action == Action.ALLOW
        assert r.check(ToolCall(tool="edit", args={"path": "c.py"})).action == Action.BLOCK
        r.begin_batch()
        # Reservations dropped, real count still 0 -> allowed again.
        assert r.check(ToolCall(tool="edit", args={"path": "d.py"})).action == Action.ALLOW

    def test_reset_clears_pending_reservations(self):
        r = SessionBudgetRule(max_file_ops=1, warn_at=1.0)
        assert r.check(ToolCall(tool="edit", args={"path": "a.py"})).action == Action.ALLOW
        assert r.check(ToolCall(tool="edit", args={"path": "b.py"})).action == Action.BLOCK
        r.reset()
        assert r.check(ToolCall(tool="edit", args={"path": "c.py"})).action == Action.ALLOW


class TestBlockRecoveryPath:

    def test_block_message_tells_agent_how_to_finish(self):
        """Exhaustion must not be a dead end: the agent needs an exit tool."""
        r = SessionBudgetRule(max_file_ops=1)
        r.record([ToolCall(tool="edit", args={"path": "a.py"})])
        result = r.check(ToolCall(tool="edit", args={"path": "b.py"}))
        assert result.action == Action.BLOCK
        assert "respond()" in result.nudge
        assert "1/1" in result.nudge

    def test_command_block_has_recovery_path(self):
        r = SessionBudgetRule(max_commands=1)
        r.record([ToolCall(tool="bash", args={"command": "ls"})])
        result = r.check(ToolCall(tool="bash", args={"command": "pwd"}))
        assert result.action == Action.BLOCK
        assert "respond()" in result.nudge
