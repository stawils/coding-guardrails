"""Tests for the duplicate write detection rule."""

import pytest

from coding_guardrails.rules.base import Action, ToolCall
from coding_guardrails.rules.dup_write import DuplicateWriteRule


@pytest.fixture
def rule():
    return DuplicateWriteRule(nudge_threshold=2, block_threshold=3)


class TestNonWriteTools:

    def test_bash_allowed(self, rule):
        call = ToolCall(tool="bash", args={"command": "echo hi"})
        result = rule.check(call)
        assert result.action == Action.ALLOW

    def test_read_allowed(self, rule):
        call = ToolCall(tool="read", args={"path": "file.py"})
        result = rule.check(call)
        assert result.action == Action.ALLOW

    def test_edit_allowed(self, rule):
        """edit tool is intentionally ignored — legitimate edits re-touch files."""
        call = ToolCall(tool="edit", args={"path": "file.py", "content": "new"})
        result = rule.check(call)
        assert result.action == Action.ALLOW


class TestFirstWriteAllowed:

    def test_first_write_allowed(self, rule):
        """First write to a path is always allowed."""
        call = ToolCall(tool="write", args={"path": "/tmp/output.txt", "content": "hello"})
        result = rule.check(call)
        assert result.action == Action.ALLOW
        rule.record([call])


class TestDuplicateDetection:

    def test_second_identical_write_nudge(self, rule):
        """2nd identical write to the same path → NUDGE."""
        call1 = ToolCall(tool="write", args={"path": "/tmp/output.txt", "content": "hello"})
        rule.check(call1)
        rule.record([call1])

        call2 = ToolCall(tool="write", args={"path": "/tmp/output.txt", "content": "hello"})
        result = rule.check(call2)
        assert result.action == Action.NUDGE
        assert "output.txt" in result.nudge

    def test_third_identical_write_blocks(self, rule):
        """3rd identical write to the same path → BLOCK."""
        call1 = ToolCall(tool="write", args={"path": "/tmp/output.txt", "content": "hello"})
        rule.check(call1)
        rule.record([call1])

        call2 = ToolCall(tool="write", args={"path": "/tmp/output.txt", "content": "hello"})
        rule.check(call2)
        rule.record([call2])

        call3 = ToolCall(tool="write", args={"path": "/tmp/output.txt", "content": "hello"})
        result = rule.check(call3)
        assert result.action == Action.BLOCK
        assert "output.txt" in result.nudge


class TestContentReset:

    def test_different_content_resets_counter(self, rule):
        """Different content resets the per-path counter."""
        call_a = ToolCall(tool="write", args={"path": "/tmp/output.txt", "content": "version A"})
        call_b = ToolCall(tool="write", args={"path": "/tmp/output.txt", "content": "version B"})

        rule.check(call_a)
        rule.record([call_a])

        # Identical to A → nudge
        result = rule.check(call_a)
        assert result.action == Action.NUDGE
        rule.record([call_a])

        # Different content → allowed, resets the consecutive-duplicate counter.
        # dup_write detects CONSECUTIVE identical writes: after B is the last
        # write, A differs from it, so this is allowed (not a re-flag of an
        # earlier value — that would be an over-aggressive "any prior content"
        # rule that false-positives on legitimate v1→v2→v1 reverts).
        result = rule.check(call_b)
        assert result.action == Action.ALLOW
        rule.record([call_b])

        # A again now differs from the last write (B) → allowed, and a second
        # consecutive A would nudge (counter restarts at the new content).
        result = rule.check(call_a)
        assert result.action == Action.ALLOW
        rule.record([call_a])

        # Second consecutive A → nudge (counter reset to 1 by the prior record).
        result = rule.check(call_a)
        assert result.action == Action.NUDGE


class TestEditIgnored:

    def test_edit_with_same_content_not_triggered(self, rule):
        """edit tool with same path and content must NOT trigger dup_write."""
        # First write — allowed
        call1 = ToolCall(tool="write", args={"path": "/tmp/file.py", "content": "hello"})
        rule.check(call1)
        rule.record([call1])

        # edit with same content — should be allowed (edit is excluded from write_tools)
        edit_call = ToolCall(tool="edit", args={"path": "/tmp/file.py", "content": "hello"})
        result = rule.check(edit_call)
        assert result.action == Action.ALLOW

    def test_create_tool_detected(self, rule):
        """'create' tool is a write tool — should be tracked."""
        call1 = ToolCall(tool="create", args={"path": "/tmp/file.txt", "content": "data"})
        rule.check(call1)
        rule.record([call1])

        call2 = ToolCall(tool="create", args={"path": "/tmp/file.txt", "content": "data"})
        result = rule.check(call2)
        assert result.action == Action.NUDGE


class TestEdgeCases:

    def test_missing_path_ignored(self, rule):
        """Write without a path arg → allowed."""
        call = ToolCall(tool="write", args={"content": "hello"})
        result = rule.check(call)
        assert result.action == Action.ALLOW

    def test_missing_content_ignored(self, rule):
        """Write without content arg → allowed."""
        call = ToolCall(tool="write", args={"path": "/tmp/file.txt"})
        result = rule.check(call)
        assert result.action == Action.ALLOW

    def test_different_path_not_confused(self, rule):
        """Writes to different paths are independent."""
        call1 = ToolCall(tool="write", args={"path": "/tmp/file1.txt", "content": "data"})
        call2 = ToolCall(tool="write", args={"path": "/tmp/file2.txt", "content": "data"})

        rule.check(call1)
        rule.record([call1])

        # Same content, different path → allowed
        result = rule.check(call2)
        assert result.action == Action.ALLOW

    def test_different_content_same_path_not_duplicate(self, rule):
        """Same path, different content → not a duplicate."""
        call1 = ToolCall(tool="write", args={"path": "/tmp/file.txt", "content": "v1"})
        call2 = ToolCall(tool="write", args={"path": "/tmp/file.txt", "content": "v2"})

        rule.check(call1)
        rule.record([call1])

        result = rule.check(call2)
        assert result.action == Action.ALLOW


class TestConfigWiring:

    def test_custom_thresholds(self, rule):
        """Custom nudge_threshold and block_threshold are respected."""
        custom = DuplicateWriteRule(nudge_threshold=4, block_threshold=7)
        assert custom.nudge_threshold == 4
        assert custom.block_threshold == 7

        # With nudge at 4 and block at 7, the 4th consecutive identical write
        # should nudge. Use `custom` (the configured rule) — not the fixture.
        calls = []
        for i in range(4):
            calls.append(ToolCall(tool="write", args={"path": "/tmp/f.txt", "content": "x"}))

        for call in calls:
            custom.check(call)
            custom.record([call])

        result = custom.check(calls[0])
        assert result.action == Action.NUDGE


class TestRuleName:

    def test_name_property(self, rule):
        assert rule.name == "dup_write"
