"""Tests for the prerequisites rule (read-before-edit)."""

from coding_guardrails.rules.base import Action, ToolCall
from coding_guardrails.rules.prerequisites import PrerequisiteRule


def test_edit_without_read_is_blocked():
    rule = PrerequisiteRule()
    call = ToolCall(tool="edit_file", args={"path": "/home/user/main.py"})

    # First violation: nudge
    result = rule.check(call)
    assert result.action == Action.NUDGE
    assert "read" in result.nudge.lower()

    # Second violation: block (max_violations=2, so 2nd = block)
    result = rule.check(call)
    assert result.action == Action.BLOCK


def test_edit_after_read_is_allowed():
    rule = PrerequisiteRule()

    # Record that the file was read
    rule.record([ToolCall(tool="read_file", args={"path": "/home/user/main.py"})])

    # Now edit should be allowed
    result = rule.check(ToolCall(tool="edit_file", args={"path": "/home/user/main.py"}))
    assert result.action == Action.ALLOW


def test_different_files_tracked_separately():
    rule = PrerequisiteRule()

    # Read file A
    rule.record([ToolCall(tool="read_file", args={"path": "/home/user/a.py"})])

    # Edit file A: allowed
    result = rule.check(ToolCall(tool="edit_file", args={"path": "/home/user/a.py"}))
    assert result.action == Action.ALLOW

    # Edit file B (not read): nudge
    result = rule.check(ToolCall(tool="edit_file", args={"path": "/home/user/b.py"}))
    assert result.action == Action.NUDGE


def test_trailing_slash_normalization():
    rule = PrerequisiteRule()

    # Read with trailing slash
    rule.record([ToolCall(tool="read_file", args={"path": "/home/user/dir/"})])

    # Edit without trailing slash: should match
    result = rule.check(ToolCall(tool="edit_file", args={"path": "/home/user/dir"}))
    assert result.action == Action.ALLOW


def test_unrelated_tool_always_allowed():
    rule = PrerequisiteRule()
    result = rule.check(ToolCall(tool="bash", args={"command": "ls"}))
    assert result.action == Action.ALLOW


def test_violation_counter_resets_after_read():
    rule = PrerequisiteRule()

    # Two violations
    call = ToolCall(tool="edit_file", args={"path": "/home/user/main.py"})
    rule.check(call)  # nudge 1
    rule.check(call)  # nudge 2

    # Read the file — should reset
    rule.record([ToolCall(tool="read_file", args={"path": "/home/user/main.py"})])

    # Should be allowed now
    result = rule.check(ToolCall(tool="edit_file", args={"path": "/home/user/main.py"}))
    assert result.action == Action.ALLOW
