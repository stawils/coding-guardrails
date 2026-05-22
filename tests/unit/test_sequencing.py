"""Tests for the sequencing rule (test-after-change nudges)."""

from coding_guardrails.rules.base import Action, ToolCall
from coding_guardrails.rules.sequencing import SequenceRule


def test_edit_triggers_allow_with_pending():
    rule = SequenceRule()
    call = ToolCall(tool="edit_file", args={"path": "/home/user/main.py"})
    result = rule.check(call)
    # Edit is allowed, but a suggestion is pending
    assert result.action == Action.ALLOW


def test_bash_after_edit_resolves_suggestion():
    rule = SequenceRule()

    # Edit triggers pending suggestion
    rule.check(ToolCall(tool="edit_file", args={"path": "/home/user/main.py"}))

    # Running tests resolves it
    rule.record([ToolCall(tool="bash", args={"command": "pytest"})])

    # No more pending suggestion
    call = ToolCall(tool="read_file", args={"path": "/home/user/other.py"})
    result = rule.check(call)
    assert result.action == Action.ALLOW


def test_hard_strength_blocks():
    rule = SequenceRule(rules=[
        {
            "trigger": "edit_file",
            "suggest": "bash",
            "strength": "hard",
            "nudge": "Run tests after editing.",
        },
    ])
    call = ToolCall(tool="edit_file", args={"path": "/home/user/main.py"})
    result = rule.check(call)
    assert result.action == Action.BLOCK


def test_soft_strength_allows():
    rule = SequenceRule(rules=[
        {
            "trigger": "edit_file",
            "suggest": "bash",
            "strength": "soft",
            "nudge": "Consider running tests.",
        },
    ])
    call = ToolCall(tool="edit_file", args={"path": "/home/user/main.py"})
    result = rule.check(call)
    assert result.action == Action.ALLOW


def test_cooldown_nudge():
    rule = SequenceRule(cooldown=2)

    # Edit triggers pending
    rule.check(ToolCall(tool="edit_file", args={"path": "/home/user/main.py"}))

    # First call after edit: no nudge yet (cooldown not reached)
    result = rule.check(ToolCall(tool="read_file", args={"path": "/home/user/test.py"}))
    assert result.action == Action.ALLOW

    # Second call: cooldown reached, should nudge
    result = rule.check(ToolCall(tool="read_file", args={"path": "/home/user/other.py"}))
    assert result.action == Action.NUDGE
    assert "test" in result.nudge.lower()


def test_unrelated_tool_no_effect():
    rule = SequenceRule()
    result = rule.check(ToolCall(tool="bash", args={"command": "ls"}))
    assert result.action == Action.ALLOW
