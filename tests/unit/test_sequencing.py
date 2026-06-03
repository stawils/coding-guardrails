"""Tests for the sequencing rule."""

import pytest

from coding_guardrails.rules.base import Action, ToolCall
from coding_guardrails.rules.sequencing import SequenceRule


@pytest.fixture
def rule():
    return SequenceRule()


@pytest.fixture
def rule_hard():
    return SequenceRule(strength="hard")


class TestToolMatching:
    """Verify prefix matching works for all agent tool naming conventions."""

    @pytest.mark.parametrize("tool_name", [
        "edit", "edit_file", "Edit", "write", "write_file", "Write",
        "create", "create_file",
    ])
    def test_edit_triggers(self, rule, tool_name):
        call = ToolCall(tool=tool_name, args={"path": "src/main.py"})
        result = rule.check(call)
        assert result.action == Action.ALLOW  # soft mode: allow but track

    @pytest.mark.parametrize("tool_name", [
        "bash", "shell", "run", "exec", "Execute", "RUN_COMMAND",
    ])
    def test_bash_satisfies(self, rule, tool_name):
        edit = ToolCall(tool="edit", args={"path": "src/main.py"})
        rule.check(edit)
        bash = ToolCall(tool=tool_name, args={"command": "pytest"})
        result = rule.check(bash)
        assert result.action == Action.ALLOW
        assert rule._pending is False  # suggestion cleared


class TestSoftNudge:

    def test_no_nudge_immediately_after_edit(self, rule):
        edit = ToolCall(tool="edit", args={"path": "src/main.py"})
        rule.check(edit)
        # Next call that isn't bash should pass (cooldown hasn't elapsed)
        read = ToolCall(tool="read", args={"path": "src/other.py"})
        result = rule.check(read)
        assert result.action == Action.ALLOW

    def test_nudge_after_cooldown(self, rule):
        edit = ToolCall(tool="edit", args={"path": "src/main.py"})
        rule.check(edit)
        # Burn through cooldown
        for i in range(rule.cooldown):
            read = ToolCall(tool="read", args={"path": f"file{i}.py"})
            result = rule.check(read)
        # Should nudge now
        assert result.action == Action.NUDGE
        assert "test" in result.nudge.lower()

    def test_bash_clears_pending(self, rule):
        edit = ToolCall(tool="edit", args={"path": "src/main.py"})
        rule.check(edit)
        bash = ToolCall(tool="bash", args={"command": "pytest"})
        rule.check(bash)
        # No pending anymore — further reads should not nudge
        read = ToolCall(tool="read", args={"path": "x.py"})
        for _ in range(rule.cooldown + 1):
            result = rule.check(read)
        assert result.action == Action.ALLOW


class TestHardMode:

    def test_hard_mode_blocks(self, rule_hard):
        edit = ToolCall(tool="edit", args={"path": "src/main.py"})
        result = rule_hard.check(edit)
        assert result.action == Action.BLOCK

    def test_record_clears_pending(self, rule):
        edit = ToolCall(tool="edit", args={"path": "src/main.py"})
        rule.check(edit)
        bash = ToolCall(tool="bash", args={"command": "pytest"})
        rule.record([bash])
        assert rule._pending is False


# ── Edge cases ──────────────────────────────────────────────────────────────

class TestEdgeCases:
    """Edge case tests for the sequencing rule."""

    def test_empty_args(self, rule):
        """Tool call with empty args should be ALLOWED."""
        call = ToolCall(tool="bash", args={})
        result = rule.check(call)
        assert result.action == Action.ALLOW

    def test_non_trigger_tool(self, rule):
        """Non-bash tool call after cooldown fires should be ALLOWED (cooldown prevents continuous nudging)."""
        edit = ToolCall(tool="edit", args={"path": "src/main.py"})
        rule.check(edit)
        # Exhaust cooldown period (cooldown=3, so need 3 reads to trigger nudge)
        for i in range(rule.cooldown):
            read = ToolCall(tool="read", args={"path": f"file{i}.py"})
            rule.check(read)
        # After the last read in the loop, cooldown fired and reset counter to 0
        # The next non-bash tool call is in the cooldown period but counter is at 1, so ALLOW
        result = rule.check(read)  # 4th call in cooldown period
        assert result.action == Action.ALLOW

    def test_non_matching_tool(self, rule):
        """Tool call for unknown_tool should be ALLOWED."""
        call = ToolCall(tool="unknown_tool", args={"some": "arg"})
        result = rule.check(call)
        assert result.action == Action.ALLOW

    def test_empty_string_command(self, rule):
        """Tool call with empty string command should be ALLOWED."""
        call = ToolCall(tool="bash", args={"command": ""})
        result = rule.check(call)
        assert result.action == Action.ALLOW
