"""Tests for the prerequisites rule."""

import pytest

from coding_guardrails.rules.base import Action, ToolCall
from coding_guardrails.rules.prerequisites import PrerequisiteRule


@pytest.fixture
def rule():
    return PrerequisiteRule()


@pytest.fixture
def rule_strict():
    return PrerequisiteRule(max_violations=1)


# ── Prefix matching ──────────────────────────────────────────────────────────

class TestToolMatching:
    """Verify prefix matching works for all agent tool naming conventions."""

    @pytest.mark.parametrize("tool_name", [
        "edit", "edit_file", "Edit", "EditFile", "EDITOR",
        "write", "write_file", "Write", "WRITE_FILE",
        "create", "create_file", "Create",
    ])
    def test_edit_tools_match(self, rule, tool_name):
        call = ToolCall(tool=tool_name, args={"path": "src/main.py"})
        result = rule.check(call)
        assert result.action in (Action.BLOCK, Action.NUDGE)

    @pytest.mark.parametrize("tool_name", [
        "read", "read_file", "Read", "READ_FILE",
        "cat", "head", "tail", "less",
    ])
    def test_read_tools_record(self, rule, tool_name):
        call = ToolCall(tool=tool_name, args={"path": "src/main.py"})
        result = rule.check(call)
        assert result.action == Action.ALLOW
        rule.record([call])
        edit = ToolCall(tool="edit", args={"path": "src/main.py"})
        assert rule.check(edit).action == Action.ALLOW

    @pytest.mark.parametrize("tool_name", [
        "bash", "shell", "run", "exec", "command",
        "grep", "find", "ls",
    ])
    def test_unrelated_tools_pass(self, rule, tool_name):
        call = ToolCall(tool=tool_name, args={"command": "ls"})
        assert rule.check(call).action == Action.ALLOW


# ── Read-before-edit enforcement ─────────────────────────────────────────────

class TestReadBeforeEdit:

    def test_edit_without_read_nudges(self, rule):
        call = ToolCall(tool="edit", args={"path": "src/main.py"})
        result = rule.check(call)
        assert result.action == Action.NUDGE

    def test_edit_without_read_twice_blocks(self, rule):
        call = ToolCall(tool="edit", args={"path": "src/main.py"})
        rule.check(call)
        result = rule.check(call)
        assert result.action == Action.BLOCK

    def test_edit_after_read_allowed(self, rule):
        read_call = ToolCall(tool="read", args={"path": "src/main.py"})
        rule.check(read_call)
        rule.record([read_call])
        edit_call = ToolCall(tool="edit", args={"path": "src/main.py"})
        assert rule.check(edit_call).action == Action.ALLOW

    def test_edit_after_read_different_path_nudges(self, rule):
        read_call = ToolCall(tool="read", args={"path": "src/main.py"})
        rule.check(read_call)
        rule.record([read_call])
        edit_call = ToolCall(tool="edit", args={"path": "src/other.py"})
        assert rule.check(edit_call).action == Action.NUDGE

    def test_strict_blocks_immediately(self, rule_strict):
        call = ToolCall(tool="edit", args={"path": "src/main.py"})
        result = rule_strict.check(call)
        assert result.action == Action.BLOCK

    def test_path_normalization(self, rule):
        read_call = ToolCall(tool="read", args={"path": "src/main.py/"})
        rule.check(read_call)
        rule.record([read_call])
        edit_call = ToolCall(tool="edit", args={"path": "src/main.py"})
        assert rule.check(edit_call).action == Action.ALLOW

    def test_violation_counter_resets_on_read(self, rule):
        edit_call = ToolCall(tool="edit", args={"path": "src/main.py"})
        rule.check(edit_call)
        read_call = ToolCall(tool="read", args={"path": "src/main.py"})
        rule.check(read_call)
        rule.record([read_call])
        edit2 = ToolCall(tool="edit", args={"path": "src/other.py"})
        result = rule.check(edit2)
        assert result.action == Action.NUDGE


# ── Smart path matching ──────────────────────────────────────────────────────

class TestSmartPathMatching:

    def test_directory_read_satisfies_child_file(self, rule):
        """Reading src/ should satisfy edit of src/main.py."""
        read_call = ToolCall(tool="read", args={"path": "src/"})
        rule.check(read_call)
        rule.record([read_call])
        # Manually add as directory (os.path.isdir won't work for non-existent paths)
        rule._read_dirs.add("src")
        edit_call = ToolCall(tool="edit", args={"path": "src/main.py"})
        assert rule.check(edit_call).action == Action.ALLOW

    def test_parent_read_satisfies_nested_file(self, rule):
        """Reading the root should satisfy any child edit."""
        rule._read_dirs.add(".")
        edit_call = ToolCall(tool="edit", args={"path": "src/deep/nested/file.py"})
        assert rule.check(edit_call).action == Action.ALLOW

    def test_sibling_file_not_satisfied(self, rule):
        """Reading src/main.py should NOT satisfy edit of src/other.py."""
        read_call = ToolCall(tool="read", args={"path": "src/main.py"})
        rule.check(read_call)
        rule.record([read_call])
        edit_call = ToolCall(tool="edit", args={"path": "src/other.py"})
        assert rule.check(edit_call).action == Action.NUDGE
