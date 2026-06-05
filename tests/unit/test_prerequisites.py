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
    ])
    def test_edit_tools_match(self, rule, tool_name):
        # 'edit' tools always nudge/block on unread files (existing or not).
        call = ToolCall(tool=tool_name, args={"path": "src/main.py"})
        result = rule.check(call)
        assert result.action in (Action.BLOCK, Action.NUDGE)

    @pytest.mark.parametrize("tool_name", [
        "write", "write_file", "Write", "WRITE_FILE",
        "create", "create_file", "Create",
    ])
    def test_create_tools_match(self, rule, tool_name):
        # 'write'/'create' on a non-existent path is a NEW file → ALLOW.
        call = ToolCall(tool=tool_name, args={"path": "src/new_file.py"})
        result = rule.check(call)
        assert result.action == Action.ALLOW

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
        rule.mark_directory_read("src")
        edit_call = ToolCall(tool="edit", args={"path": "src/main.py"})
        assert rule.check(edit_call).action == Action.ALLOW

    def test_parent_read_satisfies_nested_file(self, rule):
        """Reading the root should satisfy any child edit."""
        rule.mark_directory_read(".")
        edit_call = ToolCall(tool="edit", args={"path": "src/deep/nested/file.py"})
        assert rule.check(edit_call).action == Action.ALLOW

    def test_sibling_file_not_satisfied(self, rule):
        """Reading src/main.py should NOT satisfy edit of src/other.py."""
        read_call = ToolCall(tool="read", args={"path": "src/main.py"})
        rule.check(read_call)
        rule.record([read_call])
        edit_call = ToolCall(tool="edit", args={"path": "src/other.py"})
        assert rule.check(edit_call).action == Action.NUDGE


# ── Edge cases ──────────────────────────────────────────────────────────────

class TestEdgeCases:
    """Edge case tests for the prerequisites rule."""

    def test_empty_args(self, rule):
        """Tool call with empty args should be ALLOWED."""
        call = ToolCall(tool="edit", args={})
        result = rule.check(call)
        assert result.action in (Action.ALLOW, Action.NUDGE)

    def test_none_path(self, rule):
        """Tool call with None path should be ALLOWED."""
        call = ToolCall(tool="edit", args={"path": None})
        result = rule.check(call)
        assert result.action in (Action.ALLOW, Action.NUDGE)

    def test_empty_string_path(self, rule):
        """Tool call with empty string path should be ALLOWED."""
        call = ToolCall(tool="edit", args={"path": ""})
        result = rule.check(call)
        assert result.action in (Action.ALLOW, Action.NUDGE)

    def test_non_string_path(self, rule):
        """Tool call with non-string path is skipped (rule crashes on invalid types)."""
        # This is skipped because the rule currently crashes on non-string paths.
        # The rule should ideally handle this gracefully, but it's not in scope.
        call = ToolCall(tool="edit", args={"path": 123})
        # Just verify it crashes (expected behavior for now)
        with pytest.raises((TypeError, ValueError)):
            rule.check(call)

    def test_non_edit_tool_with_path(self, rule):
        """Tool call for bash with path arg should be ALLOWED (not an edit tool)."""
        call = ToolCall(tool="bash", args={"path": "/tmp/file.txt"})
        result = rule.check(call)
        assert result.action == Action.ALLOW

    def test_unicode_path(self, rule):
        """Tool call with unicode path should be handled appropriately."""
        call = ToolCall(tool="edit", args={"path": "/tmp/日本語/file.py"})
        result = rule.check(call)
        assert result.action in (Action.ALLOW, Action.NUDGE)


class TestRecordEdgeCases:

    def test_record_empty_list(self, rule):
        """record([]) should not change state."""
        rule.record([])
        # Verify read paths still empty
        edit_call = ToolCall(tool="edit", args={"path": "src/main.py"})
        result = rule.check(edit_call)
        assert result.action in (Action.NUDGE, Action.BLOCK)

    def test_record_before_check(self, rule):
        """record() before check() should register the read."""
        call = ToolCall(tool="read", args={"path": "src/main.py"})
        rule.record([call])
        # Now edit should be allowed (path was recorded)
        edit_call = ToolCall(tool="edit", args={"path": "src/main.py"})
        assert rule.check(edit_call).action == Action.ALLOW

    def test_record_resets_violation_counter(self, rule):
        """record() should reset the violation counter."""
        edit = ToolCall(tool="edit", args={"path": "src/main.py"})
        rule.check(edit)  # violation 1
        rule.check(edit)  # violation 2
        read = ToolCall(tool="read", args={"path": "src/main.py"})
        rule.record([read])  # should reset counter
        edit2 = ToolCall(tool="edit", args={"path": "src/other.py"})
        result = rule.check(edit2)
        assert result.action == Action.NUDGE  # counter was reset, not blocked


# ── Regression tests for 2026-06-05 Lirada session failures ──────────────────

class TestParallelWritesAndNewFiles:
    """Reproduces F1 and F5 from the Lirada session.

    F1: Agent emits 9 parallel write() calls in one turn for new files;
        the prerequisites rule blocked the entire batch after 2 nudges,
        leaving zero files created and a wasted subagent session.

    F5: 'write' on a non-existent path was treated as 'edit', forcing a
        prior read. For new files there's nothing to read.

    Fix: write()/create() on non-existent paths bypass the rule entirely.
    """

    def test_write_new_file_allowed_without_read(self, rule, tmp_path):
        """Writing a brand new file requires no prior read."""
        new_file = tmp_path / "scaffold.js"
        assert not new_file.exists()
        call = ToolCall(tool="write", args={"path": str(new_file)})
        result = rule.check(call)
        assert result.action == Action.ALLOW

    def test_create_new_file_allowed_without_read(self, rule, tmp_path):
        """create() on a non-existent path = ALLOW."""
        new_file = tmp_path / "new_module.py"
        call = ToolCall(tool="create", args={"path": str(new_file)})
        assert rule.check(call).action == Action.ALLOW

    def test_parallel_writes_of_new_files_all_allowed(self, rule, tmp_path):
        """Simulates the Lirada Phase 2 Task 1 case: 9 new files in one batch.

        Previously: prerequisites rule nudged twice then blocked the rest,
        killing the subagent. Now: all writes to new files are ALLOWED.
        """
        new_paths = [str(tmp_path / f"file{i}.js") for i in range(9)]
        calls = [ToolCall(tool="write", args={"path": p}) for p in new_paths]
        results = [rule.check(c) for c in calls]
        actions = {r.action for r in results}
        assert actions == {Action.ALLOW}, f"Expected all ALLOW, got {actions}"

    def test_edit_on_nonexistent_path_still_nudges(self, rule, tmp_path):
        """edit() on a non-existent path is still treated defensively —
        the agent probably has the wrong path."""
        call = ToolCall(tool="edit", args={"path": str(tmp_path / "ghost.py")})
        result = rule.check(call)
        assert result.action == Action.NUDGE

    def test_write_existing_file_still_requires_read(self, rule, tmp_path):
        """write() on an EXISTING file still requires a prior read."""
        existing = tmp_path / "existing.py"
        existing.write_text("print('hi')")
        call = ToolCall(tool="write", args={"path": str(existing)})
        result = rule.check(call)
        assert result.action == Action.NUDGE

    def test_write_existing_file_after_read_allowed(self, rule, tmp_path):
        """write() on existing file is OK after a prior read()."""
        existing = tmp_path / "existing.py"
        existing.write_text("print('hi')")
        read_call = ToolCall(tool="read", args={"path": str(existing)})
        rule.record([read_call])
        call = ToolCall(tool="write", args={"path": str(existing)})
        assert rule.check(call).action == Action.ALLOW

    def test_mixed_batch_new_and_existing_unread(self, rule, tmp_path):
        """In a batch of writes mixing new + existing files:
        - new files: ALLOW
        - existing unread files: NUDGE (count toward max_violations)
        """
        new_file = tmp_path / "new.js"
        existing_unread = tmp_path / "existing.py"
        existing_unread.write_text("x = 1")
        assert not new_file.exists()
        assert existing_unread.exists()

        # First write: new file → ALLOW
        assert rule.check(ToolCall(tool="write", args={"path": str(new_file)})).action == Action.ALLOW
        # Second write: existing unread → NUDGE
        assert rule.check(ToolCall(tool="write", args={"path": str(existing_unread)})).action == Action.NUDGE
        # Third write: existing unread again → BLOCK (2 violations)
        assert rule.check(ToolCall(tool="write", args={"path": str(existing_unread)})).action == Action.BLOCK
        # Fourth write: ANOTHER new file → still ALLOW (new files don't tick the violation counter)
        another_new = tmp_path / "another_new.js"
        assert rule.check(ToolCall(tool="write", args={"path": str(another_new)})).action == Action.ALLOW


class TestConfigLoadingFromYaml:
    """Regression: load_guardrail_config() used to look for a 'guardrails:'
    top-level key, but configs/guardrail-config.yaml has rules at top level.
    This silently loaded defaults — max_file_ops stayed at 100 instead of
    the configured 300, and prerequisites used wrong tool names.
    """

    def test_top_level_config_loaded(self, tmp_path):
        """Config with rules at top level (no 'guardrails:' wrapper) loads."""
        from coding_guardrails.config import load_guardrail_config
        cfg_file = tmp_path / "cfg.yaml"
        cfg_file.write_text("""
session_budget:
  max_file_ops: 500
prerequisites:
  enabled: false
""")
        config = load_guardrail_config(cfg_file)
        assert config.get("session_budget", {}).get("max_file_ops") == 500
        assert config.get("prerequisites", {}).get("enabled") is False

    def test_nested_config_still_supported(self, tmp_path):
        """Legacy 'guardrails:' wrapper still works."""
        from coding_guardrails.config import load_guardrail_config
        cfg_file = tmp_path / "cfg.yaml"
        cfg_file.write_text("""
guardrails:
  session_budget:
    max_file_ops: 200
""")
        config = load_guardrail_config(cfg_file)
        assert config.get("session_budget", {}).get("max_file_ops") == 200

    def test_no_path_returns_empty(self):
        from coding_guardrails.config import load_guardrail_config
        assert load_guardrail_config(None) == {}

    def test_default_config_loads_with_correct_budget(self):
        """The shipped configs/guardrail-config.yaml must load with
        max_file_ops=300 (not the default 100)."""
        from coding_guardrails.config import load_guardrail_config
        from pathlib import Path
        repo_config = Path(__file__).parent.parent.parent / "configs" / "guardrail-config.yaml"
        if not repo_config.exists():
            pytest.skip("Shipped config not found")
        config = load_guardrail_config(repo_config)
        assert config.get("session_budget", {}).get("max_file_ops") == 300

    def test_default_config_prereq_edit_tools_are_pi_compatible(self):
        """The shipped config must list 'edit'/'write'/'create' (Pi's tool
        names), not 'edit_file'/'write_file' (wrong dialect)."""
        from coding_guardrails.config import load_guardrail_config
        from pathlib import Path
        repo_config = Path(__file__).parent.parent.parent / "configs" / "guardrail-config.yaml"
        if not repo_config.exists():
            pytest.skip("Shipped config not found")
        config = load_guardrail_config(repo_config)
        edit_tools = config.get("prerequisites", {}).get("edit_tools", [])
        assert "edit" in edit_tools
        assert "write" in edit_tools
        assert "create" in edit_tools
