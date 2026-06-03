"""Tests for the path safety rule."""

import os

from coding_guardrails.rules.base import Action, ToolCall
from coding_guardrails.rules.path_safety import PathSafetyRule


def test_etc_passwd_blocked():
    rule = PathSafetyRule()
    result = rule.check(ToolCall(tool="read_file", args={"path": "/etc/passwd"}))
    assert result.action == Action.BLOCK
    assert "outside" in result.nudge.lower() or "blocked" in result.nudge.lower()


def test_path_traversal_blocked():
    rule = PathSafetyRule()
    result = rule.check(ToolCall(tool="read_file", args={"path": "../../etc/shadow"}))
    assert result.action == Action.BLOCK
    assert "traversal" in result.nudge.lower()


def test_home_path_allowed_without_allowlist():
    rule = PathSafetyRule()  # No allowlist configured
    result = rule.check(ToolCall(tool="read_file", args={"path": "/home/user/code/main.py"}))
    assert result.action == Action.ALLOW


def test_home_path_allowed_with_allowlist():
    rule = PathSafetyRule(allowlist=["/home/user/"])
    result = rule.check(ToolCall(tool="read_file", args={"path": "/home/user/code/main.py"}))
    assert result.action == Action.ALLOW


def test_path_outside_allowlist_blocked():
    rule = PathSafetyRule(allowlist=["/home/user/"])
    result = rule.check(ToolCall(tool="read_file", args={"path": "/tmp/other/file.py"}))
    assert result.action == Action.BLOCK


def test_env_var_expansion():
    rule = PathSafetyRule(allowlist=["/home/${USER}/"])
    import os
    os.environ.setdefault("USER", "testuser")
    result = rule.check(ToolCall(tool="read_file", args={"path": f"/home/{os.environ['USER']}/code.py"}))
    assert result.action == Action.ALLOW


def test_root_ssh_blocked():
    rule = PathSafetyRule()
    result = rule.check(ToolCall(tool="read_file", args={"path": "/root/.ssh/id_rsa"}))
    assert result.action == Action.BLOCK


def test_symlink_to_blocked_prefix_is_caught(tmp_path):
    """Symlink pointing into a blocked prefix (e.g. /etc) must be blocked."""
    rule = PathSafetyRule()
    evil = tmp_path / "evil"
    # Create a real directory to symlink to (avoid /etc permission issues in CI)
    target_dir = tmp_path / "etc_sandbox"
    target_dir.mkdir()
    (target_dir / "passwd").write_text("root:x:0:0")
    evil.symlink_to(target_dir)
    # Patch blocked_prefixes to include our sandbox target
    import os
    rule.blocked_prefixes = [os.path.join(os.fspath(target_dir), "").rstrip("/") + "/"]
    result = rule.check(ToolCall(tool="read", args={"path": str(evil / "passwd")}))
    assert result.action == Action.BLOCK
    assert "blocked prefix" in result.reason


def test_symlink_outside_allowlist_is_caught(tmp_path):
    """Symlink pointing outside the allowlist must be blocked."""
    import tempfile
    # Create a truly outside directory (sibling of tmp_path, not inside it)
    with tempfile.TemporaryDirectory(dir=os.path.dirname(tmp_path)) as outside_base:
        outside_dir = os.path.join(outside_base, "secret")
        os.makedirs(outside_dir)
        evil = tmp_path / "sneaky_link"
        evil.symlink_to(outside_base)
        rule = PathSafetyRule(allowlist=[os.fspath(tmp_path) + "/"])
        result = rule.check(ToolCall(tool="read", args={"path": str(evil / "secret")}))
        assert result.action == Action.BLOCK
        assert "not in allowlist" in result.reason


def test_unrelated_tool_allowed():
    rule = PathSafetyRule()
    result = rule.check(ToolCall(tool="bash", args={"command": "echo hello"}))
    assert result.action == Action.ALLOW


def test_no_path_arg_allowed():
    rule = PathSafetyRule()
    result = rule.check(ToolCall(tool="bash", args={"command": "ls"}))
    assert result.action == Action.ALLOW
