"""Tests for the path safety rule."""

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


def test_unrelated_tool_allowed():
    rule = PathSafetyRule()
    result = rule.check(ToolCall(tool="bash", args={"command": "echo hello"}))
    assert result.action == Action.ALLOW


def test_no_path_arg_allowed():
    rule = PathSafetyRule()
    result = rule.check(ToolCall(tool="bash", args={"command": "ls"}))
    assert result.action == Action.ALLOW
