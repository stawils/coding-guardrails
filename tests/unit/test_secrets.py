"""Tests for the secret detection rule."""

from coding_guardrails.rules.base import Action, ToolCall
from coding_guardrails.rules.secrets import SecretRule


def test_openai_key_masked():
    rule = SecretRule()
    call = ToolCall(tool="bash", args={"command": "export API_KEY=sk-abc123def456ghi789jkl012mno345"})
    result = rule.check(call)
    assert result.action == Action.NUDGE
    assert "sk-abc123" not in call.args["command"]
    assert "[REDACTED]" in call.args["command"]


def test_github_pat_masked():
    rule = SecretRule()
    pat = "ghp_" + "a" * 36
    call = ToolCall(tool="bash", args={"command": f"echo {pat}"})
    result = rule.check(call)
    assert result.action == Action.NUDGE
    assert pat not in call.args["command"]


def test_aws_key_masked():
    rule = SecretRule()
    call = ToolCall(tool="bash", args={"command": "export AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE"})
    result = rule.check(call)
    assert result.action == Action.NUDGE
    assert "AKIAIOSFODNN7EXAMPLE" not in call.args["command"]


def test_private_key_masked():
    rule = SecretRule()
    call = ToolCall(
        tool="write_file",
        args={"path": "/tmp/key.pem", "content": "-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAAKCAQEA..."},
    )
    result = rule.check(call)
    assert result.action == Action.NUDGE
    assert "BEGIN RSA PRIVATE KEY" not in call.args["content"]


def test_no_secret_allowed():
    rule = SecretRule()
    call = ToolCall(tool="bash", args={"command": "ls -la /home/user/"})
    result = rule.check(call)
    assert result.action == Action.ALLOW


def test_block_mode():
    rule = SecretRule(action="block")
    call = ToolCall(tool="bash", args={"command": "export KEY=sk-abc123def456ghi789jkl012mno345pqr678"})
    result = rule.check(call)
    assert result.action == Action.BLOCK


def test_custom_mask_value():
    rule = SecretRule(mask_value="***HIDDEN***")
    call = ToolCall(tool="bash", args={"command": "sk-abc123def456ghi789jkl012mno345pqr678stu901"})
    rule.check(call)
    assert "***HIDDEN***" in call.args["command"]


def test_clean_command_unchanged():
    rule = SecretRule()
    original = "python3 -m pytest tests/"
    call = ToolCall(tool="bash", args={"command": original})
    rule.check(call)
    assert call.args["command"] == original

class TestEdgeCases:
    def test_empty_command(self):
        rule = SecretRule()
        """Empty command should be allowed."""
        call = ToolCall(tool="bash", args={"command": ""})
        assert rule.check(call).action == Action.ALLOW

    def test_no_command_arg(self):
        rule = SecretRule()
        """Tool calls without command arg should be allowed."""
        call = ToolCall(tool="bash", args={"path": "/tmp/file"})
        assert rule.check(call).action == Action.ALLOW

    def test_safe_command_no_secrets(self):
        rule = SecretRule()
        """Commands without secrets should be allowed unchanged."""
        call = ToolCall(tool="bash", args={"command": "ls -la /home/user/"})
        result = rule.check(call)
        assert result.action == Action.ALLOW
        assert call.args["command"] == "ls -la /home/user/"

    def test_non_bash_tool_ignores(self):
        rule = SecretRule()
        """Non-bash tools like 'read' should be allowed."""
        call = ToolCall(tool="read", args={"path": "/etc/hosts"})
        assert rule.check(call).action == Action.ALLOW
