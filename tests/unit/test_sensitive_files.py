"""Tests for the sensitive file rule."""

import pytest

from coding_guardrails.rules.base import Action, ToolCall
from coding_guardrails.rules.sensitive_files import SensitiveFileRule


@pytest.fixture
def rule():
    return SensitiveFileRule()


def _write(rule, path, tool="edit"):
    return rule.check(ToolCall(tool=tool, args={"path": path}))


class TestBlockedPaths:

    @pytest.mark.parametrize("path", [
        ".git/config",
        ".git/hooks/pre-commit",
        "./.git/HEAD",
        ".ssh/authorized_keys",
        ".ssh/id_rsa",
        ".gnupg/private-keys-v1.d/key.key",
        ".github/workflows/ci.yaml",
        ".github/workflows/deploy.yml",
        ".gitlab-ci.yml",
        "Jenkinsfile",
        ".circleci/config.yml",
        ".pre-commit-config.yaml",
        ".pre-commit-config.yml",
        ".husky/pre-commit",
    ])
    def test_protected_paths_blocked(self, rule, path):
        result = _write(rule, path)
        assert result.action == Action.BLOCK, f"Should block: {path}"
        assert "protected" in result.nudge.lower() or "blocked" in result.nudge.lower()


class TestNudgePaths:

    @pytest.mark.parametrize("path", [
        ".env",
        ".env.production",
        ".env.local",
    ])
    def test_env_files_nudged(self, rule, path):
        result = _write(rule, path)
        assert result.action == Action.NUDGE, f"Should nudge: {path}"


class TestAllowedPaths:

    @pytest.mark.parametrize("path", [
        "src/main.py",
        "README.md",
        "tests/test_main.py",
        "package.json",          # not a script injection, just a write
        "pyproject.toml",
        "docs/architecture.md",
        "config/settings.yaml",
    ])
    def test_normal_files_allowed(self, rule, path):
        result = _write(rule, path)
        assert result.action == Action.ALLOW, f"Should allow: {path}"


class TestToolMatching:

    @pytest.mark.parametrize("tool", [
        "edit", "edit_file", "write", "write_file", "create", "Edit",
    ])
    def test_write_tools_checked(self, rule, tool):
        result = _write(rule, ".git/config", tool=tool)
        assert result.action == Action.BLOCK

    @pytest.mark.parametrize("tool", [
        "read", "bash", "cat", "grep",
    ])
    def test_non_write_tools_pass(self, rule, tool):
        result = _write(rule, ".git/config", tool=tool)
        assert result.action == Action.ALLOW


class TestCaseInsensitiveProtection:
    """Tests for case-insensitive protection against .GIT, .SSH, etc. bypass."""

    @pytest.mark.parametrize("path", [
        ".GIT/config",
        ".GIT/hooks/pre-commit",
        ".SSH/authorized_keys",
        ".SSH/id_rsa",
        ".GNUPG/private-keys-v1.d/key.key",
        ".GITHUB/workflows/ci.yaml",
        ".GITHUB/workflows/deploy.yml",
        ".GITLAB-ci.yml",
        ".jenkinsfile",
        ".circleci/config.yml",
    ])
    def test_uppercase_bypass_blocked(self, rule, path):
        result = _write(rule, path)
        assert result.action == Action.BLOCK, f"Should block uppercase bypass: {path}"
        assert "protected" in result.nudge.lower() or "blocked" in result.nudge.lower()

    @pytest.mark.parametrize("path", [
        ".Git/config",
        ".Ssh/id_rsa",
        ".Gnupg/private-keys-v1.d/key.key",
        ".Github/workflows/ci.yaml",
        ".Gitlab-ci.yml",
    ])
    def test_mixed_case_bypass_blocked(self, rule, path):
        result = _write(rule, path)
        assert result.action == Action.BLOCK, f"Should block mixed-case bypass: {path}"
        assert "protected" in result.nudge.lower() or "blocked" in result.nudge.lower()


class TestExtraProtected:

    def test_extra_protected_paths(self):
        rule = SensitiveFileRule(
            extra_protected=[
                (r"^my-secrets/", "Custom secrets dir", "block"),
            ],
        )
        result = _write(rule, "my-secrets/api-keys.json")
        assert result.action == Action.BLOCK

    def test_extra_nudge_paths(self):
        rule = SensitiveFileRule(
            extra_protected=[
                (r"^local\.env$", "Local env", "nudge"),
            ],
        )
        result = _write(rule, "local.env")
        assert result.action == Action.NUDGE
