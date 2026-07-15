"""Unit tests for the lint gate rule (coding_guardrails.rules.lint)."""

from __future__ import annotations

import subprocess

from coding_guardrails.rules.base import Action, ToolCall
from coding_guardrails.rules.lint import LinterSpec, LintRule, default_linters, workspace_from_env


def _call(tool: str = "edit", **args) -> ToolCall:
    return ToolCall(tool=tool, args=args)


def _fake_run_factory(*, returncode: int = 0, stdout: str = "", calls: list | None = None):
    """Return a subprocess.run replacement that records calls and returns a result."""
    def _fake(cmd, **kwargs):
        if calls is not None:
            calls.append(cmd)
        return subprocess.CompletedProcess(args=cmd, returncode=returncode, stdout=stdout, stderr="")
    return _fake


def _raise_factory(exc, calls: list | None = None):
    def _fake(cmd, **kwargs):
        if calls is not None:
            calls.append(cmd)
        raise exc
    return _fake


# --- trigger + allow paths -------------------------------------------------


class TestTriggering:
    def test_non_edit_tool_not_linted(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            "coding_guardrails.rules.lint.subprocess.run", _fake_run_factory(calls=calls)
        )
        rule = LintRule(workspace=".")
        assert rule.check(_call("bash", command="ls")).action == Action.ALLOW
        assert rule.check(_call("read", path="x.py")).action == Action.ALLOW
        assert calls == []  # linter never invoked

    def test_missing_path_arg_skipped(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            "coding_guardrails.rules.lint.subprocess.run", _fake_run_factory(calls=calls)
        )
        rule = LintRule(workspace=".")
        # edit with no recognizable path arg
        assert rule.check(_call("edit", content="x")).action == Action.ALLOW
        assert calls == []

    def test_nonexistent_file_skipped(self, tmp_path, monkeypatch):
        calls = []
        monkeypatch.setattr(
            "coding_guardrails.rules.lint.subprocess.run", _fake_run_factory(calls=calls)
        )
        rule = LintRule(workspace=str(tmp_path))
        result = rule.check(_call("edit", path="nope.py"))
        assert result.action == Action.ALLOW
        assert calls == []


# --- path resolution + sandbox --------------------------------------------


class TestPathResolution:
    def test_relative_path_without_workspace_skipped(self, monkeypatch, tmp_path):
        calls = []
        monkeypatch.setattr(
            "coding_guardrails.rules.lint.subprocess.run", _fake_run_factory(calls=calls)
        )
        rule = LintRule(workspace=None)  # cannot resolve relative paths
        f = tmp_path / "a.py"
        f.write_text("x = 1\n")
        assert rule.check(_call("edit", path="a.py")).action == Action.ALLOW
        assert calls == []

    def test_relative_path_resolved_against_workspace(self, monkeypatch, tmp_path):
        calls = []
        monkeypatch.setattr(
            "coding_guardrails.rules.lint.subprocess.run",
            _fake_run_factory(returncode=0, calls=calls),
        )
        f = tmp_path / "a.py"
        f.write_text("x = 1\n")
        rule = LintRule(workspace=str(tmp_path))
        assert rule.check(_call("edit", path="a.py")).action == Action.ALLOW
        assert len(calls) == 1
        assert str(f) in " ".join(calls[0])

    def test_path_escaping_workspace_skipped(self, monkeypatch, tmp_path):
        calls = []
        monkeypatch.setattr(
            "coding_guardrails.rules.lint.subprocess.run", _fake_run_factory(calls=calls)
        )
        rule = LintRule(workspace=str(tmp_path))
        # absolute path outside workspace must be refused
        assert rule.check(_call("edit", path="/etc/passwd")).action == Action.ALLOW
        assert calls == []


# --- findings → nudge / block ---------------------------------------------


class TestFindings:
    def test_clean_file_allowed(self, monkeypatch, tmp_path):
        calls = []
        monkeypatch.setattr(
            "coding_guardrails.rules.lint.subprocess.run",
            _fake_run_factory(returncode=0, calls=calls),
        )
        f = tmp_path / "clean.py"
        f.write_text("x = 1\n")
        rule = LintRule(workspace=str(tmp_path))
        assert rule.check(_call("edit", path="clean.py")).action == Action.ALLOW

    def test_findings_nudge_mode(self, monkeypatch, tmp_path):
        calls = []
        monkeypatch.setattr(
            "coding_guardrails.rules.lint.subprocess.run",
            _fake_run_factory(
                returncode=1, stdout="clean.py:1:1: F401 'os' imported but unused", calls=calls
            ),
        )
        f = tmp_path / "clean.py"
        f.write_text("x = 1\n")
        rule = LintRule(workspace=str(tmp_path), mode="nudge")
        result = rule.check(_call("edit", path="clean.py"))
        assert result.action == Action.NUDGE
        assert "F401" in (result.nudge or "")
        assert "held" not in (result.nudge or "")  # nudge, not block wording

    def test_findings_block_mode(self, monkeypatch, tmp_path):
        calls = []
        monkeypatch.setattr(
            "coding_guardrails.rules.lint.subprocess.run",
            _fake_run_factory(
                returncode=1, stdout="dirty.py:2:5: F401 'os' imported but unused", calls=calls
            ),
        )
        f = tmp_path / "dirty.py"
        f.write_text("x = 1\n")
        rule = LintRule(workspace=str(tmp_path), mode="block")
        result = rule.check(_call("edit", path="dirty.py"))
        assert result.action == Action.BLOCK
        assert "held" in (result.nudge or "")
        assert "F401" in (result.nudge or "")
        assert result.reason.endswith("lint findings")

    def test_findings_other_path_args(self, monkeypatch, tmp_path):
        """file_path / file arg names are recognized."""
        monkeypatch.setattr(
            "coding_guardrails.rules.lint.subprocess.run",
            _fake_run_factory(returncode=1, stdout="a.py:1:1: F401 x"),
        )
        (tmp_path / "a.py").write_text("x")
        rule = LintRule(workspace=str(tmp_path), mode="nudge")
        assert rule.check(_call("write", file_path="a.py")).action == Action.NUDGE
        assert rule.check(_call("create", file="a.py")).action == Action.NUDGE


# --- linter unavailable / timeout -----------------------------------------


class TestLinterFailures:
    def test_linter_not_installed_allowed(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            "coding_guardrails.rules.lint.subprocess.run",
            _raise_factory(FileNotFoundError(), []),
        )
        (tmp_path / "a.py").write_text("x")
        rule = LintRule(workspace=str(tmp_path), mode="block")
        # missing linter must NOT block (treat as clean)
        assert rule.check(_call("edit", path="a.py")).action == Action.ALLOW

    def test_linter_timeout_allowed(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            "coding_guardrails.rules.lint.subprocess.run",
            _raise_factory(subprocess.TimeoutExpired(cmd=[], timeout=10), []),
        )
        (tmp_path / "a.py").write_text("x")
        rule = LintRule(workspace=str(tmp_path), mode="block")
        assert rule.check(_call("edit", path="a.py")).action == Action.ALLOW

    def test_internal_error_no_stdout_allowed(self, monkeypatch, tmp_path):
        """Non-zero exit but empty stdout (ruff internal error) → don't block."""
        monkeypatch.setattr(
            "coding_guardrails.rules.lint.subprocess.run",
            _fake_run_factory(returncode=2, stdout=""),
        )
        (tmp_path / "a.py").write_text("x")
        rule = LintRule(workspace=str(tmp_path), mode="block")
        assert rule.check(_call("edit", path="a.py")).action == Action.ALLOW


# --- workspace_from_env ---------------------------------------------------


class TestWorkspaceEnv:
    def test_config_value_wins(self, monkeypatch):
        monkeypatch.setenv("CG_LINT_WORKSPACE", "/env/path")
        assert workspace_from_env("/cfg/path") == "/cfg/path"

    def test_env_fallback(self, monkeypatch):
        monkeypatch.setenv("CG_LINT_WORKSPACE", "/env/path")
        assert workspace_from_env(None) == "/env/path"

    def test_nothing_returns_none(self, monkeypatch):
        monkeypatch.delenv("CG_LINT_WORKSPACE", raising=False)
        assert workspace_from_env(None) is None


class TestLanguageSelection:
    def _capture(self, records, *, returncode: int = 0, stdout: str = ""):
        def _fake(cmd, **kwargs):
            records.append((cmd, kwargs.get("cwd")))
            return subprocess.CompletedProcess(args=cmd, returncode=returncode, stdout=stdout, stderr="")
        return _fake

    def test_go_file_gofmt_stdout_mode(self, monkeypatch, tmp_path):
        # gofmt -l always exits 0 but lists unformatted files on stdout
        records: list = []
        monkeypatch.setattr(
            "coding_guardrails.rules.lint.subprocess.run",
            self._capture(records, returncode=0, stdout="main.go"),
        )
        (tmp_path / "main.go").write_text("x")
        rule = LintRule(workspace=str(tmp_path), mode="nudge")
        result = rule.check(_call("edit", path="main.go"))
        assert result.action == Action.NUDGE
        assert "Go" in (result.nudge or "")
        assert records and records[0][0][0] == "gofmt"

    def test_go_file_clean_allowed(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            "coding_guardrails.rules.lint.subprocess.run",
            self._capture([], returncode=0, stdout=""),
        )
        (tmp_path / "main.go").write_text("x")
        assert LintRule(workspace=str(tmp_path)).check(_call("edit", path="main.go")).action == Action.ALLOW

    def test_typescript_uses_biome(self, monkeypatch, tmp_path):
        records: list = []
        monkeypatch.setattr(
            "coding_guardrails.rules.lint.subprocess.run",
            self._capture(records, returncode=1, stdout="a.ts:1:1 lint/suspicious"),
        )
        (tmp_path / "a.ts").write_text("x")
        result = LintRule(workspace=str(tmp_path), mode="nudge").check(_call("edit", path="a.ts"))
        assert result.action == Action.NUDGE
        assert records[0][0][0] == "biome"

    def test_unknown_extension_skipped(self, monkeypatch, tmp_path):
        records: list = []
        monkeypatch.setattr(
            "coding_guardrails.rules.lint.subprocess.run", self._capture(records),
        )
        (tmp_path / "README.md").write_text("x")
        rule = LintRule(workspace=str(tmp_path), mode="block")
        assert rule.check(_call("edit", path="README.md")).action == Action.ALLOW
        assert records == []

    def test_default_linters_cover_py_js_ts_go(self):
        exts = set()
        for spec in default_linters():
            exts.update(spec.extensions)
        for e in (".py", ".js", ".jsx", ".ts", ".tsx", ".go"):
            assert e in exts


class TestProjectMode:
    def _capture(self, records, *, returncode: int = 0, stdout: str = ""):
        def _fake(cmd, **kwargs):
            records.append((cmd, kwargs.get("cwd")))
            return subprocess.CompletedProcess(args=cmd, returncode=returncode, stdout=stdout, stderr="")
        return _fake

    def test_project_mode_without_workspace_skipped(self, monkeypatch, tmp_path):
        records: list = []
        monkeypatch.setattr(
            "coding_guardrails.rules.lint.subprocess.run",
            self._capture(records, returncode=1, stdout="warn"),
        )
        f = tmp_path / "lib.rs"
        f.write_text("x")
        rule = LintRule(workspace=None, mode="block", linters=(
            LinterSpec("Rust", (".rs",), ("cargo", "clippy"), path_mode="project"),
        ))
        assert rule.check(_call("edit", path=str(f))).action == Action.ALLOW
        assert records == []

    def test_project_mode_runs_in_workspace_cwd(self, monkeypatch, tmp_path):
        records: list = []
        monkeypatch.setattr(
            "coding_guardrails.rules.lint.subprocess.run",
            self._capture(records, returncode=1, stdout="warn"),
        )
        (tmp_path / "lib.rs").write_text("x")
        rule = LintRule(workspace=str(tmp_path), mode="block", linters=(
            LinterSpec("Rust", (".rs",), ("cargo", "clippy"), path_mode="project"),
        ))
        result = rule.check(_call("edit", path="lib.rs"))
        assert result.action == Action.BLOCK
        assert records and records[0][0] == ["cargo", "clippy"]  # no path appended
        assert records[0][1] == str(tmp_path)  # cwd = workspace


class TestCustomLinters:
    def _capture(self, records, *, returncode: int = 0, stdout: str = ""):
        def _fake(cmd, **kwargs):
            records.append(cmd)
            return subprocess.CompletedProcess(args=cmd, returncode=returncode, stdout=stdout, stderr="")
        return _fake

    def test_custom_linters_replace_defaults(self, monkeypatch, tmp_path):
        records: list = []
        monkeypatch.setattr(
            "coding_guardrails.rules.lint.subprocess.run",
            self._capture(records, returncode=1, stdout="x"),
        )
        (tmp_path / "a.py").write_text("x")
        rule = LintRule(workspace=str(tmp_path), linters=(
            LinterSpec("Ruby", (".rb",), ("rubocop", "-f", "compact")),
        ))
        # .py no longer matched (defaults replaced) → allowed, linter not called
        assert rule.check(_call("edit", path="a.py")).action == Action.ALLOW
        assert records == []
        # .rb matched by the custom spec
        (tmp_path / "b.rb").write_text("x")
        assert rule.check(_call("edit", path="b.rb")).action == Action.NUDGE
        assert records[0][0] == "rubocop"
