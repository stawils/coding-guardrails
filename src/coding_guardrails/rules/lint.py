"""Lint gate — run the right linter on files the agent edits, any language.

Noticing pre-existing lint defects is unreliable for small local models: they clean
their *own* output but read past pre-existing nits in files they only partially edit
(verified 2026-07-15 on Qwen3.5-9B). This rule offloads noticing to a deterministic
tool — when the agent edits/writes a file, run that language's linter and surface findings.

Each file extension maps to a linter (``LinterSpec``). Defaults cover the most common
languages with fast, file-level tools:

- Python (``.py``)        → ``ruff check --select=F,E9``  (defects; non-zero exit = findings)
- JS/TS (``.js .ts ...``) → ``biome check``               (defects; non-zero exit = findings)
- Go (``.go``)            → ``gofmt -l``                  (formatting; non-empty stdout = findings)

Linters that aren't installed are skipped (the call is allowed). Project-scoped linters
(e.g. ``cargo clippy``, ``golangci-lint``) can be added via config with ``path_mode:
project``. The full set is configurable in guardrail-config.yaml.

Modes:
- ``nudge`` (default): advisory — the call proceeds, findings are logged.
- ``block``: the edit is held and findings returned as text (the only mode Pi-streamed
  agents heed, since their nudges are otherwise silently logged).

Path resolution is sandboxed: relative paths resolve against ``workspace`` and must stay
inside it; with no workspace, relative paths are skipped.
"""

from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from coding_guardrails.rules.base import RuleResult, ToolCall

logger = logging.getLogger("coding_guardrails.layer2")

_DEFAULT_EDIT_TOOLS = ("edit", "write", "create")
_DEFAULT_PATH_ARGS = ("path", "file_path", "file", "filename")


def _tool_matches(tool: str, prefixes: tuple[str, ...]) -> bool:
    tool_l = tool.lower()
    return any(tool_l == p or tool_l.startswith(p) for p in prefixes)


@dataclass
class LinterSpec:
    """One linter for a set of file extensions.

    Attributes:
        name: Human-readable label (Python, Go, ...).
        extensions: File extensions this linter handles, lowercase (e.g. (".py",)).
        command: Linter argv prefix; the target is appended per ``path_mode``.
        path_mode: "file" (append the file path), "dir" (append its parent directory),
            "project" (run in the workspace root with no path appended — for tools like
            ``cargo clippy`` that operate on a manifest, not a single file).
        findings_mode: How findings are detected. "exitcode" (default) — a non-zero exit
            code means findings, reported via stdout. "stdout" — any non-empty stdout
            means findings regardless of exit code (e.g. ``gofmt -l`` always exits 0 but
            lists unformatted files).
        enabled: If False, this spec is skipped during selection.
    """

    name: str
    extensions: tuple[str, ...]
    command: tuple[str, ...]
    path_mode: str = "file"
    findings_mode: str = "exitcode"
    enabled: bool = True


def default_linters() -> tuple[LinterSpec, ...]:
    """Fast, file-level linters for the most common languages.

    Each degrades gracefully: if the binary isn't on PATH, the file is allowed.
    Override or extend via the ``lint.linters`` config section.
    """
    return (
        LinterSpec(
            "Python",
            (".py",),
            ("ruff", "check", "--select=F,E9", "--output-format=concise"),
        ),
        LinterSpec(
            "JavaScript/TypeScript",
            (".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".mts", ".cts"),
            ("biome", "check"),
        ),
        LinterSpec(
            "Go",
            (".go",),
            ("gofmt", "-l"),
            findings_mode="stdout",
        ),
    )


@dataclass
class LintRule:
    """Run the matching language linter on files the agent edits; surface findings.

    Attributes:
        edit_tools: Tool-name prefixes that trigger a lint check.
        path_args: Argument names tried, in order, for the target file path.
        workspace: Root directory for resolving relative paths, the sandbox boundary,
            and the cwd for ``path_mode: project`` linters. If None, relative paths and
            project-mode linters are skipped.
        mode: "nudge" (advisory) or "block" (hold the edit until the file is clean).
        timeout: Max seconds for a linter subprocess.
        linters: Ordered specs; the first whose extensions contain the file's suffix wins.
    """

    edit_tools: tuple[str, ...] = _DEFAULT_EDIT_TOOLS
    path_args: tuple[str, ...] = _DEFAULT_PATH_ARGS
    workspace: str | None = None
    mode: str = "nudge"
    timeout: float = 10.0
    linters: tuple[LinterSpec, ...] = field(default_factory=default_linters)

    @property
    def name(self) -> str:
        return "lint"

    # --- path handling -------------------------------------------------

    def _extract_path(self, call: ToolCall) -> str | None:
        for arg in self.path_args:
            value = call.args.get(arg)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    def _resolve(self, raw: str) -> Path | None:
        """Resolve to an absolute path inside the workspace, or None to skip."""
        p = Path(raw)
        if not p.is_absolute():
            if not self.workspace:
                return None
            p = Path(self.workspace) / p
        try:
            p = p.resolve(strict=False)
        except (OSError, RuntimeError):
            return None
        if self.workspace:
            try:
                root = Path(self.workspace).resolve(strict=False)
            except (OSError, RuntimeError):
                return None
            try:
                p.relative_to(root)
            except ValueError:
                return None  # escapes workspace sandbox — skip
        return p

    # --- linter selection + execution ---------------------------------

    def _select(self, target: Path) -> LinterSpec | None:
        suffix = target.suffix.lower()
        for spec in self.linters:
            if spec.enabled and suffix in spec.extensions:
                return spec
        return None

    def _run(self, spec: LinterSpec, target: Path) -> str:
        """Run the linter; return its findings text (empty if clean/unavailable)."""
        cwd: str | None = None
        if spec.path_mode == "project":
            if not self.workspace:
                return ""  # no project root to run in
            cmd = list(spec.command)
            cwd = self.workspace
        elif spec.path_mode == "dir":
            cmd = [*spec.command, str(target.parent)]
        else:  # file
            cmd = [*spec.command, str(target)]

        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=self.timeout, check=False, cwd=cwd,
            )
        except FileNotFoundError:
            logger.debug("lint: %s not installed — skipping %s", spec.command[0], spec.name)
            return ""
        except subprocess.TimeoutExpired:
            logger.warning("lint: %s timed out after %ss on %s", spec.name, self.timeout, target)
            return ""

        stdout = (proc.stdout or "").strip()
        if spec.findings_mode == "stdout":
            return stdout  # gofmt -l style: non-empty stdout == findings
        # exitcode mode: findings iff non-zero exit (guard against empty stdout on errors)
        return stdout if proc.returncode != 0 else ""

    # --- rule API ------------------------------------------------------

    def check(self, call: ToolCall) -> RuleResult:
        if not _tool_matches(call.tool, self.edit_tools):
            return RuleResult.allow(call.tool)

        raw = self._extract_path(call)
        if raw is None:
            return RuleResult.allow(call.tool)

        target = self._resolve(raw)
        if target is None or not target.is_file():
            return RuleResult.allow(call.tool)

        spec = self._select(target)
        if spec is None:
            return RuleResult.allow(call.tool)  # no linter for this file type

        findings = self._run(spec, target)
        if not findings:
            return RuleResult.allow(call.tool)

        preview = findings if len(findings) <= 800 else findings[:797] + "..."
        rel = self._relative_for_display(target)

        if self.mode == "block":
            nudge = (
                f"{spec.name} lint findings in {rel} (your edit is held). "
                f"Clean the file, then re-issue your edit.\n\n{preview}"
            )
            return RuleResult.block(call.tool, nudge=nudge, reason=f"{spec.name} lint findings")

        nudge = f"{spec.name} lint findings in {rel} (not auto-applied):\n\n{preview}"
        return RuleResult.nudge(call.tool, message=nudge)

    def record(self, calls: list[ToolCall]) -> None:
        """Lint is stateless — nothing to record."""
        return None

    # --- helpers -------------------------------------------------------

    def _relative_for_display(self, target: Path) -> str:
        if self.workspace:
            try:
                return str(target.relative_to(Path(self.workspace).resolve(strict=False)))
            except ValueError:
                pass
        return str(target)


def workspace_from_env(config_value: str | None) -> str | None:
    """Resolve the lint workspace: explicit config, else $CG_LINT_WORKSPACE."""
    if config_value:
        return config_value
    env = os.environ.get("CG_LINT_WORKSPACE")
    return env or None
