"""Lint gate — run the project linter on files the agent edits.

Noticing pre-existing lint defects is unreliable for small local models: they clean
their *own* output but read past pre-existing nits in files they only partially edit
(verified 2026-07-15 on Qwen3.5-9B). This rule offloads noticing to a deterministic
tool — when the agent edits/writes a file, run ``ruff check`` on it and surface findings.

Modes:
- ``nudge`` (default): advisory — the call proceeds, findings are logged. Visible in
  Forge/eval runners that inject nudges.
- ``block``: the edit is held and the findings returned as a text nudge. This is the
  ONLY mode that reliably changes behavior for Pi-streamed agents, whose nudges are
  otherwise silently logged. The held-edit nudge preserves intent and tells the agent
  how to clean the file, then redo the edit.

Path resolution is sandboxed: relative paths resolve against ``workspace`` and must
stay inside it. If no workspace is configured, relative paths are skipped (the rule
cannot resolve them safely from the proxy's cwd).
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
class LintRule:
    """Run a linter on files the agent edits; surface findings.

    Attributes:
        edit_tools: Tool-name prefixes that trigger a lint check.
        path_args: Argument names tried, in order, for the target file path.
        workspace: Root directory for resolving relative paths and the sandbox
            boundary. If None, relative paths are skipped.
        mode: "nudge" (advisory) or "block" (hold the edit until the file is clean).
        timeout: Max seconds for the linter subprocess.
        command: Linter command prefix; the resolved path is appended as the last arg.
    """

    edit_tools: tuple[str, ...] = _DEFAULT_EDIT_TOOLS
    path_args: tuple[str, ...] = _DEFAULT_PATH_ARGS
    workspace: str | None = None
    mode: str = "nudge"
    timeout: float = 10.0
    command: tuple[str, ...] = field(default_factory=lambda: ("ruff", "check", "--select=F,E9", "--output-format=concise"))

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

    # --- linter --------------------------------------------------------

    def _run_linter(self, target: Path) -> str:
        """Run the linter on target; return its stdout (empty if clean/unavailable)."""
        cmd = [*self.command, str(target)]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=self.timeout, check=False,
            )
        except FileNotFoundError:
            logger.debug("lint: %s not installed — skipping", self.command[0])
            return ""
        except subprocess.TimeoutExpired:
            logger.warning("lint: timed out after %ss on %s", self.timeout, target)
            return ""
        # ruff: 0 = clean, 1 = findings, >1 = internal error.
        if proc.returncode == 0:
            return ""
        return (proc.stdout or "").strip()

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

        findings = self._run_linter(target)
        if not findings:
            return RuleResult.allow(call.tool)

        preview = findings if len(findings) <= 800 else findings[:797] + "..."
        rel = self._relative_for_display(target)

        if self.mode == "block":
            nudge = (
                f"Lint findings in {rel} (your edit is held). Clean the file, e.g. "
                f"`ruff check --fix {rel}`, then re-issue your edit.\n\n{preview}"
            )
            return RuleResult.block(call.tool, nudge=nudge, reason="lint findings")

        nudge = f"Lint findings in {rel} (not auto-applied): consider `ruff --fix`.\n\n{preview}"
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
