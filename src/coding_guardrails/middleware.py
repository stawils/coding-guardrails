"""Guardrail middleware — composes all rules into a single check/record API.

The middleware holds all configured rules and provides:
- check(): inspect tool calls against all rules
- record(): update rule state after execution
- check_result(): inspect tool results for resolution nudges
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, replace

from coding_guardrails.rules.base import (
    Action,
    CheckResult,
    Rule,
    RuleResult,
    ToolCall,
)
from coding_guardrails.rules.commands import CommandSafetyRule
from coding_guardrails.rules.loop_detection import LoopDetectionRule
from coding_guardrails.rules.network import NetworkRule
from coding_guardrails.rules.path_safety import PathSafetyRule
from coding_guardrails.rules.prerequisites import PrerequisiteRule
from coding_guardrails.rules.secrets import SecretRule
from coding_guardrails.rules.dup_write import DuplicateWriteRule
from coding_guardrails.rules.lint import LinterSpec, LintRule, default_linters, workspace_from_env
from coding_guardrails.rules.sensitive_files import SensitiveFileRule
from coding_guardrails.rules.sequencing import SequenceRule
from coding_guardrails.rules.session_budget import SessionBudgetRule
from coding_guardrails.rules.thoroughness import ThoroughnessRule
from coding_guardrails.rules.tool_resolution import ToolResolutionRule

logger = logging.getLogger("coding_guardrails.layer2")


def _short(text: str, width: int = 60) -> str:
    if len(text) <= width:
        return text
    return text[:width - 3] + "..."


def _fmt_call(call: ToolCall, width: int = 60) -> str:
    """Format a tool call preview: tool(arg1=val1, arg2=val2)."""
    parts = []
    for k, v in list(call.args.items())[:3]:
        s = str(v)
        if len(s) > 20:
            s = s[:17] + "..."
        parts.append(f"{k}={s}")
    result = f"{call.tool}({', '.join(parts)})"
    if len(result) > width:
        return result[:width - 3] + "..."
    return result


@dataclass
class CodingGuardrails:
    """Composes all coding guardrail rules.

    Attributes:
        prerequisites: Read-before-edit enforcement.
        path_safety: Path traversal blocking.
        command_safety: Destructive command blocking.
        secrets: Secret detection and masking.
        sequencing: Test-after-change nudges.
        tool_resolution: Empty/error result handling.
        dup_write: Duplicate identical write detection.
    """

    prerequisites: PrerequisiteRule | None = None
    path_safety: PathSafetyRule | None = None
    command_safety: CommandSafetyRule | None = None
    network: NetworkRule | None = None
    sensitive_files: SensitiveFileRule | None = None
    secrets: SecretRule | None = None
    loop_detection: LoopDetectionRule | None = None
    session_budget: SessionBudgetRule | None = None
    thoroughness: ThoroughnessRule | None = None
    sequencing: SequenceRule | None = None
    tool_resolution: ToolResolutionRule | None = None
    dup_write: DuplicateWriteRule | None = None
    lint: LintRule | None = None

    @classmethod
    def from_config(cls, config: dict) -> CodingGuardrails:
        """Create guardrails from a config dict.

        Args:
            config: The "guardrails" section of the YAML config.

        Returns:
            Configured CodingGuardrails instance.
        """
        rules: dict[str, Rule | None] = {}

        # Prerequisites
        prereq_cfg = config.get("prerequisites", {})
        if prereq_cfg.get("enabled", True):
            rules["prerequisites"] = PrerequisiteRule(
                edit_tools=tuple(prereq_cfg.get("edit_tools", (
                    "edit", "write", "create",
                ))),
                read_tools=tuple(prereq_cfg.get("read_tools", (
                    "read", "cat", "head", "tail", "less",
                ))),
                match_arg=prereq_cfg.get("match_arg", "path"),
                max_violations=prereq_cfg.get("max_violations", 2),
            )

        # Path safety
        path_cfg = config.get("path_safety", {})
        if path_cfg.get("enabled", True):
            rules["path_safety"] = PathSafetyRule(
                allowlist=path_cfg.get("allowlist", []),
                blocked_prefixes=path_cfg.get("blocked_prefixes", None),
                blocked_patterns=path_cfg.get("blocked_patterns", None),
            )

        # Command safety
        cmd_cfg = config.get("command_safety", {})
        if cmd_cfg.get("enabled", True):
            kwargs = {}
            if "blocked" in cmd_cfg:
                kwargs["blocked"] = cmd_cfg["blocked"]
            if "blocked_patterns" in cmd_cfg:
                kwargs["blocked_patterns"] = cmd_cfg["blocked_patterns"]
            if "require_confirmation" in cmd_cfg:
                kwargs["require_confirmation"] = cmd_cfg["require_confirmation"]
            rules["command_safety"] = CommandSafetyRule(**kwargs)

        # Network
        net_cfg = config.get("network", {})
        if net_cfg.get("enabled", True):
            rules["network"] = NetworkRule(
                block_uploads=net_cfg.get("block_uploads", True),
                block_metadata=net_cfg.get("block_metadata", True),
                block_private_ips=net_cfg.get("block_private_ips", False),
                allowed_hosts=net_cfg.get("allowed_hosts", [
                    "localhost", "127.0.0.1", "0.0.0.0", "::1",
                ]),
            )

        # Sensitive files
        sf_cfg = config.get("sensitive_files", {})
        if sf_cfg.get("enabled", True):
            extra = []
            for entry in sf_cfg.get("extra_protected", []):
                extra.append((entry["pattern"], entry["label"], entry.get("action", "block")))
            rules["sensitive_files"] = SensitiveFileRule(
                extra_protected=extra,
            )

        # Secrets
        secrets_cfg = config.get("secrets", {})
        if secrets_cfg.get("enabled", True):
            # Support both "strength" (from config YAML) and "action" (legacy)
            strength = secrets_cfg.get("strength", "hard")
            action = secrets_cfg.get("action", "block" if strength == "hard" else "mask")
            rules["secrets"] = SecretRule(
                action=action,
                mask_value=secrets_cfg.get("mask_value", "[REDACTED]"),
            )

        # Sequencing
        seq_cfg = config.get("sequencing", {})
        if seq_cfg.get("enabled", True):
            rules["sequencing"] = SequenceRule(
                trigger_prefixes=tuple(seq_cfg.get("trigger_tools", (
                    "edit", "write", "create",
                ))),
                suggest_prefixes=tuple(seq_cfg.get("suggest_tools", (
                    "bash", "shell", "run", "exec",
                ))),
                strength=seq_cfg.get("strength", "soft"),
                nudge=seq_cfg.get("nudge", "Consider running tests to verify your changes."),
                cooldown=seq_cfg.get("cooldown", 3),
            )

        # Loop detection
        loop_cfg = config.get("loop_detection", {})
        if loop_cfg.get("enabled", True):
            rules["loop_detection"] = LoopDetectionRule(
                window=loop_cfg.get("window", 10),
                nudge_threshold=loop_cfg.get("nudge_threshold", 3),
                block_threshold=loop_cfg.get("block_threshold", 5),
                stagnation_threshold=loop_cfg.get("stagnation_threshold", 14),
            )

        # Session budget
        budget_cfg = config.get("session_budget", {})
        if budget_cfg.get("enabled", True):
            rules["session_budget"] = SessionBudgetRule(
                max_file_ops=budget_cfg.get("max_file_ops", 100),
                max_commands=budget_cfg.get("max_commands", 200),
                max_reads=budget_cfg.get("max_reads", 0),
                warn_at=budget_cfg.get("warn_at", 0.8),
            )

        # Thoroughness
        thor_cfg = config.get("thoroughness", {})
        if thor_cfg.get("enabled", True):
            rules["thoroughness"] = ThoroughnessRule(
                min_tools=thor_cfg.get("min_tools", 3),
                cooldown=thor_cfg.get("cooldown", 2),
            )

        # Tool resolution
        res_cfg = config.get("tool_resolution", {})
        if res_cfg.get("enabled", True):
            empty_cfg = res_cfg.get("empty_result", {})
            error_cfg = res_cfg.get("error_output", {})
            rules["tool_resolution"] = ToolResolutionRule(
                empty_result_nudge=empty_cfg.get("nudge", "Query returned no results. Try broadening your search."),
                error_output_nudge=error_cfg.get("nudge", "Command produced errors. Read the error output before proceeding."),
            )

        # Duplicate write
        dw_cfg = config.get("dup_write", {})
        if dw_cfg.get("enabled", True):
            rules["dup_write"] = DuplicateWriteRule(
                nudge_threshold=dw_cfg.get("nudge_threshold", 2),
                block_threshold=dw_cfg.get("block_threshold", 3),
            )

        # Lint gate — run the project linter on edited files (noticing offload).
        # workspace/mode fall back to $CG_LINT_WORKSPACE / $CG_LINT_MODE so the
        # rule can be enabled for a config-less deployment via env vars. ``linters``
        # maps file extensions to language linters (defaults: ruff/biome/gofmt).
        lint_cfg = config.get("lint", {})
        if lint_cfg.get("enabled", True):
            linters_cfg = lint_cfg.get("linters")
            if linters_cfg:
                linters = tuple(
                    LinterSpec(
                        name=lc.get("name", "custom"),
                        extensions=tuple(lc.get("extensions", ())),
                        command=tuple(lc.get("command", ())),
                        path_mode=lc.get("path_mode", "file"),
                        findings_mode=lc.get("findings_mode", "exitcode"),
                        enabled=lc.get("enabled", True),
                    )
                    for lc in linters_cfg
                )
            else:
                linters = default_linters()
            rules["lint"] = LintRule(
                workspace=workspace_from_env(lint_cfg.get("workspace")),
                mode=lint_cfg.get("mode", os.environ.get("CG_LINT_MODE", "nudge")),
                timeout=lint_cfg.get("timeout", 10.0),
                linters=linters,
            )

        return cls(**rules)

    @classmethod
    def defaults(cls) -> CodingGuardrails:
        """Create guardrails with all defaults enabled."""
        return cls(
            prerequisites=PrerequisiteRule(),
            path_safety=PathSafetyRule(),
            command_safety=CommandSafetyRule(),
            network=NetworkRule(),
            sensitive_files=SensitiveFileRule(),
            secrets=SecretRule(action="block"),
            loop_detection=LoopDetectionRule(),
            session_budget=SessionBudgetRule(),
            thoroughness=ThoroughnessRule(),
            sequencing=SequenceRule(),
            tool_resolution=ToolResolutionRule(),
            dup_write=DuplicateWriteRule(),
            lint=LintRule(),
        )

    def _active_rules(self) -> list[Rule]:
        """Return list of non-None rules."""
        return [r for r in [
            self.prerequisites,
            self.path_safety,
            self.command_safety,
            self.network,
            self.sensitive_files,
            self.secrets,
            self.loop_detection,
            self.session_budget,
            self.thoroughness,
            self.sequencing,
            self.tool_resolution,
            self.dup_write,
            self.lint,
        ] if r is not None]

    def check(self, calls: list[ToolCall]) -> CheckResult:
        """Check tool calls against all rules.

        For each call, runs through all rules. Collects:
        - Blocks: hard stops (call may not execute)
        - Nudges: soft suggestions (call proceeds with a message)
        - Allowed: calls that passed all rules

        If any call is blocked, all blocked calls are returned and
        the agent should handle them before proceeding.

        Args:
            calls: Tool calls from the LLM.

        Returns:
            CheckResult with blocked, nudged, and allowed calls.
        """
        result = CheckResult()
        rules = self._active_rules()
        debug = logger.isEnabledFor(logging.DEBUG)

        for call in calls:
            call_blocked = False
            call_nudges: list[RuleResult] = []

            if debug:
                logger.debug("  CALL %s", _fmt_call(call))

            for rule in rules:
                rule_result = rule.check(call)
                annotated = replace(rule_result, rule_name=rule.name)

                if annotated.action == Action.BLOCK:
                    result.blocked.append(annotated)
                    call_blocked = True
                    reason = annotated.reason or "policy violation"
                    logger.info(
                        "BLOCK %s | %s - %s",
                        _fmt_call(call), rule.name, reason,
                    )
                    if debug and annotated.nudge:
                        logger.debug("    nudge: %s", _short(annotated.nudge))
                    break  # No point checking further rules

                elif annotated.action == Action.NUDGE:
                    call_nudges.append(annotated)
                    result.nudges.append(annotated)
                    reason = annotated.reason or "advisory"
                    logger.info(
                        "NUDGE %s | %s - %s",
                        _fmt_call(call), rule.name, reason,
                    )
                    if debug and annotated.nudge:
                        logger.debug("    message: %s", _short(annotated.nudge))

            # Debug: single summary line per call

            if not call_blocked:
                result.allowed.append(call)
                if call_nudges:
                    nudge_names = ", ".join(
                        n.rule_name for n in call_nudges if n.rule_name
                    )
                    logger.debug(
                        "  %s | allowed (nudged: %s)", call.tool, nudge_names,
                    )
                else:
                    logger.debug("  %s | allowed", call.tool)

        # DEBUG: single summary line
        if debug:
            logger.debug("%s", result.summary())
        return result

    def record(self, calls: list[ToolCall]) -> None:
        """Record executed tool calls to update rule state.

        Call this after tool execution to keep rules in sync
        (e.g. tracking which files have been read).

        Args:
            calls: Tool calls that were executed.
        """
        for rule in self._active_rules():
            rule.record(calls)

    def check_tool_result(self, tool: str, result_text: str) -> RuleResult | None:
        """Check a tool result for empty/error patterns.

        Only the tool_resolution rule uses this.

        Args:
            tool: Tool name that produced the result.
            result_text: The result text.

        Returns:
            RuleResult with nudge if pattern detected, None otherwise.
        """
        if self.tool_resolution:
            return self.tool_resolution.check_result(tool, result_text)
        return None
