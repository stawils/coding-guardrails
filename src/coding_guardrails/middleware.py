"""Guardrail middleware — composes all rules into a single check/record API.

The middleware holds all configured rules and provides:
- check(): inspect tool calls against all rules
- record(): update rule state after execution
- check_result(): inspect tool results for resolution nudges
"""

from __future__ import annotations

from dataclasses import dataclass, field

from coding_guardrails.rules.base import (
    Action,
    CheckResult,
    Rule,
    RuleResult,
    ToolCall,
)
from coding_guardrails.rules.commands import CommandSafetyRule
from coding_guardrails.rules.path_safety import PathSafetyRule
from coding_guardrails.rules.prerequisites import PrerequisiteRule
from coding_guardrails.rules.secrets import SecretRule
from coding_guardrails.rules.sequencing import SequenceRule
from coding_guardrails.rules.tool_resolution import ToolResolutionRule


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
    """

    prerequisites: PrerequisiteRule | None = None
    path_safety: PathSafetyRule | None = None
    command_safety: CommandSafetyRule | None = None
    secrets: SecretRule | None = None
    sequencing: SequenceRule | None = None
    tool_resolution: ToolResolutionRule | None = None

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
            rules["command_safety"] = CommandSafetyRule(
                blocked=cmd_cfg.get("blocked", None),
                blocked_patterns=cmd_cfg.get("blocked_patterns", None),
                require_confirmation=cmd_cfg.get("require_confirmation", None),
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

        # Tool resolution
        res_cfg = config.get("tool_resolution", {})
        if res_cfg.get("enabled", True):
            empty_cfg = res_cfg.get("empty_result", {})
            error_cfg = res_cfg.get("error_output", {})
            rules["tool_resolution"] = ToolResolutionRule(
                empty_result_nudge=empty_cfg.get("nudge", "Query returned no results. Try broadening your search."),
                error_output_nudge=error_cfg.get("nudge", "Command produced errors. Read the error output before proceeding."),
            )

        return cls(**rules)

    @classmethod
    def defaults(cls) -> CodingGuardrails:
        """Create guardrails with all defaults enabled."""
        return cls(
            prerequisites=PrerequisiteRule(),
            path_safety=PathSafetyRule(),
            command_safety=CommandSafetyRule(),
            secrets=SecretRule(),
            sequencing=SequenceRule(),
            tool_resolution=ToolResolutionRule(),
        )

    def _active_rules(self) -> list[Rule]:
        """Return list of non-None rules."""
        return [r for r in [
            self.prerequisites,
            self.path_safety,
            self.command_safety,
            self.secrets,
            self.sequencing,
            self.tool_resolution,
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

        for call in calls:
            call_blocked = False
            call_nudges: list[str] = []

            for rule in rules:
                rule_result = rule.check(call)

                if rule_result.action == Action.BLOCK:
                    result.blocked.append(rule_result)
                    call_blocked = True
                    break  # No point checking further rules

                elif rule_result.action == Action.NUDGE:
                    result.nudges.append(rule_result)

            if not call_blocked:
                result.allowed.append(call)

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
