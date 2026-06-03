"""Sensitive file protection — block writes to critical paths.

Prevents agents from overwriting:
- Secrets (.env, .ssh/, .gnupg/)
- Git internals (.git/)
- CI/CD pipelines (.github/workflows/, .gitlab-ci.yml, Jenkinsfile)
- Package manager hooks (package.json scripts, pre-commit config)
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from coding_guardrails.rules.base import Action, RuleResult, ToolCall
from coding_guardrails.rules.prerequisites import _tool_matches

# Tool prefixes that write files.
_WRITE_TOOLS = ("edit", "write", "create")

# Protected path patterns: (regex, label, default_action)
# action: "block" = always block, "nudge" = warn but allow
# Patterns are matched case-insensitively (see check())
_DEFAULT_PROTECTED: list[tuple[str, str, str]] = [
    # Git internals
    (r"^(\./)?\.git/", "Git internal files", "block"),
    # SSH/GPG keys
    (r"^(\./)?\.ssh/", "SSH directory", "block"),
    (r"^(\./)?\.gnupg/", "GPG directory", "block"),
    # CI/CD pipelines
    (r"^(\./)?\.github/workflows/", "GitHub Actions workflow", "block"),
    (r"^(\./)?\.gitlab-ci\.yml$", "GitLab CI config", "block"),
    (r"^(\./)?Jenkinsfile$|^(\./)?jenkinsfile$", "Jenkins pipeline", "block"),
    (r"^(\./)?\.circleci/", "CircleCI config", "block"),
    # Pre-commit / git hooks
    (r"^(\./)?\.pre-commit-config\.ya?ml$", "Pre-commit config", "block"),
    (r"^(\./)?\.husky/", "Husky git hooks", "block"),
    # Secrets
    (r"^(\./)?\.env$", "Environment secrets file", "nudge"),
    (r"^(\./)?\.env\.", "Environment secrets file", "nudge"),
]

# Additional nested path patterns that must be blocked (e.g., subdir/.git/config)
# These are checked separately after stripping ./ prefix
_NESTED_PROTECTED_PATTERNS: list[tuple[str, str, str]] = [
    # Match .git/ anywhere in the path (nested directories)
    (r"/\.git/", "Git internal files", "block"),
    # Match .ssh/ anywhere in the path
    (r"/\.ssh/", "SSH directory", "block"),
    (r"/\.gnupg/", "GPG directory", "block"),
    # Match .github/workflows/ anywhere
    (r"/\.github/workflows/", "GitHub Actions workflow", "block"),
    (r"/\.gitlab-ci\.yml$", "GitLab CI config", "block"),
    (r"/(Jenkinsfile|jenkinsfile)$", "Jenkins pipeline", "block"),
    (r"/\.circleci/", "CircleCI config", "block"),
    # Match .pre-commit-config.yaml file anywhere
    (r"/\.pre-commit-config\.ya?ml$", "Pre-commit config", "block"),
    (r"/\.husky/", "Husky git hooks", "block"),
    # Match .env file anywhere (nudge, not block)
    (r"/\.env$", "Environment secrets file", "nudge"),
    (r"/\.env\.", "Environment secrets file", "nudge"),
]


@dataclass
class SensitiveFileRule:
    """Block writes to sensitive files and directories.

    Attributes:
        write_tools: Tool name prefixes that write files.
        path_arg: Argument name containing the file path.
        protected: List of (regex_pattern, label, action) tuples for root-level paths.
            action is "block" or "nudge".
        extra_protected: Additional protected paths to add.
        nested_protected: List of (regex_pattern, label, action) tuples for nested paths.
    """

    write_tools: tuple[str, ...] = _WRITE_TOOLS
    path_arg: str = "path"
    protected: list[tuple[str, str, str]] = field(
        default_factory=lambda: list(_DEFAULT_PROTECTED)
    )
    extra_protected: list[tuple[str, str, str]] = field(default_factory=list)
    nested_protected: list[tuple[str, str, str]] = field(
        default_factory=lambda: list(_NESTED_PROTECTED_PATTERNS)
    )

    @property
    def name(self) -> str:
        return "sensitive_files"

    def check(self, call: ToolCall) -> RuleResult:
        if not _tool_matches(call.tool, self.write_tools):
            return RuleResult.allow(call.tool)

        path = call.args.get(self.path_arg, "")
        if not path or not isinstance(path, str):
            return RuleResult.allow(call.tool)

        # Normalize: expand user, strip leading ./
        normalized = os.path.normpath(os.path.expanduser(path))
        # Keep relative form for matching
        rel = normalized.lstrip("./")
        # Normalize to lowercase for additional case-insensitive protection
        rel_lower = rel.lower()
        normalized_lower = normalized.lower()

        # Check root-level patterns first (paths starting with .)
        all_root_protected = list(self.protected) + list(self.extra_protected)
        for pattern, label, action in all_root_protected:
            if (re.search(pattern, rel_lower) or re.search(pattern, normalized_lower)) or (
                re.search(pattern, rel) or re.search(pattern, normalized)
            ):
                if action == "block":
                    return RuleResult.block(
                        call.tool,
                        nudge=f"Write to {label} blocked: '{path}' is a protected path.",
                        reason=f"sensitive file: {path} ({label})",
                    )
                else:
                    return RuleResult.nudge(
                        call.tool,
                        message=f"⚠️ Writing to {label}: '{path}'. "
                        "Make sure this doesn't expose secrets.",
                    )

        # Check nested patterns (paths like subdir/.git/config)
        for pattern, label, action in self.nested_protected:
            if re.search(pattern, rel_lower):
                if action == "block":
                    return RuleResult.block(
                        call.tool,
                        nudge=f"Write to {label} blocked: '{path}' is a protected path.",
                        reason=f"sensitive file: {path} ({label})",
                    )
                else:
                    return RuleResult.nudge(
                        call.tool,
                        message=f"⚠️ Writing to {label}: '{path}'. "
                        "Make sure this doesn't expose secrets.",
                    )

        return RuleResult.allow(call.tool)

    def record(self, calls: list[ToolCall]) -> None:
        """Stateless — nothing to record."""
        pass
