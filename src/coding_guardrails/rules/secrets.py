"""Secret detection and masking.

Detects API keys, tokens, private keys, and other secrets in tool
call arguments. Masks them with [REDACTED].
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from coding_guardrails.rules.base import Action, RuleResult, ToolCall


# Built-in secret patterns: (regex, label, flags)
_BUILTIN_PATTERNS: list[tuple[str, str, int]] = [
    # OpenAI API keys
    (r"sk-[a-zA-Z0-9]{20,}", "OpenAI API key", 0),
    # GitHub personal access tokens
    (r"ghp_[a-zA-Z0-9]{36}", "GitHub PAT", 0),
    # GitHub OAuth tokens
    (r"gho_[a-zA-Z0-9]{36}", "GitHub OAuth", 0),
    # GitHub fine-grained PATs
    (r"github_pat_[a-zA-Z0-9_]{22,}", "GitHub fine-grained PAT", 0),
    # AWS access keys
    (r"AKIA[0-9A-Z]{16}", "AWS access key", 0),
    # AWS secret keys (high entropy after known prefix)
    (r"aws_secret_access_key\s*=\s*[A-Za-z0-9/+=]{40}", "AWS secret key", re.IGNORECASE),
    # Private keys
    (r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----", "Private key", 0),
    # Slack tokens
    (r"xox[baprs]-[a-zA-Z0-9-]{10,}", "Slack token", 0),
    # Generic high-entropy token patterns
    (r"(?:api[_-]?key|token|secret|password)\s*[:=]\s*['\"]?[A-Za-z0-9+/=_-]{32,}['\"]?", "Generic secret", re.IGNORECASE),
]

_MASK_VALUE = "[REDACTED]"


@dataclass
class SecretRule:
    """Detect and mask secrets in tool call arguments.

    Scans all string arguments for known secret patterns.
    When found, masks the secret with [REDACTED] and returns a nudge.

    Attributes:
        action: What to do when secrets are found ("mask" or "block").
        extra_patterns: Additional (regex, label) tuples.
        mask_value: Replacement string for detected secrets.
    """

    action: str = "mask"
    extra_patterns: list[tuple[str, str]] = field(default_factory=list)
    mask_value: str = _MASK_VALUE

    @property
    def name(self) -> str:
        return "secrets"

    def check(self, call: ToolCall) -> RuleResult:
        all_patterns: list[tuple[str, str, int]] = list(_BUILTIN_PATTERNS)
        for pat, label in self.extra_patterns:
            all_patterns.append((pat, label, 0))

        secrets_found: list[str] = []
        masked_args: dict = {}

        for key, value in call.args.items():
            if not isinstance(value, str):
                masked_args[key] = value
                continue

            masked = value
            for pattern, label, flags in all_patterns:
                matches = re.findall(pattern, value, flags)
                if matches:
                    secrets_found.append(label)
                    masked = re.sub(pattern, self.mask_value, masked, flags=flags)

            masked_args[key] = masked

        if not secrets_found:
            return RuleResult.allow(call.tool)

        # Update the call's args with masked values
        call.args.update(masked_args)

        labels = ", ".join(set(secrets_found))
        if self.action == "block":
            return RuleResult.block(
                call.tool,
                nudge=f"Secret detected and blocked: {labels}. "
                "Do not include secrets in tool arguments.",
                reason=f"secrets detected: {labels}",
            )

        return RuleResult.nudge(
            call.tool,
            message=f"Secret detected and masked for safety: {labels}.",
        )

    def record(self, calls: list[ToolCall]) -> None:
        """Secret detection is stateless — nothing to record."""
        pass
