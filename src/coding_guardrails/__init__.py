"""Coding guardrails — safe, reliable local coding agent backend.

Layer 1: Forge (mechanical reliability — rescue parsing, retries, validation).
Layer 2: Coding guardrails (read-before-edit, path safety, command blocking,
         secret masking, test-after-change, tool resolution).
"""

from coding_guardrails.middleware import CodingGuardrails
from coding_guardrails.rules.base import Action, CheckResult, RuleResult, ToolCall
from coding_guardrails.config import load_config, load_guardrail_config

__all__ = [
    "CodingGuardrails",
    "Action",
    "CheckResult",
    "RuleResult",
    "ToolCall",
    "load_config",
    "load_guardrail_config",
]
