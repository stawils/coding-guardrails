"""Value fidelity rule — block terminal calls with placeholder/unresolved values.

When the model submits a "report" or "answer" tool call, checks for two
failure modes common with small models:

1. **Placeholder leakage**: The report contains bracketed placeholder values
   like [RESTRICTED], [PROTECTED], [UNAVAILABLE] that were redirect hints
   in earlier tool results, not actual data. The model copied the placeholder
   instead of the resolved value.

2. **Missing key values**: Tool results contained codes/IDs/classifications
   that don't appear anywhere in the report. The model paraphrased or
   omitted them.

Mode 1 (placeholders) triggers a hard BLOCK with the correct values.
Mode 2 (missing values) triggers a NUDGE to re-check.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from coding_guardrails.rules.base import Action, RuleResult, ToolCall, Rule


# Tool names that indicate a final submission / terminal call
_TERMINAL_PATTERNS = (
    "submit", "report", "respond", "answer", "present",
    "summarize", "diagnose", "recommend", "complete",
)


def _is_terminal_call(call: ToolCall) -> bool:
    """Check if this looks like a terminal/submission tool call."""
    name_lower = call.tool.lower()
    return any(p in name_lower for p in _TERMINAL_PATTERNS)


def _extract_key_values(text: str) -> set[str]:
    """Extract plausible key values from tool result text.

    Focus on high-signal values that are easy to paraphrase:
    - Codes/classifications: L3, B7, Confidential
    - IDs with prefixes: TX-1001, E-1847, U-1001
    - Phone numbers

    NOT generic numbers/dates which appear differently in reports.
    """
    values: set[str] = set()

    # Alphanumeric codes with prefix (L3, B7, TX-1001, E-1847, U-1001)
    for m in re.finditer(r'\b([A-Z]{1,3}-?\d{1,5})\b', text):
        val = m.group(1)
        if any(c.isalpha() for c in val):
            values.add(val)

    # Classification levels (exact word match)
    for word in ("confidential", "secret", "classified"):
        if word in text.lower():
            values.add(word)

    # Phone numbers
    for m in re.finditer(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', text):
        values.add(m.group())

    return values


def _find_placeholders(text: str) -> list[str]:
    """Find bracketed placeholder values like [RESTRICTED], [PROTECTED].

    These are redirect hints from tool results — the model should have
    resolved them via subsequent tool calls. If they appear in a final
    report, the model copied the placeholder verbatim.
    """
    placeholders = []
    for m in re.finditer(r'\[([A-Z][A-Z\s-]{2,})\]', text):
        val = m.group(1).strip()
        # Filter out common non-placeholder bracketed content
        _SKIP = {
            "UNAVAILABLE IN THIS RECORD", "MANAGED EXTERNALLY",
            "PROTECTED", "RESTRICTED", "REQUEST VIA",
        }
        if val.upper() in _SKIP or any(s in val.upper() for s in ("REQUEST VIA",)):
            placeholders.append(val)
    return placeholders


@dataclass
class ValueFidelityRule:
    """Block/nudge when terminal calls have unresolved or missing values.

    This rule inspects terminal tool calls (submit, report, respond, etc.)
    and checks for:
    1. Placeholder values (hard block with correct values)
    2. Missing key values from tool results (nudge)

    The handler feeds it recent tool result text via ``set_recent_results()``.
    """

    nudge_threshold: int = 2  # Missing >= this many key values triggers nudge
    _recent_results_text: str = field(default="", repr=False)

    @property
    def name(self) -> str:
        return "value_fidelity"

    def set_recent_results(self, texts: list[str]) -> None:
        """Feed recent tool result texts for comparison."""
        self._recent_results_text = " ".join(texts)

    def check(self, call: ToolCall) -> RuleResult:
        if not _is_terminal_call(call):
            return RuleResult.allow(call.tool)

        call_text = " ".join(str(v) for v in call.args.values())

        # ── Mode 1: Placeholder detection (hard block) ──
        placeholders = _find_placeholders(call_text)
        if placeholders:
            # Build a correction hint from tool results
            # Look for the key that had the placeholder and find the resolved value
            corrections: list[str] = []
            result_values = _extract_key_values(self._recent_results_text)
            for ph in placeholders:
                # Try to find what the placeholder was for
                ph_lower = ph.lower()
                for line in self._recent_results_text.split("\n"):
                    if ":" not in line:
                        continue
                    key, _, val = line.partition(":")
                    key = key.strip()
                    val = val.strip()
                    # If the key matches the placeholder context, offer the real value
                    if (ph_lower in key.lower() or ph_lower in val.lower()
                            or val.lower().startswith(ph_lower)):
                        # This line is the resolved version
                        if not val.startswith("["):
                            corrections.append(f"{key}: {val}")
                            break

            correction_text = ""
            if corrections:
                correction_text = " Correct values: " + "; ".join(corrections) + "."

            return RuleResult.block(
                call.tool,
                nudge=(
                    f"Your report contains unresolved placeholder values "
                    f"({', '.join(f'[{p}]' for p in placeholders)}). "
                    f"These are redirect hints from earlier tool results, "
                    f"not actual data. You must use the RESOLVED values "
                    f"from subsequent tool calls instead.{correction_text} "
                    f"Rewrite your report with the actual values."
                ),
                reason=f"value_fidelity: {len(placeholders)} placeholder values: {', '.join(placeholders)}",
            )

        # ── Mode 2: Missing key values (nudge) ──
        if not self._recent_results_text:
            return RuleResult.allow(call.tool)

        result_values = _extract_key_values(self._recent_results_text)
        if not result_values:
            return RuleResult.allow(call.tool)

        call_text_lower = call_text.lower()
        missing = []
        for val in sorted(result_values):
            if val.lower() not in call_text_lower and len(val) >= 2:
                missing.append(val)

        if len(missing) >= self.nudge_threshold:
            missing_str = ", ".join(missing[:6])
            return RuleResult.nudge(
                call.tool,
                message=(
                    f"Your report may be missing key values from tool results. "
                    f"Verify these appear exactly as returned: {missing_str}. "
                    f"Use exact values (e.g. 'L3' not 'Level 3'). "
                    f"Rewrite your report including ALL of these values."
                ),
            )

        return RuleResult.allow(call.tool)

    def record(self, calls: list[ToolCall]) -> None:
        """No state to update."""
        pass
