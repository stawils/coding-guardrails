"""Acceptance-report fencing — completes the F9 prefill fix.

Pi's subagent runtime requires the final acceptance report in one of two
shapes (see pi-subagents ``runs/shared/acceptance-reports.ts``):

  1. a fenced block:  ```acceptance-report\\n{...}\\n```
  2. a marker line:   ACCEPTANCE_REPORT: {...}

The F9 prefill (``client.SafeLlamafileClient._inject_acceptance_prefill``)
makes local models emit the report as valid JSON by seeding a trailing
assistant prefix. But models like Qwen3.5-9B drop the fence and emit the
report *bare* (``{...}``), which Pi's parser rejects with
``Structured acceptance report not found.``

This module wraps any bare acceptance-report JSON object found in the model's
text response into the fenced shape Pi accepts. It is pure response shaping:
no I/O, no model calls, and a no-op when no bare report is present.

The report-field set mirrors Pi's ``isAcceptanceReport`` so we only ever
fence objects that Pi will actually accept.
"""
from __future__ import annotations

import json
from typing import Any

# Fields Pi treats as acceptance-report evidence (mirror of isAcceptanceReport
# in pi-subagents/src/runs/shared/acceptance-reports.ts).
_REPORT_FIELDS = frozenset({
    "criteriaSatisfied",
    "changedFiles",
    "testsAddedOrUpdated",
    "commandsRun",
    "validationOutput",
    "residualRisks",
    "noStagedFiles",
    "diffSummary",
    "reviewFindings",
    "manualNotes",
    "notes",
})
# Pi's parser also accepts {"acceptance": {...}} wrapping.
_WRAPPER_KEY = "acceptance"

_FENCE_OPEN = "```acceptance-report"
_FENCE_CLOSE = "```"
_MARKER = "ACCEPTANCE_REPORT:"


def _looks_like_report(obj: Any) -> bool:
    """True if ``obj`` is a dict carrying at least one acceptance-report field.

    Handles the optional ``{"acceptance": {...}}`` wrapper that Pi also
    accepts.
    """
    if not isinstance(obj, dict):
        return False
    target: dict[str, Any] = obj
    wrapped = obj.get(_WRAPPER_KEY)
    if isinstance(wrapped, dict):
        target = wrapped
    return any(key in target for key in _REPORT_FIELDS)


def _extract_balanced_json(text: str, start: int) -> str | None:
    """Return the balanced ``{...}`` substring beginning at ``start``.

    Handles nested objects and string escapes. Mirrors the
    ``extractBalancedJson`` helper in Pi's acceptance-reports.ts so the two
    agree on object boundaries. Returns None if the braces never balance.
    """
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _find_bare_report(text: str) -> tuple[int, int] | None:
    """Locate the last bare JSON object that parses as an acceptance report.

    Returns ``(start, end_exclusive)`` of the object substring, or None.
    The model's report is normally the final JSON object in the response, so
    the last matching candidate wins; this also avoids wrapping an example
    object the model quoted earlier in its answer.
    """
    last: tuple[int, int] | None = None
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        obj = _extract_balanced_json(text, i)
        if obj is None:
            continue
        try:
            parsed = json.loads(obj)
        except (json.JSONDecodeError, ValueError):
            continue
        if _looks_like_report(parsed):
            last = (i, i + len(obj))
    return last


def wrap_bare_acceptance_report(text: str) -> str:
    """Wrap a bare acceptance-report JSON object in the fence Pi requires.

    - If ``text`` already carries a fenced block or an ``ACCEPTANCE_REPORT:``
      marker, it is returned unchanged (idempotent / respects valid output).
    - Otherwise, if a bare JSON object with acceptance-report fields is found,
      that object is re-serialized (``indent=2``) and wrapped in a
      ```` ```acceptance-report ```` fence, preserving any surrounding prose.
    - If no report is found, the text is returned unchanged.
    """
    if not text:
        return text
    if _FENCE_OPEN in text or _MARKER in text:
        return text
    found = _find_bare_report(text)
    if found is None:
        return text
    start, end = found
    try:
        parsed = json.loads(text[start:end])
    except (json.JSONDecodeError, ValueError):
        return text
    fenced = _FENCE_OPEN + "\n" + json.dumps(parsed, indent=2) + "\n" + _FENCE_CLOSE
    return text[:start] + fenced + text[end:]
