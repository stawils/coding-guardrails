"""Tests for acceptance-report fencing.

The F9 prefill makes the model emit the report as bare JSON; this module
wraps it in the fence Pi requires. The gold-standard test re-implements
Pi's real parser (parseAcceptanceReport from pi-subagents) and asserts that
wrap_bare_acceptance_report(raw) produces output Pi would ACCEPT.
"""
from __future__ import annotations

import json
from typing import Any

from coding_guardrails.proxy.acceptance import wrap_bare_acceptance_report


# ── Real-shaped fixtures (modeled on observed Qwen3.5-9B finalization output) ──

BARE_REPORT = (
    '{"criteriaSatisfied": [{"id": "all-tests", "status": "satisfied", '
    '"evidence": "cargo test: 8 passed"}], "changedFiles": '
    '["src/store.rs"], "commandsRun": [{"command": "cargo test", '
    '"result": "passed", "summary": "8 passed"}], "residualRisks": []}'
)

NON_REPORT_JSON = '{"name": "kvstore", "version": "0.1.0"}'


# ── Core wrapping behaviour ───────────────────────────────────────────────────


def test_bare_report_gets_fenced() -> None:
    out = wrap_bare_acceptance_report(BARE_REPORT)
    assert out.startswith("```acceptance-report\n")
    assert out.rstrip().endswith("```")
    # The JSON inside the fence is valid and preserves the report fields.
    body = out.split("```acceptance-report\n", 1)[1].rsplit("```", 1)[0].strip()
    parsed = json.loads(body)
    assert parsed["changedFiles"] == ["src/store.rs"]


def test_bare_report_with_surrounding_prose_preserves_prose() -> None:
    text = "Here is my final report.\n\n" + BARE_REPORT + "\n\nThanks."
    out = wrap_bare_acceptance_report(text)
    assert out.startswith("Here is my final report.")
    assert out.endswith("Thanks.")
    assert "```acceptance-report" in out


def test_already_fenced_is_unchanged_idempotent() -> None:
    fenced = "```acceptance-report\n" + json.dumps(
        {"changedFiles": ["a.rs"]}, indent=2
    ) + "\n```"
    assert wrap_bare_acceptance_report(fenced) == fenced


def test_marker_form_is_unchanged() -> None:
    text = "ACCEPTANCE_REPORT: " + BARE_REPORT
    assert wrap_bare_acceptance_report(text) == text


def test_no_report_returns_unchanged() -> None:
    assert wrap_bare_acceptance_report("All 8 tests pass. Done.") == "All 8 tests pass. Done."


def test_non_report_json_is_not_wrapped() -> None:
    # A package.json-shaped object must not be mistaken for a report.
    assert wrap_bare_acceptance_report(NON_REPORT_JSON) == NON_REPORT_JSON


def test_empty_string_is_unchanged() -> None:
    assert wrap_bare_acceptance_report("") == ""


def test_wrapped_acceptance_key_form_is_recognized() -> None:
    # Pi also accepts {"acceptance": {...}}; the wrapper should fence it too.
    text = '{"acceptance": {"changedFiles": ["a.rs"], "residualRisks": []}}'
    out = wrap_bare_acceptance_report(text)
    assert "```acceptance-report" in out


def test_last_report_chosen_when_multiple_json_objects() -> None:
    # Model quoted a non-report object first, then the real report last.
    text = NON_REPORT_JSON + "\n" + BARE_REPORT
    out = wrap_bare_acceptance_report(text)
    # The non-report object is untouched, the report is fenced.
    assert NON_REPORT_JSON in out
    assert out.count("```acceptance-report") == 1


def test_nested_objects_and_string_escapes_parsed_correctly() -> None:
    # Strings containing braces/quotes must not break balanced extraction.
    report = (
        '{"commandsRun": [{"command": "echo \\"}{\\" ", "result": "passed", '
        '"summary": "has } braces"}], "residualRisks": []}'
    )
    out = wrap_bare_acceptance_report(report)
    body = out.split("```acceptance-report\n", 1)[1].rsplit("```", 1)[0].strip()
    parsed = json.loads(body)
    assert parsed["commandsRun"][0]["summary"] == "has } braces"


def test_idempotent_on_its_own_output() -> None:
    out = wrap_bare_acceptance_report(BARE_REPORT)
    assert wrap_bare_acceptance_report(out) == out


# ── Gold standard: Pi's real parser accepts the wrapped output ────────────────


def _pi_is_report(value: Any) -> bool:
    """Port of isAcceptanceReport from pi-subagents acceptance-reports.ts."""
    if not isinstance(value, dict):
        return False
    keys = set(value.keys())
    report_fields = {
        "criteriaSatisfied", "changedFiles", "testsAddedOrUpdated", "commandsRun",
        "validationOutput", "residualRisks", "noStagedFiles", "diffSummary",
        "reviewFindings", "manualNotes", "notes",
    }
    return bool(keys & report_fields)


def _pi_parse_acceptance_report(output: str) -> bool:
    """Port of parseAcceptanceReport success path (fenced form only)."""
    import re
    matches = re.findall(r"```acceptance-report\s*\n([\s\S]*?)```", output, re.IGNORECASE)
    for body in matches:
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            continue
        report = parsed.get("acceptance", parsed) if isinstance(parsed, dict) else parsed
        if _pi_is_report(report):
            return True
    return False


def test_gold_standard_pi_parser_accepts_wrapped_bare_report() -> None:
    # The bare report alone would be REJECTED by Pi's parser...
    assert not _pi_parse_acceptance_report(BARE_REPORT)
    # ...but after wrapping, Pi accepts it.
    assert _pi_parse_acceptance_report(wrap_bare_acceptance_report(BARE_REPORT))
