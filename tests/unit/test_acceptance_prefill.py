"""Tests for acceptance-finalization prefill (F9 fix).

The model often does correct work but refuses to emit the structured
acceptance-report JSON, narrating in prose instead. The fix appends a JSON
prefill as a trailing assistant message when the conversation contains an
acceptance-finalization prompt, forcing the model to continue in JSON.

These tests pin the detection logic: marker matching, content-shape tolerance
(string vs OpenAI content-parts list), trailing-turn guard, and idempotency.
"""
from __future__ import annotations

from coding_guardrails.proxy.client import SafeLlamafileClient

MARKER = "Acceptance Finalization"


def _client() -> SafeLlamafileClient:
    """Build a client without network init — _inject_acceptance_prefill is pure."""
    return SafeLlamafileClient.__new__(SafeLlamafileClient)


def test_no_marker_leaves_messages_untouched() -> None:
    c = _client()
    msgs = [
        {"role": "user", "content": "Write a unit test for Foo."},
        {"role": "assistant", "content": "Done."},
    ]
    out = c._inject_acceptance_prefill(msgs)
    assert out == msgs
    assert len(out) == 2


def test_marker_string_content_appends_prefill() -> None:
    c = _client()
    msgs = [
        {"role": "user", "content": "Write a unit test for Foo."},
        {"role": "assistant", "content": "Created foo.spec.ts."},
        {"role": "user", "content": f"## {MARKER}\nEmit the report."},
    ]
    out = c._inject_acceptance_prefill(msgs)
    assert len(out) == 4
    assert out[-1]["role"] == "assistant"
    assert out[-1]["content"].startswith('{"criteriaSatisfied"')


def test_marker_in_content_parts_list_is_detected() -> None:
    """Forge/OpenAI may send content as a list of parts, not a plain string.
    Detection must flatten both shapes — this was the live-path miss."""
    c = _client()
    msgs = [
        {"role": "user", "content": [{"type": "text", "text": f"## {MARKER}\nturn 1 of 3"}]},
    ]
    out = c._inject_acceptance_prefill(msgs)
    assert len(out) == 2
    assert out[-1]["role"] == "assistant"


def test_marker_anywhere_in_history_appends_prefill() -> None:
    """Forge may interleave tool results after the finalization prompt;
    detection scans all user turns, not just the last few."""
    c = _client()
    msgs = [
        {"role": "user", "content": f"## {MARKER}\nEmit report."},
        {"role": "assistant", "content": [{"type": "text", "text": "thinking"}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "x", "content": "ok"}]},
    ]
    out = c._inject_acceptance_prefill(msgs)
    assert out[-1]["role"] == "assistant"


def test_trailing_assistant_turn_is_not_reprimed() -> None:
    """If the last message is already an assistant turn, the request is a
    tool-result follow-up or retry — a prefill would corrupt it."""
    c = _client()
    msgs = [
        {"role": "user", "content": f"## {MARKER}\nEmit report."},
        {"role": "assistant", "content": '{"criteriaSatisfied":'[:5]},
    ]
    out = c._inject_acceptance_prefill(msgs)
    assert out == msgs


def test_does_not_mutate_input() -> None:
    c = _client()
    msgs = [{"role": "user", "content": f"## {MARKER}\nEmit report."}]
    original_len = len(msgs)
    _ = c._inject_acceptance_prefill(msgs)
    assert len(msgs) == original_len  # input list untouched


def test_empty_messages_is_safe() -> None:
    assert _client()._inject_acceptance_prefill([]) == []


def test_prefill_seeds_contract_criterion_id() -> None:
    """When the nudge lists a real criterion id ('Criteria:\n- comment: ...'),
    the prefill must seed THAT id so the model's report matches the contract
    (otherwise Pi rejects with 'Required criterion <id> was not reported')."""
    c = _client()
    msgs = [
        {"role": "user", "content": f"## {MARKER}\n## Acceptance Contract\nCriteria:\n- comment: src has the marker"},
    ]
    out = c._inject_acceptance_prefill(msgs)
    assert out[-1]["role"] == "assistant"
    assert out[-1]["content"] == '{"criteriaSatisfied": [{"id": "comment", "status": "'


def test_prefill_falls_back_when_no_criterion_id() -> None:
    """Nudges without a 'Criteria:'/'- <id>:' section fall back to the generic
    prefill (backward-compatible with the older detection tests)."""
    c = _client()
    msgs = [{"role": "user", "content": f"## {MARKER}\nEmit the report."}]
    out = c._inject_acceptance_prefill(msgs)
    assert out[-1]["content"] == '{"criteriaSatisfied": [{"id": "'


def test_prefill_ignores_example_block_criterion_id() -> None:
    """The nudge's fenced EXAMPLE uses a generic 'criterion-1' JSON id; the
    resolver must pick the real markdown-listed id, not the example's."""
    c = _client()
    nudge = (
        f"## {MARKER}\n"
        "## Acceptance Contract\nCriteria:\n- all-tests: cargo test passes\n"
        "```acceptance-report\n{\"criteriaSatisfied\": [{\"id\": \"criterion-1\"}]}\n```"
    )
    out = c._inject_acceptance_prefill([{"role": "user", "content": nudge}])
    assert '"id": "all-tests"' in out[-1]["content"]
    assert "criterion-1" not in out[-1]["content"]
