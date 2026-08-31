"""Convergence nudge tests — deep tool loops get a conditional finalize reminder.

Targets the "keeps going" terminal-discipline failure: the model re-verifies
instead of writing the deliverable + final report when the conversation is deep.
The nudge is conditional ("if the task is complete...") so it never forces
premature termination.
"""

from __future__ import annotations

from coding_guardrails.proxy.handler import _convergence_nudge_openai


def _assistant_with_tool_calls(i: int) -> dict:
    return {"role": "assistant", "content": "x", "tool_calls": [{"id": f"c{i}"}]}


def _plain_assistant() -> dict:
    return {"role": "assistant", "content": "no calls"}


def _user() -> dict:
    return {"role": "user", "content": "u"}


def test_under_threshold_no_nudge():
    msgs = [_user(), _assistant_with_tool_calls(1), _user(),
            _assistant_with_tool_calls(2), _user()]
    assert _convergence_nudge_openai(msgs, after=8) is None


def test_at_threshold_nudge():
    msgs = [ _assistant_with_tool_calls(i) for i in range(8) ]
    nudge = _convergence_nudge_openai(msgs, after=8)
    assert nudge is not None
    assert "STOP calling tools" in nudge
    assert "If the task is complete" in nudge


def test_disabled_with_zero():
    msgs = [ _assistant_with_tool_calls(i) for i in range(20) ]
    assert _convergence_nudge_openai(msgs, after=0) is None


def test_plain_assistant_turns_do_not_count():
    msgs = [ _plain_assistant() for _ in range(12) ]
    assert _convergence_nudge_openai(msgs, after=8) is None


def test_mixed_history_counts_tool_call_turns_only():
    msgs = []
    for i in range(7):
        msgs.append(_assistant_with_tool_calls(i))
        msgs.append(_user())
    msgs.append(_plain_assistant())
    msgs.append(_plain_assistant())
    assert _convergence_nudge_openai(msgs, after=8) is None
    # one more tool-call turn → crosses the line
    msgs.append(_assistant_with_tool_calls(9))
    assert _convergence_nudge_openai(msgs, after=8) is not None