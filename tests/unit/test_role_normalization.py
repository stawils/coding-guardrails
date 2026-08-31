"""Role normalization tests — system/developer messages hoisted to the front.

The Qwen3.5/3.8 jinja template (llama-server --jinja parser generation) hard-fails
with "System message must be at the beginning" when a developer-role message appears
mid-stream. The proxy normalizes so OpenAI client streams (which emit developer
mid-conversation) still satisfy the template.
"""

from __future__ import annotations

from coding_guardrails.proxy.handler import _normalize_message_roles


def _msg(role: str, content: str = "x") -> dict:
    return {"role": role, "content": content}


def test_already_starts_with_system_is_unchanged():
    msgs = [_msg("system", "sys"), _msg("user", "u"), _msg("assistant", "a")]
    assert _normalize_message_roles(msgs) == msgs


def test_already_starts_with_developer_is_unchanged():
    msgs = [_msg("developer", "d"), _msg("user", "u")]
    assert _normalize_message_roles(msgs) == msgs


def test_mid_stream_developer_hoisted_to_front():
    msgs = [
        _msg("system", "sys"),
        _msg("user", "task"),
        _msg("developer", "runtime instr"),
        _msg("assistant", "a"),
    ]
    out = _normalize_message_roles(msgs)
    roles = [m["role"] for m in out]
    assert roles == ["system", "developer", "user", "assistant"]
    assert out[0]["content"] == "sys"
    assert out[1]["content"] == "runtime instr"
    # user/assistant relative order preserved
    assert out[2] == _msg("user", "task")
    assert out[3] == _msg("assistant", "a")


def test_multiple_system_dev_keep_relative_order():
    msgs = [
        _msg("user", "u"),
        _msg("system", "s1"),
        _msg("developer", "d1"),
        _msg("system", "s2"),
        _msg("user", "u2"),
    ]
    out = _normalize_message_roles(msgs)
    roles = [m["role"] for m in out]
    assert roles == ["system", "developer", "system", "user", "user"]
    asserts = ["s1", "d1", "s2", "u", "u2"]
    assert [m["content"] for m in out] == asserts


def test_no_system_messages_is_unchanged():
    msgs = [_msg("user", "u"), _msg("assistant", "a"), _msg("user", "u2")]
    assert _normalize_message_roles(msgs) == msgs


def test_empty_list():
    assert _normalize_message_roles([]) == []