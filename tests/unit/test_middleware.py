"""Tests for the middleware (rule composition)."""

from coding_guardrails.rules.base import Action, ToolCall
from coding_guardrails.middleware import CodingGuardrails


def test_defaults_all_rules_active():
    gw = CodingGuardrails.defaults()
    assert gw.prerequisites is not None
    assert gw.path_safety is not None
    assert gw.command_safety is not None
    assert gw.secrets is not None
    assert gw.sequencing is not None
    assert gw.tool_resolution is not None


def test_edit_without_read_nudge():
    gw = CodingGuardrails.defaults()
    result = gw.check([ToolCall(tool="edit_file", args={"path": "/home/user/main.py"})])
    # First attempt: nudge (prerequisites)
    assert result.has_nudges
    assert len(result.allowed) == 1  # Soft nudge still allows execution


def test_destructive_command_blocked():
    gw = CodingGuardrails.defaults()
    result = gw.check([ToolCall(tool="bash", args={"command": "rm -rf /"})])
    assert result.has_blocks
    assert len(result.allowed) == 0


def test_path_traversal_blocked():
    gw = CodingGuardrails.defaults()
    result = gw.check([ToolCall(tool="read_file", args={"path": "../../../etc/shadow"})])
    assert result.has_blocks


def test_read_edit_workflow():
    gw = CodingGuardrails.defaults()

    # Step 1: Read file — allowed
    read_call = ToolCall(tool="read_file", args={"path": "/home/user/main.py"})
    result = gw.check([read_call])
    assert len(result.allowed) == 1

    # Record the read
    gw.record([read_call])

    # Step 2: Edit file — now allowed (prereq satisfied)
    edit_call = ToolCall(tool="edit_file", args={"path": "/home/user/main.py"})
    result = gw.check([edit_call])
    assert len(result.allowed) == 1


def test_multiple_calls_mixed_results():
    gw = CodingGuardrails.defaults()

    calls = [
        ToolCall(tool="read_file", args={"path": "/home/user/main.py"}),  # ok
        ToolCall(tool="bash", args={"command": "rm -rf / "}),  # blocked
        ToolCall(tool="edit_file", args={"path": "/home/user/other.py"}),  # nudge (not read)
    ]

    result = gw.check(calls)
    assert result.has_blocks  # rm -rf /
    assert result.has_nudges  # edit without read
    # read_file allowed, edit_file allowed (nudge is soft), bash blocked
    assert len(result.allowed) == 2


def test_from_config():
    config = {
        "prerequisites": {"enabled": True, "max_violations": 3},
        "path_safety": {"enabled": True, "allowlist": ["/home/user/"]},
        "command_safety": {"enabled": True},
        "secrets": {"enabled": False},
        "sequencing": {"enabled": False},
        "tool_resolution": {"enabled": False},
    }
    gw = CodingGuardrails.from_config(config)
    assert gw.prerequisites is not None
    assert gw.path_safety is not None
    assert gw.secrets is None
    assert gw.sequencing is None


def test_config_disable_all():
    config = {
        "prerequisites": {"enabled": False},
        "path_safety": {"enabled": False},
        "command_safety": {"enabled": False},
        "secrets": {"enabled": False},
        "sequencing": {"enabled": False},
        "tool_resolution": {"enabled": False},
    }
    gw = CodingGuardrails.from_config(config)
    # All disabled — everything allowed
    result = gw.check([ToolCall(tool="bash", args={"command": "rm -rf /"})])
    assert not result.has_blocks
    assert len(result.allowed) == 1


def test_check_tool_result():
    gw = CodingGuardrails.defaults()
    result = gw.check_tool_result("bash", "")
    assert result is not None
    assert result.action == Action.NUDGE

    result = gw.check_tool_result("bash", "all good")
    assert result is None
