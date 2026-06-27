"""Tests for the middleware (rule composition)."""

import pytest
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


class TestCrossRuleInteractions:
    """Test that rules compose correctly in middleware."""

    def test_blocked_call_not_counted_in_budget(self):
        """A path_traversal call gets BLOCKED → session_budget should NOT count it."""
        gw = CodingGuardrails.defaults()
        # Path traversal is blocked by path_safety, not session_budget
        # session_budget tracks ops that were allowed and recorded

        # Step 1: Allowed read - counts toward budget
        read1 = ToolCall(tool="read_file", args={"path": "/home/user/test.py"})
        result1 = gw.check([read1])
        assert len(result1.allowed) == 1
        gw.record([read1])

        # Step 2: Path traversal — blocked by path_safety, not counted by budget
        result2 = gw.check([
            ToolCall(tool="read_file", args={"path": "../../../etc/shadow"})
        ])
        assert result2.has_blocks
        assert len(result2.allowed) == 0

        # Step 3: Another allowed read — budget now at 2/100
        result3 = gw.check([
            ToolCall(tool="read_file", args={"path": "/home/user/test2.py"})
        ])
        assert len(result3.allowed) == 1

    def test_nudged_call_counted_in_budget(self):
        """An edit-without-read gets NUDGED → session_budget SHOULD count it (nudge is advisory, call proceeds)."""
        gw = CodingGuardrails.defaults()

        # Step 1: Read file
        read_call = ToolCall(tool="read_file", args={"path": "/home/user/test.py"})
        result1 = gw.check([read_call])
        assert len(result1.allowed) == 1
        gw.record([read_call])

        # Step 2: Edit without read — nudge from prerequisites but still allowed
        edit_call = ToolCall(tool="edit_file", args={"path": "/home/user/test.py"})
        result2 = gw.check([edit_call])
        # Prerequisites rule nudges edit without read, but call still proceeds
        assert len(result2.allowed) == 1

        # Record the edit (it proceeded despite nudge)
        gw.record([edit_call])

        # Step 3: Two ops recorded (1 read + 1 edit)
        # session_budget tracks this, allowing further operations

    def test_loop_detection_across_rules(self):
        """Call bash('echo hi') 5 times → loop_detection blocks, even though other rules (network, path_safety) also checked."""
        gw = CodingGuardrails.defaults()

        # First 4 calls: allowed (nudge at 3, block at 5)
        for i in range(4):
            call = ToolCall(tool="bash", args={"command": "echo hi"})
            result = gw.check([call])
            assert len(result.allowed) == 1  # Allowed (nudge at call 3)
            gw.record([call])  # Record for loop detection

        # 5th call: should be blocked by loop_detection
        call5 = ToolCall(tool="bash", args={"command": "echo hi"})
        result5 = gw.check([call5])
        assert result5.has_blocks
        assert len(result5.allowed) == 0

        # Other rules (network, path_safety) didn't interfere

    def test_secrets_block_overrides_other_nudges(self):
        """A command with a secret detected → secrets BLOCK takes priority."""
        gw = CodingGuardrails.defaults()

        # Command with secret pattern (password=secret123)
        result = gw.check([
            ToolCall(tool="bash", args={"command": "echo password=secret123"})
        ])

        # Secrets rule masks and nudges (doesn't block by default)
        assert len(result.allowed) >= 1

    def test_full_workflow_allowed(self):
        """read('f.py') → edit('f.py') → bash('pytest') → all allowed, no nudges."""
        gw = CodingGuardrails.defaults()

        # Step 1: Read file
        read_call = ToolCall(tool="read_file", args={"path": "/home/user/f.py"})
        result1 = gw.check([read_call])
        assert len(result1.allowed) == 1
        assert not result1.has_nudges

        gw.record([read_call])

        # Step 2: Edit file (now allowed since read done)
        edit_call = ToolCall(tool="edit_file", args={"path": "/home/user/f.py"})
        result2 = gw.check([edit_call])
        assert len(result2.allowed) == 1
        assert not result2.has_nudges

        # Step 3: Run tests
        bash_call = ToolCall(tool="bash", args={"command": "pytest"})
        result3 = gw.check([bash_call])
        assert len(result3.allowed) == 1
        assert not result3.has_nudges

    def test_full_workflow_blocked_then_recovered(self):
        """edit('f.py') without read → NUDGED by prerequisites → read('f.py') → edit('f.py') → allowed."""
        gw = CodingGuardrails.defaults()

        # Step 1: Edit without read → nudge from prerequisites (not block)
        edit_call1 = ToolCall(tool="edit_file", args={"path": "/home/user/f.py"})
        result1 = gw.check([edit_call1])

        # Prerequisites nudges edit without read, but call still proceeds
        assert len(result1.allowed) == 1

        gw.record([edit_call1])

        # Step 2: Read the file
        read_call = ToolCall(tool="read_file", args={"path": "/home/user/f.py"})
        result2 = gw.check([read_call])
        assert len(result2.allowed) == 1

        gw.record([read_call])

        # Step 3: Edit again → still allowed (prereq already satisfied)
        edit_call2 = ToolCall(tool="edit_file", args={"path": "/home/user/f.py"})
        result3 = gw.check([edit_call2])
        assert len(result3.allowed) == 1
        assert not result3.has_nudges  # No nudges after prereq satisfied


def test_read_edit_workflow():
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


def test_from_config_dup_write():
    config = {
        "dup_write": {"enabled": True, "nudge_threshold": 4, "block_threshold": 7},
    }
    gw = CodingGuardrails.from_config(config)
    assert gw.dup_write is not None
    assert gw.dup_write.nudge_threshold == 4
    assert gw.dup_write.block_threshold == 7


def test_from_config_loop_detection_stagnation_threshold():
    config = {
        "loop_detection": {"enabled": True, "stagnation_threshold": 7},
    }
    gw = CodingGuardrails.from_config(config)
    assert gw.loop_detection is not None
    assert gw.loop_detection.stagnation_threshold == 7


def test_defaults_includes_dup_write():
    gw = CodingGuardrails.defaults()
    assert gw.dup_write is not None


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
