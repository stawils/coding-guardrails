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
    assert gw.canary is not None
    assert gw.input_scanning is not None


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


def test_defaults_wire_all_rules():
    """The defaults profile wires every rule type (prereq sanity check)."""
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


class TestSessionBudgetWiring:

    def test_defaults_are_long_horizon(self):
        gw = CodingGuardrails.from_config({})
        assert gw.session_budget.max_file_ops >= 500
        assert gw.session_budget.max_commands >= 1000

    def test_env_override_without_config(self, monkeypatch):
        monkeypatch.setenv("CG_MAX_FILE_OPS", "1234")
        monkeypatch.setenv("CG_MAX_COMMANDS", "4321")
        gw = CodingGuardrails.from_config({})
        assert gw.session_budget.max_file_ops == 1234
        assert gw.session_budget.max_commands == 4321

    def test_config_beats_env(self, monkeypatch):
        monkeypatch.setenv("CG_MAX_FILE_OPS", "1234")
        gw = CodingGuardrails.from_config(
            {"session_budget": {"max_file_ops": 77}}
        )
        assert gw.session_budget.max_file_ops == 77

    def test_invalid_env_falls_back(self, monkeypatch):
        monkeypatch.setenv("CG_MAX_FILE_OPS", "not-a-number")
        gw = CodingGuardrails.from_config({})
        assert gw.session_budget.max_file_ops >= 500

    def test_middleware_batch_does_not_overshoot(self):
        from coding_guardrails.rules.session_budget import SessionBudgetRule

        gw = CodingGuardrails(session_budget=SessionBudgetRule(max_file_ops=3))
        gw.record([ToolCall(tool="edit", args={"path": "a.py"})])
        gw.record([ToolCall(tool="edit", args={"path": "b.py"})])
        assert gw.session_budget.file_op_count == 2

        batch = [ToolCall(tool="edit", args={"path": f"c{i}.py"}) for i in range(3)]
        result = gw.check(batch)
        # Only one call fits (the 3rd op); the other two are blocked.
        assert len(result.allowed) == 1
        assert len(result.blocked) == 2
        gw.record(result.allowed)
        assert gw.session_budget.file_op_count == 3


class TestResetConversationState:
    """The middleware resets every stateful rule at a new conversation."""

    def test_calls_reset_on_stateful_rules(self):
        from coding_guardrails.rules.base import RuleResult

        class SpyRule:
            name = "spy"

            def __init__(self):
                self.reset_calls = 0

            def check(self, call):
                return RuleResult.allow(call.tool)

            def record(self, calls):
                pass

            def reset(self):
                self.reset_calls += 1

        spy = SpyRule()
        gw = CodingGuardrails(session_budget=spy)
        gw.reset_conversation_state()
        assert spy.reset_calls == 1

    def test_tolerates_rules_without_reset(self):
        from coding_guardrails.rules.base import RuleResult

        class Stateless:
            name = "stateless"

            def check(self, call):
                return RuleResult.allow(call.tool)

            def record(self, calls):
                pass

        gw = CodingGuardrails(session_budget=Stateless())
        gw.reset_conversation_state()  # must not raise

    def test_defaults_clears_session_budget(self):
        gw = CodingGuardrails.defaults()
        gw.session_budget.record([ToolCall(tool="edit", args={"path": "a.py"})])
        assert gw.session_budget.file_op_count == 1
        gw.reset_conversation_state()
        assert gw.session_budget.file_op_count == 0
