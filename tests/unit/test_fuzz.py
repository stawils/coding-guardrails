"""Property-based fuzz tests for guardrail rules."""
import string
from hypothesis import given, settings, strategies as st
from hypothesis.strategies import text as st_text
from coding_guardrails.rules.base import Action, ToolCall
from coding_guardrails.rules.commands import CommandSafetyRule
from coding_guardrails.rules.path_safety import PathSafetyRule
from coding_guardrails.rules.network import NetworkRule
from coding_guardrails.rules.secrets import SecretRule


# Hypothesis strategy for random text input
text_strategy = st_text(min_size=0, max_size=500)


class TestNeverCrashes:
    """
    Never-Crashes Tests (highest priority).
    For each rule: given any string input, check() must return a valid RuleResult
    and never raise.
    """

    @settings(max_examples=200, deadline=None)
    @given(command=text_strategy)
    def test_command_safety_never_crashes(self, command: str):
        """Random text as command - must not crash."""
        rule = CommandSafetyRule()
        result = rule.check(ToolCall(tool="bash", args={"command": command}))
        assert result is not None
        assert result.action in (Action.ALLOW, Action.BLOCK, Action.NUDGE)

    @settings(max_examples=200, deadline=None)
    @given(path=st_text(min_size=0, max_size=500).filter(lambda p: '\x00' not in p))
    def test_path_safety_never_crashes(self, path: str):
        """Random text as path - must not crash. Excludes null bytes."""
        rule = PathSafetyRule()
        result = rule.check(ToolCall(tool="read_file", args={"path": path}))
        assert result is not None
        assert result.action in (Action.ALLOW, Action.BLOCK, Action.NUDGE)

    @settings(max_examples=200, deadline=None)
    @given(command=text_strategy)
    def test_network_never_crashes(self, command: str):
        """Random text as command - must not crash."""
        rule = NetworkRule()
        result = rule.check(ToolCall(tool="bash", args={"command": command}))
        assert result is not None
        assert result.action in (Action.ALLOW, Action.BLOCK, Action.NUDGE)

    @settings(max_examples=200, deadline=None)
    @given(command=text_strategy)
    def test_secrets_never_crashes(self, command: str):
        """Random text as command - must not crash."""
        rule = SecretRule()
        result = rule.check(ToolCall(tool="bash", args={"command": command}))
        assert result is not None
        assert result.action in (Action.ALLOW, Action.BLOCK, Action.NUDGE)


class TestObfuscationAwareness:
    """
    Obfuscation-Aware Tests.
    Tests that noise/random characters don't bypass security checks.
    """

    @settings(max_examples=200, deadline=None)
    @given(payload=st_text(min_size=1, max_size=50).filter(lambda p: all(c in string.ascii_letters + string.digits + " ._-/" for c in p)))
    def test_rm_rf_with_noise_still_blocks(self, payload: str):
        """rm -rf /{payload} with random ascii -> always BLOCK."""
        rule = CommandSafetyRule()
        # Match exact pattern: rm -rf / followed by space and anything
        command = f"rm -rf / {payload}"
        result = rule.check(ToolCall(tool="bash", args={"command": command}))
        assert result is not None
        assert result.action == Action.BLOCK, (
            f"Expected BLOCK for 'rm -rf / {payload}', got {result.action}"
        )
        assert "blocked" in result.nudge.lower() or "dangerous" in result.nudge.lower()

    @settings(max_examples=200, deadline=None)
    @given(path=st_text(min_size=1, max_size=100).filter(lambda p: '\x00' not in p and all(c in string.ascii_letters + string.digits + "./" for c in p)))
    def test_path_traversal_with_noise_blocks(self, path: str):
        """../../etc/{payload} -> always BLOCK."""
        rule = PathSafetyRule()
        full_path = f"../../etc/{path}"
        result = rule.check(ToolCall(tool="read_file", args={"path": full_path}))
        assert result is not None
        assert result.action == Action.BLOCK, (
            f"Expected BLOCK for path traversal '{full_path}', got {result.action}"
        )
        assert "blocked" in result.nudge.lower() or "outside" in result.nudge.lower()

    def test_secret_prefixes_caught(self):
        """Secrets with known prefixes (sk-, ghp_, AKIA, etc.) -> always NUDGE or BLOCK."""
        rule = SecretRule()
        
        # Generate test cases with known secret prefixes
        test_cases = [
            ("sk-", "abcdefghijklmnopqrstuvwxyz123456"),
            ("ghp_", "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"),
            ("AKIA", "ABCDEFGHIJ1234567890123456789012"),
        ]
        
        for prefix, body in test_cases:
            secret = f"{prefix}{body}"
            call = ToolCall(tool="bash", args={"command": f"echo {secret}"})
            result = rule.check(call)
            assert result is not None, f"Result is None for secret: {secret}"
            # Should either nudge (mask) or block
            assert result.action in (Action.NUDGE, Action.BLOCK), (
                f"Expected NUDGE or BLOCK for secret pattern '{secret}', got {result.action}"
            )
            if result.action == Action.NUDGE:
                # Masked secrets should not contain the original
                assert secret not in call.args.get("command", "")


class TestDeterminism:
    """
    Invariants: Same input must produce same output.
    """

    @settings(max_examples=200, deadline=None)
    @given(command=text_strategy)
    def test_command_safety_deterministic(self, command: str):
        """Calling check() twice with same input gives same result."""
        rule = CommandSafetyRule()
        result1 = rule.check(ToolCall(tool="bash", args={"command": command}))
        result2 = rule.check(ToolCall(tool="bash", args={"command": command}))
        assert result1.action == result2.action, (
            f"CommandSafetyRule is non-deterministic: {result1.action} != {result2.action}"
        )
        assert result1.nudge == result2.nudge

    @settings(max_examples=200, deadline=None)
    @given(path=st_text(min_size=0, max_size=500).filter(lambda p: '\x00' not in p))
    def test_path_safety_deterministic(self, path: str):
        """Calling check() twice with same input gives same result."""
        rule = PathSafetyRule()
        result1 = rule.check(ToolCall(tool="read_file", args={"path": path}))
        result2 = rule.check(ToolCall(tool="read_file", args={"path": path}))
        assert result1.action == result2.action, (
            f"PathSafetyRule is non-deterministic: {result1.action} != {result2.action}"
        )
        assert result1.nudge == result2.nudge
