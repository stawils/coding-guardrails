"""Tests for the thoroughness rule — nudges premature terminal submissions.

The ThoroughnessRule compares tools used vs tools available and nudges when
a terminal call (submit, report, respond) happens after low exploration.
It pulls conversation state via set_context() before each check().
"""

import pytest

from coding_guardrails.rules.base import Action, ToolCall
from coding_guardrails.rules.thoroughness import ThoroughnessRule


# ── Fixtures and helpers ──────────────────────────────────────────


@pytest.fixture
def rule():
    """Fresh rule with defaults: min_tools=3, min_ratio=0.4, cooldown=2."""
    return ThoroughnessRule()


def _terminal(tool="submit", **args):
    return ToolCall(tool=tool, args=args or {"answer": "done"})


def _action(tool="bash", **args):
    return ToolCall(tool=tool, args=args or {"command": "ls"})


def _msgs(*tool_names):
    """Build assistant messages with tool_calls using the given tool names."""
    return [
        {
            "role": "assistant",
            "tool_calls": [{"function": {"name": t}} for t in tool_names],
        }
    ]


def _ctx(rule, used_tools, available_tools):
    """Set context: tools used (deduped) + the full set available."""
    rule.set_context(_msgs(*used_tools), set(available_tools))


# ── Terminal detection ────────────────────────────────────────────


class TestTerminalDetection:
    """Non-terminal calls should always pass through without nudging."""

    def test_non_terminal_call_allowed(self, rule):
        _ctx(rule, used_tools=["bash"], available_tools={"bash", "read", "edit"})
        result = rule.check(_action())
        assert result.action == Action.ALLOW

    def test_terminal_patterns_detected(self, rule):
        # All documented terminal keywords should trigger the rule's path.
        # Use a fresh rule per name to avoid the cooldown masking later ones.
        available = {"a", "b", "c", "d", "read", "edit", "bash", "grep"}
        for name in ("submit", "report", "respond", "answer",
                     "present", "summarize", "diagnose", "recommend", "complete"):
            r = ThoroughnessRule()
            _ctx(r, used_tools=["read"], available_tools=available)
            # With only 1 tool used of 7 non-terminal, every terminal should nudge.
            result = r.check(_terminal(tool=name))
            assert result.action == Action.NUDGE, f"{name} should nudge"

    def test_terminal_case_insensitive(self, rule):
        available = {"a", "b", "c", "d", "read", "edit", "bash", "grep"}
        for name in ("SUBMIT", "Report"):
            r = ThoroughnessRule()
            _ctx(r, used_tools=["read"], available_tools=available)
            assert r.check(_terminal(tool=name)).action == Action.NUDGE


# ── Nudge conditions ──────────────────────────────────────────────


class TestNudgeOnLowExploration:
    """Nudge when few tools have been explored before a terminal call."""

    def test_nudge_when_too_few_tools_used(self, rule):
        available = {"read", "edit", "write", "bash", "grep", "glob", "list"}
        _ctx(rule, used_tools=["read"], available_tools=available)
        result = rule.check(_terminal())
        assert result.action == Action.NUDGE
        assert "1 of" in result.nudge

    def test_nudge_when_ratio_below_threshold(self, rule):
        # 2 of 6 = 0.33 < min_ratio(0.4), even though 2 < min_tools(3) already
        available = {"a", "b", "c", "d", "e", "f"}
        _ctx(rule, used_tools=["a", "b"], available_tools=available)
        result = rule.check(_terminal())
        assert result.action == Action.NUDGE

    def test_nudge_lists_unused_tools(self, rule):
        available = {"read", "edit", "write", "bash", "grep", "glob", "list"}
        _ctx(rule, used_tools=["read"], available_tools=available)
        result = rule.check(_terminal())
        # Should mention tools not yet tried
        assert "not yet tried" in result.nudge.lower() or "Tools not yet tried" in result.nudge


class TestAllowOnSufficientExploration:
    """No nudge when enough tools have been explored."""

    def test_allow_when_enough_tools_and_ratio(self, rule):
        # 4 of 5 = 0.8 >= 0.4, and 4 >= min_tools(3) → allow
        available = {"a", "b", "c", "d", "e"}
        _ctx(rule, used_tools=["a", "b", "c", "d"], available_tools=available)
        result = rule.check(_terminal())
        assert result.action == Action.ALLOW

    def test_allow_ignores_terminal_tools_in_used_count(self, rule):
        # Terminal tools in history shouldn't count toward exploration.
        available = {"a", "b", "c", "d", "e", "f"}
        # used = [a, b, submit] → only 2 non-terminal, below threshold
        _ctx(rule, used_tools=["a", "b", "submit"], available_tools=available)
        result = rule.check(_terminal())
        assert result.action == Action.NUDGE

    def test_allow_ignores_terminal_tools_in_available_count(self, rule):
        # Available set including terminal tools shouldn't pad the denominator.
        available = {"a", "b", "c", "d", "submit", "report"}
        # non-terminal available = {a,b,c,d} = 4. min_tools=3 → 3 used is enough
        _ctx(rule, used_tools=["a", "b", "c"], available_tools=available)
        result = rule.check(_terminal())
        # 3 of 4 = 0.75 >= 0.4 and 3 >= 3 → allow
        assert result.action == Action.ALLOW

    def test_no_nudge_when_few_tools_available(self, rule):
        # If only 2 tools available (<= min_tools), don't nag.
        available = {"read", "edit"}
        _ctx(rule, used_tools=["read"], available_tools=available)
        result = rule.check(_terminal())
        assert result.action == Action.ALLOW


# ── Empty / unset context ─────────────────────────────────────────


class TestNoContext:
    """Without conversation context, the rule stays silent (safe default)."""

    def test_no_context_allows(self, rule):
        result = rule.check(_terminal())
        assert result.action == Action.ALLOW

    def test_empty_messages_allows(self, rule):
        rule.set_context([], {"a", "b", "c", "d"})
        result = rule.check(_terminal())
        assert result.action == Action.ALLOW

    def test_empty_available_tools_allows(self, rule):
        rule.set_context(_msgs("read"), set())
        result = rule.check(_terminal())
        assert result.action == Action.ALLOW


# ── Cooldown ──────────────────────────────────────────────────────


class TestCooldown:
    """Rule goes silent after firing `cooldown` times."""

    def test_cooldown_silences_after_limit(self, rule):
        available = {"a", "b", "c", "d", "e", "f"}
        _ctx(rule, used_tools=["a"], available_tools=available)

        first = rule.check(_terminal())
        second = rule.check(_terminal())
        third = rule.check(_terminal())

        assert first.action == Action.NUDGE
        assert second.action == Action.NUDGE
        # cooldown=2 → third call is allowed silently
        assert third.action == Action.ALLOW

    def test_custom_cooldown(self):
        rule = ThoroughnessRule(cooldown=1)
        available = {"a", "b", "c", "d", "e", "f"}
        _ctx(rule, used_tools=["a"], available_tools=available)

        assert rule.check(_terminal()).action == Action.NUDGE
        assert rule.check(_terminal()).action == Action.ALLOW

    def test_reset_clears_fire_count(self, rule):
        available = {"a", "b", "c", "d", "e", "f"}
        _ctx(rule, used_tools=["a"], available_tools=available)
        rule.check(_terminal())
        rule.check(_terminal())  # fire_count = 2 (at cooldown)

        rule.reset()

        # After reset, fires again
        result = rule.check(_terminal())
        assert result.action == Action.NUDGE


# ── Configurable thresholds ───────────────────────────────────────


class TestConfigurableThresholds:
    """Custom min_tools / min_ratio settings change behavior."""

    def test_custom_min_tools(self):
        rule = ThoroughnessRule(min_tools=2)
        available = {"a", "b", "c", "d"}
        # 2 of 4 = 0.5 >= 0.4 and 2 >= 2 → allow (with default it would nudge)
        _ctx(rule, used_tools=["a", "b"], available_tools=available)
        assert rule.check(_terminal()).action == Action.ALLOW

    def test_high_min_tools_forces_nudge(self):
        rule = ThoroughnessRule(min_tools=5)
        available = {"a", "b", "c", "d", "e", "f"}
        # 4 of 6 = 0.66 >= 0.4 but 4 < 5 → nudge
        _ctx(rule, used_tools=["a", "b", "c", "d"], available_tools=available)
        assert rule.check(_terminal()).action == Action.NUDGE

    def test_custom_min_ratio(self):
        rule = ThoroughnessRule(min_ratio=0.7)
        available = {"a", "b", "c", "d", "e"}
        # 3 of 5 = 0.6 < 0.7 → nudge (with default 0.4 it would allow)
        _ctx(rule, used_tools=["a", "b", "c"], available_tools=available)
        assert rule.check(_terminal()).action == Action.NUDGE


# ── Metadata ──────────────────────────────────────────────────────


class TestMetadata:
    def test_name(self, rule):
        assert rule.name == "thoroughness"

    def test_record_is_noop(self, rule):
        # record() should not raise and should not affect behavior
        rule.record([_action(), _terminal()])
        result = rule.check(_action())
        assert result.action == Action.ALLOW
