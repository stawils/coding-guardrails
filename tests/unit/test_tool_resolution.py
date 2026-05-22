"""Tests for the tool resolution rule."""

from coding_guardrails.rules.base import Action
from coding_guardrails.rules.tool_resolution import ToolResolutionRule


def test_empty_result_nudge():
    rule = ToolResolutionRule()
    result = rule.check_result("bash", "")
    assert result is not None
    assert result.action == Action.NUDGE
    assert "no results" in result.nudge.lower() or "broadening" in result.nudge.lower()


def test_whitespace_only_result_nudge():
    rule = ToolResolutionRule()
    result = rule.check_result("bash", "   \n\t  ")
    assert result is not None
    assert result.action == Action.NUDGE


def test_error_result_nudge():
    rule = ToolResolutionRule()
    result = rule.check_result("bash", "Error: file not found")
    assert result is not None
    assert result.action == Action.NUDGE
    assert "error" in result.nudge.lower()


def test_permission_error_nudge():
    rule = ToolResolutionRule()
    result = rule.check_result("bash", "Permission denied: /root/secret")
    assert result is not None
    assert result.action == Action.NUDGE


def test_normal_result_no_nudge():
    rule = ToolResolutionRule()
    result = rule.check_result("bash", "file1.py\nfile2.py\nfile3.py")
    assert result is None


def test_successful_command_no_nudge():
    rule = ToolResolutionRule()
    result = rule.check_result("bash", "Ran 5 tests, all passed.")
    assert result is None
