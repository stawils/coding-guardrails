"""Tests for guardrail config loading."""

import pytest

from coding_guardrails.middleware import CodingGuardrails


class TestFromConfig:

    def test_empty_config_loads_all_defaults(self):
        guardrails = CodingGuardrails.from_config({})
        assert guardrails.prerequisites is not None
        assert guardrails.path_safety is not None
        assert guardrails.command_safety is not None
        assert guardrails.secrets is not None
        assert guardrails.sequencing is not None
        assert guardrails.tool_resolution is not None

    def test_disable_individual_rules(self):
        cfg = {
            "prerequisites": {"enabled": False},
            "sequencing": {"enabled": False},
        }
        guardrails = CodingGuardrails.from_config(cfg)
        assert guardrails.prerequisites is None
        assert guardrails.sequencing is None
        assert guardrails.command_safety is not None

    def test_prerequisites_custom_tools(self):
        cfg = {
            "prerequisites": {
                "edit_tools": ["modify", "update"],
                "read_tools": ["inspect", "view"],
                "max_violations": 5,
            }
        }
        guardrails = CodingGuardrails.from_config(cfg)
        assert guardrails.prerequisites.edit_tools == ("modify", "update")
        assert guardrails.prerequisites.read_tools == ("inspect", "view")
        assert guardrails.prerequisites.max_violations == 5

    def test_sequencing_custom_config(self):
        cfg = {
            "sequencing": {
                "trigger_tools": ["modify", "patch"],
                "suggest_tools": ["test", "validate"],
                "strength": "hard",
                "cooldown": 5,
            }
        }
        guardrails = CodingGuardrails.from_config(cfg)
        assert guardrails.sequencing.trigger_prefixes == ("modify", "patch")
        assert guardrails.sequencing.suggest_prefixes == ("test", "validate")
        assert guardrails.sequencing.strength == "hard"
        assert guardrails.sequencing.cooldown == 5

    def test_secrets_action_block(self):
        cfg = {"secrets": {"strength": "hard"}}
        guardrails = CodingGuardrails.from_config(cfg)
        assert guardrails.secrets.action == "block"

    def test_secrets_action_mask(self):
        cfg = {"secrets": {"strength": "soft"}}
        guardrails = CodingGuardrails.from_config(cfg)
        assert guardrails.secrets.action == "mask"


class TestDefaults:

    def test_defaults_creates_all_rules(self):
        guardrails = CodingGuardrails.defaults()
        assert guardrails.prerequisites is not None
        assert guardrails.path_safety is not None
        assert guardrails.command_safety is not None
        assert guardrails.secrets is not None
        assert guardrails.sequencing is not None
        assert guardrails.tool_resolution is not None
