"""Tests for config loading."""

import os
import tempfile

import pytest

from coding_guardrails.config import load_config, load_guardrail_config


def test_load_valid_yaml():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("guardrails:\n  prerequisites:\n    enabled: true\n")
        f.flush()
        config = load_config(f.name)
    assert config["guardrails"]["prerequisites"]["enabled"] is True
    os.unlink(f.name)


def test_env_var_expansion():
    os.environ["TEST_USER"] = "alice"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("guardrails:\n  path_safety:\n    allowlist:\n      - /home/${TEST_USER}/\n")
        f.flush()
        config = load_config(f.name)
    assert config["guardrails"]["path_safety"]["allowlist"][0] == "/home/alice/"
    os.unlink(f.name)
    del os.environ["TEST_USER"]


def test_load_missing_file():
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/config.yaml")


def test_load_guardrail_config():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("forge:\n  max_retries: 3\nguardrails:\n  secrets:\n    enabled: false\n")
        f.flush()
        config = load_guardrail_config(f.name)
    assert config == {"secrets": {"enabled": False}}
    os.unlink(f.name)


def test_load_guardrail_config_none():
    config = load_guardrail_config(None)
    assert config == {}


def test_empty_yaml():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("")
        f.flush()
        config = load_config(f.name)
    assert config == {}
    os.unlink(f.name)
