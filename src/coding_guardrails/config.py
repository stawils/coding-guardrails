"""YAML config loading with environment variable expansion."""

from __future__ import annotations

import os
import re
from pathlib import Path

import yaml


def _expand_env(value: str) -> str:
    """Expand ${VAR} and $VAR environment variables in a string."""
    return os.path.expandvars(value)


def _expand_config(config: dict | list | str) -> dict | list | str:
    """Recursively expand environment variables in config values."""
    if isinstance(config, dict):
        return {k: _expand_config(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [_expand_config(item) for item in config]
    elif isinstance(config, str):
        return _expand_env(config)
    return config


def load_config(path: str | Path) -> dict:
    """Load a YAML config file with environment variable expansion.

    Args:
        path: Path to the YAML config file.

    Returns:
        Parsed config dict with env vars expanded.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        yaml.YAMLError: If config is invalid YAML.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    text = config_path.read_text()
    data = yaml.safe_load(text)

    if data is None:
        return {}

    return _expand_config(data)


def load_guardrail_config(path: str | Path | None = None) -> dict:
    """Load just the guardrails section of a config file.

    Args:
        path: Path to the YAML config file. If None, returns defaults.

    Returns:
        The "guardrails" section of the config, or empty dict for defaults.
    """
    if path is None:
        return {}

    full_config = load_config(path)
    return full_config.get("guardrails", {})
