"""YAML config loading with environment variable expansion."""

from __future__ import annotations

import os
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
    """Load the guardrails config from a YAML file.

    Supports two layouts:
      1. Rules at top level (preferred, matches configs/guardrail-config.yaml):
         ```
         prerequisites: {...}
         session_budget: {...}
         ```
      2. Rules nested under a `guardrails:` key (legacy):
         ```
         guardrails:
           prerequisites: {...}
           session_budget: {...}
         ```

    If both layouts are present, the nested form wins (more explicit).

    Args:
        path: Path to the YAML config file. If None, returns defaults (empty dict).

    Returns:
        Config dict consumed by `CodingGuardrails.from_config()`.
    """
    if path is None:
        return {}

    full_config = load_config(path)
    if not isinstance(full_config, dict):
        return {}

    nested = full_config.get("guardrails")
    if isinstance(nested, dict):
        return nested
    return full_config
