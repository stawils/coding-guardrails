"""Filesystem layout for the cg-owned llama.cpp server stack.

All cg-managed artifacts live under one XDG-aware data directory, decoupled
from LM Studio / HuggingFace caches::

    ~/.local/share/coding-guardrails/
    |-- llama.cpp/                 git checkout (pinned commit)
    |   `-- build/bin/llama-server the compiled binary
    |-- models/                    GGUF model cache (cg-owned, primary)
    `-- run/
        |-- llama-server.pid
        `-- llama-server.log
"""

from __future__ import annotations

import os
from pathlib import Path


def data_dir() -> Path:
    """Root data directory for cg-owned artifacts (XDG_DATA_HOME aware)."""
    base = os.environ.get("XDG_DATA_HOME")
    root = Path(base) if base else Path.home() / ".local" / "share"
    return root / "coding-guardrails"


def source_dir() -> Path:
    """The pinned llama.cpp git checkout."""
    return data_dir() / "llama.cpp"


def build_dir() -> Path:
    """CMake build directory."""
    return source_dir() / "build"


def binary_path() -> Path:
    """The compiled llama-server binary."""
    return build_dir() / "bin" / "llama-server"


def models_dir() -> Path:
    """cg-owned GGUF cache (primary search path).

    lm-studio / huggingface caches remain as fallbacks in the registry, but a
    cg user never needs them.
    """
    return data_dir() / "models"


def run_dir() -> Path:
    """Runtime files: PID, logs."""
    return data_dir() / "run"


def pid_file() -> Path:
    """PID file for the running llama-server."""
    return run_dir() / "llama-server.pid"


def log_file() -> Path:
    """stdout/stderr log for the running llama-server."""
    return run_dir() / "llama-server.log"


def proxy_pid_file() -> Path:
    """PID file for the managed proxy (``cg up``)."""
    return run_dir() / "proxy.pid"


def proxy_log_file() -> Path:
    """stdout/stderr log for the managed proxy."""
    return run_dir() / "proxy.log"


def pin_file() -> Path:
    """Record of the pinned commit the build was performed against."""
    return data_dir() / "llama.cpp.pin"
