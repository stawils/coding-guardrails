"""Unit tests for path computation functions in coding_guardrails.server.paths."""

import os
from pathlib import Path

import pytest

from coding_guardrails.server.paths import (
    binary_path,
    build_dir,
    data_dir,
    log_file,
    models_dir,
    pin_file,
    pid_file,
    proxy_log_file,
    proxy_pid_file,
    run_dir,
    source_dir,
)


@pytest.fixture
def fake_home(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set up a fake HOME environment and clean XDG_DATA_HOME for tests that need it."""
    monkeypatch.setenv("HOME", "/fake/home")
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)


class TestDataDir:
    """Tests for data_dir() function."""

    def test_default_data_dir(self, fake_home) -> None:
        """data_dir() defaults to ~/.local/share/coding-guardrails."""
        expected = Path("/fake/home/.local/share/coding-guardrails")
        assert data_dir() == expected

    def test_custom_xdg_data_home(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """data_dir() honors XDG_DATA_HOME when set."""
        custom_xdg = "/custom/xdg/data"
        monkeypatch.setenv("XDG_DATA_HOME", custom_xdg)
        expected = Path(custom_xdg) / "coding-guardrails"
        assert data_dir() == expected


class TestSourceDir:
    """Tests for source_dir() function."""

    def test_source_dir_composed_from_data_dir(self, fake_home) -> None:
        """source_dir() = data_dir() / 'llama.cpp'."""
        expected = Path("/fake/home/.local/share/coding-guardrails/llama.cpp")
        assert source_dir() == expected


class TestBuildDir:
    """Tests for build_dir() function."""

    def test_build_dir_composed_from_source_dir(self, fake_home) -> None:
        """build_dir() = source_dir() / 'build'."""
        expected = Path("/fake/home/.local/share/coding-guardrails/llama.cpp/build")
        assert build_dir() == expected


class TestBinaryPath:
    """Tests for binary_path() function."""

    def test_binary_path_composed_from_build_dir(self, fake_home) -> None:
        """binary_path() = build_dir() / 'bin' / 'llama-server'."""
        expected = Path("/fake/home/.local/share/coding-guardrails/llama.cpp/build/bin/llama-server")
        assert binary_path() == expected


class TestModelsDir:
    """Tests for models_dir() function."""

    def test_models_dir_composed_from_data_dir(self, fake_home) -> None:
        """models_dir() = data_dir() / 'models'."""
        expected = Path("/fake/home/.local/share/coding-guardrails/models")
        assert models_dir() == expected


class TestRunDir:
    """Tests for run_dir() function."""

    def test_run_dir_composed_from_data_dir(self, fake_home) -> None:
        """run_dir() = data_dir() / 'run'."""
        expected = Path("/fake/home/.local/share/coding-guardrails/run")
        assert run_dir() == expected


class TestPidFile:
    """Tests for pid_file() function."""

    def test_pid_file_composed_from_run_dir(self, fake_home) -> None:
        """pid_file() = run_dir() / 'llama-server.pid'."""
        expected = Path("/fake/home/.local/share/coding-guardrails/run/llama-server.pid")
        assert pid_file() == expected


class TestLogFile:
    """Tests for log_file() function."""

    def test_log_file_composed_from_run_dir(self, fake_home) -> None:
        """log_file() = run_dir() / 'llama-server.log'."""
        expected = Path("/fake/home/.local/share/coding-guardrails/run/llama-server.log")
        assert log_file() == expected


class TestProxyPidFile:
    """Tests for proxy_pid_file() function."""

    def test_proxy_pid_file_composed_from_run_dir(self, fake_home) -> None:
        """proxy_pid_file() = run_dir() / 'proxy.pid'."""
        expected = Path("/fake/home/.local/share/coding-guardrails/run/proxy.pid")
        assert proxy_pid_file() == expected


class TestProxyLogFile:
    """Tests for proxy_log_file() function."""

    def test_proxy_log_file_composed_from_run_dir(self, fake_home) -> None:
        """proxy_log_file() = run_dir() / 'proxy.log'."""
        expected = Path("/fake/home/.local/share/coding-guardrails/run/proxy.log")
        assert proxy_log_file() == expected


class TestPinFile:
    """Tests for pin_file() function."""

    def test_pin_file_composed_from_data_dir(self, fake_home) -> None:
        """pin_file() = data_dir() / 'llama.cpp.pin'."""
        expected = Path("/fake/home/.local/share/coding-guardrails/llama.cpp.pin")
        assert pin_file() == expected
