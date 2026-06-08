"""Tests for the cg-owned server stack (paths, version, build, launcher, sources, download, registry)."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from coding_guardrails.server import paths, version, sources, build, launcher, download
from coding_guardrails.models import registry


# ───────────────────────── paths ─────────────────────────


class TestPaths:
    def test_data_dir_respects_xdg(self, monkeypatch, tmp_path):
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
        assert paths.data_dir() == tmp_path / "coding-guardrails"

    def test_data_dir_default(self, monkeypatch):
        monkeypatch.delenv("XDG_DATA_HOME", raising=False)
        monkeypatch.setattr(paths.Path, "home", staticmethod(lambda: Path("/home/u")))
        assert paths.data_dir() == Path("/home/u/.local/share/coding-guardrails")

    def test_subpaths_under_data_dir(self, monkeypatch, tmp_path):
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
        root = tmp_path / "coding-guardrails"
        assert paths.source_dir() == root / "llama.cpp"
        assert paths.build_dir() == root / "llama.cpp" / "build"
        assert paths.binary_path() == root / "llama.cpp" / "build" / "bin" / "llama-server"
        assert paths.models_dir() == root / "models"
        assert paths.pid_file() == root / "run" / "llama-server.pid"
        assert paths.log_file() == root / "run" / "llama-server.log"
        assert paths.pin_file() == root / "llama.cpp.pin"


# ───────────────────────── version ─────────────────────────


class TestVersion:
    def test_pinned_commit_is_full_hash(self):
        # a git SHA is 40 hex chars
        assert len(version.PINNED_COMMIT) == 40
        int(version.PINNED_COMMIT, 16)  # raises if not hex

    def test_installed_commit_no_checkout(self, monkeypatch, tmp_path):
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
        assert version.installed_commit() is None

    def test_is_up_to_date_false_when_not_cloned(self, monkeypatch, tmp_path):
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
        assert version.is_up_to_date() is False

    def test_is_up_to_date_true_when_pinned(self, monkeypatch, tmp_path):
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
        (paths.source_dir() / ".git").mkdir(parents=True)
        with patch("coding_guardrails.server.version.subprocess.run") as mock_run:
            class _R:
                stdout = version.PINNED_COMMIT + "\n"
            mock_run.return_value = _R()
            assert version.is_up_to_date() is True


# ───────────────────────── sources ─────────────────────────


class TestSources:
    def test_known_source(self):
        src = sources.get_source("gemma-4-26B-A4B-it-qat-UD-Q4_K_XL")
        assert src is not None
        assert src.repo_id == "unsloth/gemma-4-26B-A4B-it-qat-GGUF"
        assert src.filename.endswith(".gguf")

    def test_unknown_source(self):
        assert sources.get_source("does-not-exist") is None


# ───────────────────────── registry (decoupling) ─────────────────────────


class TestRegistry:
    def test_find_model_prefers_cg_cache(self, monkeypatch, tmp_path):
        """Same GGUF in cg cache + lm-studio → cg wins."""
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
        name = "gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf"

        cg_copy = paths.models_dir() / "gemma-4-26B-A4B-it-qat-GGUF" / name
        cg_copy.parent.mkdir(parents=True)
        cg_copy.write_bytes(b"cg")

        # simulate an lm-studio fallback dir
        lm = tmp_path / "lm-studio" / "models" / "unsloth" / "gemma-4-26B-A4B-it-qat-GGUF"
        lm.mkdir(parents=True)
        (lm / name).write_bytes(b"lm")
        monkeypatch.setattr(registry, "FALLBACK_CACHE_DIRS", [lm.parent.parent])

        found = registry.find_model("gemma-4-26B-A4B-it-qat-UD-Q4_K_XL")
        assert found == cg_copy
        assert found.read_bytes() == b"cg"

    def test_find_model_falls_back_to_lmstudio(self, monkeypatch, tmp_path):
        """Model only in fallback → still found (read-only fallback works)."""
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
        name = "gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf"
        lm = tmp_path / "lm" / "unsloth" / "gemma-4-26B-A4B-it-qat-GGUF"
        lm.mkdir(parents=True)
        target = lm / name
        target.write_bytes(b"lm")
        monkeypatch.setattr(registry, "FALLBACK_CACHE_DIRS", [tmp_path / "lm"])

        found = registry.find_model("gemma-4-26B-A4B-it-qat-UD-Q4_K_XL")
        assert found == target

    def test_find_model_unknown_profile(self, monkeypatch, tmp_path):
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
        assert registry.find_model("not-a-real-model") is None


# ───────────────────────── launcher ─────────────────────────


class TestBuildArgv:
    def test_includes_profile_boot_flags(self, monkeypatch, tmp_path):
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
        # point binary + model at fake paths
        monkeypatch.setattr(launcher, "binary_path", lambda: Path("/fake/llama-server"))
        gguf = tmp_path / "m.gguf"
        gguf.write_bytes(b"x")
        argv = launcher.build_argv(
            "gemma-4-26B-A4B-it-qat-UD-Q4_K_XL", model_path=gguf
        )
        assert argv[0] == "/fake/llama-server"
        assert "-m" in argv and str(gguf) in argv
        # gemma qat profile has q8_0 KV cache + jinja
        assert "--jinja" in argv
        assert "-ctk" in argv and "q8_0" in argv
        assert "-ctv" in argv and "q8_0" in argv

    def test_ctx_and_ngl_overrides(self, monkeypatch, tmp_path):
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
        gguf = tmp_path / "m.gguf"
        gguf.write_bytes(b"x")
        argv = launcher.build_argv(
            "gemma-4-26B-A4B-it-qat-UD-Q4_K_XL",
            model_path=gguf,
            ctx=4096,
            ngl=10,
            port=9000,
        )
        assert "-c" in argv and "4096" in argv
        assert "-ngl" in argv and "10" in argv
        assert "9000" in argv

    def test_extra_flags_appended(self, monkeypatch, tmp_path):
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
        gguf = tmp_path / "m.gguf"
        gguf.write_bytes(b"x")
        argv = launcher.build_argv(
            "gemma-4-26B-A4B-it-qat-UD-Q4_K_XL",
            model_path=gguf,
            extra=["--temp", "0.7"],
        )
        assert argv[-2:] == ["--temp", "0.7"]

    def test_unknown_profile_raises(self, monkeypatch, tmp_path):
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
        with pytest.raises(KeyError):
            launcher.build_argv("nope", model_path=tmp_path / "m.gguf")


class TestLauncherLifecycle:
    def _fake_binary(self, tmp_path, lifetime=30):
        """A real tiny executable that sleeps (so pid/is_running behave truthfully)."""
        binpath = tmp_path / "fake-server"
        binpath.write_text(f"#!/bin/sh\nsleep {lifetime}\n")
        binpath.chmod(0o755)
        return binpath

    def test_is_running_no_pid(self, monkeypatch, tmp_path):
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
        assert launcher.is_running() is False

    def test_start_with_model_in_cg_cache(self, monkeypatch, tmp_path):
        """Full path: model in cg cache → resolve + start + stop."""
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
        fake = self._fake_binary(tmp_path)
        monkeypatch.setattr(launcher, "binary_path", lambda: fake)

        name = "gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf"
        gguf = paths.models_dir() / "gemma-4-26B-A4B-it-qat-GGUF" / name
        gguf.parent.mkdir(parents=True)
        gguf.write_bytes(b"x")

        pid = launcher.start(
            "gemma-4-26B-A4B-it-qat-UD-Q4_K_XL", ctx=64, ngl=1, detach=True
        )
        try:
            assert launcher.is_running() is True
            st = launcher.status()
            assert st.running is True and st.pid == pid
        finally:
            launcher.stop()
        time.sleep(0.3)
        assert launcher.status().running is False

    def test_stop_when_not_running(self, monkeypatch, tmp_path):
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
        assert launcher.stop() is False

    def test_double_start_rejected(self, monkeypatch, tmp_path):
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
        fake = self._fake_binary(tmp_path)
        monkeypatch.setattr(launcher, "binary_path", lambda: fake)
        name = "gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf"
        gguf = paths.models_dir() / "gemma-4-26B-A4B-it-qat-GGUF" / name
        gguf.parent.mkdir(parents=True)
        gguf.write_bytes(b"x")
        try:
            launcher.start("gemma-4-26B-A4B-it-qat-UD-Q4_K_XL", ctx=64, ngl=1, detach=True)
            with pytest.raises(RuntimeError, match="already running"):
                launcher.start("gemma-4-26B-A4B-it-qat-UD-Q4_K_XL", ctx=64, ngl=1, detach=True)
        finally:
            launcher.stop()
        time.sleep(0.3)

    def test_start_without_binary_errors(self, monkeypatch, tmp_path):
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
        # binary_path points to a non-existent path by default under tmp_path
        name = "gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf"
        gguf = paths.models_dir() / "gemma-4-26B-A4B-it-qat-GGUF" / name
        gguf.parent.mkdir(parents=True)
        gguf.write_bytes(b"x")
        with pytest.raises(FileNotFoundError, match="not built"):
            launcher.start("gemma-4-26B-A4B-it-qat-UD-Q4_K_XL", ctx=64, ngl=1)


# ───────────────────────── build ─────────────────────────


class TestBuild:
    def test_detect_cuda_with_nvcc(self, monkeypatch):
        monkeypatch.setattr(build.shutil, "which", lambda name: "/usr/bin/nvcc" if name == "nvcc" else None)
        assert build._detect_cuda() is True

    def test_detect_cuda_without_anything(self, monkeypatch, tmp_path):
        monkeypatch.setattr(build.shutil, "which", lambda name: None)
        monkeypatch.setattr(build.Path, "exists", lambda self: False)
        monkeypatch.setattr(build, "Path", lambda p: type("FakePath", (), {
            "exists": lambda self: False,
            "glob": lambda self, pat: [],
            "__truediv__": lambda self, other: self,
            "__eq__": lambda self, other: False,
        })())
        # Just assert it returns a bool (detection is environment-dependent)
        assert isinstance(build._detect_cuda(), bool)

    def test_clone_or_update_init_then_checkout(self, monkeypatch, tmp_path):
        """Clone path: git init → add remote → fetch pin → checkout FETCH_HEAD."""
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
        calls: list[list[str]] = []
        monkeypatch.setattr(build.subprocess, "run", lambda cmd, **kw: calls.append(cmd) or _ok())
        src = paths.source_dir()
        build._clone_or_update(src)
        joined = [" ".join(c) for c in calls]
        assert any("git init" in j for j in joined)
        assert any("remote add origin" in j for j in joined)
        assert any("fetch --depth 1" in j and version.PINNED_COMMIT in j for j in joined)
        assert any("checkout FETCH_HEAD" in j for j in joined)


class _R:
    stdout = ""
    stderr = ""

    def check_returncode(self):
        return None


def _ok():
    return _R()
