"""Launch / stop / inspect the cg-owned llama-server.

The server runs detached (own process group) so it survives the CLI exiting.
Its PID and a combined stdout/stderr log live under ``server run_dir``.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from coding_guardrails.models.profiles import get_profile
from coding_guardrails.server.paths import (
    binary_path,
    log_file,
    pid_file,
    run_dir,
)

logger = logging.getLogger("coding_guardrails.server.launcher")


@dataclass(frozen=True)
class ServerStatus:
    """Snapshot of the server's state."""

    running: bool
    pid: int | None
    binary_present: bool
    binary_version: str | None


def _read_pid() -> int | None:
    pf = pid_file()
    if not pf.exists():
        return None
    try:
        return int(pf.read_text().strip())
    except (ValueError, OSError):
        return None


def _is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def is_running() -> bool:
    """True if the recorded PID is a live process."""
    pid = _read_pid()
    return pid is not None and _is_alive(pid)


def _clear_stale_pid() -> None:
    pf = pid_file()
    if pf.exists():
        try:
            pf.unlink()
        except OSError:
            pass


def _resolve_model(profile_name: str) -> Path:
    """Locate the model GGUF for a profile name (cg cache first)."""
    from coding_guardrails.models.registry import find_model

    path = find_model(profile_name)
    if path is None:
        profile = get_profile(profile_name)
        hint = f"Profile {profile_name!r} is unknown." if profile is None else ""
        raise FileNotFoundError(
            f"No GGUF found for model {profile_name!r}. {hint}\n"
            f"Try: cg server download {profile_name}"
        )
    return path


def build_argv(
    profile_name: str,
    *,
    model_path: Path | None = None,
    ctx: int | None = None,
    ngl: int = 99,
    host: str = "0.0.0.0",
    port: int = 8080,
    extra: list[str] | None = None,
) -> list[str]:
    """Assemble the llama-server argv from a profile.

    Args:
        profile_name: key in PROFILES.
        model_path: override model resolution.
        ctx: context window (default: profile.context_tokens).
        ngl: GPU layers (default: 99 = all).
        host/port: bind address.
        extra: additional raw flags appended verbatim.
    """
    profile = get_profile(profile_name)
    if profile is None:
        raise KeyError(f"unknown model profile: {profile_name!r}")
    gguf = model_path if model_path is not None else _resolve_model(profile_name)

    argv = [
        str(binary_path()),
        "-m",
        str(gguf),
        "-c",
        str(ctx or profile.context_tokens),
        "-ngl",
        str(ngl),
        "--host",
        host,
        "--port",
        str(port),
    ]
    argv.extend(profile.boot_flags)
    if extra:
        argv.extend(extra)
    return argv


def start(
    profile_name: str,
    *,
    ctx: int | None = None,
    ngl: int = 99,
    host: str = "0.0.0.0",
    port: int = 8080,
    extra: list[str] | None = None,
    detach: bool = True,
) -> int:
    """Launch llama-server. Returns the PID.

    Args:
        detach: if True (default), the server runs in its own process group so
            it outlives the CLI. If False, the caller is responsible for the
            child (used by tests).
    """
    binary = binary_path()
    if not binary.exists():
        raise FileNotFoundError(
            f"llama-server binary not built. Run: cg server build"
        )
    if is_running():
        raise RuntimeError(
            f"server already running (pid {_read_pid()}). Run: cg server stop"
        )

    run_dir().mkdir(parents=True, exist_ok=True)
    argv = build_argv(
        profile_name,
        ctx=ctx,
        ngl=ngl,
        host=host,
        port=port,
        extra=extra,
    )
    logger.info("Launch: %s", " ".join(argv))

    log_handle = log_file().open("a", buffering=1)
    log_handle.write(f"\n=== cg server start {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    log_handle.write(" ".join(argv) + "\n")
    log_handle.flush()

    popen_kwargs: dict = {
        "stdout": log_handle,
        "stderr": subprocess.STDOUT,
        "stdin": subprocess.DEVNULL,
    }
    if detach:
        popen_kwargs["start_new_session"] = True

    proc = subprocess.Popen(argv, **popen_kwargs)
    pid_file().write_text(f"{proc.pid}\n", encoding="utf-8")
    logger.info("Started pid %d -> %s", proc.pid, log_file())
    return proc.pid


def stop(timeout: float = 10.0) -> bool:
    """SIGTERM the running server. Returns True if it stopped."""
    pid = _read_pid()
    if pid is None or not _is_alive(pid):
        _clear_stale_pid()
        return False
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        _clear_stale_pid()
        return False

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _is_alive(pid):
            _clear_stale_pid()
            return True
        time.sleep(0.2)
    # Still alive: escalate.
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    _clear_stale_pid()
    return True


def status() -> ServerStatus:
    """Return a snapshot of server + binary state."""
    from coding_guardrails.server.version import binary_version

    pid = _read_pid()
    running = pid is not None and _is_alive(pid)
    if pid and not running:
        _clear_stale_pid()
        pid = None
    return ServerStatus(
        running=running,
        pid=pid,
        binary_present=binary_path().exists(),
        binary_version=binary_version() if binary_path().exists() else None,
    )
