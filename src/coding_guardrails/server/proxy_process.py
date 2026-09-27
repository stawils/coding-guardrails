"""Discover and identify the managed ``coding-guardrails`` proxy process.

The pid file is the fast path, but it can be missing or stale while a proxy is
still running — an orphan that keeps holding the port. When that happens,
``cg up`` starts a new proxy which dies on bind, and systemd (Type=forking +
PIDFile) enters a restart loop.

These helpers make ``cg up``/``cg down``/``cg status`` find the real process
and verify its identity before signalling it, so:

* a stale pid file can never make ``cg up`` refuse to start (PID reuse), and
* ``cg down`` never SIGKILLs an unrelated process that happens to reuse a pid.

Identity is decided from ``/proc/<pid>/cmdline``: a managed proxy is a process
whose command line contains ``coding_guardrails`` and the ``serve`` subcommand
(the ``up``/``down`` CLI processes are short-lived and excluded).
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path


def is_proxy_cmdline(cmdline: str) -> bool:
    """True if a command line belongs to a long-running managed proxy."""
    if "coding_guardrails" not in cmdline:
        return False
    return "serve" in cmdline.split()


def read_cmdline(pid: int) -> str | None:
    """Read a process command line from /proc, or None if it is gone."""
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return None
    return raw.replace(b"\x00", b" ").decode("utf-8", "replace").strip()


def pid_is_proxy(pid: int) -> bool:
    """True if ``pid`` is alive and is a managed coding-guardrails proxy."""
    if pid <= 0:
        return False
    cmdline = read_cmdline(pid)
    return cmdline is not None and is_proxy_cmdline(cmdline)


def proxy_processes() -> list[tuple[int, str]]:
    """Running managed proxies as ``[(pid, cmdline)]``.

    Best-effort via ``pgrep``; returns an empty list if pgrep is unavailable.
    """
    try:
        proc = subprocess.run(
            ["pgrep", "-af", "coding_guardrails"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:  # noqa: BLE001 - best-effort discovery
        return []

    procs: list[tuple[int, str]] = []
    for line in proc.stdout.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        pid_s, _, cmdline = line.partition(" ")
        if not pid_s.isdigit():
            continue
        if is_proxy_cmdline(cmdline):
            procs.append((int(pid_s), cmdline))
    return procs


def _is_zombie(pid: int) -> bool:
    """True if ``pid`` exists only as an unreaped zombie (already dead)."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return False
    # The comm field can contain spaces/parens; state is the token after the
    # last ')'. A zombie has state 'Z' and os.kill(pid, 0) still succeeds.
    rparen = stat.rfind(")")
    if rparen == -1:
        return False
    fields = stat[rparen + 1:].split()
    return bool(fields) and fields[0] == "Z"


def terminate(pid: int, term_timeout: float = 6.0, kill_timeout: float = 2.0) -> bool:
    """Stop ``pid`` with SIGTERM, escalating to SIGKILL. True once it is gone.

    Never raises if the process has already exited.
    """
    for sig, timeout in ((signal.SIGTERM, term_timeout), (signal.SIGKILL, kill_timeout)):
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            return True
        except PermissionError:
            return False
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return True
            if _is_zombie(pid):
                return True
            time.sleep(0.1)
    return False
