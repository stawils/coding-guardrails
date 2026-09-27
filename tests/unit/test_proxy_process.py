"""Tests for managed-proxy process discovery and identity checks."""

import os
import subprocess
import sys
import time

from coding_guardrails.server.proxy_process import (
    is_proxy_cmdline,
    pid_is_proxy,
    proxy_processes,
    read_cmdline,
    terminate,
)


class TestIsProxyCmdline:

    def test_serve_cmdline_is_proxy(self):
        assert is_proxy_cmdline(
            "python3 -m coding_guardrails serve --manage-backend --port 8081"
        )

    def test_up_cmdline_is_not_proxy(self):
        assert not is_proxy_cmdline("coding-guardrails up --model Qwen3.8-27B")

    def test_down_cmdline_is_not_proxy(self):
        assert not is_proxy_cmdline("coding-guardrails down")

    def test_llama_server_is_not_proxy(self):
        assert not is_proxy_cmdline("llama-server -m model.gguf --port 8080")

    def test_requires_coding_guardrails_marker(self):
        assert not is_proxy_cmdline("python -m other_tool serve")

    def test_pgrep_itself_is_not_proxy(self):
        assert not is_proxy_cmdline("pgrep -af coding_guardrails")


class TestReadCmdline:

    def test_read_own_cmdline(self):
        cmdline = read_cmdline(os.getpid())
        assert cmdline is not None
        assert cmdline  # non-empty

    def test_dead_pid_returns_none(self):
        # A pid far above the pid_max ceiling cannot exist.
        assert read_cmdline(2**30) is None

    def test_invalid_pid_returns_none(self):
        assert read_cmdline(0) is None
        assert read_cmdline(-1) is None


class TestPidIsProxy:

    def test_pytest_is_not_proxy(self):
        assert not pid_is_proxy(os.getpid())

    def test_dead_pid_is_not_proxy(self):
        assert not pid_is_proxy(2**30)


class TestProxyProcesses:

    def test_returns_list(self):
        procs = proxy_processes()
        assert isinstance(procs, list)
        for pid, cmdline in procs:
            assert isinstance(pid, int)
            assert is_proxy_cmdline(cmdline)

    def test_pytest_process_not_included(self):
        assert os.getpid() not in [p for p, _ in proxy_processes()]


class TestTerminate:

    def test_terminates_a_live_process(self):
        proc = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"]
        )
        try:
            assert terminate(proc.pid, term_timeout=5.0, kill_timeout=2.0)
            proc.wait(timeout=5)
            assert proc.poll() is not None
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)

    def test_already_dead_process_returns_true(self):
        proc = subprocess.Popen([sys.executable, "-c", "pass"])
        proc.wait(timeout=5)
        # Reap so the pid is gone, then terminating it must be a no-op success.
        time.sleep(0.1)
        assert terminate(proc.pid, term_timeout=0.5, kill_timeout=0.5)
