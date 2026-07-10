"""Tests for the managed backend (node 1.3): VRAM-gate + queue + idle-unload + dedupe.

All tests run without a GPU — the launcher, VRAM query, and health check are faked
via monkeypatch so the lifecycle logic is exercised deterministically.
"""

from __future__ import annotations

import asyncio

import pytest

from coding_guardrails.server import manager as mgr_mod
from coding_guardrails.server.manager import BackendConfig, BackendManager, BackendUnavailable

PROFILE = "Qwen3.5-9B-UD-Q4_K_XL"  # vram_required_gb = 18.1 → need 20.1 with default margin


class FakeLauncher:
    """Stand-in for coding_guardrails.server.launcher."""

    def __init__(self) -> None:
        self.starts = 0
        self.stops = 0
        self._running = False

    def start(self, profile, **kw):  # noqa: ANN001 — mirrors launcher.start signature
        self.starts += 1
        self._running = True
        return 12345

    def stop(self, timeout: float = 10.0) -> bool:
        self.stops += 1
        self._running = False
        return True

    def is_running(self) -> bool:
        return self._running


@pytest.fixture
def fake(monkeypatch):
    fl = FakeLauncher()
    monkeypatch.setattr(mgr_mod, "launcher", fl)
    state = {"free": 24.0}  # GB free — controllable; 24 ≥ 20.1 needed → "free"
    monkeypatch.setattr(mgr_mod, "free_vram_gb", lambda: state["free"])
    monkeypatch.setattr(mgr_mod, "gpu_holders", lambda: [(99, "comfyui", 18.0)])

    # Health == "loaded?" (True once the fake launcher has started).
    async def fake_health(self, timeout):  # noqa: ANN001
        return fl._running

    monkeypatch.setattr(BackendManager, "_await_health", fake_health)
    return fl, state


def _cfg(**over) -> BackendConfig:
    base = dict(profile=PROFILE, idle_timeout=99.0, poll_interval=0.01, queue_timeout=2.0)
    base.update(over)
    return BackendConfig(**base)


async def test_acquire_loads_when_vram_free(fake):
    fl, _ = fake
    m = BackendManager(_cfg(idle_timeout=0.05))
    await m.acquire()
    assert fl.starts == 1
    assert m.is_loaded
    assert m.refcount == 1

    await m.release()
    assert m.refcount == 0
    # idle-unload fires after idle_timeout
    await asyncio.sleep(0.15)
    assert fl.stops == 1
    assert not m.is_loaded


async def test_acquire_queues_when_busy_then_frees(fake):
    fl, state = fake
    state["free"] = 5.0  # busy (held by ComfyUI in the fake holders)
    m = BackendManager(_cfg())

    async def free_it():
        await asyncio.sleep(0.05)
        state["free"] = 24.0

    asyncio.create_task(free_it())
    await m.acquire()  # queues, then loads once VRAM frees
    assert fl.starts == 1
    assert m.is_loaded
    await m.release()
    await m.unload_now()


async def test_queue_timeout_raises_and_never_loads(fake):
    fl, state = fake
    state["free"] = 5.0  # stays busy
    m = BackendManager(_cfg(queue_timeout=0.05))

    with pytest.raises(BackendUnavailable):
        await m.acquire()
    assert fl.starts == 0          # never loaded
    assert m.refcount == 0         # refcount rolled back
    assert not m.is_loaded


async def test_concurrent_acquires_dedupe_to_one_load(fake):
    fl, _ = fake
    m = BackendManager(_cfg())

    await asyncio.gather(m.acquire(), m.acquire(), m.acquire())
    assert fl.starts == 1          # one shared load
    assert m.is_loaded
    assert m.refcount == 3

    for _ in range(3):
        await m.release()
    await m.unload_now()


async def test_adopt_already_running_healthy(fake):
    fl, _ = fake
    fl._running = True  # backend already up (manual `cg server start`)
    m = BackendManager(_cfg())

    await m.acquire()
    assert fl.starts == 0          # adopted, not re-started
    assert m.is_loaded
    await m.release()
    await m.unload_now()


async def test_acquire_during_idle_cancels_unload(fake):
    fl, _ = fake
    m = BackendManager(_cfg(idle_timeout=0.05))

    await m.acquire()
    await m.release()              # idle timer armed
    await asyncio.sleep(0.02)      # partway through idle window
    await m.acquire()              # new request cancels the idle-unload
    await asyncio.sleep(0.15)      # past the original idle deadline
    assert fl.stops == 0           # was NOT unloaded — still loaded
    assert m.is_loaded
    await m.release()
    await m.unload_now()


def test_free_vram_fails_open(monkeypatch):
    """If nvidia-smi is unavailable, the gate fails OPEN (allows load)."""
    from coding_guardrails.server import manager_vram

    def boom(*a, **k):
        raise FileNotFoundError("no nvidia-smi")

    monkeypatch.setattr(manager_vram.subprocess, "run", boom)
    assert manager_vram.free_vram_gb() == float("inf")
    assert manager_vram.gpu_holders() == []


async def test_unload_now_is_idempotent(fake):
    fl, _ = fake
    m = BackendManager(_cfg())
    await m.unload_now()           # nothing loaded — no-op, no crash
    assert fl.stops == 0
    await m.acquire()
    assert m.is_loaded
    await m.unload_now()
    assert fl.stops == 1
    assert not m.is_loaded
