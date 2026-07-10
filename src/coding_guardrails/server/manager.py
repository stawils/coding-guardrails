"""Backend manager — lazy llama-server lifecycle behind the always-on proxy.

Architecture (node 1.3 of the L1-engine plan):
- The proxy (``cg serve``, :8081) is lightweight and **always-on** — Forge +
  guardrails, no GPU. Agents point at ``:8081/v1`` permanently.
- The heavy GPU model (the 18 GB backend on :8080) is **lazy**: loaded on demand
  and unloaded when idle, so the GPU is shared fairly with ComfyUI / Ollama / other
  workloads between requests.

Policy:
- ``acquire()`` — VRAM-gate + **QUEUE** (wait for free VRAM; never refuse/evict) +
  load + health-wait, with **dedupe** (one load in flight; concurrent acquires share it).
- ``release()`` — mark a request done; (re)start the idle timer.
- **idle-unload** — after ``idle_timeout`` with no in-flight requests, unload (free VRAM).
- **queue-timeout** — if VRAM never frees within ``queue_timeout``, raise
  ``BackendUnavailable`` → the proxy returns 503 → the fleet escalates to L2 (zi-02).
  The local tier degrades gracefully; it never blocks forever and never evicts.
"""

from __future__ import annotations

import asyncio
import logging
import time
import urllib.request
from dataclasses import dataclass

from coding_guardrails.server import launcher
from coding_guardrails.server.manager_vram import free_vram_gb, gpu_holders, llama_processes

logger = logging.getLogger("coding_guardrails.server.manager")


class BackendUnavailable(RuntimeError):
    """Backend could not be acquired (e.g. VRAM busy past the queue-timeout)."""


@dataclass
class BackendConfig:
    """Lifecycle parameters for the managed backend."""

    profile: str
    host: str = "0.0.0.0"
    port: int = 8080
    ctx: int | None = None
    ngl: int = 99
    # timing (seconds)
    idle_timeout: float = 90.0
    queue_timeout: float = 120.0
    poll_interval: float = 3.0
    health_timeout: float = 180.0  # an 18 GB load can take a while
    vram_margin_gb: float = 2.0


class BackendManager:
    """Owns the acquire / idle-unload lifecycle of one llama-server backend."""

    def __init__(self, cfg: BackendConfig) -> None:
        self.cfg = cfg
        self._lock = asyncio.Lock()
        self._loaded = asyncio.Event()
        self._refcount = 0
        self._load_task: asyncio.Task | None = None
        self._idle_task: asyncio.Task | None = None

    @property
    def is_loaded(self) -> bool:
        return self._loaded.is_set()

    @property
    def refcount(self) -> int:
        return self._refcount

    async def acquire(self) -> None:
        """Ensure the backend is loaded + healthy before a request.

        VRAM-gates + queues + dedupes. Raises ``BackendUnavailable`` on
        queue-timeout / load failure (caller surfaces 503 → fleet L2 fallback).
        """
        async with self._lock:
            self._refcount += 1
            self._cancel_idle_timer()
            if self._loaded.is_set():
                logger.debug("acquire: already loaded (refcount=%d)", self._refcount)
                return
            if self._load_task is None or self._load_task.done():
                self._load_task = asyncio.create_task(self._load())
            load_task = self._load_task

        # Await the (possibly shared) load outside the lock.
        try:
            await load_task
        except BaseException:
            # We bumped refcount but won't use the backend — back it out, and let
            # the next acquire retry (only the first clearer resets _load_task).
            async with self._lock:
                self._refcount = max(0, self._refcount - 1)
                if self._load_task is load_task:
                    self._load_task = None
                self._maybe_idle()
            raise

    async def release(self) -> None:
        """Mark a request complete; start the idle-unload timer if now idle."""
        async with self._lock:
            self._refcount = max(0, self._refcount - 1)
            self._maybe_idle()

    async def unload_now(self) -> None:
        """Force-unload (e.g. ``cg down``). Cancels any in-flight load/idle."""
        async with self._lock:
            self._cancel_idle_timer()
            await self._do_unload(reason="forced (unload_now)")

    def verify_clean(self) -> dict:
        """Post-unload cleanliness report: orphan llama-server procs + free VRAM.

        ``clean`` is True only when no orphan llama-server remains. Used by
        ``cg status`` and node-1.4 acceptance (clean unload = VRAM freed + no orphans).
        """
        procs = llama_processes()
        return {
            "orphans": len(procs),
            "orphan_pids": [pid for pid, _ in procs],
            "free_vram_gb": free_vram_gb(),
            "clean": len(procs) == 0,
        }

    # ── internals ────────────────────────────────────────────────────────────

    async def _load(self) -> None:
        """VRAM-gate (queue) → start → health-wait → mark loaded."""
        cfg = self.cfg

        # Adopt an already-running backend (manual `cg server start`, or a stale
        # managed one) instead of double-starting.
        if launcher.is_running():
            logger.info("acquire: backend process already running — awaiting health")
            if await self._await_health(cfg.health_timeout):
                logger.info("acquire: adopted already-running backend")
                self._loaded.set()
                return
            # Running but not becoming healthy — tear down and reload fresh.
            logger.warning("acquire: running but unhealthy; restarting")
            await asyncio.to_thread(launcher.stop)

        # 1) VRAM-gate + QUEUE
        need = self._vram_needed()
        deadline = time.monotonic() + cfg.queue_timeout
        while True:
            free = free_vram_gb()
            if free >= need:
                logger.info("VRAM-gate OK: %.1f GB free ≥ %.1f needed", free, need)
                break
            holders = gpu_holders()
            who = ", ".join(f"{n}({g:.1f}GB)" for _, n, g in holders) or "unknown"
            if time.monotonic() >= deadline:
                raise BackendUnavailable(
                    f"VRAM busy after {cfg.queue_timeout:.0f}s queue: "
                    f"{free:.1f} GB free, {need:.1f} needed (held by: {who})"
                )
            logger.info(
                "VRAM-gate WAIT: %.1f GB free < %.1f needed (held by: %s) — queuing",
                free, need, who,
            )
            await asyncio.sleep(cfg.poll_interval)

        # 2) Start (sync Popen) off the event loop
        await asyncio.to_thread(
            launcher.start,
            cfg.profile,
            ctx=cfg.ctx,
            ngl=cfg.ngl,
            host=cfg.host,
            port=cfg.port,
        )

        # 3) Ready-before-serving: no traffic until /health 200
        if not await self._await_health(cfg.health_timeout):
            await asyncio.to_thread(launcher.stop)
            raise BackendUnavailable(
                f"backend failed to become healthy within {cfg.health_timeout:.0f}s"
            )

        self._loaded.set()
        logger.info("acquire: backend loaded + healthy")

    async def _do_unload(self, reason: str = "idle") -> None:
        """Stop the backend + clear state. Caller holds ``_lock``."""
        if not self._loaded.is_set() and not launcher.is_running():
            return
        await asyncio.to_thread(launcher.stop)
        self._loaded.clear()
        if self._load_task is not None and not self._load_task.done():
            self._load_task.cancel()
        self._load_task = None
        report = self.verify_clean()
        if report["orphans"]:
            logger.warning(
                "backend unloaded (%s) but %d orphan llama-server remain: %s",
                reason, report["orphans"], report["orphan_pids"],
            )
        else:
            logger.info(
                "backend unloaded (%s) — VRAM freed (%.1f GB free), no orphans",
                reason, report["free_vram_gb"],
            )

    def _maybe_idle(self) -> None:
        """Arm the idle-unload timer if idle + loaded. Caller holds ``_lock``."""
        if self._refcount == 0 and self._loaded.is_set() and self._idle_task is None:
            self._idle_task = asyncio.create_task(self._idle_unload())

    async def _idle_unload(self) -> None:
        try:
            await asyncio.sleep(self.cfg.idle_timeout)
        except asyncio.CancelledError:
            return
        async with self._lock:
            self._idle_task = None
            if self._refcount == 0 and self._loaded.is_set():
                await self._do_unload(reason=f"idle {self.cfg.idle_timeout:.0f}s")

    def _cancel_idle_timer(self) -> None:
        """Caller holds ``_lock``."""
        if self._idle_task is not None and not self._idle_task.done():
            self._idle_task.cancel()
        self._idle_task = None

    def _vram_needed(self) -> float:
        from coding_guardrails.models.profiles import get_profile

        profile = get_profile(self.cfg.profile)
        base = profile.vram_required_gb if profile else 18.0
        return base + self.cfg.vram_margin_gb

    async def _await_health(self, timeout: float) -> bool:
        url = f"http://127.0.0.1:{self.cfg.port}/health"
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                urllib.request.urlopen(url, timeout=3)
                return True
            except Exception:  # noqa: BLE001 — not healthy yet
                await asyncio.sleep(1.0)
        return False
