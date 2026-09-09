"""Unit tests for ``BackendManager`` — lazy llama-server lifecycle.

No GPU, no subprocess: every external seam is monkeypatched at the module
paths the functions are imported into (manager's own namespace for the
``manager_vram`` helpers, the launcher module for start/stop/is_running, and
an instance-level patch for ``_await_health``).
"""

import pytest

from coding_guardrails.server import launcher
from coding_guardrails.server import manager as manager_mod
from coding_guardrails.server.manager import BackendConfig, BackendManager, BackendUnavailable, MultiBackendManager


def _fast_cfg(profile: str, margin: float = 0.0) -> BackendConfig:
    return BackendConfig(
        profile=profile,
        queue_timeout=0.2,
        poll_interval=0.01,
        health_timeout=0.5,
        vram_margin_gb=margin,
    )


def _make_manager(profile: str, margin: float = 0.0) -> BackendManager:
    return BackendManager(_fast_cfg(profile, margin))


def _patch_vram(monkeypatch: pytest.MonkeyPatch, free_value: float) -> None:
    """VRAM seams are imported INTO manager's namespace — patch them there."""
    monkeypatch.setattr(manager_mod, "free_vram_gb", lambda: free_value)
    monkeypatch.setattr(manager_mod, "gpu_holders", lambda: [])


def _patch_launcher(
    monkeypatch: pytest.MonkeyPatch, *, is_running: bool = False
) -> tuple[list, list]:
    """Patch the sync launcher seams; return (start_calls, stop_calls) logs."""
    start_calls: list = []
    stop_calls: list = []
    monkeypatch.setattr(launcher, "is_running", lambda: is_running)
    monkeypatch.setattr(launcher, "start", lambda *a, **k: start_calls.append((a, k)))
    monkeypatch.setattr(launcher, "stop", lambda *a, **k: stop_calls.append((a, k)))
    return start_calls, stop_calls


def _patch_health(monkeypatch: pytest.MonkeyPatch, mgr: BackendManager, ok: bool) -> None:
    async def fake(self):
        return ok

    monkeypatch.setattr(mgr, "_await_health", fake)


def test_vram_needed_math() -> None:
    # Profile gate: Qwen3.8-27B-UD-Q3_K_XL declares vram_required_gb=18.0 @120K ctx.
    mgr = _make_manager("Qwen3.8-27B-UD-Q3_K_XL", margin=0.0)
    assert mgr._vram_needed() == pytest.approx(18.0, rel=0.01)

    # Unknown profile falls back to the 18.0 GB baseline + margin.
    mgr2 = _make_manager("does-not-exist", margin=2.0)
    assert mgr2._vram_needed() == pytest.approx(20.0, rel=0.01)


@pytest.mark.asyncio
async def test_gate_ok_loads_and_releases(monkeypatch: pytest.MonkeyPatch) -> None:
    mgr = _make_manager("Qwen3.8-27B-UD-Q3_K_XL")
    _patch_vram(monkeypatch, 18.5)  # 18.5 >= 18.0 gate
    start_calls, _ = _patch_launcher(monkeypatch, is_running=False)
    _patch_health(monkeypatch, mgr, True)

    await mgr.acquire()
    assert mgr.is_loaded is True
    assert mgr._refcount == 1
    assert len(start_calls) == 1  # backend started exactly once

    await mgr.release()
    assert mgr._refcount == 0


@pytest.mark.asyncio
async def test_queue_timeout_raises_backend_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    mgr = _make_manager("Qwen3.8-27B-UD-Q3_K_XL")
    _patch_vram(monkeypatch, 0.0)  # VRAM never frees → queue times out
    start_calls, _ = _patch_launcher(monkeypatch, is_running=False)
    _patch_health(monkeypatch, mgr, True)

    with pytest.raises(BackendUnavailable) as e:
        await mgr.acquire()

    assert "VRAM busy" in str(e.value)
    assert start_calls == []  # start must NOT be called while VRAM-gated
    assert mgr.is_loaded is False
    assert mgr._refcount == 0  # refcount backed out on failure


@pytest.mark.asyncio
async def test_health_timeout_raises_and_stops_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    mgr = _make_manager("Qwen3.8-27B-UD-Q3_K_XL")
    _patch_vram(monkeypatch, 18.5)
    start_calls, stop_calls = _patch_launcher(monkeypatch, is_running=False)
    _patch_health(monkeypatch, mgr, False)  # backend never becomes healthy

    with pytest.raises(BackendUnavailable, match="failed to become healthy"):
        await mgr.acquire()

    assert len(start_calls) == 1  # it was started...
    assert len(stop_calls) >= 1  # ...then torn down again
    assert mgr.is_loaded is False


@pytest.mark.asyncio
async def test_unload_now_clears_loaded(monkeypatch: pytest.MonkeyPatch) -> None:
    mgr = _make_manager("Qwen3.8-27B-UD-Q3_K_XL")
    _patch_vram(monkeypatch, 18.5)
    _, _ = _patch_launcher(monkeypatch, is_running=False)
    _patch_health(monkeypatch, mgr, True)

    await mgr.acquire()
    assert mgr.is_loaded is True

    await mgr.unload_now()
    assert mgr.is_loaded is False


# ── MultiBackendManager: single-port unload-before-load swap ────────────────


def _multi_manager() -> MultiBackendManager:
    configs = {
        "Qwen3.8-27B-UD-Q3_K_XL": BackendConfig(
            profile="Qwen3.8-27B-UD-Q3_K_XL",
            queue_timeout=0.2, poll_interval=0.01, health_timeout=0.5,
            vram_margin_gb=0.0,
        ),
        "MiniCPM5-2B-Q4_K_M": BackendConfig(
            profile="MiniCPM5-2B-Q4_K_M",
            queue_timeout=0.2, poll_interval=0.01, health_timeout=0.5,
            vram_margin_gb=0.0,
        ),
    }
    return MultiBackendManager(configs, default="Qwen3.8-27B-UD-Q3_K_XL")


@pytest.mark.asyncio
async def test_multi_served_model_ids_and_resolve(monkeypatch: pytest.MonkeyPatch) -> None:
    mgr = _multi_manager()
    assert mgr.model_ids == ["Qwen3.8-27B-UD-Q3_K_XL", "MiniCPM5-2B-Q4_K_M"]
    assert mgr.default_model == "Qwen3.8-27B-UD-Q3_K_XL"
    # bare id, provider-prefixed id, default-when-empty
    assert mgr._resolve("Qwen3.8-27B-UD-Q3_K_XL") == "Qwen3.8-27B-UD-Q3_K_XL"
    assert mgr._resolve("coding-guardrails/MiniCPM5-2B-Q4_K_M") == "MiniCPM5-2B-Q4_K_M"
    assert mgr._resolve("") == "Qwen3.8-27B-UD-Q3_K_XL"
    with pytest.raises(BackendUnavailable):
        mgr._resolve("not-a-model")


@pytest.mark.asyncio
async def test_multi_unload_before_load_swap(monkeypatch: pytest.MonkeyPatch) -> None:
    """Requesting a different model unloads the current backend before loading."""
    mgr = _multi_manager()
    _patch_vram(monkeypatch, 20.0)  # plenty of VRAM for either model
    start_calls, stop_calls = _patch_launcher(monkeypatch, is_running=False)

    # Patch health at the class level so every inner BackendManager is healthy.
    async def fake_health(self, timeout):
        return True
    monkeypatch.setattr(BackendManager, "_await_health", fake_health)

    # First request: Qwen3.8 loads.
    async with mgr.use("Qwen3.8-27B-UD-Q3_K_XL"):
        assert mgr.current_model == "Qwen3.8-27B-UD-Q3_K_XL"
        assert mgr.is_loaded is True
    assert len(start_calls) == 1
    assert len(stop_calls) == 0

    # Swap to MiniCPM5: the Qwen backend must be unloaded (stop) before the new loads.
    async with mgr.use("coding-guardrails/MiniCPM5-2B-Q4_K_M"):
        assert mgr.current_model == "MiniCPM5-2B-Q4_K_M"
        assert mgr.is_loaded is True
    assert len(start_calls) == 2  # one fresh start per model
    assert len(stop_calls) == 1  # exactly one unload (the Qwen backend)

    # Back to Qwen3.8: another swap — stop MiniCPM5, start Qwen again.
    async with mgr.use("Qwen3.8-27B-UD-Q3_K_XL"):
        assert mgr.current_model == "Qwen3.8-27B-UD-Q3_K_XL"
    assert len(start_calls) == 3
    assert len(stop_calls) == 2

    await mgr.unload_now()
    assert mgr.is_loaded is False


@pytest.mark.asyncio
async def test_multi_unknown_model_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    mgr = _multi_manager()
    _patch_vram(monkeypatch, 20.0)
    _, stop_calls = _patch_launcher(monkeypatch, is_running=False)

    with pytest.raises(BackendUnavailable, match="unknown model"):
        async with mgr.use("does-not-exist"):
            pass

    assert mgr.current_model is None
    assert stop_calls == []  # nothing was loaded, so nothing was unloaded


@pytest.mark.asyncio
async def test_multi_single_model_backward_compat(monkeypatch: pytest.MonkeyPatch) -> None:
    """A one-config MultiBackendManager behaves the same as a single BackendManager."""
    mgr = MultiBackendManager(
        {"Qwen3.8-27B-UD-Q3_K_XL": BackendConfig(
            profile="Qwen3.8-27B-UD-Q3_K_XL",
            queue_timeout=0.2, poll_interval=0.01, health_timeout=0.5,
            vram_margin_gb=0.0,
        )},
        default="Qwen3.8-27B-UD-Q3_K_XL",
    )
    _patch_vram(monkeypatch, 18.5)
    start_calls, _ = _patch_launcher(monkeypatch, is_running=False)
    async def fake_health(self, timeout):
        return True
    monkeypatch.setattr(BackendManager, "_await_health", fake_health)

    async with mgr.use("whatever-the-agent-sends"):
        assert mgr.current_model == "Qwen3.8-27B-UD-Q3_K_XL"
    assert len(start_calls) == 1
