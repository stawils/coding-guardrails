"""GPU VRAM introspection for the backend manager.

Best-effort queries via nvidia-smi. All functions fail OPEN (return a permissive
value) if nvidia-smi is unavailable, so a missing/odd GPU state never hard-blocks
inference — it lets llama-server decide (and OOM-handle) rather than the proxy
refusing to try.
"""

from __future__ import annotations

import logging
import subprocess

logger = logging.getLogger("coding_guardrails.server.vram")


def _run_nvidia_smi(fields: str) -> str | None:
    """Run an nvidia-smi --query, returning stdout or None on any failure."""
    try:
        proc = subprocess.run(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return proc.stdout.strip()
    except Exception as exc:  # noqa: BLE001 — any failure means "can't query"
        logger.warning("nvidia-smi query (%s) failed: %s", fields, exc)
        return None


def free_vram_gb() -> float:
    """Free GPU memory in GB.

    Returns ``float('inf')`` if nvidia-smi is unavailable so the VRAM-gate
    fails OPEN (allows the load) rather than deadlocking on a query error.
    """
    out = _run_nvidia_smi("memory.free")
    if out is None or not out:
        return float("inf")
    try:
        return float(out.splitlines()[0]) / 1024.0
    except (ValueError, IndexError):
        return float("inf")


def gpu_holders() -> list[tuple[int, str, float]]:
    """Processes currently holding GPU memory: ``[(pid, process_name, gb)]``.

    Used only for human-readable queue messages (who is holding the GPU).
    Best-effort; returns ``[]`` on any failure.
    """
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except Exception:  # noqa: BLE001
        return []

    holders: list[tuple[int, str, float]] = []
    for line in proc.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            holders.append((int(parts[0]), parts[1], float(parts[2]) / 1024.0))
        except (ValueError, IndexError):
            continue
    return holders
