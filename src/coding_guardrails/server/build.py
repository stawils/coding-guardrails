"""Build llama.cpp from the pinned commit into cg's data directory.

Reproducible: every user gets the same binary (same commit, same flags). CUDA
is auto-detected; pass ``--cpu`` to force a CPU-only build.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path

from coding_guardrails.server.paths import (
    binary_path,
    build_dir,
    pin_file,
    run_dir,
    source_dir,
)
from coding_guardrails.server.version import PINNED_COMMIT, PINNED_SHORT

logger = logging.getLogger("coding_guardrails.server.build")

REPO_URL = "https://github.com/ggml-org/llama.cpp.git"


def _detect_cuda() -> bool:
    """True if a CUDA toolkit looks usable."""
    if shutil.which("nvcc"):
        return True
    local = Path("/usr/local")
    if (local / "cuda").exists():
        return True
    return any(p.name.startswith("cuda-") for p in local.glob("cuda-*"))


def _clone_or_update(src: Path) -> None:
    """Ensure ``src`` is a shallow checkout at the pinned commit."""
    if (src / ".git").exists():
        logger.info("Updating existing checkout -> %s", PINNED_SHORT)
        subprocess.run(
            ["git", "-C", str(src), "fetch", "--depth", "1", "origin", PINNED_COMMIT],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(src), "checkout", "FETCH_HEAD"], check=True
        )
        return

    src.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Cloning llama.cpp (pin %s) -> %s", PINNED_SHORT, src)
    subprocess.run(["git", "init", str(src)], check=True)
    subprocess.run(
        ["git", "-C", str(src), "remote", "add", "origin", REPO_URL], check=True
    )
    subprocess.run(
        ["git", "-C", str(src), "fetch", "--depth", "1", "origin", PINNED_COMMIT],
        check=True,
    )
    subprocess.run(["git", "-C", str(src), "checkout", "FETCH_HEAD"], check=True)


def build(force_cuda: bool | None = None, jobs: int | None = None) -> Path:
    """Clone/update + configure + build llama-server.

    Args:
        force_cuda: override detection. True=force CUDA on, False=CPU only,
            None=auto-detect.
        jobs: parallel build jobs (default: CPU count).

    Returns:
        Path to the compiled binary.

    Raises:
        RuntimeError: if the build completes but the binary is missing.
        subprocess.CalledProcessError: if any git/cmake step fails.
    """
    run_dir().mkdir(parents=True, exist_ok=True)
    src = source_dir()

    _clone_or_update(src)

    bdir = build_dir()
    bdir.mkdir(parents=True, exist_ok=True)

    use_cuda = _detect_cuda() if force_cuda is None else force_cuda

    configure = [
        "cmake",
        "-B",
        str(bdir),
        "-S",
        str(src),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DLLAMA_BUILD_TOOLS=ON",
    ]
    if use_cuda:
        configure.append("-DGGML_CUDA=ON")
        logger.info("CUDA: ON")
    else:
        logger.info("CUDA: OFF (CPU-only build)")
    if shutil.which("ninja"):
        configure.append("-GNinja")

    logger.info("Configure: %s", " ".join(configure))
    subprocess.run(configure, check=True)

    n = jobs or max(1, os.cpu_count() or 4)
    build_cmd = [
        "cmake",
        "--build",
        str(bdir),
        "--config",
        "Release",
        "-j",
        str(n),
    ]
    logger.info("Build: %s", " ".join(build_cmd))
    # Stream build output (cmake/ninja progress) to the caller's stdout.
    subprocess.run(build_cmd, check=True)

    if not binary_path().exists():
        raise RuntimeError(
            f"build finished but binary not found at {binary_path()}"
        )
    binary_path().chmod(0o755)

    pin_file().write_text(f"{PINNED_COMMIT}\n", encoding="utf-8")
    logger.info("OK: llama-server (%s) at %s", PINNED_SHORT, binary_path())
    return binary_path()
