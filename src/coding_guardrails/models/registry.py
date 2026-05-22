"""Model registry — auto-discover models from common cache directories."""

from __future__ import annotations

import logging
from pathlib import Path

from coding_guardrails.models.profiles import ModelProfile, get_profile

logger = logging.getLogger("coding_guardrails.models")

# Common GGUF cache locations
CACHE_DIRS = [
    Path.home() / ".cache" / "lm-studio" / "models",
    Path.home() / ".cache" / "huggingface" / "hub",
    Path.home() / "models",
]


def discover_models(extra_dirs: list[str] | None = None) -> list[tuple[Path, ModelProfile]]:
    """Scan for GGUF files and match them to known profiles.

    Returns list of (gguf_path, profile) tuples.
    """
    dirs = CACHE_DIRS.copy()
    if extra_dirs:
        dirs.extend(Path(d) for d in extra_dirs)

    found: list[tuple[Path, ModelProfile]] = []

    for cache_dir in dirs:
        if not cache_dir.exists():
            continue
        for gguf in cache_dir.rglob("*.gguf"):
            name = gguf.stem
            profile = get_profile(name)
            if profile:
                found.append((gguf, profile))
            else:
                logger.debug("Unknown model: %s", gguf)

    return found
