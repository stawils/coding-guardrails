"""Model registry — discover GGUFs from cg's own cache and common fallbacks.

cg's own cache (``~/.local/share/coding-guardrails/models``) is searched first;
LM Studio and HuggingFace caches are kept as read-only fallbacks so existing
installs keep working, but a cg user never needs them.
"""

from __future__ import annotations

import logging
from pathlib import Path

from coding_guardrails.models.profiles import ModelProfile, get_profile
from coding_guardrails.server.paths import models_dir

logger = logging.getLogger("coding_guardrails.models")

# cg's own cache is primary; the rest are read-only fallbacks.
FALLBACK_CACHE_DIRS = [
    Path.home() / ".cache" / "lm-studio" / "models",
    Path.home() / ".cache" / "huggingface" / "hub",
    Path.home() / "models",
]


def _search_dirs(extra_dirs: list[str] | None = None) -> list[Path]:
    dirs = [models_dir()] + FALLBACK_CACHE_DIRS.copy()
    if extra_dirs:
        dirs.extend(Path(d) for d in extra_dirs)
    return dirs


def discover_models(
    extra_dirs: list[str] | None = None,
) -> list[tuple[Path, ModelProfile]]:
    """Scan for GGUF files and match them to known profiles.

    Returns list of (gguf_path, profile) tuples. cg's cache wins on ties.
    """
    found: list[tuple[Path, ModelProfile]] = []
    seen_profiles: set[str] = set()

    for cache_dir in _search_dirs(extra_dirs):
        if not cache_dir.exists():
            continue
        for gguf in cache_dir.rglob("*.gguf"):
            profile = get_profile(gguf.stem)
            if profile is None:
                continue
            if profile.name in seen_profiles:
                continue  # earlier (higher-priority) dir already found it
            found.append((gguf, profile))
            seen_profiles.add(profile.name)
    return found


def find_model(profile_name: str, extra_dirs: list[str] | None = None) -> Path | None:
    """Return the GGUF path for ``profile_name``, or None if not found.

    Resolves the profile (fuzzy), then searches cg's cache first.
    """
    profile = get_profile(profile_name)
    if profile is None:
        return None
    # Candidate filenames to look for. The canonical one is the registered
    # source filename (what download.py writes); the profile-name stem is a
    # fallback for models whose GGUF is named after the profile. Source wins
    # so discovery and download always agree, even when the upstream GGUF name
    # differs in casing from the profile name (e.g. Ornith's lowercase files).
    from coding_guardrails.server.sources import get_source

    candidates = [f"{profile.name}.gguf"]
    src = get_source(profile_name)
    if src is not None:
        candidates.insert(0, src.filename)
    for cache_dir in _search_dirs(extra_dirs):
        if not cache_dir.exists():
            continue
        for fname in candidates:
            for gguf in cache_dir.rglob(fname):
                return gguf
    return None
