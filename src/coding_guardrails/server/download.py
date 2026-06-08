"""Download model GGUFs into cg's own cache via huggingface_hub."""

from __future__ import annotations

import logging
from pathlib import Path

from coding_guardrails.server.paths import models_dir
from coding_guardrails.server.sources import ModelSource, get_source

logger = logging.getLogger("coding_guardrails.server.download")


def target_path(profile_name: str) -> Path:
    """Where a profile's GGUF will live in the cg cache."""
    src = get_source(profile_name)
    if src is None:
        raise KeyError(
            f"no download source registered for {profile_name!r}. "
            "Add an entry to server/sources.py SOURCES."
        )
    return models_dir() / src.repo_id.split("/")[-1] / src.filename


def download(profile_name: str) -> Path:
    """Download the GGUF for ``profile_name`` into the cg cache.

    Returns the path to the downloaded file. Re-downloads are skipped by
    huggingface_hub's cache only if its own cache is used; here we download
    directly into the cg models dir, so an existing complete file is kept.
    """
    src: ModelSource | None = get_source(profile_name)
    if src is None:
        raise KeyError(
            f"no download source registered for {profile_name!r}."
        )
    dest = target_path(profile_name)
    if dest.exists() and dest.stat().st_size > 0:
        logger.info("Already present: %s", dest)
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - dep present in cg env
        raise RuntimeError(
            "huggingface_hub is required for downloads. Install it (pip install "
            "huggingface_hub) and retry."
        ) from exc

    logger.info("Downloading %s/%s -> %s", src.repo_id, src.filename, dest.parent)
    fetched = hf_hub_download(
        repo_id=src.repo_id,
        filename=src.filename,
        local_dir=str(dest.parent),
        local_dir_use_symlinks=False,
    )
    fetched_path = Path(fetched)
    # hf_hub_download may place it under a nested local_dir layout; normalize.
    if fetched_path.resolve() != dest.resolve():
        fetched_path.replace(dest)
    logger.info("OK: %s (%.1f GB)", dest, dest.stat().st_size / 1e9)
    return dest
