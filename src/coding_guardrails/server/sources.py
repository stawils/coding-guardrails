"""HuggingFace source mapping for downloadable models.

Keeps download provenance reproducible without touching ``models/profiles.py``.
Each entry maps a profile name (the key in ``PROFILES``) to the exact GGUF to
fetch. Only sources verified against an existing on-disk cache are listed;
add new entries as you verify them.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSource:
    """Where to download a model's GGUF from."""

    repo_id: str
    filename: str
    license: str = ""


# Keyed by profile name (must match PROFILES in models/profiles.py).
# Filenames verified against the local cache.
SOURCES: dict[str, ModelSource] = {
    "gemma-4-26B-A4B-it-qat-UD-Q4_K_XL": ModelSource(
        repo_id="unsloth/gemma-4-26B-A4B-it-qat-GGUF",
        filename="gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf",
        license="Gemma",
    ),
    "gemma-4-26B-A4B-it-UD-Q3_K_XL": ModelSource(
        repo_id="unsloth/gemma-4-26B-A4B-it-GGUF",
        filename="gemma-4-26B-A4B-it-UD-Q3_K_XL.gguf",
        license="Gemma",
    ),
    "gemma-4-12b-it-UD-Q4_K_XL": ModelSource(
        repo_id="unsloth/gemma-4-12b-it-GGUF",
        filename="gemma-4-12b-it-UD-Q4_K_XL.gguf",
        license="Gemma",
    ),
    "Qwen3.5-9B-UD-Q4_K_XL": ModelSource(
        repo_id="unsloth/Qwen3.5-9B-MTP-GGUF",
        filename="Qwen3.5-9B-UD-Q4_K_XL.gguf",
        license="Apache-2.0",
    ),
    "Ornith-1.0-9B-Q8_0": ModelSource(
        repo_id="deepreinforce-ai/Ornith-1.0-9B-GGUF",
        filename="ornith-1.0-9b-Q8_0.gguf",
        license="MIT",
    ),
    "Qwen3.6-27B-UD-Q4_K_XL": ModelSource(
        repo_id="unsloth/Qwen3.6-27B-MTP-GGUF",
        filename="Qwen3.6-27B-UD-Q4_K_XL.gguf",
        license="Apache-2.0",
    ),
}


def get_source(profile_name: str) -> ModelSource | None:
    """Return the download source for a profile name, or None if unmapped."""
    return SOURCES.get(profile_name)
