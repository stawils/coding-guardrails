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
    "LFM2.5-2.6B-BF16": ModelSource(
        repo_id="LiquidAI/LFM2.5-2.6B-GGUF",
        filename="LFM2.5-2.6B-BF16.gguf",
        license="LFM Open License v1.0",
    ),
    "MiniCPM5-2B-Q4_K_M": ModelSource(
        repo_id="openbmb/MiniCPM5-2B-GGUF",
        filename="MiniCPM5-2B-Q4_K_M.gguf",
        license="Apache-2.0",
    ),
    "Qwen3-8B-Q8_0": ModelSource(
        repo_id="Qwen/Qwen3-8B-GGUF",
        filename="Qwen3-8B-Q8_0.gguf",
        license="Apache-2.0",
    ),
    "Qwen3.8-27B-UD-Q3_K_XL": ModelSource(
        repo_id="unsloth/Qwen3.8-27B-GGUF",
        filename="Qwen3.8-27B-UD-Q3_K_XL.gguf",
        license="Apache-2.0",
    ),
    "flint-UD-Q3_K_XL": ModelSource(
        repo_id="unsloth/flint-GGUF",
        filename="flint-UD-Q3_K_XL.gguf",
        license="Apache-2.0",
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


@dataclass(frozen=True)
class DrafterSource:
    """A speculative-decoding draft model paired with a target profile.

    GGUFs only — llama.cpp needs ``-md <path>`` plus the matching ``--spec-type``.
    ``n_max`` is the number of tokens drafted per verify step and is hardware
    dependent: it belongs to the drafter entry only as a default that a profile
    may override via ``spec_n_max``.
    """

    repo_id: str
    filename: str
    spec_type: str
    n_max: int
    vram_gb: float  # extra VRAM the drafter + verification state costs
    license: str = ""


# Keyed by drafter id (referenced from ModelProfile.drafter).
DRAFTERS: dict[str, DrafterSource] = {
    # DSpine for Qwen3.8-27B: 5-layer parallel drafter, 16-token block, reads
    # hidden states from target layers [1,16,31,46,61]. Rides llama.cpp's
    # existing `draft-dflash` implementation type. Measured 2026-11-09 on an
    # RTX 3090 Ti with the UD-Q3_K_XL target: short-ctx decode 46.3 -> 94.5
    # tok/s; full 150-run Forge eval 3846.6s -> 2030.7s (1.89x on this profile's
    # 120K ctx; 2.16x at 98K q8_0 KV) with correctness 146/150 -> 149/150.
    # n_max 4 beat 15 (small target quant makes
    # verification relatively expensive) and saves 1.7 GB of context budget.
    "Qwen3.8-27B-DSpine-b16-Q4_K_M": DrafterSource(
        repo_id="aj9o9/Qwen3.8-27B-DSpine-b16-GGUF",
        filename="Qwen3.8-27B-DSpine-b16-Q4_K_M.gguf",
        spec_type="draft-dflash",
        n_max=4,
        vram_gb=1.4,
        license="Apache-2.0",
    ),
}


def get_drafter(drafter_id: str) -> DrafterSource | None:
    """Return the drafter source for an id, or None if unmapped."""
    return DRAFTERS.get(drafter_id)
