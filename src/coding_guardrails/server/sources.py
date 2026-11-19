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
    # The pi-adapted profile is a LoRA over the same UD-Q3_K_XL base — it has
    # no GGUF of its own, so it points at the identical file (one download
    # serves both profiles; the adapter is a separate 319 MB artifact, see
    # LORAS below).
    "Qwen3.8-27B-pi-adapted": ModelSource(
        repo_id="unsloth/Qwen3.8-27B-GGUF",
        filename="Qwen3.8-27B-UD-Q3_K_XL.gguf",
        license="Apache-2.0",
    ),
    # Thinking-off sibling: same base GGUF and same adapter, only the reasoning
    # flag differs (profiles.py pins --reasoning off there).
    "Qwen3.8-27B-pi-adapted-nothink": ModelSource(
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


@dataclass(frozen=True)
class LoraSource:
    """A LoRA adapter (GGUF) applied on top of a profile's target model.

    An adapter is trained against one specific base model and does nothing
    useful on another, so ``base_profile`` records the profile whose GGUF it
    must sit over; the launcher loads it with ``--lora <path>`` and the manager
    gate adds ``vram_gb`` on top of the base footprint.

    Unlike a drafter (which trades no quality for speed), an adapter changes
    behaviour: Qwen3.8-27B-pi-adapted was RL-trained inside the pi harness and
    is far terser and more tool-directed than the base. A missing file is
    therefore a loud warning, not a silent fallback — the backend still boots
    (an always-on tier must not die over an optional artifact), but it is
    serving the *base* behaviour and the log says so.
    """

    repo_id: str
    filename: str
    base_profile: str
    vram_gb: float  # adapter buffers + apply overhead, on top of the target
    license: str = ""
    notes: str = ""


# Keyed by adapter id (referenced from ModelProfile.lora).
LORAS: dict[str, LoraSource] = {
    # harness-hub, released 2026-11-06: Qwen3.8-27B RL-trained (AsyncGRPO, TRL)
    # inside unmodified pi — 750 scale800 + 300 TermGrade Harbor tasks, LoRA
    # r=32 alpha=64 all-linear, 160 steps, thinking OFF at training time.
    # Card (corrected, one attempt/task, pi harness, 100-step budget):
    #   SWE-bench Pro hard-51 39.2% -> 64.7% (+25.5 +/- 14.2, p=0.0023),
    #   Terminal-Bench 2.1 39.5% -> 55.8% (+16.3 +/- 9.1, p=0.0013).
    #   Turns per solved task 23 vs 59 (SWE-Pro); generated tokens 6.9k vs 9.6k.
    # MEASURED 2026-11-19 on this box (RTX 3090 Ti, UD-Q3_K_XL target): loads
    #   together with the DSpine drafter — adapter scale 1.0 AND draft-dflash
    #   n_max=4 in one process (acceptance 0.58-0.80, mean len 3.2-4.2).
    #   Greedy A/B scale 0 vs 1 diverges (adapter picks read_file/path where the
    #   base picks read/limit=100), so the adapter is genuinely applied under
    #   speculation. Cost: decode 70.6 -> 60-63 tok/s (~12%) at 128K ctx, which
    #   is the price of loading LoRA in the verify graph. End-to-end pi run
    #   fixed a failing 4-test repo in 67s.
    # Caveat: trained on BF16; the Q3_K_XL base is far from it, so absolute
    #   card numbers will not reproduce here — the plumbing is verified, the
    #   quality delta is not. logprobs are unavailable with the drafter on.
    "Qwen3.8-27B-pi-adapted": LoraSource(
        repo_id="harness-hub/Qwen3.8-27B-pi-adapted-GGUF",
        filename="Qwen3.8-27B-pi-lora-F16.gguf",
        base_profile="Qwen3.8-27B-UD-Q3_K_XL",
        vram_gb=0.3,
        license="Apache-2.0",
        # Thinking defaults ON (`--reasoning auto` -> the Qwen3.8 template
        # thinks), matching the base profile. The -nothink sibling pins
        # `--reasoning off`, the mode the adapter was trained and benchmarked
        # in: ~4.5x fewer generated tokens, but also ~10 points behind on
        # SWE-bench Pro hard-51 (64.7 vs 74.5, card, p=0.23).
    ),
}


def get_lora(lora_id: str) -> LoraSource | None:
    """Return the adapter source for an id, or None if unmapped."""
    return LORAS.get(lora_id)
