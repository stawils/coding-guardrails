"""Model profiles — sampling defaults and hardware requirements.

Each profile maps a model identifier to its characteristics for the proxy.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelProfile:
    """A model's hardware and sampling characteristics."""

    name: str
    family: str  # e.g. "Qwen3.6"
    quant: str  # e.g. "Q3_K_M"
    file_size_gb: float
    vram_required_gb: float
    context_tokens: int
    architecture: str  # "dense" or "moe"
    active_params_b: float  # active parameters in billions
    swe_bench_verified: float | None  # SWE-bench Verified score (0-100)
    sampling: dict  # default sampling parameters
    boot_flags: list[str]  # extra llama-server flags


# fmt: off
PROFILES: dict[str, ModelProfile] = {
    "Qwen3.6-35B-A3B-UD-Q3_K_M": ModelProfile(
        name="Qwen3.6-35B-A3B-UD-Q3_K_M",
        family="Qwen3.6",
        quant="Q3_K_M",
        file_size_gb=15.9,
        vram_required_gb=22.0,
        context_tokens=16384,
        architecture="moe",
        active_params_b=3.5,
        swe_bench_verified=73.4,
        sampling={"temperature": 1.0, "top_k": 20, "top_p": 0.95},
        boot_flags=["--jinja", "--fit", "on", "--flash-attn", "auto",
                     "--spec-type", "draft-mtp", "-np", "1"],
    ),
    "Qwen3.6-35B-A3B-UD-Q4_K_S": ModelProfile(
        name="Qwen3.6-35B-A3B-UD-Q4_K_S",
        family="Qwen3.6",
        quant="Q4_K_S",
        file_size_gb=19.7,
        vram_required_gb=24.5,
        context_tokens=8192,
        architecture="moe",
        active_params_b=3.5,
        swe_bench_verified=73.4,
        sampling={"temperature": 1.0, "top_k": 20, "top_p": 0.95},
        boot_flags=["--jinja", "--fit", "on", "--flash-attn", "auto",
                     "--spec-type", "draft-mtp"],
    ),
    # ── Qwen3.6-27B (Gated DeltaNet hybrid, 64 layers: 48 DeltaNet + 16 GQA) ──
    # KV cache: 64 KB/token (only 16 attention layers; DeltaNet has fixed recurrent
    # state). MTP GGUFs include draft tensors for ~1.5-2x speedup via --spec-type
    # draft-mtp. Context budgets assume 24GB GPU with 1.5GB CUDA overhead:
    #   Q4_K_M  (16.8 GB): 65K ctx, 22.3/24 GB
    #   UD-Q4_K_XL MTP (17.5 GB): 49K ctx, 22.8/24 GB
    #   Q4_K_S  (15.9 GB): 65K ctx, 21.4/24 GB
    "Qwen3.6-27B-Q4_K_M": ModelProfile(
        name="Qwen3.6-27B-Q4_K_M",
        family="Qwen3.6",
        quant="Q4_K_M",
        file_size_gb=16.8,
        vram_required_gb=22.3,
        context_tokens=65536,
        architecture="dense",
        active_params_b=27.0,
        swe_bench_verified=77.2,
        sampling={"temperature": 1.0, "top_k": 20, "top_p": 0.95},
        boot_flags=["--jinja", "--flash-attn", "auto"],
    ),
    "Qwen3.6-27B-UD-Q3_K_XL": ModelProfile(
        name="Qwen3.6-27B-UD-Q3_K_XL",
        family="Qwen3.6",
        quant="UD-Q3_K_XL (MTP)",
        file_size_gb=14.5,
        vram_required_gb=22.5,
        context_tokens=81920,
        architecture="dense",
        active_params_b=27.0,
        swe_bench_verified=77.2,
        sampling={"temperature": 1.0, "top_k": 20, "top_p": 0.95},
        boot_flags=["--jinja", "--flash-attn", "auto",
                     "--spec-type", "draft-mtp", "-np", "1"],
    ),
    "Qwen3.6-27B-UD-Q4_K_XL": ModelProfile(
        name="Qwen3.6-27B-UD-Q4_K_XL",
        family="Qwen3.6",
        quant="UD-Q4_K_XL (MTP)",
        file_size_gb=17.0,
        vram_required_gb=22.4,
        context_tokens=32768,
        architecture="dense",
        active_params_b=27.0,
        swe_bench_verified=77.2,
        sampling={"temperature": 1.0, "top_k": 20, "top_p": 0.95},
        boot_flags=["--jinja", "--flash-attn", "auto",
                     "--spec-type", "draft-mtp", "-np", "1"],
    ),
    "Qwen3.6-27B-Q4_K_S": ModelProfile(
        name="Qwen3.6-27B-Q4_K_S",
        family="Qwen3.6",
        quant="Q4_K_S",
        file_size_gb=15.9,
        vram_required_gb=21.4,
        context_tokens=65536,
        architecture="dense",
        active_params_b=27.0,
        swe_bench_verified=77.2,
        sampling={"temperature": 1.0, "top_k": 20, "top_p": 0.95},
        boot_flags=["--jinja", "--flash-attn", "auto"],
    ),
    "Qwen3.5-9B-UD-Q4_K_XL": ModelProfile(
        name="Qwen3.5-9B-UD-Q4_K_XL",
        family="Qwen3.5",
        quant="UD-Q4_K_XL (MTP)",
        file_size_gb=5.7,
        vram_required_gb=18.1,
        context_tokens=200000,
        architecture="dense",
        active_params_b=9.0,
        swe_bench_verified=None,
        sampling={"temperature": 0.7, "top_k": 20, "top_p": 0.9},
        boot_flags=["--jinja", "--flash-attn", "auto",
                     "--spec-type", "draft-mtp", "-np", "1"],
    ),
    "gemma-4-26B-A4B-it-UD-Q3_K_XL": ModelProfile(
        name="gemma-4-26B-A4B-it-UD-Q3_K_XL",
        family="Gemma4",
        quant="UD-Q3_K_XL",
        file_size_gb=12.0,
        vram_required_gb=21.1,
        context_tokens=200000,
        architecture="moe",
        active_params_b=4.0,
        swe_bench_verified=None,
        sampling={"temperature": 1.0, "top_k": 64, "top_p": 0.95},
        boot_flags=["--jinja", "--flash-attn", "auto", "-np", "1"],
    ),
}
# fmt: on


def get_profile(model_name: str) -> ModelProfile | None:
    """Look up a model profile by name (exact or fuzzy)."""
    if model_name in PROFILES:
        return PROFILES[model_name]
    # Fuzzy match: check if the name is a substring
    for key, profile in PROFILES.items():
        if model_name in key or key in model_name:
            return profile
    return None


def list_profiles() -> list[ModelProfile]:
    """Return all profiles with recommended first."""
    # Recommended order: Qwen3.6-27B UD-Q4_K_XL MTP first (best coding + speed)
    order = {"Qwen3.6-27B-UD-Q4_K_XL": 0}
    return sorted(PROFILES.values(), key=lambda p: (order.get(p.name, 1), p.vram_required_gb))
