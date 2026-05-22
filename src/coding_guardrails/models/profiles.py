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
    "Qwen_Qwen3.6-27B-Q4_K_M": ModelProfile(
        name="Qwen_Qwen3.6-27B-Q4_K_M",
        family="Qwen3.6",
        quant="Q4_K_M",
        file_size_gb=17.6,
        vram_required_gb=22.0,
        context_tokens=4096,
        architecture="dense",
        active_params_b=27.0,
        swe_bench_verified=77.2,
        sampling={"temperature": 1.0, "top_k": 20, "top_p": 0.95},
        boot_flags=["--jinja", "--fit", "on", "--flash-attn", "auto",
                     "--spec-type", "draft-mtp"],
    ),
    "Qwen_Qwen3.6-27B-Q4_K_S": ModelProfile(
        name="Qwen_Qwen3.6-27B-Q4_K_S",
        family="Qwen3.6",
        quant="Q4_K_S",
        file_size_gb=15.8,
        vram_required_gb=20.0,
        context_tokens=8192,
        architecture="dense",
        active_params_b=27.0,
        swe_bench_verified=77.2,
        sampling={"temperature": 1.0, "top_k": 20, "top_p": 0.95},
        boot_flags=["--jinja", "--fit", "on", "--flash-attn", "auto",
                     "--spec-type", "draft-mtp"],
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
    # Recommended order: Q3_K_M MoE first (best for 24GB cards)
    order = {"Qwen3.6-35B-A3B-UD-Q3_K_M": 0}
    return sorted(PROFILES.values(), key=lambda p: (order.get(p.name, 1), p.vram_required_gb))
