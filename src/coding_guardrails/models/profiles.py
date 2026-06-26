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
    # ── Gemma 4 26B A4B QAT (MoE, 25.23B total / 3.8B active) ──
    # Quantization-Aware Training: 14.25 GB at Q4 with ~85.6% top-1 vs BF16
    # (vs 70.2% for naive Q4_0 — QAT lattice needs Unsloth UD-Q4_K_XL).
    # Native 256K context; runs 200K on 24 GB GPU with q8_0 KV cache.
    # Measured: 19.75 GB VRAM at 200K ctx (q8_0 KV), 2.8 GB headroom.
    # No MTP for Gemma 4 (llama.cpp #22747). Sliding-window attn keeps
    # KV cache tiny: only 5 global layers of 30 hold full sequence.
    "gemma-4-26B-A4B-it-qat-UD-Q4_K_XL": ModelProfile(
        name="gemma-4-26B-A4B-it-qat-UD-Q4_K_XL",
        family="Gemma4",
        quant="UD-Q4_K_XL (QAT)",
        file_size_gb=14.25,
        vram_required_gb=19.8,
        context_tokens=200000,
        architecture="moe",
        active_params_b=3.8,
        swe_bench_verified=None,
        sampling={"temperature": 1.0, "top_k": 64, "top_p": 0.95},
        boot_flags=["--jinja", "--flash-attn", "auto",
                     "-ctk", "q8_0", "-ctv", "q8_0", "-np", "1"],
    ),
    # ── Ornith-1.0-9B (Dense, qwen3_5 arch, 256K ctx, reasoning) ──
    # DeepReinforce RL post-train on Qwen3.5-9B (same hybrid linear/full
    # attention, same vocab). Reasoning model: <think>...</think> +
    # reasoning_content, which SafeLlamafileClient already captures.
    # Official GGUF only — NO Unsloth UD, NO MTP tensors, so do NOT pass
    # --spec-type draft-mtp. Sampling from the model card (agentic).
    # Benchmarks are disputed — this profile exists for local testing.
    "Ornith-1.0-9B-Q8_0": ModelProfile(
        name="Ornith-1.0-9B-Q8_0",
        family="Qwen3.5",
        quant="Q8_0",
        file_size_gb=9.5,
        vram_required_gb=18.0,
        context_tokens=262144,
        architecture="dense",
        active_params_b=9.0,
        swe_bench_verified=69.4,
        sampling={"temperature": 0.6, "top_k": 20, "top_p": 0.95},
        boot_flags=["--jinja", "--flash-attn", "auto", "-np", "1"],
    ),
    # ── Qwen3.5-9B (Dense, 9B params, 200K ctx, MTP) ──
    # Fastest option with proven tool-use reliability. 18 GB VRAM with MTP.
    # Boot: llama-server with --spec-type draft-mtp for ~1.5-2x speedup.
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
    """Return all profiles with Qwen (default) first."""
    order = {"Qwen3.5-9B-UD-Q4_K_XL": 0, "gemma-4-26B-A4B-it-qat-UD-Q4_K_XL": 1}
    return sorted(PROFILES.values(), key=lambda p: (order.get(p.name, 99), p.vram_required_gb))
