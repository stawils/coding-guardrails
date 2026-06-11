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
    # ── Gemma 4 12B Unified (Dense, encoder-free multimodal, 48 layers) ──
    # 11.95B params, 256K max context, hybrid sliding window (1024) + global attn.
    # Encoder-free: projects image/audio directly into LLM embedding space.
    # Very efficient KV cache due to sliding window layers.
    # No MTP support yet (llama.cpp issue #22747 — Gemma4AssistantForCausalLM
    # not convertible to GGUF yet). Use without --spec-type draft-mtp.
    # Sampling: Google recommends temp=1.0, top_k=64, top_p=0.95 for Gemma 4.
    #
    # VRAM budgets on 24 GB GPU (1.5 GB CUDA overhead):
    #   UD-Q4_K_XL (~6.7 GB): 15.8 GB KV headroom → 256K ctx fits easily
    #   Q4_K_M    (~6.7 GB): same
    #   UD-Q3_K_XL (~5.3 GB): 17.2 GB KV headroom → 256K ctx with max headroom
    "gemma-4-12b-it-UD-Q4_K_XL": ModelProfile(
        name="gemma-4-12b-it-UD-Q4_K_XL",
        family="Gemma4",
        quant="UD-Q4_K_XL",
        file_size_gb=6.7,
        vram_required_gb=8.2,
        context_tokens=256000,
        architecture="dense",
        active_params_b=12.0,
        swe_bench_verified=None,
        sampling={"temperature": 1.0, "top_k": 64, "top_p": 0.95},
        boot_flags=["--jinja", "--flash-attn", "auto", "-np", "1"],
    ),
    "gemma-4-12b-it-Q4_K_M": ModelProfile(
        name="gemma-4-12b-it-Q4_K_M",
        family="Gemma4",
        quant="Q4_K_M",
        file_size_gb=6.7,
        vram_required_gb=8.2,
        context_tokens=256000,
        architecture="dense",
        active_params_b=12.0,
        swe_bench_verified=None,
        sampling={"temperature": 1.0, "top_k": 64, "top_p": 0.95},
        boot_flags=["--jinja", "--flash-attn", "auto", "-np", "1"],
    ),
    "gemma-4-12b-it-UD-Q3_K_XL": ModelProfile(
        name="gemma-4-12b-it-UD-Q3_K_XL",
        family="Gemma4",
        quant="UD-Q3_K_XL",
        file_size_gb=5.3,
        vram_required_gb=6.8,
        context_tokens=256000,
        architecture="dense",
        active_params_b=12.0,
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
    order = {"gemma-4-26B-A4B-it-qat-UD-Q4_K_XL": 0}
    return sorted(PROFILES.values(), key=lambda p: (order.get(p.name, 1), p.vram_required_gb))
