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
        vram_required_gb=15.0,  # 16K ctx: 14.25 weights + ~0.5GB KV (MoE sliding-window) — fits 24GB shared GPU
        context_tokens=16384,
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
    # ── Qwen3.8-27B (Dense, 27B, 256K ctx, MTP, qwen3_5 arch) ──
    # Released Aug 2026. Same qwen3_5 hybrid arch as Qwen3.5-9B (MTP supported).
    # 64 layers: 48 Gated DeltaNet (recurrent, ~no KV) + 16 full-attention (KV
    # grows only there) — cheap KV for long ctx. Native 262144 ctx, 1M via YaRN.
    # Multimodal (vision+video) in the source model; text-only GGUF for agent use.
    # 27B BF16 = ~54 GB — impossible on 24 GB. UD-Q3_K_XL (13.44 GB) is the max
    # ctx play: 131072 (128K) with q4_0 KV + MTP draft fits the 24 GB card with
    # ~2 GB headroom (measured: 20.5 GB used / 2.1 free). q8_0 KV + MTP OOMs at
    # 128K (model loaded but decode graph exceeded free VRAM); q4_0 KV halves KV
    # (16 KiB/token) and frees room for the MTP draft. UD-Q4_K_XL (17.9 GB)
    # would cap ctx at ~16-24K — wasted for a 256K-native long-horizon model.
    # Sampling = official generation_config (temp 1.0, top_k 20, top_p 0.95).
    # Vendor: SWE-bench Pro 61.7, Terminal-Bench 73.0, LiveCodeBench 90.3,
    # GPQA 89.2 — sizable agentic gains over Qwen3.6-27B. Apache-2.0.
    "Qwen3.8-27B-UD-Q3_K_XL": ModelProfile(
        name="Qwen3.8-27B-UD-Q3_K_XL",
        family="Qwen3.8",
        quant="UD-Q3_K_XL",
        file_size_gb=13.44,
        vram_required_gb=19.0,  # incremental @128K q4_0 KV + MTP + mmproj (~18.9 GB
        # measured with vision projector auto-attached); gate compares FREE VRAM
        context_tokens=131072,
        architecture="dense",
        active_params_b=27.0,
        swe_bench_verified=61.7,
        sampling={"temperature": 1.0, "top_k": 20, "top_p": 0.95},
        boot_flags=["--jinja", "--flash-attn", "auto",
                     "-ctk", "q4_0", "-ctv", "q4_0",
                     "--spec-type", "draft-mtp", "-np", "1"],
    ),
    # ── LFM2.5-2.6B (Dense hybrid, 2.6B params, 128K ctx, BF16) ──
    # Liquid AI LFM2.5 — agentic post-trained (RL inside agentic harnesses).
    # Hybrid arch: 22 double-gated short-conv blocks + 8 GQA attention layers.
    # Official BF16 GGUF — maximum precision, zero quantization (~5.4 GB).
    # Context: GGUF train ctx is 128000 (llama-server warns/caps above that) —
    # 128000 is the max usable ctx; 8 GQA layers hold KV, so the 128K KV cache
    # is tiny (~2 GB at f16) — fits 24 GB easily.
    # Card: "pure reasoning model" (always thinks), tool-use via llama.cpp jinja.
    # Card explicitly says NOT recommended for agentic coding / knowledge-heavy
    # tasks — validated as a worker before trusting (see docs/models.md).
    # No MTP. Sampling from the model card: temp 0.1, top_k 50, rep_penalty 1.1.
    "LFM2.5-2.6B-BF16": ModelProfile(
        name="LFM2.5-2.6B-BF16",
        family="LFM2.5",
        quant="BF16",
        file_size_gb=5.4,
        vram_required_gb=9.0,  # 5.0 weights + ~2 GB KV @128K f16 + buffers
        context_tokens=128000,
        architecture="dense",
        active_params_b=2.6,
        swe_bench_verified=None,
        sampling={"temperature": 0.1, "top_k": 50, "repeat_penalty": 1.1},
        boot_flags=["--jinja", "--flash-attn", "auto", "-np", "1",
                     "--temp", "0.1", "--top-k", "50", "--repeat-penalty", "1.1"],
    ),
    # ── Qwen3.5-9B (Dense, 9B params, 200K ctx, MTP) ──
    # Fastest option with proven tool-use reliability. Boot: llama-server with
    # --spec-type draft-mtp for ~1.5-2x speedup.
    # MEASURED 2026-07-17: actual footprint at 200K ctx (with MTP) ≈ 13-15.3 GB
    # (hybrid 3:1 linear:full attention → only ~1/4 of layers hold growing KV, so
    # the 200K KV cache stays small). The old 18.1 GB gate was ~3 GB over-
    # conservative and deadlocked on the shared 24 GB card whenever the agent's
    # embedding model (embeddinggemma, ~0.9 GB) was also loaded — every turn needs
    # both. Gate lowered to 16.5 GB (measured + ~1.2 GB margin): coexists with the
    # embedding model, no context cut, no KV-quant change. Re-raise if real peak
    # usage during long agentic turns is observed above ~16 GB.
    # ── Qwen3.6-27B (Dense hybrid, 27B params, 262K native ctx, MTP file) ──
    # qwen3next arch (Gated DeltaNet SSM + Gated Attention hybrid): 64 layers in
    # 16×(3×DeltaNet→FFN → 1×Attention→FFN) blocks — only 16 layers hold growing
    # KV, so KV is 65,536 B/token f16 / 32,768 B/token q8_0.
    # Native ctx 262,144 (extensible to 1M); Unsloth UD-Q4_K_XL = 17.9 GB file.
    # MEASURED 2026-08-07 (RTX 3090 Ti, 24 GB, shared desktop):
    #   - Weights + compute peak at ~20.25 GB; q8_0 KV @ 48K = +1.54 GB → 22.35 GB
    #     total, 7/7 boots OK. 56K+ q8_0 KV needs a >1.79 GB single block → fails
    #     in the fragmented allocator state (driver VA fragmentation after many
    #     large load/unload cycles on this host). 32K f16 KV needs a 2.05 GB block
    #     (same failure); MTP spec decoding needs the same → MTP off at 48K.
    #   - `-ub 256` keeps the flash-attn compute buffer small (~150 MiB vs ~1.6 GB
    #     without flash-attn); default ubatch (2048) intermittently fails to find a
    #     contiguous block. q8_0 KV required — f16 KV doesn't fit 40K+.
    #   - Tool-calling verified through llama-server: clean read/respond calls.
    # Sampling from Forge registry (model card): temp=1.0, top_k=20, top_p=0.95.
    "Qwen3.6-27B-UD-Q4_K_XL": ModelProfile(
        name="Qwen3.6-27B-UD-Q4_K_XL",
        family="Qwen3.6",
        quant="UD-Q4_K_XL",
        file_size_gb=17.9,
        vram_required_gb=19.5,  # weights 17.9 + compute ~1 + q8_0 KV 1.54 @48K; fits 19-19.7 GB free on shared desktop
        context_tokens=49152,   # max reliable: 48K q8_0 KV (7/7 boots); 56K+ fails (allocator block limit)
        architecture="dense",
        active_params_b=27.0,
        swe_bench_verified=None,
        sampling={"temperature": 1.0, "top_k": 20, "top_p": 0.95},
        boot_flags=["--jinja", "--flash-attn", "auto", "-np", "1",
                     "-ub", "256", "-ctk", "q8_0", "-ctv", "q8_0"],
    ),
    "Qwen3.5-9B-UD-Q4_K_XL": ModelProfile(
        name="Qwen3.5-9B-UD-Q4_K_XL",
        family="Qwen3.5",
        quant="UD-Q4_K_XL (MTP)",
        file_size_gb=5.7,
        vram_required_gb=16.5,
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
