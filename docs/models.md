# Supported Models

coding-guardrails works with any OpenAI-compatible backend. These profiles are
optimized for local inference with llama-server on consumer GPUs.

## Recommended for 24 GB GPUs (RTX 3090 / 3090 Ti)

| Model | Quant | Size | VRAM | Context | Active | Arch | Speed |
|---|---|---|---|---|---|---|---|
| **Qwen3.5-9B** ⭐ | UD-Q4_K_XL | 5.7 GB | 18.1 GB | **200K** | 9B | Dense | ~53 tok/s |
| **Gemma 4 26B-A4B** | UD-Q3_K_XL | 12.0 GB | 21.1 GB | **200K** | 4B | MoE | ~50 tok/s |
| Qwen3.6-35B-A3B | Q3_K_M | 15.9 GB | 22.5 GB | 32K | 3.5B | MoE | ~22 tok/s |

**Why Qwen3.5-9B is recommended:**
- Dense 9B active params (2.5x more than MoE models)
- 200K context fits Pi's system prompt + long tool-use sessions
- Built-in MTP for ~2x faster inference
- Only 18 GB VRAM at 200K — 6 GB headroom

**Why Gemma 4 26B-A4B as alternative:**
- Multimodal (vision support)
- 200K context with Sliding Window Attention (efficient KV cache)
- Google's latest architecture (Gated DeltaNet)

## Boot Commands

### Qwen3.5-9B (200K context + MTP)

```bash
llama-server \
  -m Qwen3.5-9B-UD-Q4_K_XL.gguf \
  --jinja --flash-attn auto \
  --port 8080 -c 200000 \
  --spec-type draft-mtp -np 1
```

### Gemma 4 26B-A4B (200K context)

```bash
llama-server \
  -m gemma-4-26B-A4B-it-UD-Q3_K_XL.gguf \
  --jinja --flash-attn auto \
  --port 8080 -c 200000 -np 1
```

### Qwen3.6-35B-A3B (32K context + MTP, legacy)

```bash
llama-server \
  -m Qwen3.6-35B-A3B-UD-Q3_K_M.gguf \
  --jinja --flash-attn auto \
  --port 8080 -c 32768 \
  --spec-type draft-mtp -np 1
```

**Key flags:**
- `--jinja` — enables native function calling
- `--flash-attn auto` — FlashAttention when available
- `--spec-type draft-mtp` — multi-token prediction for ~2x faster inference
- `-np 1` — single slot (maximizes GPU layers for MoE models)
- `-c 200000` — 200K context window

## All Profiles

| Model | Quant | Size | VRAM | Context | Arch |
|---|---|---|---|---|---|
| Qwen3.5-9B | UD-Q4_K_XL (MTP) | 5.7 GB | 18.1 GB | 200K | Dense |
| Gemma 4 26B-A4B | UD-Q3_K_XL | 12.0 GB | 21.1 GB | 200K | MoE |
| Qwen3.6-35B-A3B | Q3_K_M | 15.9 GB | 22.0 GB | 32K | MoE |
| Qwen3.6-35B-A3B | Q4_K_S | 19.7 GB | 24.5 GB | 8K | MoE |
| Qwen3.6-27B | Q4_K_M | 17.6 GB | 22.0 GB | 4K | Dense |
| Qwen3.6-27B | Q4_K_S | 15.8 GB | 20.0 GB | 8K | Dense |

## Adding New Models

Add a profile in `src/coding_guardrails/models/profiles.py`:

```python
"my-model-name": ModelProfile(
    name="my-model-name",
    family="MyModel",
    quant="Q4_K_M",
    file_size_gb=14.0,
    vram_required_gb=18.0,
    context_tokens=8192,
    architecture="dense",
    active_params_b=14.0,
    swe_bench_verified=None,
    sampling={"temperature": 0.7, "top_p": 0.9},
    boot_flags=["--jinja", "--flash-attn", "auto", "-np", "1"],
),
```
