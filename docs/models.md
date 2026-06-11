# Supported Models

coding-guardrails works with any OpenAI-compatible backend. These profiles are
optimized for local inference with llama-server on consumer GPUs.

## Recommended for 24 GB GPUs (RTX 3090 / 3090 Ti)

| Model | Quant | Size | VRAM | Context | Active | Arch | Speed |
|---|---|---|---|---|---|---|---|
| **Gemma 4 26B A4B QAT** ⭐ | UD-Q4_K_XL (QAT) | 14.25 GB | 19.8 GB | **200K** | 3.8B | MoE | ~40+ tok/s |
| **Qwen3.5-9B** | UD-Q4_K_XL | 5.7 GB | 18.1 GB | **200K** | 9B | Dense | ~53 tok/s |
| **Qwen3.6-27B** | UD-Q4_K_XL (MTP) | 17.0 GB | 22.4 GB | 32K | 27B | Dense | ~28 tok/s |
| **Gemma 4 12B** 🆕 | UD-Q4_K_XL | 6.7 GB | 8.2 GB | **256K** | 12B | Dense | ~45 tok/s |
| Gemma 4 26B-A4B | UD-Q3_K_XL | 12.0 GB | 21.1 GB | **200K** | 4B | MoE | ~50 tok/s |

**Why Gemma 4 26B A4B QAT is recommended:** ⭐
- Highest capability on a single 24 GB GPU — 88.3% AIME 2026, 77.1% LiveCodeBench v6, 82.6% MMLU Pro
- MoE: 25.2B total / **3.8B active** — runs at ~4B-class speed while carrying 26B-class knowledge
- Native 256K context (run at 200K) with sliding-window attention — only 5 of 30 layers hold full KV, so the cache is tiny even at long context
- QAT-trained weights: ~72% smaller than BF16 with near-original quality — **but only via Unsloth UD-Q4_K_XL** (naive Q4_0 loses 15.4pp top-1)
- q8_0 KV cache (`-ctk q8_0 -ctv q8_0`) required for 200K to fit 24 GB (~20 GB used, 2.8 GB headroom)
- No MTP support (llama.cpp issue #22747). Sampling: temp=1.0, top_k=64, top_p=0.95

**Why Qwen3.5-9B as alternative:**
- 200K context fits Pi's system prompt + long tool-use sessions
- MTP draft tensors for ~1.5-2x speedup (~53 tok/s — fastest option)
- Only 18 GB VRAM — 6 GB headroom, leaves room for other GPU work
- Proven reliable tool-use through the proxy

**Why Gemma 4 12B as alternative:** 🆕
- **256K context** — largest context window of any supported model
- Encoder-free unified architecture — native text + image + audio multimodal
- Only 8 GB VRAM — **16 GB headroom** on 24 GB cards
- Hybrid sliding window (1024 tokens) + global attention — very efficient KV cache
- Native function calling support (built for agentic workflows)
- Apache 2.0 license
- No MTP support yet (llama.cpp issue #22747)

**Why Gemma 4 26B-A4B as alternative:**
- Multimodal (vision support)
- 200K context with Sliding Window Attention (efficient KV cache)
- Only 21 GB VRAM

## Boot Commands

### Qwen3.6-27B (32K context + MTP)

```bash
llama-server \
  -m Qwen3.6-27B-UD-Q4_K_XL.gguf \
  --jinja --flash-attn auto \
  --port 8080 -c 32768 \
  --spec-type draft-mtp -np 1
```

### Gemma 4 26B A4B QAT (200K context) ⭐

```bash
llama-server \
  -m gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf \
  --jinja --flash-attn auto \
  --port 8080 -c 200000 \
  -ctk q8_0 -ctv q8_0 \
  --temp 1.0 --top-p 0.95 --top-k 64 -np 1
```

> **Notes:** No `--spec-type draft-mtp` — no MTP for Gemma 4 (llama.cpp #22747).
> **q8_0 KV cache is required** for 200K to fit 24 GB (~20 GB used, 2.8 GB headroom).
> Use the **Unsloth UD-Q4_K_XL QAT** GGUF only — naive Q4_0 conversion loses
> 15.4pp top-1 accuracy (QAT lattice needs Unsloth's dynamic method).

### Qwen3.5-9B (200K context + MTP)

```bash
llama-server \
  -m Qwen3.5-9B-UD-Q4_K_XL.gguf \
  --jinja --flash-attn auto \
  --port 8080 -c 200000 \
  --spec-type draft-mtp -np 1
```

### Gemma 4 12B Unified (256K context) 🆕

```bash
llama-server \
  -m gemma-4-12b-it-UD-Q4_K_XL.gguf \
  --jinja --flash-attn auto \
  --port 8080 -c 256000 -np 1
```

> **Note:** No `--spec-type draft-mtp` — Gemma 4 MTP is not yet supported in
> llama.cpp (issue #22747). The assistant/draft model architecture
> (`Gemma4AssistantForCausalLM`) cannot be converted to GGUF yet.

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
- `--spec-type draft-mtp` — multi-token prediction for ~1.5-2x faster inference
- `-np 1` — single slot (maximizes GPU layers for dense models)
- `-c 32768` — 32K context window

## All Profiles

| Model | Quant | Size | VRAM | Context | Arch | Notes |
|---|---|---|---|---|---|---|
| **Gemma 4 26B A4B QAT** ⭐ | UD-Q4_K_XL (QAT) | 14.25 GB | 19.8 GB | 200K | MoE | Highest capability; q8_0 KV |
| **Qwen3.5-9B** | UD-Q4_K_XL (MTP) | 5.7 GB | 18.1 GB | 200K | Dense | Fastest, proven tool-use |
| **Qwen3.6-27B** | UD-Q4_K_XL (MTP) | 17.0 GB | 22.4 GB | 32K | Dense | Best dense quality |
| Qwen3.6-27B | UD-Q3_K_XL (MTP) | 14.5 GB | 22.5 GB | 82K | Dense | Max context |
| Qwen3.6-27B | Q4_K_M | 16.8 GB | 22.3 GB | 65K | Dense | Good balance |
| Qwen3.6-27B | Q4_K_S | 15.9 GB | 21.4 GB | 65K | Dense | Lightest 27B |
| **Gemma 4 12B** 🆕 | UD-Q4_K_XL | 6.7 GB | 8.2 GB | **256K** | Dense | 256K ctx, multimodal |
| Gemma 4 12B 🆕 | Q4_K_M | 6.7 GB | 8.2 GB | **256K** | Dense | Standard Q4 |
| Gemma 4 12B 🆕 | UD-Q3_K_XL | 5.3 GB | 6.8 GB | **256K** | Dense | Max headroom |
| Gemma 4 26B-A4B | UD-Q3_K_XL | 12.0 GB | 21.1 GB | 200K | MoE | Non-QAT, vision |
| Qwen3.6-35B-A3B | Q3_K_M | 15.9 GB | 22.0 GB | 16K | MoE | Legacy |
| Qwen3.6-35B-A3B | Q4_K_S | 19.7 GB | 24.5 GB* | 8K | MoE | ⚠️ >24 GB |

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
