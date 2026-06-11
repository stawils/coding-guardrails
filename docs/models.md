# Supported Models

coding-guardrails works with any OpenAI-compatible backend. These profiles are
optimized for local inference with llama-server on consumer GPUs.

## Profiles

| Model | Quant | Size | VRAM | Context | Active | Arch | Speed |
|---|---|---|---|---|---|---|---|
| **Gemma 4 26B A4B QAT** ⭐ | UD-Q4_K_XL (QAT) | 14.25 GB | 19.8 GB | **200K** | 3.8B | MoE | ~40+ tok/s |
| **Qwen3.5-9B** | UD-Q4_K_XL (MTP) | 5.7 GB | 18.1 GB | **200K** | 9B | Dense | ~53 tok/s |

## Gemma 4 26B A4B QAT (Default) ⭐

- Highest capability on a single 24 GB GPU — 88.3% AIME 2026, 77.1% LiveCodeBench v6, 82.6% MMLU Pro
- MoE: 25.2B total / **3.8B active** — runs at ~4B-class speed while carrying 26B-class knowledge
- Native 256K context (run at 200K) with sliding-window attention — only 5 of 30 layers hold full KV, so the cache is tiny even at long context
- QAT-trained weights: ~72% smaller than BF16 with near-original quality — **but only via Unsloth UD-Q4_K_XL** (naive Q4_0 loses 15.4pp top-1)
- q8_0 KV cache (`-ctk q8_0 -ctv q8_0`) required for 200K to fit 24 GB (~20 GB used, 2.8 GB headroom)
- No MTP support (llama.cpp issue #22747). Sampling: temp=1.0, top_k=64, top_p=0.95

## Qwen3.5-9B (Alternative)

- 200K context fits Pi's system prompt + long tool-use sessions
- MTP draft tensors for ~1.5-2x speedup (~53 tok/s — fastest option)
- Only 18 GB VRAM — 6 GB headroom, leaves room for other GPU work
- Proven reliable tool-use through the proxy

## Boot Commands

### Gemma 4 26B A4B QAT (200K context, default) ⭐

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

### Qwen3.5-9B (200K context, alternative)

```bash
llama-server \
  -m Qwen3.5-9B-UD-Q4_K_XL.gguf \
  --jinja --flash-attn auto \
  --port 8080 -c 200000 \
  --spec-type draft-mtp -np 1
```

## Key flags

- `--jinja` — enables native function calling
- `--flash-attn auto` — FlashAttention when available
- `--spec-type draft-mtp` — multi-token prediction for ~1.5-2x faster inference (Qwen only)
- `-np 1` — single slot (maximizes GPU layers)
- `-ctk q8_0 -ctv q8_0` — q8_0 KV cache required for Gemma 4 26B at 200K context
