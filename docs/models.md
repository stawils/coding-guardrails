# Supported Models

coding-guardrails works with any OpenAI-compatible backend. These profiles are
optimized for local inference with llama-server on consumer GPUs.

## Profiles

| Model | Quant | Size | VRAM | Context | Active | Arch | Speed |
|---|---|---|---|---|---|---|---|
| **Qwen3.5-9B** ⭐ | UD-Q4_K_XL (MTP) | 5.7 GB | 18.1 GB | **200K** | 9B | Dense | ~53 tok/s |
| **Gemma 4 26B A4B QAT** | UD-Q4_K_XL (QAT) | 14.25 GB | 19.8 GB | **200K** | 3.8B | MoE | ~40+ tok/s |
| **Ornith-1.0-9B** | Q8_0 | 9.5 GB | 18.0 GB | **200K** | 9B | Dense | ~50 tok/s |
| **LFM2.5-2.6B** | **BF16** | 5.4 GB | 9.0 GB | **128K** | 2.6B | Dense (hybrid) | ~fast (tiny) |

## LFM2.5-2.6B (BF16 — maximum precision, 128K context)

- Liquid AI LFM2.5, agentic post-trained. **2.69B params, native 131072 (128K) context**.
- **Official BF16 GGUF** (`LiquidAI/LFM2.5-2.6B-GGUF`) — no quantization, max precision.
- Hybrid architecture: 22 double-gated short-conv blocks + 8 GQA attention layers. Only 8 layers
  hold KV, so the 128K cache is tiny (~2 GB f16) — ~9 GB total VRAM, huge headroom on 24 GB.
- **Pure reasoning model** — always thinks before answering (`<think>` in the chat template).
- Tool use via llama.cpp `--jinja` (Pythonic `<|tool_call_start|>`…`<|tool_call_end|>` format).
- Card sampling: temp 0.1, top_k 50, rep_penalty 1.1 (baked into boot flags).
- ⚠️ **Card explicitly says NOT recommended for agentic coding / knowledge-heavy tasks** —
  **measured 2026-08-05 (Forge eval, proxy mode): 138/150 completion (92%), 100/140 correctness (71%)**
  vs Qwen3.5-9B's 140/150 (93%) / 132/140 (94%). Completion near parity and ~1.9× faster, but
  **-23pp correctness**, concentrated in data-heavy scenarios (`data_gap_recovery_extended`,
  `argument_transformation`, `inconsistent_api_recovery`, `grounded_synthesis` — all 0/10) and
  terminal-tool commitment (`tool_selection` 0/10). Full analysis:
  [reports/../plans/2026-08-05_lfm2.5-2.6b-worker-assessment.md](../plans/2026-08-05_lfm2.5-2.6b-worker-assessment.md).
- License: LFM Open License v1.0. No MTP.

## Qwen3.5-9B (Default) ⭐

- **Default model.** Reliable tool-use, no degenerate loops, consistent clean completion.
- 200K context fits Pi's system prompt + long tool-use sessions
- MTP draft tensors for ~1.5-2x speedup (~53 tok/s — fastest option)
- Only 18 GB VRAM — 6 GB headroom, leaves room for other GPU work
- Proven reliable tool-use through the proxy

## Gemma 4 26B A4B QAT (Alternative)

- Higher capability for complex tasks — 88.3% AIME 2026, 77.1% LiveCodeBench v6, 82.6% MMLU Pro
- MoE: 25.2B total / **3.8B active** — runs at ~4B-class speed while carrying 26B-class knowledge
- Native 256K context (run at 200K) with sliding-window attention — only 5 of 30 layers hold full KV, so the cache is tiny even at long context
- QAT-trained weights: ~72% smaller than BF16 with near-original quality — **but only via Unsloth UD-Q4_K_XL** (naive Q4_0 loses 15.4pp top-1)
- q8_0 KV cache (`-ctk q8_0 -ctv q8_0`) required for 200K to fit 24 GB (~20 GB used, 2.8 GB headroom)
- No MTP support (llama.cpp issue #22747). Sampling: temp=1.0, top_k=64, top_p=0.95
- ⚠️ Prone to degenerate thinking loops on finalization — work is correct on disk but the agent may not return cleanly

## Ornith-1.0-9B (Alternative)

- DeepReinforce RL post-train **on Qwen3.5-9B** — same hybrid linear/full attention
  architecture (`qwen3_5`), same vocab. Dense 9B, runs at Qwen3.5-class speed.
- **Reasoning model** — opens with `<think>…</think>`, returns `reasoning_content`, which
  `SafeLlamafileClient` already captures (no proxy changes needed).
- **Measured locally (2026-06-27, Forge 30-scenario eval, Q8_0, proxy mode):**
  140/150 completion (93%), 132/140 correctness (94%) — **parity with Qwen3.5-9B**,
  not a gain. The RL post-train does not improve agentic reliability here. Full
  report: [reports/2026-06-27_ornith-assessment.md](../reports/2026-06-27_ornith-assessment.md).
- Vendor benchmarks (69.4 SWE-bench Verified, 43.1 Terminal-Bench 2.1) are disputed
  and did not reproduce as a reliability advantage. MIT-licensed.
- Official GGUF only (`deepreinforce-ai/Ornith-1.0-9B-GGUF`) — **no Unsloth UD, no MTP tensors**, so
  do **not** pass `--spec-type draft-mtp`.
- Sampling (from card): temp=0.6, top_k=20, top_p=0.95.
- ⚠️ **Prefers prose answers over terminal tool calls.** On the Forge eval Ornith
  called `respond()` only 2× in 150 runs, answering in plain text instead. This is
  fatal for workflows that require an explicit final tool call to terminate (e.g.
  `tool_selection`, 0/5) but harmless otherwise. The `dup_write` rule catches the
  related output-file re-emission loop.

## Boot Commands

### Qwen3.5-9B (200K context, default) ⭐

```bash
llama-server \
  -m Qwen3.5-9B-UD-Q4_K_XL.gguf \
  --jinja --flash-attn auto \
  --port 8080 -c 200000 \
  --spec-type draft-mtp -np 1
```

### Gemma 4 26B A4B QAT (200K context, alternative)

```bash
llama-server \
  -m gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf \
  --jinja --flash-attn auto \
  --port 8080 -c 200000 \
  -ctk q8_0 -ctv q8_0 \
  --temp 1.0 --top-p 0.95 --top-k 64 -np 1
```

### Ornith-1.0-9B (200K context, alternative)

```bash
llama-server \
  -m ornith-1.0-9b-Q8_0.gguf \
  --jinja --flash-attn auto \
  --port 8080 -c 200000 \
  --temp 0.6 --top-p 0.95 --top-k 20 -np 1
```

### LFM2.5-2.6B (128K context, max precision)

```bash
llama-server \
  -m LFM2.5-2.6B-BF16.gguf \
  --jinja --flash-attn auto \
  --port 8080 -c 128000 \
  --temp 0.1 --top-k 50 --repeat-penalty 1.1 -np 1
```

> **Notes:** No `--spec-type draft-mtp` — the official Ornith GGUF has no MTP tensors.
> Reasoning model: enable a reasoning parser if driving llama-server directly; the
> cg proxy captures `reasoning_content` automatically via `SafeLlamafileClient`.

> **Notes:** No `--spec-type draft-mtp` — no MTP for Gemma 4 (llama.cpp #22747).
> **q8_0 KV cache is required** for 200K to fit 24 GB (~20 GB used, 2.8 GB headroom).
> Use the **Unsloth UD-Q4_K_XL QAT** GGUF only — naive Q4_0 conversion loses
> 15.4pp top-1 accuracy (QAT lattice needs Unsloth's dynamic method).

## Key flags

- `--jinja` — enables native function calling
- `--flash-attn auto` — FlashAttention when available
- `--spec-type draft-mtp` — multi-token prediction for ~1.5-2x faster inference (Qwen only)
- `-np 1` — single slot (maximizes GPU layers)
- `-ctk q8_0 -ctv q8_0` — q8_0 KV cache required for Gemma 4 26B at 200K context
