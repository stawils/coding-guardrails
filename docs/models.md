# Supported Models

coding-guardrails works with any OpenAI-compatible backend. These profiles are
optimized for local inference with llama-server on consumer GPUs.

## Profiles

| Model | Quant | Size | VRAM | Context | Active | Arch | Speed |
|---|---|---|---|---|---|---|---|
| **Qwen3.6-27B** | UD-Q4_K_XL | 17.9 GB | 19.5 GB | **48K** (q8_0 KV) | 27B | Dense (hybrid) | ~20-30 tok/s |
| **Qwen3.5-9B** ⭐ | UD-Q4_K_XL (MTP) | 5.7 GB | 18.1 GB | **200K** | 9B | Dense | ~53 tok/s |
| **Gemma 4 26B A4B QAT** | UD-Q4_K_XL (QAT) | 14.25 GB | 19.8 GB | **200K** | 3.8B | MoE | ~40+ tok/s |
| **Ornith-1.0-9B** | Q8_0 | 9.5 GB | 18.0 GB | **200K** | 9B | Dense | ~50 tok/s |
| **LFM2.5-2.6B** | **BF16** | 5.4 GB | 9.0 GB | **128K** | 2.6B | Dense (hybrid) | ~fast (tiny) |
| **Qwen3.8-27B** | UD-Q3_K_XL | 13.44 GB | 18.5 GB | **128K** | 27B | Dense (hybrid) | MTP ~fast |

## Forge eval results (v0.16.1 — respond() pass-through fix, 150 runs, proxy mode)

All numbers corrected under the v0.16.1 fix (2026-08-08 re-eval). The v0.7.4 bug
(proxy converting declared respond() calls to text) had zeroed tool_selection 0/5
for every model — earlier "prose quirk" diagnoses for Qwen3.5-9B/Ornith/Qwen3.6-27B
were wrong. LFM2.5's tool_selection 0/10 is **genuine** (never calls respond()).

| Model | Completion | Accuracy | Run |
|---|---|---|---|
| **Qwen3.5-9B** | **150/150 (100%)** | 138/150 (92%) | 2026-08-08_142633Z |
| **Ornith-1.0-9B** | **150/150 (100%)** | **143/150 (95%)** — top | 2026-08-08_144837Z |
| **Qwen3.6-27B** | **149/150 (99.3%)** | 141/150 (94%) | 2026-08-07_151927Z |
| **LFM2.5-2.6B** | 139/150 (92.7%) | 100/140 (71%) — card caveat confirmed | 2026-08-08_153811Z |

Shared weakness across all local models: data-heavy recovery scenarios
(`data_gap_recovery_extended`, `argument_transformation`, `inconsistent_api_recovery`,
`grounded_synthesis` all score ~0-60% accuracy; LFM2.5 0%). Qwen3.5-9B stays the
default (MTP speed, 200K ctx, 100% completion); Ornith leads accuracy.


## LFM2.5-2.6B (BF16 — maximum precision, 128K context)

- Liquid AI LFM2.5, agentic post-trained. **2.69B params, native 131072 (128K) context**.
- **Official BF16 GGUF** (`LiquidAI/LFM2.5-2.6B-GGUF`) — no quantization, max precision.
- Hybrid architecture: 22 double-gated short-conv blocks + 8 GQA attention layers. Only 8 layers
  hold KV, so the 128K cache is tiny (~2 GB f16) — ~9 GB total VRAM, huge headroom on 24 GB.
- **Pure reasoning model** — always thinks before answering (`<think>` in the chat template).
- Tool use via llama.cpp `--jinja` (Pythonic `<|tool_call_start|>`…`<|tool_call_end|>` format).
- Card sampling: temp 0.1, top_k 50, rep_penalty 1.1 (baked into boot flags).
- ⚠️ **Card explicitly says NOT recommended for agentic coding / knowledge-heavy tasks** —
  **re-eval 2026-08-08 (v0.16.1): 139/150 completion (92.7%), 100/140 correctness (71%)** —
  unchanged from the buggy 2026-08-05 run (138/150): **tool_selection 0/10 is GENUINE** (the
  model never calls respond(); 0 pass-throughs/0 conversions in the proxy log) and the
  data-heavy failures stand (all 0%: `data_gap_recovery_extended`, `argument_transformation`,
  `inconsistent_api_recovery`, `grounded_synthesis`). The card caveat is confirmed: fine for
  bounded tasks, not for terminal-tool or data-heavy agentic work. Full analysis:
  [reports/../plans/2026-08-05_lfm2.5-2.6b-worker-assessment.md](../plans/2026-08-05_lfm2.5-2.6b-worker-assessment.md).
- License: LFM Open License v1.0. No MTP.

## Qwen3.6-27B (Newest, highest capability)

- **27B dense hybrid** (`qwen3next` arch): 64 layers in 16×(3×Gated DeltaNet→FFN → 1×Gated Attention→FFN) blocks.
  Only **16 of 64 layers hold growing KV** (65,536 B/token f16, 32,768 B/token q8_0) — the DeltaNet layers
  keep fixed-size recurrent state.
- **Native 262K context** (extensible to 1M), agentic-coding focused release, MTP-trained (tensors present).
- **Unsloth UD-Q4_K_XL** GGUF, 17.9 GB — Apache-2.0.
- **MEASURED 2026-08-07 (RTX 3090 Ti 24 GB, shared desktop):** weights+compute peak ~20.25 GB; q8_0 KV
  @48K adds 1.54 GB → **22.35 GB total**, 7/7 boots OK. **48K is the reliable max**: 56K+ needs a single
  KV block >1.79 GB, which fails in the driver's fragmented allocator state (VA fragmentation after many
  large load/unload cycles). 32K f16 KV (2.05 GB block) and MTP spec decoding hit the same wall.
- **Boot requirements baked into the profile:** `-ctk q8_0 -ctv q8_0` (q8_0 KV — f16 KV doesn't fit 40K+)
  and `-ub 256` (keeps the flash-attn compute buffer ~150 MiB; default ubatch intermittently can't find a
  contiguous block). **No `--spec-type draft-mtp`** at 48K — the 2 GB spec/KV blocks don't fit.
- **Verified:** generation + clean `read`/`respond` tool calls through the proxy.
- **Forge 30-scenario eval (2026-08-07, proxy mode, 150 runs):** **149/150 completion (99.3%),
  141/150 accuracy (94%)** — measured under v0.16.1 (respond() pass-through fix). The morning
  run (138/150, 92%) was inflated by the v0.7.4 respond()-conversion bug, which cost every
  model 10 tool_selection points. The only completion loss: one relevance_detection timeout.
  Accuracy weakness: `data_gap_recovery_extended` (+stateful) 40%/0% — detailed reports
  missing required recovered fields (same scenario class that sank LFM2.5 0/10). Everything
  else ≥80% acc, 25 of 30 scenarios at 100%.
- **v0.16.1 fix (2026-08-07): the proxy ate respond() calls.** Since v0.7.4 (2026-06-01) the
  proxy converted respond()→text even when the agent declared a respond tool. Tool_selection
  went from 5/5 (pre-v0.7.4) to 0/5 for every model — terminal-tool workflows looked like
  prose failures. Fixed: declared respond tools now pass the call through (plus terminal
  enforcement + terminal-aware retry nudge). Verified: tool_selection + stateful 0/5 → **5/5
  each; full eval 138/150 → 149/150** (99.3% completion).
- Sampling (model card, via Forge registry): temp=1.0, top_k=20, top_p=0.95.
- ⚠️ **GPU allocator state matters:** if `cg server start` fails with `cudaMalloc failed: out of memory`
  on small buffers, the driver's memory is fragmented (many prior large load/unload cycles). A reboot
  clears it; 56K-64K q8_0 KV becomes loadable again in a fresh state.

## Qwen3.8-27B (128K context on 24 GB, MTP)

- **Released Aug 2026**, Apache-2.0. Same `qwen3_5` hybrid arch as Qwen3.5-9B — MTP
  speculative decoding works (`--spec-type draft-mtp`).
- **Measured 2026-08-15 (Forge eval, proxy mode, 150 runs): 150/150 completion (100%),
  148/150 correctness (99%) — best result ever on this harness** (vs Qwen3.5-9B
  150/150/100% completion, 92% correctness; Qwen3.6-27B 99.3%/94%). Scored 100% on
  every family smaller models fail: `tool_selection`, `data_gap_recovery_extended`,
  `argument_transformation`, `inconsistent_api_recovery`, `grounded_synthesis`. Only 2
  misses total (compaction_chain_p1/p2, 4/5 each).
  Full analysis: [../plans/2026-08-05_qwen3.8-27b-research.md](../plans/2026-08-05_qwen3.8-27b-research.md).
- Dense 27B, 64 layers: 48 Gated DeltaNet (recurrent, ~no KV) + 16 full-attention
  (only these grow with ctx) → KV cache stays cheap at long context.
- Native **262,144 ctx** (1M via YaRN). On the 24 GB card with **UD-Q3_K_XL (13.44 GB)**
  we run **131072 (128K)** with **q4_0 KV + MTP draft** (~20.5 GB VRAM, ~2 GB headroom;
  measured 2026-08-05). q8_0 KV + MTP OOMs at 128K; q4_0 KV halves KV and fits the draft.
  UD-Q4_K_XL (17.9 GB) would cap ctx at ~16-24K — chose context over quant quality.
- Native vision (images+video) in the source model; we run the text GGUF for agent work.
- Thinking on by default, per-request disable (`enable_thinking`), `reasoning_effort`,
  `preserve_thinking` — the proxy already handles these.
- Sampling (official `generation_config`): temp 1.0, top_k 20, top_p 0.95.
- Vendor benchmarks vs Qwen3.6-27B: SWE-bench Pro 61.7 (vs 53.5), Terminal-Bench 2.1
  73.0 (vs 63.4), DeepSWE 1.1 42.2 (vs 13.3), QwenSWEBench 79.0 (vs 49.3),
  LiveCodeBench v6 90.3, GPQA Diamond 89.2.

## Qwen3.5-9B (Default) ⭐

- **Default model.** Reliable tool-use, no degenerate loops, consistent clean completion.
- 200K context fits Pi's system prompt + long tool-use sessions
- MTP draft tensors for ~1.5-2x speedup (~53 tok/s — fastest option)
- Only 18 GB VRAM — 6 GB headroom, leaves room for other GPU work
- **Forge eval (2026-08-08 re-eval, v0.16.1): 150/150 completion (100%), 138/150 accuracy (92%)** —
  the old "93% (140/150)" was the v0.7.4 respond() bug (tool_selection 0/5); pre-bug May 31 runs
  also hit 150/150. Weak spots mirror all local models: `data_gap_recovery_extended` 40%,
  `argument_transformation` 60%.

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
- **Re-eval 2026-08-08 (v0.16.1, 150 runs): 150/150 completion (100%), 143/150 accuracy (95%) —
  top accuracy of the lineup.** The earlier "93% (140/150)" was the v0.7.4 respond() bug
  (tool_selection 0/5); the "prefers prose" diagnosis was wrong. Full report:
  [reports/2026-06-27_ornith-assessment.md](../reports/2026-06-27_ornith-assessment.md).
- Vendor benchmarks (69.4 SWE-bench Verified, 43.1 Terminal-Bench 2.1) are disputed
  and did not reproduce as a reliability advantage. MIT-licensed.
- Official GGUF only (`deepreinforce-ai/Ornith-1.0-9B-GGUF`) — **no Unsloth UD, no MTP tensors**, so
  do **not** pass `--spec-type draft-mtp`.
- Sampling (from card): temp=0.6, top_k=20, top_p=0.95.
- ⚠️ **Terminal-tool note (retired v0.16.1):** the earlier "prefers prose over terminal
  respond()" diagnosis was **wrong** — the 2026-06-27 eval ran under the v0.7.4 respond()-
  conversion bug, so respond() calls were eaten by the proxy and counted as prose. The
  re-eval (2026-08-08) shows **150/150 completion** — the prose behavior was entirely the bug.

## Boot Commands

### Qwen3.6-27B (48K context, q8_0 KV — max reliable)

```bash
llama-server \
  -m Qwen3.6-27B-UD-Q4_K_XL.gguf \
  --jinja --flash-attn auto \
  --port 8080 -c 49152 -np 1 \
  -ub 256 -ctk q8_0 -ctv q8_0
```

or via cg (profile-driven, same flags):

```bash
cg server start -m Qwen3.6-27B-UD-Q4_K_XL
```

> **Notes:** No `--spec-type draft-mtp` — the MTP spec buffers need a contiguous 2 GB block that
> doesn't fit alongside 48K of KV on the fragmented 24 GB allocator. `-ub 256` and q8_0 KV are
> required (see profile comment). Max measured: 48K. After a GPU driver reset/reboot, 56K-64K may
> load.

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

### Qwen3.8-27B (128K context on 24 GB, MTP)

```bash
llama-server \
  -m Qwen3.8-27B-UD-Q3_K_XL.gguf \
  --jinja --flash-attn auto \
  --port 8080 -c 131072 \
  -ctk q4_0 -ctv q4_0 \
  --spec-type draft-mtp -np 1
```

> **Notes:** q4_0 KV is required for 128K + MTP on 24 GB (q8_0 KV + MTP OOMs).

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
