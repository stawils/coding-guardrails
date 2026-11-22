# Supported Models

coding-guardrails works with any OpenAI-compatible backend. These profiles are
optimized for local inference with llama-server on consumer GPUs.

## Profiles

| Model | Quant | Size | VRAM | Context | Active | Arch | Speed |
|---|---|---|---|---|---|---|---|
| **Qwen3.8-27B-pi-adapted** ⭐ default | UD-Q3_K_XL + **pi LoRA** (**DSpine** draft) | 13.76 GB | 18.3 GB | **160K** (q4_0 KV) | 27B | Dense (hybrid) | ~60-63 tok/s |
| Qwen3.8-27B (base, no adapter) | UD-Q3_K_XL (**DSpine** draft) | 13.44 GB | 17.8 GB | **120K** (q4_0 KV) | 27B | Dense (hybrid) | **~84 tok/s** |
| **Qwen3.6-27B** | UD-Q4_K_XL | 17.9 GB | 19.5 GB | **48K** (q8_0 KV) | 27B | Dense (hybrid) | ~20-30 tok/s |
| **Qwen3.5-9B** | UD-Q4_K_XL (MTP) | 5.7 GB | 18.1 GB | **200K** | 9B | Dense | ~53 tok/s |
| **Gemma 4 26B A4B QAT** | UD-Q4_K_XL (QAT) | 14.25 GB | 19.8 GB | **200K** | 3.8B | MoE | ~40+ tok/s |
| **Ornith-1.0-9B** | Q8_0 | 9.5 GB | 18.0 GB | **200K** | 9B | Dense | ~50 tok/s |
| **Flint** | UD-Q3_K_XL | 18.49 GB | 19.9 GB | **128K** (q8_0 KV) | 2.5B | MoE (39.8B total) | ~107 tok/s |
| **LFM2.5-2.6B** | **BF16** | 5.4 GB | 9.0 GB | **128K** | 2.6B | Dense (hybrid) | ~fast (tiny) |
| **MiniCPM5-2B** | Q4_K_M | 1.56 GB | ~3.0 GB | **131K** | 2.52B | Dense | ~182 tok/s |
| **Inkling 2 Mini** | IQ4_NL | 17.63 GB | 19.0 GB | **98K** (q4_0 KV) | 5.4B (26.4B total) | MoE (looped 3x) | ~40-53 tok/s |
| **LFM3-1.7B** | Q8_0 | 1.72 GB | ~4.5 GB | **256K** (q8_0 KV) | 1.73B | Dense (tri-hybrid) | ~150 tok/s |

## Forge eval results (150 runs, proxy mode)

All numbers are recorded runs on the same local GPU. The v0.7.4 respond() bug
(proxy converting declared respond() calls to text) had zeroed tool_selection
0/5 for every model — earlier "prose quirk" diagnoses for Qwen3.5-9B/Ornith/
Qwen3.6-27B were wrong; the v0.16.1 fix re-evaluated them (2026-08-08).
LFM2.5's tool_selection 0/10 is **genuine** (never calls respond()).

| Model | Completion | Accuracy | Run |
|---|---|---|---|
| **Inkling-2-Mini (IQ4_NL)** | **150/150 (100%)** | **150/150 (100%)** — best accuracy on record | 2026-11-22_155839Z |
| **Qwen3.8-27B (default)** | **150/150 (100%)** | **148/150 (99%)** | 2026-08-15_000804Z |
| **Qwen3.8-27B + DSpine** | **150/150 (100%)** | **149/150 (99%)** | 2026-11-09_081753Z |
| **Qwen3.5-9B** | **150/150 (100%)** | 138/150 (92%) | 2026-08-08_142633Z |
| **Ornith-1.0-9B** | **150/150 (100%)** | **143/150 (95%)** | 2026-08-08_144837Z |
| **Qwen3.6-27B** | **149/150 (99.3%)** | 141/150 (94%) | 2026-08-07_151927Z |
| **Flint** | 145/150 (96.7%) | 135/150 (90%) — fastest 40B-class run (24 min); tool_selection 40% (prose instead of respond()) | 2026-10-22_144116Z |
| **LFM2.5-2.6B** | 139/150 (92.7%) | 100/140 (71%) — card caveat confirmed | 2026-08-08_153811Z |
| **MiniCPM5-2B** | **150/150 (100%)** | **128/150 (85%)** — tool_selection 5/5 (fixes LFM2.5 0/5); data_gap_recovery_extended 100% | 2026-09-08_111510Z |
| **LFM3-1.7B** | 144/150 (96%) | 129/150 (86%) — ties MiniCPM5 on accuracy at 2/3 the params and 2x ctx, but ~2x wall time; 4 of 6 misses are one retry-context quoting quirk | 2026-10-28_143636Z |

### Wall time, and the trap in comparing it

All 150-run evals, proxy mode, same box. **These are not interchangeable baselines** —
see the note under the table.

| Model / config | Run | Wall | vs 08-15 base |
|---|---|---|---|
| Qwen3.8-27B (no drafter) | 2026-08-15_000804Z | 2276.9 s (37.9 min) | 1.00x |
| Qwen3.8-27B (no drafter, DSpine-era control) | 2026-11-09_084814Z | 3846.6 s (64.1 min) | **1.69x** |
| Qwen3.8-27B + DSpine, 120K q4_0 | 2026-11-09_101147Z | 2030.7 s (33.8 min) | 0.89x |
| Qwen3.8-27B + DSpine, 98K q8_0 | 2026-11-09_081753Z | 1781.2 s (29.7 min) | 0.78x |
| **Inkling 2 Mini IQ4_NL** | 2026-11-22_155839Z | **4172.8 s (69.5 min)** | 1.83x |
| Qwen3.6-27B UD-Q4_K_XL | 2026-08-07_151927Z | 6958.0 s (116.0 min) | — |
| Ornith-1.0-9B Q8_0 | 2026-08-08_144837Z | 2731.7 s (45.5 min) | — |
| Qwen3.5-9B UD-Q4_K_XL | 2026-08-08_142633Z | 1278.2 s (21.3 min) | — |
| Flint UD-Q3_K_XL | 2026-10-22_144116Z | 1438.0 s (24.0 min) | — |
| LFM3-1.7B Q8_0 | 2026-10-28_143636Z | 1444.2 s (24.1 min) | — |
| LFM2.5-2.6B BF16 | 2026-08-08_153811Z | 1755.7 s (29.3 min) | — |
| MiniCPM5-2B Q4_K_M | 2026-09-08_111510Z | 765.5 s (12.8 min) | — |

**The DSpine number is 1.89x only against its own same-day control** (3846.6 s ->
2030.7 s). Against the 08-15 un-drafted run it looks like a mere 1.12x — because the
un-drafted baseline itself got **1.69x slower** between 08-15 and 11-09. Iteration
counts are identical across those runs (basic_2step 2.0 everywhere, argument_fidelity
3.0, compaction_chain 10.0), so it is not extra retries: the per-turn cost changed
(prompt length / context budget / KV type all differed between the two eras).

That difference is **not attributable from the artifacts**: `run.json` records the
model and wall time but not `context_tokens`, `context_budget`, KV type or boot flags.
Any cross-date wall-time comparison in this repo is therefore unattributed — compare
only runs whose profile you can state, or record the config first.


Shared weakness across the smaller models: data-heavy recovery scenarios
(`data_gap_recovery_extended`, `argument_transformation`, `inconsistent_api_recovery`,
`grounded_synthesis` score ~0-60% accuracy; LFM2.5 0%). **Qwen3.8-27B closes that
gap — 100% on all four families** — which is why it is the new default worker
(128K ctx, MTP speed, vision-capable). Qwen3.5-9B remains the fast 200K fallback.


## MiniCPM5-2B (Q4_K_M — tiny-tier, 131K context, Apache-2.0)

- OpenBMB MiniCPM5-2B, released 2026-09-05, **Apache-2.0**. **2.52B dense params**
  (2,516,756,480), standard **LlamaForCausalLM** arch (42 layers, GQA 16 Q / 2 KV) —
  llama.cpp loads it directly with no custom kernels; boots on the 2026-07 build.
- **Official Q4_K_M GGUF** (`openbmb/MiniCPM5-2B-GGUF`), 1.56 GB on disk. 2 KV heads →
  the 131072 (128K) KV cache is tiny → **~3 GB total VRAM** on a 24 GB card.
- Native 131072 (128K) context, en/zh. **Reasoning model** — emits/receives
  `reasoning_content`; thinks at length before answering, so give `max_tokens` real
  headroom (a budget as small as 120 can be eaten entirely by thinking).
- Tool calling via `--jinja`: the model emits XML-style tool calls which the llama.cpp/server
  (July build) forms into clean OpenAI `tool_calls`; full tool loop (call → result → grounded
  answer) verified end-to-end.
- **Measured 2026-09-08** (direct + through the coding-guardrails proxy, Q4_K_M, RTX 3090 Ti):
  ~182 tok/s. Direct subset 30/30 (100%) completion, 25/30 (83%) accuracy. Proxy-mode subset
  15/15 (**100%**) — **tool_selection 5/5** (does call `respond()`; fixes LFM2.5's genuine 0/5)
  and data-heavy `data_gap_recovery_extended` 60% (LFM2.5 0%). **Full 150-run proxy eval:
  150/150 (100%) completion, 128/150 (85%) accuracy** — tool_selection 5/5 (+stateful 5/5), data_gap
  family 80-100%, all compaction_chain 100%; remaining small-model weak spots are
  `argument_transformation` 0-20% and `inconsistent_api_recovery` 40% (vs Qwen3.8-27B's 100%).
- Card sampling: temperature 1.0, top_p 0.95. License Apache-2.0. No MTP tensors in this file
  (the official DSpark speculative decoder is a separate repo/GGUF).
- This slots in as the fast/edge tier replacement for LFM2.5-2.6B (better tool-select
  discipline + data-heavy recovery at ~1/3 the size and Apache-2.0).

### MiniCPM5-2B boot command
```bash
$LLAMA \
  -m ~/.local/share/coding-guardrails/models/MiniCPM5-2B-Q4_K_M.gguf \
  -c 131072 -ngl 99 --host 127.0.0.1 --port 8080 \
  --jinja --flash-attn auto -np 1 -v
# ~182 tok/s, ~3 GB VRAM. Reasoning model: keep max_tokens generous.
```

## LFM3-1.7B (Q8_0 — 256K long-context tiny tier, LFM Open License v1.0)

- Liquid AI **LFM3-1.7B** (family released 2026-10-21+, first tri-hybrid generation).
  **1.73B dense params**, 128K vocab, 16 languages, native 262144 (256K) context.
- **Tri-hybrid architecture**: 24 layers = 13 gated short convolutions + 5 sliding-window
  attention + 6 global attention. Only the 6 global layers hold growing KV, so **the full
  256K context costs ~4.5 GB of VRAM total** (weights 1.72 GB + q8_0 KV).
- **Requires llama.cpp ≥ b11960** (`LLM_ARCH_LFM3`; cg pin `39febd2ad`). Older builds reject
  the GGUF. Tool calling works through `--jinja` (template ships in the GGUF); the model emits
  `<tool_call><function=…>` blocks that llama-server forms into clean OpenAI `tool_calls`.
- **Measured 2026-10-28** (proxy mode, 150 runs, RTX 3090 Ti): **144/150 completion (96%),
  129/150 accuracy (86%), 24 min wall, ~150 tok/s**. Head-to-head with MiniCPM5-2B:
  wins `inconsistent_api_recovery` **100/100** (vs 40/80) and `conditional_routing` 100/100
  (vs 60/100), `data_gap_*` 100%; loses `error_recovery` 60/60 (vs 100/80),
  `grounded_synthesis` 60/20 (vs 60/80), `sequential_reasoning_stateful` 60 (vs 100).
- ⚠️ **The 6 misses are mostly one quirk, not reasoning**: after a tool raises `TypeError`,
  string arguments come back double-quoted — `count: "0010"` re-sent as `"\"0010\""`.
  Direct calls render correctly, so it is retry-context drift. The other 2 are
  `_ToolCallExhausted` on `sequential_reasoning_stateful`.
- ⚠️ **Card caveat (validated)**: "not recommended for agentic coding or complex multi-step
  reasoning" — use it as a long-context tiny tier, not as a coding worker.
- Card sampling (agentic): temp 0.1, top_k 50, repeat_penalty 1.1 (baked into boot flags).
  For math/science the card prefers temp 1.0 / top_p 0.95 / top_k 20.
- The `LFM3-1.7B-QAD-Q4_0.gguf` (quantization-aware-distilled 4-bit, ≈1 GB) is downloaded but
  **not profiled** — untested on the harness.

### LFM3-1.7B boot command
```bash
$LLAMA \
  -m ~/.local/share/coding-guardrails/models/LFM3-1.7B-GGUF/LFM3-1.7B-Q8_0.gguf \
  -c 262144 -ngl 99 --host 127.0.0.1 --port 8080 \
  --jinja --flash-attn auto -ctk q8_0 -ctv q8_0 -np 1 \
  --temp 0.1 --top-k 50 --repeat-penalty 1.1 -v
# ~150 tok/s, ~4.5 GB VRAM at the full 256K ctx.
# Manager equivalent: cg server start -m LFM3-1.7B-Q8_0
```

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

## Qwen3.8-27B (120K context on 24 GB, DSpine draft)

- **Released Aug 2026**, Apache-2.0. Same `qwen3_5` hybrid arch as Qwen3.5-9B.
- **Speculative decoding: DSpine, not MTP** (since v0.26.0). [DSpine](https://arxiv.org/abs/2609.36173)
  (Tencent Hunyuan) is a 5-layer parallel drafter that reads hidden states from five
  target layers (1/16/31/46/61), drafts a 16-token block, and injects each position's
  predicted predecessor *code* into its successor at every layer. It rides llama.cpp's
  existing `draft-dflash` implementation, so cg only sets `-md` + `--spec-type draft-dflash`.
- **Measured 2026-11-09 on this box (RTX 3090 Ti, UD-Q3_K_XL target):**
  - short-prompt decode **46.3 → 94.5 tok/s** (n-max 4); production config at 120K ctx
    measures **84 tok/s** average across code/math/tool/prose/summarize.
  - **`--spec-draft-n-max 4`, not 15.** With a Q3 target, verification is relatively
    expensive: n=4 peaks, n=15 is the same speed but costs 1.7 GB of context. n=2 and
    n=7 are both slower. (On the porter's RTX 3090 + Q4_K_M target, n=15 wins — the
    optimum is a function of how cheaply *your* card verifies 16 tokens.)
  - full 150-run Forge eval: **3846.6s → 1781.2s wall (2.16x)** at n-max 4 with 98K
    q8_0 KV, and **2030.7s (1.89x)** on the shipped production config (120K q4_0 KV,
    systemd managed proxy). Completion 150/150 both ways, correctness **146/150 →
    149/150** (no regression). `tool_selection` and every `*_stateful` family stay 100%
    — the draft chain does not disturb tool calls.
  - 128K ctx still fits at n-max 3 (22.4 GB); at n-max 4 the 120K profile is 20.9 GB
    used on a 3.4 GB desktop baseline (~17.5 GB incremental).
- Dense 27B, 64 layers: 48 Gated DeltaNet (recurrent, ~no KV) + 16 full-attention
  (only these grow with ctx) → KV cache stays cheap at long context.
- Native **262,144 ctx** (1M via YaRN). On the 24 GB card with **UD-Q3_K_XL (13.44 GB)**
  we run **120000 (120K)** with **q4_0 KV**; the profile gate is 16.4 GB + 1.4 GB for
  the drafter. UD-Q4_K_XL (17.9 GB) would cap ctx at ~16-24K — chose context over quant.
- Native vision (images+video) in the source model; we run the text GGUF for agent work.
  The launcher auto-attaches a sibling `mmproj-*.gguf`, and the proxy captions inbound
  images (`[image: caption]` text blocks) so vision works through the text-only guardrail
  pipeline — verified 2026-08-15 through :8081 ("Red square; text: SQUARE = 42").
  Hour-scale video is vLLM/SGLang territory; unsupported on llama.cpp.
- Thinking on by default, per-request disable (`enable_thinking`), `reasoning_effort`,
  `preserve_thinking` — the proxy already handles these.
- Sampling (official `generation_config`): temp 1.0, top_k 20, top_p 0.95.
- Vendor benchmarks vs Qwen3.6-27B: SWE-bench Pro 61.7 (vs 53.5), Terminal-Bench 2.1
  73.0 (vs 63.4), DeepSWE 1.1 42.2 (vs 13.3), QwenSWEBench 79.0 (vs 49.3),
  LiveCodeBench v6 90.3, GPQA Diamond 89.2.

## Qwen3.8-27B-pi-adapted (DEFAULT — LoRA over Qwen3.8-27B, 160K context)

A **319 MB F16 LoRA** over the `Qwen3.8-27B-UD-Q3_K_XL` base, released 2026-11-06 by
harness-hub and RL-trained (TRL AsyncGRPO, LoRA r=32 alpha=64 on all linear layers,
160 steps) **inside unmodified pi** on Harbor SWE/terminal tasks. It is not a new base
model: it reuses the same 13.44 GB GGUF, so adopting it costs one small download.

- Card (pi harness, one attempt/task, 100 steps, thinking off): **SWE-bench Pro hard-51
  39.2% → 64.7%** (+25.5 ± 14.2, p=0.0023), **Terminal-Bench 2.1 39.5% → 55.8%**
  (+16.3 ± 9.1, p=0.0013), turns per solved task **23 vs 59**. An earlier "29.4% → 62.7%"
  was retracted by the authors (nine base-model rollouts died on a dead tunnel).
- Thinking: **ON by default** (`--reasoning auto`). The adapter was trained thinking-off
  and reasons in short visible bursts rather than hidden chains (`<think>` in <1 turn per
  1,000). The card's own thinking-on run scored **74.5%** on SWE-bench Pro vs 64.7% off, at
  ~4.5x the generated tokens — so off is the *cheap* mode, not the accurate one.
  `Qwen3.8-27B-pi-adapted-nothink` pins `--reasoning off`.
- Sampling: **temp 0.8, top_p 1.0** (the adapter's training sampling — not the base
  model's 1.0/0.95/20).
- Composes with the DSpine drafter. Verified 2026-11-19: adapter `scale 1.0` and
  `draft-dflash n_max=4` in one process, acceptance 0.58-0.80, mean len 3.2-4.2.
  Cost of the LoRA in the verify graph: **60-63 tok/s vs 70.6** (~12%).
- **Context 160000** (raised from 120000 once a local TTS server freed ~1 GB). Measured
  KV cost ~**26 KB per token**: 160K → 21,068 MiB used / ~1.4-1.5 GB free; 192K → 21,880
  MiB / 704 MiB free (boots, but too tight on a shared desktop). Gate 16.9 + 1.4 drafter.
- **`context_budget=120000`**, because the base model's tool-call cliff does not reproduce
  here: 4/4 tool calls at 16K/24K/32K/49K/65K/98K prompt tokens backend-direct (24/24), and
  19/20 through the proxy at a 120K budget (the single miss uncorrelated with length, 98K
  clean). Compaction therefore fires at ~90K instead of ~9K.
- Caveat: trained on BF16, served here at Q3_K_XL — treat the card's absolute numbers as
  directional. `logprobs` are unavailable while the drafter is on.

### Qwen3.8-27B-pi-adapted boot command

```bash
coding-guardrails server start -m Qwen3.8-27B-pi-adapted --wait
coding-guardrails serve --backend-url http://localhost:8080 \
  --model Qwen3.8-27B-pi-adapted --port 8081 --context-budget 120000
```

Raw equivalent (what the profile builds):

```bash
$LLAMA \
  -m ~/.local/share/coding-guardrails/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q3_K_XL.gguf \
  --lora ~/.local/share/coding-guardrails/models/Qwen3.8-27B-pi-adapted-GGUF/Qwen3.8-27B-pi-lora-F16.gguf \
  -md ~/models/dspine/Qwen3.8-27B-DSpine-b16-Q4_K_M.gguf \
  --spec-type draft-dflash --spec-draft-n-max 4 --spec-draft-ngl 99 \
  --spec-draft-type-k q8_0 --spec-draft-type-v q8_0 \
  -c 160000 -ngl 99 --host 127.0.0.1 --port 8080 --jinja --flash-attn auto \
  -ctk q4_0 -ctv q4_0 -np 1
```

Two operational notes: `--reasoning off` (not `--chat-template-kwargs
'{"enable_thinking": false}'`, which llama.cpp now deprecates) is how the `-nothink`
sibling is expressed; and a **missing** adapter file does not stop the boot — it logs
"serving the BASE model, WITHOUT the adapter", because a dead always-on tier is worse
than an unadapted one.

## Flint (MoE 39.8B-A2.5B — fast agentic-coding tier, 128K context, Apache-2.0)

- Recursive Flint, released 2026-10-21, **Apache-2.0**. **39.8B total / ~2.5B active**
  MoE: 49 layers (2 dense + 47 MoE, 256 routed experts, 6/token + shared expert).
  35 sliding-window layers (window 513) + 14 full-attention layers — only the 14
  full layers grow KV, so long context is cheap. Per-head QK RMSNorm + attention
  sinks; no positional encoding on the full-attention layers.
- **Needs llama.cpp ≥ `d77256999`** (`LLM_ARCH_FLINT`, upstream #30928). cg's pin
  was bumped from `5a6a0dd7e` (2026-06-26) to `d77256999` (2026-10-22) for this;
  the rebuild was smoke-tested against the Qwen3.8-27B default (MTP + mmproj OK).
- **Measured 2026-11-01 (Forge eval, proxy mode, 150 runs): 145/150 completion
  (96.7%), 135/150 accuracy (90%), 24 min wall** — the fastest 40B-class run on
  this harness (Qwen3.8-27B: 38 min). 100% on every `compaction_chain_*` and
  `data_gap_*` scenario, and on most `*_stateful` families.
- **Weak spots (documented, not proxy bugs):** `tool_selection` 40% — Flint often
  answers in prose instead of calling the declared `respond()` tool, so Forge's
  validator retries to exhaustion (same failure class as LFM2.5's tool_selection,
  but partial). `argument_transformation` (aggregation arithmetic) and
  `inconsistent_api_recovery` ~60%. One `error_recovery` run entered a repetition
  loop and came back as a backend 500.
- **VRAM:** 18.49 GB UD-Q3_K_XL + ~1.5 GB q8_0 KV/compute @128K ≈ **20.0 GB
  incremental** (22.3 GB used with a 2.5 GB desktop baseline, ~0.7 GB free).
  Profile gate = 19.9 GB. Default ctx 131072 of the native 262144.
- Sampling (Recursive): temp 1.0, top_p 0.95, top_k 20. No MTP tensors — do NOT
  pass `--spec-type draft-mtp`. Emits `reasoning_content` (captured by the proxy).
- Vendor benchmarks: Terminal-Bench 2.1 54.9, SWE-bench Pro v2 89.7,
  SWE-bench Verified 74.2, LiveCodeBench v6 80.3, IFBench 89.2, IFEval 95.1.

### Flint boot command

```bash
# Manager (preferred): cg server start -m flint-UD-Q3_K_XL
# Raw equivalent:
$LLAMA \
  -m ~/.local/share/coding-guardrails/models/flint-GGUF/flint-UD-Q3_K_XL.gguf \
  -c 131072 -ngl 99 --host 127.0.0.1 --port 8080 \
  --jinja --flash-attn auto -ctk q8_0 -ctv q8_0 -np 1
# ~107 tok/s decode, ~20 GB VRAM. Reasoning model — keep max_tokens generous
# (hard problems: >= 32K output budget per the model card).
```

## Inkling 2 Mini (IQ4_NL — looped MoE, 98K context, best Forge accuracy)

- Thinking Machines Lab, released 2026-11-20, **Apache-2.0**. **26.4B total /
  3.4B active** (5.4B effective). A 24-layer MoE whose **middle 12 layers run
  three times with shared weights** — 48 layer applications per token from 24
  stored layers (arch `inkling2_loop`). 8-of-128 experts per layer, hybrid
  sliding-window (1024) + full attention, **natively multimodal**, 1M ctx.
- **Measured 2026-11-22 (Forge eval, proxy mode, 150 runs): 150/150 completion,
  150/150 accuracy (100%) — the best accuracy on this harness, 69.5 min wall.**
  Every scenario 5/5 at 100%, including all `compaction_chain_*`, every
  `data_gap_*` and all `*_stateful` families. Wall time is ~2x Qwen3.8-27B+DSpine
  (33.8 min): the model reasons more per turn.
- **Long context does NOT collapse.** `probe_longctx.py` against :8080 direct:
  **10/10 tool-call successes at 2K/16K/32K/65K/98K**, `finish_reason=tool_calls`
  every time, no prose fallback. This is the opposite of the base Qwen3.8-27B
  cliff (~20-27K) — do not inherit that belief here.
- **PPL 6.7260 (`-fa off`) / 6.7618 (`-fa on`)** on cg's own docs. Upstream CUDA
  NaN bug #32710 does NOT reproduce on this 3090 Ti without its `-ncmoe 8` flag.
- **Cost is prefill, not decode:** 426 t/s prefill (19K prompt ≈ 45 s to first
  token, ~150 s at 98K) vs 39-53 t/s decode. A long agent transcript pays that
  every turn; llama-server's prompt cache only covers the shared prefix.
- **Thinking cannot be fully suppressed cheaply.** `reasoning_effort` in
  `chat_template_kwargs` works ("none" → 80 tokens + correct content on a
  factual ask), but a coding prompt at `max_tokens=700` spent all 700 on
  reasoning and returned EMPTY content. Budget output tokens generously. The
  template also preserves reasoning across turns by default
  (`--no-reasoning-preserve` disables it).
- **VRAM:** 17.63 GB IQ4_NL; measured **20,070 MiB @32K / 20,905 @98K /
  21,030 @163K** used (2.1 GB desktop) → 17.9-18.9 GB incremental. 16.33 GB of
  weights on CUDA0 + 0.47 GB host-mapped. Profile gate = 19.0 GB.
- **`-ub 128` is load-bearing.** At the default ubatch the compute buffer
  reached ~13.5 GB at 128K on other setups and silently pushed the model onto
  the CPU. With `-ub 128`: 838 MiB @32K, 2.4 GB @98K.
- **No drafter.** Inkling's DSpark draft needs upstream #32218 (open) and this
  pin rejects that GGUF (`markov_w1` shape / unknown head tensors).
- **GGUF source matters.** The pin (`4703b7d6`) predates upstream #32704, so
  **Unsloth's NVFP4 quants emit gibberish** (passes 2-3 run the experts
  unscaled; PPL ~168 vs ~8.5) while still loading and generating at full speed.
  Only `ggml-org/Inkling-2-Mini-GGUF` is safe on this pin.
- Sampling: temp 1.0, top_p 1.0, top_k 0, min_p 0.0 (Unsloth run guide).

### Inkling 2 Mini boot command

```bash
# Manager (preferred): cg server start -m Inkling-2-Mini-IQ4_NL
# Raw equivalent:
$LLAMA \
  -m ~/.local/share/coding-guardrails/models/Inkling-2-Mini-GGUF/Inkling-2-Mini-IQ4_NL.gguf \
  -c 98304 --host 127.0.0.1 --port 8080 \
  --jinja --flash-attn auto -ctk q4_0 -ctv q4_0 -ub 128 -np 1
# ~40-53 tok/s decode, ~19 GB incremental @98K.
```

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

### MiniCPM5-2B (131K context, Q4_K_M — tiny/edge tier)

```bash
llama-server \
  -m ~/.local/share/coding-guardrails/models/MiniCPM5-2B-Q4_K_M.gguf \
  --jinja --flash-attn auto \
  --port 8080 -c 131072 \
  --temp 1.0 --top-p 0.95 -np 1
# ~182 tok/s, ~3 GB VRAM; reasoning model — keep max_tokens generous.
# Manager equivalent: cg server start -m MiniCPM5-2B-Q4_K_M
```

### Qwen3.8-27B (120K context on 24 GB, DSpine draft)

```bash
llama-server \
  -m Qwen3.8-27B-UD-Q3_K_XL.gguf \
  -md Qwen3.8-27B-DSpine-b16-Q4_K_M.gguf \
  --jinja --flash-attn auto \
  --port 8080 -c 120000 \
  -ctk q4_0 -ctv q4_0 \
  --spec-type draft-dflash --spec-draft-n-max 4 --spec-draft-ngl 99 \
  --spec-draft-type-k q8_0 --spec-draft-type-v q8_0 -np 1
```

> **Notes:** the drafter needs the `dspine` branch of cg's pinned fork — stock
> llama.cpp refuses the GGUF rather than running it wrong. `--spec-draft-n-max 4`
> is the peak on this card; raise it only after sweeping (see docs/models.md).
> Manager equivalent: `cg server start -m Qwen3.8-27B-UD-Q3_K_XL`.

### Flint (128K context, q8_0 KV — fast MoE tier)

```bash
llama-server \
  -m flint-UD-Q3_K_XL.gguf \
  --jinja --flash-attn auto \
  --port 8080 -c 131072 \
  -ctk q8_0 -ctv q8_0 -np 1
```

> **Notes:** No `--spec-type draft-mtp` — the Flint GGUF has no MTP tensors.
> Requires llama.cpp ≥ `d77256999`; ~20 GB VRAM at 128K.

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
