# Ornith-1.0-9B — Local Capability Assessment

**Date:** 2026-06-27
**Model:** Ornith-1.0-9B (DeepReinforce) — Q8_0 quant, 200K context
**Backend:** coding-guardrails proxy → cg-owned llama.cpp (pin `5a6a0dd`)
**Status:** Complete. **Verdict: parity with the Qwen3.5-9B baseline; one specific, trace-verified weakness.**

Companion to [docs/models.md](../docs/models.md). Raw run artifacts: `eval/runs/2026-06-27_122017Z/`.

## TL;DR

Ornith-1.0-9B is an RL post-train **on Qwen3.5-9B**. On the Forge 30-scenario
eval (30 × 5 = 150 runs) it **matches** the Qwen3.5-9B default — 140/150
completion (93%), 132/140 correctness (94%) — and passed a real surgical coding
task first-try through cg. The RL post-train produces **no reliability gain**;
the vendor's disputed benchmarks did not reproduce as an agentic advantage.

Its single failure mode is **terminal-tool commitment**, not tool selection: it
computes correct answers but emits them as **plain prose instead of calling the
required terminal tool**, so it fails workflows that require an explicit final
tool call. It is a viable alternate backend, not an upgrade.

## Prong 1 — Real delegated task (qualitative)

**Task:** plumb `loop_detection.stagnation_threshold` through config in
`middleware.py` (a genuine, previously-unplumbed config key).

**Result: PASSED first try**, verified independently:
- 1-line edit in `from_config()` matching the sibling-threshold style exactly.
- One meaningful unit test; 488 tests green (was 487).
- No stray files, no rework. Sole blemish: a structurally-rejected acceptance
  report (known Pi-runtime formatting quirk, not a capability issue).

## Prong 2 — Forge benchmark (quantitative)

| Model | Completion | Correctness (of completed) | Wall |
|---|---|---|---|
| **Ornith-1.0-9B Q8_0** | **140/150 (93.3%)** | **132/140 (94.3%)** | 51.3 min |
| Qwen3.5-9B (baseline) | 140/150 (93.3%) | 140/150 (93.0%) | — |

cg's guardrails were **fully transparent** to Ornith: **0 blocks, 0 nudges, 0
loop detections** across all 150 runs — a clean capability read, no proxy artifact.

## Failure analysis — terminal-tool commitment (trace-verified)

The only failures are `tool_selection` + `tool_selection_stateful` (0/5 each).
The scenario name is a red herring — tracing one run disproves "tool selection"
as the cause:

| Iter | msgs | Ornith's action | Verdict |
|---|---|---|---|
| 1 | 2 | `lookup_user(name=Alice)` | ✅ correct tool, picked out of 8 |
| 2 | 4 | `get_permissions(user_id=U-1001)` | ✅ correct tool, ID extracted from free text |
| 3 | 6 | text: "Alice has… read, write, admin" | ❌ should call `respond()` |
| 4–8 | 8→16 | text × 6 more | ❌ answers in prose, never `respond()` |

So Ornith:
1. **Picked the right tools** among 8 distractors (lookup_user, get_permissions) — *not* a tool-picking failure.
2. **Formatted args perfectly**, extracting `U-1001` from prose — *not* a formatting failure.
3. **Computed the correct answer** (read/write/admin — matches the validator).
4. But emitted the answer as **plain text 6× in a row**, never calling `respond(answer=...)`.

Why fatal here only: most Forge scenarios terminate on the model's final text
`content`. But `tool_selection`'s workflow declares `terminal_tool="respond"`
and its validator reads `args.get("answer")` from the `respond()` call — so
prose-only cannot terminate → 8-iteration cap → `ToolCallError`
(`terminal_args: None` in the result row).

This is Ornith's general behavior, not a tool-selection quirk: across all 150
runs `respond()` was validated only **2 times** while the model passed through
prose **110 times**. Ornith strongly prefers to answer in prose rather than
commit to a terminal tool call. (Contrast: `argument_fidelity`, the *same*
2-hop extract-an-ID pattern with 3 tools, passes 5/5 in both variants — so the
chaining and extraction are fine; only the crowded-terminal requirement breaks.)

### Ornith-specific, not a cg/eval artifact (head-to-head vs Qwen3.5-9B)

The decisive comparison: the **Qwen3.5-9B base model** (which Ornith is an RL
post-train of) on the *same* `tool_selection` scenario, from prior baseline
runs (2026-05-31):

| Model | tool_selection (clean proxy runs) |
|---|---|
| **Qwen3.5-9B** (base) | **15/15 completed, 15/15 correct** (3 runs × 5) |
| **Ornith-1.0-9B** (RL post-train) | **0/5 completed, 0/5 correct** |

Qwen reliably calls `respond(answer=...)` to terminate; Ornith does not. So the
regression was introduced by the RL post-train — it sharpened something
(prose-answer quality) at the cost of terminal-tool commitment. (One early
Qwen run, 2026-05-31_174605Z, went 0/5 but with a different signature —
`MaxIterationsError` at 15 iters, an older harness config — and is excluded as a
warmup artifact; the three clean proxy-mode runs are the apples-to-apples read.)

**Fixability note:** this is a frozen-model behavior, not a cg defect — cg's
proxy deliberately does not force `respond()` (it's a measured escape-hatch for
local models) and only enforces tool-use for real coding agents (bash/read/edit/write),
which the eval scenarios don't use. So there is no cg-side "fix" that wouldn't
fight the design. The practical guidance: don't use Ornith for workflows that
require explicit terminal tool calls; use Qwen3.5-9B.

## Tooling changes shipped with this assessment

- `eval/scripts/run_forge_eval.py`: added a `--model` flag (was hardcoded to
  `Qwen3.5-9B-UD-Q4_K_XL`). Sets the run label + request model id; the booted
  model is whatever runs on :8080. **Local-only** — `eval/` is gitignored, so
  this is a dev convenience not shipped with the package. Uses default guardrail
  budgets (no `--config`) for apples-to-apples vs the 93% baseline.
- `middleware.py` + `tests/unit/test_middleware.py`: the Prong-1
  `stagnation_threshold` fix (a real bug, independently eligible for release).

## Recommendation

Keep **Qwen3.5-9B as the default** (MTP-accelerated ~53 tok/s, proven
tool-calling). Ornith is a fine **alternate** — its terminal-tool-in-prose habit
won't bite typical cg-worker delegation (small bash/read/edit/write namespace
where the agent's runner consumes prose fine), but Qwen's speed and terminal-tool
discipline tip the default. Do **not** adopt Ornith expecting the disputed
benchmark gains — they do not reproduce in agentic reliability.
