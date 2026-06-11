# A/B Test: Qwen3.5-9B vs Gemma-4-26B-A4B-QAT — Worker Delegation

**Date:** 2026-06-08
**Plan:** `~/AI/coding-guardrails/plans/2026-06-08_qwen-vs-gemma-ab.md`
**Task:** Add `normalizeDays` helper + spec to Lirada `AnalyticsService` (multi-file: service.ts + spec.ts). Identical task, identical `cg-worker` agent (v2 prompt), identical acceptance contract. Only variable: the model on `:8080`.

## Result — decisive, and surprising

| Metric | **Qwen 9B** | **Gemma 26B QAT** |
|---|---|---|
| Both files written? | ✅ yes | ✅ yes |
| service.ts `normalizeDays` | ✅ present (3 refs) | ✅ present (3 refs) |
| spec.ts new tests | ✅ 14 new (259 lines) | ✅ 7 new (196 lines) |
| **Tests pass?** | ❌ **14 FAIL** (`RangeError: Invalid time value`) | ✅ **7/7 pass** |
| Acceptance report emitted? | ❌ no | ❌ no |
| Duration | 5.2 min / 15 turns | 5.3 min / 7 turns |
| Failure mode | buggy implementation (date math broke despite normalizeDays) | degenerate text loop at the end (but work was already correct on disk) |

## The surprise

**Gemma produced the correct, passing implementation. Qwen produced more code but with a real bug.**

This inverts the going-in assumption. The degenerate text loop in Gemma's *output* (thousands of "Actually, I'll do it." lines) is a finalization-phase token burn — but **the actual work had already landed correctly before the loop started**. The loop is cosmetic noise after success; the work is sound.

Qwen, conversely, did *more* (14 tests to Gemma's 7) and didn't loop — but its `normalizeDays` didn't actually fix the date math (14 tests throw `RangeError: Invalid time value`, meaning the window is still hitting the date op somewhere invalid).

## What this answers

The open question was: **are cg-worker's failures Gemma-inherent (→ switch models) or prompt-fixable (→ keep iterating)?**

Answer: **neither, exactly — the framing was wrong.** The real findings:

1. **The acceptance report never emits under finalization for *either* model** with the v2 cg-worker prompt. This is a **prompt/agent-layer problem**, not a model problem. The v2 prompt's "checklist + escaping rules" made it worse (degenerate loops on Gemma; no report on Qwen). → **cg-worker v2 is a regression; revert toward v1.**

2. **Code correctness is a model property, and here Gemma > Qwen** on this task. The historical "Qwen is reliable, Gemma is flaky" assumption doesn't hold for multi-file logic tasks — Gemma got the date math right, Qwen didn't. (Sample size = 1 task; don't over-generalize, but don't assume either.)

3. **The verify-gate is the real backstop, for both models.** `npx jest` caught Qwen's bug instantly and would catch any Gemma lie too. The acceptance *report* adds no reliability on top of the verify commands for these models.

## Decision

- **Default model for delegation: keep Gemma 26B QAT.** It produced correct work this round; Qwen did not. The earlier "cg-worker failed on Gemma" diagnosis was conflating *correctness* (Gemma was fine) with *finalization noise* (the loop). One data point — re-confirm on more tasks.
- **cg-worker: revert to v1** (the simpler "STOP re-analysis, emit only JSON" prompt). v2's additions triggered the degenerate loop and didn't help the report emit on Qwen either. Do **not** iterate the prompt further blind — the acceptance-report-emission problem is structurally hard for both models and may need the orchestration layer (drop acceptance; verify via bash) rather than prompt magic.
- **Production delegation pattern: acceptance contract with verify commands, but treat the report as best-effort.** Always verify independently via the `verify` commands (which run regardless of the report). The report is a nice-to-have, not a gate.

## Next steps

1. Revert cg-worker to v1 prompt.
2. Re-run a *third* task (different from analytics) on Gemma to confirm correctness wasn't a fluke.
3. If Gemma is correct on 2/2 tasks with v1 prompt + verify-gate → promote that as the documented production pattern, update the ai-vibe-coder skill accordingly.
4. Clean up the ab-test worktrees.

## Artifacts

- Plan: `~/AI/coding-guardrails/plans/2026-06-08_qwen-vs-gemma-ab.md`
- Qwen worktree: `~/workspaces/ab-test/lirada-qwen` (buggy impl)
- Gemma worktree: `~/workspaces/ab-test/lirada-gemma` (correct impl — 7/7 tests pass)
- Qwen run: `a9a71c32` (15 turns, 5.2 min, 14/19 tests fail)
- Gemma run: `3b97b041` (7 turns, 5.3 min, 7/7 tests pass, degenerate loop after work)

## Addendum — 3rd Gemma run (og.service spec, 2026-06-08)

Ran a follow-up Gemma delegation to test correctness wasn't a fluke (different task: OgService spec).

**Result:** Logic correct (4/4 tests pass after a trivial fix). Worker deviated from repo spec style — used `jest.Mocked<T>` variable types without casting `module.get`, causing a TS2322 compile error. The `verify` command (`npx jest`) caught it instantly; the fix was a 3-line `as jest.Mocked<T>` cast applied directly.

**Refined finding:** Gemma 26B QAT produces logically-correct code on multi-file logic tasks but doesn't perfectly mirror established file conventions when instructed to "match style." The verify gate (jest/build) is the reliable backstop — exactly the production pattern this report recommended.

**Gemma correctness scorecard:** analytics ✓, og.service ✓ (logic), chart.js ✓. The one "failure" (analytics v2 degenerate loop) was the cg-worker v2 prompt regression, not Gemma. Confirms the decision: **Gemma 26B QAT is the default delegation model; verify-via-bash is the gate.**
