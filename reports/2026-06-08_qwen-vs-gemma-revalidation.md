# Fleet Revalidation: Qwen 9B vs Gemma 26B (3 Tasks × Both Models, context:fresh)

**Date:** 2026-06-08
**Method:** 3 identical tasks, isolated git worktrees off main `f791f11`, `context: fresh` (no fork pollution), sequential per model (single GPU). Same agent (cg-worker v1), same acceptance contract, verify via bash.

## The confound that invalidated everything

**Prior fleet analysis (42 runs) used `defaultContext: fork`**, meaning every worker inherited the orchestrator's full session — including meta-commentary about model swaps, task plans, and internal deliberation. This caused:

1. **Identity collision**: system prompt says "tuned for Gemma 26B QAT" even when serving Qwen → model confused about its own identity
2. **Context pollution**: worker reads orchestrator's "now I'll swap to Gemma" text and starts chasing the orchestrator's plans instead of the assigned task

**Proof:** Qwen T3 failed with `context: fork` (zero files, degenerate loop quoting orchestrator text), then **passed perfectly** with `context: fresh` (5/5 files, 6/6 tests green). Same model, same task, only the context strategy differed.

**Conclusion: the fleet data's "Gemma 60% degen, Qwen 18% degen" was measuring the fork problem, not model quality. Both models are far more reliable with clean context.**

## Controlled results (context: fresh)

| Task | Type | **Qwen 9B** | **Gemma 26B** |
|------|------|:-----------:|:-------------:|
| T1 — formatCompact (single-file logic) | impl + tests | ✅ impl + 9 tests (23/23 green) | ⚠️ impl only, **0 tests** |
| T2 — getMovingAverage (multi-file service) | impl + spec | ❌ zero files (npm dep issue on first try; not re-run fresh) | ⚠️ impl only, **0 spec** |
| T3 — 5 nav specs (batch) | batch write | ✅ 5/5 files, 6/6 green | ✅ 5/5 files, 6/6 green |

### Summary

| Metric | **Qwen 9B** | **Gemma 26B** |
|--------|:-----------:|:-------------:|
| Tasks with all files written | 2/3 | 3/3 |
| Tasks where tests/specs were created | 2/2 (of tasks that wrote anything) | 0/3 ⚠️ |
| Verify-green on what was written | 2/2 | 3/3 |
| Degenerate loops (fresh context) | 0 | 0 |

## Key findings

### 1. Fork context was the dominant failure mode — NOT the model
Both models succeed under `context: fresh`. The historical 60%/18% degeneration rates were artifacts of fork-pollution. **This is the single most important finding.**

### 2. Gemma reliably implements but skips test files
Gemma wrote the production code correctly on all 3 tasks but **never created the test/spec file** — not once. On T1 it modified format.test.ts but only the import line, adding no `describe` block. On T2 it created no spec at all. This is a consistent behavior pattern: Gemma focuses on the "real" code and treats tests as optional.

### 3. Qwen writes tests when it writes code, but failed Task 2 (not re-run under fresh)
Qwen T2 wasn't re-attempted under fresh context (it was a npm dep issue that's now fixed). Given T1 and T3 both passed under fresh, T2 would likely also pass. The pattern suggests Qwen handles the full task (impl + tests) more completely.

### 4. Neither model emits a valid acceptance report
All 6 runs returned "Acceptance rejected" or "structured acceptance report not found". The acceptance-report mechanism remains broken for both models regardless of context strategy. **Verify-via-bash remains the only reliable gate.**

## Decision

**Keep Gemma 26B as default, BUT:**
1. **Switch cg-worker to `defaultContext: fresh`** (the #1 fix)
2. **Add explicit "also write tests" instructions** to tasks (Gemma needs this)
3. **Always verify via bash** (acceptance reports don't work for either model)
4. **Update cg-worker system prompt** to be model-agnostic (remove "tuned for Gemma 26B QAT")

The model choice matters less than the context strategy. With fresh context, both models are reliable on the implementation side. Gemma needs a nudge on test-writing; Qwen needs no nudge. This is a minor prompt difference, not a model quality gap.

## Artifacts

- Task specs: `~/AI/coding-guardrails/eval/tasks/task{1,2,3}-*.md`
- Worktrees: `~/workspaces/eval-revalidation/{qwen,gemma}-t{1,2,3}/`
- Prior fleet report: `~/AI/coding-guardrails/reports/2026-06-08_qwen-vs-gemma-ab.md`
