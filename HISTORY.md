# Session History

## 2026-06-27 — Ornith-1.0-9B assessment
- Booted Ornith-1.0-9B Q8_0 (200K ctx) on cg's llama.cpp :8080 + guardrails proxy :8081; smoke-tested tool calls through both layers (green)
- Prong 1 (real task): delegated the open `stagnation_threshold` config-plumbing bug to Ornith via cg-worker — PASSED first try (1-line edit + 1 test, 488 green, no rework). Independently verified
- Prong 2 (benchmark): patched `run_forge_eval.py` to take `--model`, ran Forge 30-scenario eval (150 runs, proxy mode). Result: **140/150 (93%), 94% correctness = parity with Qwen3.5-9B**. cg fully transparent (0 blocks/nudges)
- **Root-cause traced:** Ornith's only failures (tool_selection ×2, 0/5 each) are NOT tool-picking or arg-formatting — it picks tools and extracts args perfectly. It **answers in prose instead of calling the terminal `respond()` tool** (respond() fired only 2×/150 runs). Fatal for strict-terminal workflows, harmless otherwise
- Doc updates: README.md, docs/models.md, CLAUDE.md (Ornith now evidence-based); new reports/2026-06-27_ornith-assessment.md; this HISTORY entry
- Uncommitted working changes: run_forge_eval.py `--model` flag, middleware.py stagnation_threshold fix + test. Verdict: keep Qwen3.5-9B default; Ornith is a viable alternate, not an upgrade

## 2026-06-15 (session 2) — Autopilot
- Reviewed last session findings: reliability campaign (Qwen 5/5), security fixes (3 exfil bypasses), Gemma 4 12B Coder rejected
- Infrastructure: Qwen3.5-9B on :8080, guardrails proxy on :8081, both healthy
- Uncommitted changes from last session: network rule fixes, defaults() fix, dead code cleanup, tests, docs
- Created BACKLOG.md with 2 active investigations + 1 active task (uncommitted work)
- Next: commit uncommitted work, then start exfil vector investigation
