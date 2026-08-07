# Session History

## 2026-08-07 — Qwen3.6-27B added (max-ctx profile, 48K q8_0 KV)
- Downloaded Qwen3.6-27B-UD-Q4_K_XL (17.9 GB, unsloth/Qwen3.6-27B-MTP-GGUF) into cg cache
- Empirical max-ctx sweep on RTX 3090 Ti (24 GB shared desktop): weights+compute peak ~20.25 GB;
  q8_0 KV @48K = +1.54 GB → 22.35 GB total, **7/7 boots OK**. 56K+/64K fail (single KV block
  >1.79 GB can't allocate — driver VA fragmentation after many large load/unload cycles);
  32K f16 KV (2.05 GB block) and MTP spec decoding hit the same wall
- Key findings: `-ub 256` keeps the flash-attn compute buffer ~150 MiB (default ubatch 2048
  intermittently fails to find a contiguous block); q8_0 KV required (f16 doesn't fit 40K+);
  no MTP at 48K; small `cudaMalloc failed` errors = fragmented allocator state, reboot fixes
- Verified: generation + clean read/respond tool calls through the proxy (smoke test green)
- Added profile `Qwen3.6-27B-UD-Q4_K_XL` (48K ctx, q8_0 KV, -ub 256) + docs/models.md + CLAUDE.md
- Forge eval subset (4 scenarios × 3 runs, proxy mode): **8/12 completion, 9/9 (100%) accuracy**
  on completed runs. tool_selection 0/3: correct tool sequence (lookup_user→get_permissions) but
  answers in prose instead of calling the terminal respond() tool (Ornith-class quirk)
- **Worker test (2026-08-07, cg-worker via isolated stack :8090/:8082):** 2/2 real tasks passed
  first try — README model table (committed d5afbab) + FORGE_EVAL_BACKEND_URL env override
  (local, eval/ gitignored). Terminated cleanly in real delegation (prose quirk did NOT manifest).
  Verdict: viable worker; Qwen3.5-9B stays default. Full report:
  plans/2026-08-07_qwen3.6-27b-worker-assessment.md

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
