# Coding Guardrails

An LLM proxy with safety guardrails, built on [Forge](https://github.com/antoinezambelli/forge) (vendored at `.vendors/forge/`).

## Quick Start

```bash
source .venv/bin/activate
pytest tests/unit/ -q          # 421 tests, ~2s
uv pip install -e ".[dev]"     # refresh editable install
```

## Architecture

Two-layer proxy between agent and LLM:

```
Agent → :8081 (our proxy)
          ├── Layer 1: Forge (rescue, validate, retry, thinking capture)
          ├── Layer 2: Coding Guardrails (11 rules)
          └── → :8080 (llama-server / LLM backend)
```

### Request Flow

1. Agent sends OpenAI-compatible request to `:8081`
2. **Preprocessing**: empty user messages fixed, stale assistant text stripped
3. **Tool enforcement**: for real coding agents (bash/read/edit/write tools), inject guidance
4. **Layer 1 (Forge)**: run inference with rescue + retry + thinking capture
5. **Layer 2 (Guardrails)**: check tool calls against 11 rules
6. **Response**: blocked calls return text nudge to agent; allowed calls pass through

### Key Modules

| File | Purpose |
|------|---------|
| `proxy/handler.py` | Request pipeline: preprocessing → L1 → L2 → response |
| `proxy/layer1.py` | Instrumented Forge wrapper with thinking capture |
| `proxy/client.py` | `SafeLlamafileClient` — thinking tokens, max_tokens |
| `proxy/server.py` | Asyncio HTTP server, `/v1/chat/completions` |
| `middleware.py` | Composes all rules, `check()` / `record()` API |
| `cli.py` | `coding-guardrails serve` CLI |
| `rules/` | 11 guardrail rules |

### Guardrail Rules

| Rule | Action | Purpose |
|------|--------|---------|
| `prerequisites` | block | Read-before-edit |
| `path_safety` | block | Path traversal |
| `command_safety` | block | Destructive commands |
| `network` | block | Network access |
| `sensitive_files` | block | Protect sensitive files |
| `secrets` | block/mask | Secret detection |
| `loop_detection` | nudge→block | Repeated identical calls (3 nudge, 5 block) |
| `session_budget` | nudge | File/command budgets |
| `thoroughness` | nudge | Premature terminal submission |
| `sequencing` | nudge | Test-after-change |
| `tool_resolution` | nudge | Empty/error results |

## Production Rules

### Block Response Design

When Layer 2 blocks a tool call, the proxy returns a **text response** (not an empty tool call).

**Why**: Returning an empty tool call with the blocked tool name confuses agents — they see a tool call, execute it with empty args, get nothing, and retry the same call. Returning text makes the nudge visible to the agent and the model, providing a clear escape path.

**Loop detection blocks specifically** tell the model: "If the task is done, call respond() with your final answer. Otherwise try a completely different approach." This gives the model an explicit exit instead of vague guidance.

### Tool Enforcement

Only injected for **real coding agents** (requests that include bash/read/edit/write tools). The enforcement says:

> "Respond by calling tools. If the task is complete, call respond(). If unsure, call bash with 'echo ready'."

**Never** say "NEVER respond with plain text" — the model needs to finish tasks. The `respond()` tool is the proper terminal action.

### For eval/workflow requests (no coding tools)

No enforcement, no injection, no modification. The proxy is a transparent pass-through.

### Thinking Capture & Injection

Layer 1 captures thinking tokens from the model's response. On retry (failed validation), the thinking is injected into the retry nudge so the model doesn't re-think from scratch.

### Nudges vs Blocks

- **Nudge**: advisory — the call proceeds, message is logged. Only visible if the agent's runner supports nudge injection (Forge does for its own evals). For streaming agents (Pi), nudges are silently logged.
- **Block**: hard stop — the call is NOT passed through. The proxy returns a text response to the agent explaining why. This is the only reliable way to change model behavior mid-conversation.

## Running the Proxy

### tmux Sessions

| Session | Purpose | Port |
|---------|---------|------|
| `llm-server` | `llama-server` with model | :8080 |
| `guardrails` | `coding-guardrails serve` | :8081 |

No workspace session — delegation is done via Pi subagents (`subagent()` tool) pointing at the proxy.

```bash
# Session 1: LLM backend (use LM Studio's llama-server — supports DeltaNet models)
LLAMA=~/.cache/lm-studio/extensions/backends/llama.cpp-linux-x86_64-nvidia-cuda12-avx2-2.18.0/llama-server

# Qwen3.5-9B (200K ctx, MTP)
$LLAMA \
  -m ~/.cache/lm-studio/models/unsloth/Qwen3.5-9B-MTP-GGUF/Qwen3.5-9B-UD-Q4_K_XL.gguf \
  -c 200000 -ngl 99 --host 0.0.0.0 --port 8080 \
  --jinja --flash-attn auto --spec-type draft-mtp -np 1 -v

# Qwen3.6-27B UD-Q4_K_XL (32K ctx)
$LLAMA \
  -m ~/.cache/lm-studio/models/unsloth/Qwen3.6-27B-MTP-GGUF/Qwen3.6-27B-UD-Q4_K_XL.gguf \
  -c 32768 -ngl 99 --host 0.0.0.0 --port 8080 \
  --jinja --flash-attn auto -v

# Qwen3.6-27B UD-Q3_K_XL (82K ctx, needs 24+ GB free VRAM)
$LLAMA \
  -m ~/.cache/lm-studio/models/unsloth/Qwen3.6-27B-MTP-GGUF/Qwen3.6-27B-UD-Q3_K_XL.gguf \
  -c 81920 -ngl 99 --host 0.0.0.0 --port 8080 \
  --jinja --flash-attn auto -v

# Gemma 4 12B Unified UD-Q4_K_XL (256K ctx, ~8 GB VRAM)
$LLAMA \
  -m ~/.cache/lm-studio/models/unsloth/gemma-4-12b-it-GGUF/gemma-4-12b-it-UD-Q4_K_XL.gguf \
  -c 256000 -ngl 99 --host 0.0.0.0 --port 8080 \
  --jinja --flash-attn auto -np 1 -v

# Gemma 4 26B A4B QAT UD-Q4_K_XL (200K ctx, ~20 GB VRAM, current default)
#   MoE: 25.2B total / 3.8B active. q8_0 KV cache required for 200K to fit 24 GB.
#   Use ONLY the Unsloth QAT GGUF — naive Q4_0 loses 15.4pp accuracy.
$LLAMA \
  -m ~/.cache/lm-studio/models/unsloth/gemma-4-26B-A4B-it-qat-GGUF/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf \
  -c 200000 -ngl 99 --host 0.0.0.0 --port 8080 \
  --jinja --flash-attn auto -ctk q8_0 -ctv q8_0 \
  --temp 1.0 --top-p 0.95 --top-k 64 -np 1 -v

# Session 2: Guardrails proxy (with config for increased budgets)
source .venv/bin/activate
coding-guardrails serve \
  --backend-url http://localhost:8080 \
  --model <model-name> \
  --port 8081 \
  --config configs/guardrail-config.yaml \
  -v

# No workspace session needed — use Pi subagents for delegation:
#   subagent({ agent: "worker", model: "coding-guardrails/<model>", ... })
```

### Notes

- **Use LM Studio's llama-server** (`~/.cache/lm-studio/extensions/.../2.18.0/llama-server`), NOT `~/llama.cpp/llama-server`. The LM Studio build supports DeltaNet/SSM tensors (Qwen3.6); the local build does not.
- `~/llama.cpp/llama-server` is older (build 8276) and fails on Qwen3.6-27B models (missing `ssm_conv1d` tensors).
- Qwen3.6-27B Q3 model OOMs at 82K ctx on RTX 3090 Ti — reduce to ≤49K or use Q4_K_XL at 32K.
- **Gemma 4 12B Unified**: Dense 12B, 256K ctx, encoder-free multimodal (text+image+audio). Only ~8 GB VRAM at Q4 — massive headroom on 24 GB cards. **No MTP yet** (llama.cpp issue #22747). Sampling: temp=1.0, top_k=64, top_p=0.95.
- **Gemma 4 26B A4B QAT**: MoE (25.2B total / 3.8B active), native 256K ctx (run at 200K). ~14.25 GB weights, **~20 GB VRAM at 200K** with q8_0 KV cache — needs `-ctk q8_0 -ctv q8_0`. Sliding-window attention (5 global of 30 layers) keeps the KV cache tiny. Highest capability (88.3% AIME, 77.1% LiveCodeBench). **No MTP.** Use the **Unsloth UD-Q4_K_XL** QAT GGUF only — naive Q4_0 loses 15.4pp top-1.
- `SafeLlamafileClient` needs `gguf_path` (stem = model name): use `/tmp/<model-name>.gguf`
- `recommended_sampling=False` — model not in Forge's registry
- any md file writing should be in gitignored plans/ folder.
- use one subagent at a time , no parallel agents.

## Testing

```bash
pytest tests/unit/ -q              # All 421 tests
pytest tests/unit/ -q -k "loop"    # Specific rule
```

All 421 tests must pass before committing.

## Eval

```bash
# Full Forge 30-scenario benchmark (proxy mode)
python eval/scripts/run_forge_eval.py --mode proxy --runs 5

# Direct vs proxy comparison
python eval/scripts/run_forge_eval.py --mode both --runs 5

# Specific scenarios
python eval/scripts/run_forge_eval.py --mode proxy --scenario data_gap_recovery_extended

# Layer 2 guardrails
python eval/scripts/run_layer2_eval.py
```

Results go to `eval/runs/<timestamp>/` (gitignored).

**Best result: 93% (140/150)** on Forge 30-scenario eval, +9pp over Forge's 84% baseline.

## Development Guidelines

- **Do NOT hack Forge source** — extend via public API, subclassing, wrapping
- All 385 unit tests must pass
- No hardcoded scenario-specific logic
- Block responses must return **text**, not empty tool calls
- Enforcement prompts must mention `respond()` as the exit tool
- No injection/modification for non-coding-agent requests

## Versioning & Releases

SemVer: `MAJOR.MINOR.PATCH`

- **PATCH**: Bug fixes, no new features
- **MINOR**: New features, new rules, new model profiles, backward-compatible
- **MAJOR**: Breaking API changes

### Full Release Process

Every release follows these steps **in order**. Do not skip any step.

#### 1. Bump version

```bash
# Edit pyproject.toml — change version to the new number
# PATCH example: 0.9.0 → 0.9.1 (bug fix)
# MINOR example: 0.9.1 → 0.10.0 (new feature/model)
# MAJOR example: 0.10.0 → 1.0.0 (breaking change)
```

#### 2. Run tests

```bash
source .venv/bin/activate
pytest tests/unit/ -q          # All 421 tests MUST pass
```

If any test fails → **stop**, fix, re-run. Do not proceed.

#### 3. Update docs

Update `CLAUDE.md` and `docs/models.md` if the change affects:
- Model profiles
- Boot commands
- Running instructions
- Test counts

#### 4. Stage and commit

```bash
git add pyproject.toml src/ docs/ CLAUDE.md  # only changed files
git commit -m "vX.Y.Z: short description"
```

Commit message format: `vX.Y.Z: description` (imperative mood).

#### 5. Push to main

```bash
git push origin main
```

Verify push succeeded. This triggers CI (test + lint) but does NOT publish.

#### 6. Tag the release

```bash
git tag vX.Y.Z
git push origin vX.Y.Z
```

This triggers the full CI pipeline: test → build → verify CLI → **publish to PyPI**.

#### 7. Verify CI passed

```bash
gh run list --limit 3
gh run view <run-id>   # all 3 jobs green: test, build, publish-pypi
```

Wait for all jobs to show ✅. If anything fails → fix, bump PATCH, restart from step 1.

#### 8. Create GitHub Release

**GitHub Releases are NOT automatic.** CI publishes to PyPI, but the Releases page
needs a manual step. Without this, the GitHub Releases tab shows stale versions.

```bash
gh release create vX.Y.Z \
  --title "vX.Y.Z" \
  --notes "## What's New

- Change 1
- Change 2

### Full Changelog
https://github.com/stawils/coding-guardrails/compare/vPREV...vX.Y.Z"
```

#### 9. Refresh local install

```bash
uv pip install -e ".[dev]"
coding-guardrails --version    # verify it matches vX.Y.Z
```

### Quick Reference (copy-paste)

```bash
# Replace X.Y.Z and PREV with actual versions
VERSION=X.Y.Z PREV=X.PREV.Z
source .venv/bin/activate
pytest tests/unit/ -q
git add -A && git commit -m "v${VERSION}: description"
git push origin main
git tag v${VERSION} && git push origin v${VERSION}
# Wait for CI...
gh run list --limit 1  # confirm green
gh release create v${VERSION} --title "v${VERSION}" --notes "## What's New\n\n- Description\n\n### Full Changelog\nhttps://github.com/stawils/coding-guardrails/compare/v${PREV}...v${VERSION}"
uv pip install -e ".[dev]" && coding-guardrails --version
```

### CI Pipeline

- **Push to `main`**: tests + lint (no publish)
- **Tag `v*`**: test → build → verify CLI → publish to PyPI (trusted publishing, no tokens)
- **GitHub Release**: manual step via `gh release create` (PyPI is automatic, GitHub Releases is not)

### Common Mistakes

| Mistake | Result | Fix |
|---------|--------|-----|
| Committing but not pushing | Tags point to unreachable commits | Always `git push origin main` BEFORE tagging |
| Tagging without pushing main first | CI can't find the commit | Push main first, then tag |
| Forgetting `gh release create` | GitHub Releases page shows old version | Always create the release after CI passes |
| Skipping tests | Broken release on PyPI | Never skip step 2 |
