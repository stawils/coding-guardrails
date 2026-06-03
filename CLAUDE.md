# Coding Guardrails

An LLM proxy with safety guardrails, built on [Forge](https://github.com/antoinezambelli/forge) (vendored at `.vendors/forge/`).

## Quick Start

```bash
source .venv/bin/activate
pytest tests/unit/ -q          # 233 tests, ~0.1s
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

```bash
# Start proxy
source .venv/bin/activate
coding-guardrails serve \
  --backend-url http://localhost:8080 \
  --model Qwen3.6-27B-UD-Q4_K_XL \
  --port 8081 -v
```

### Client Quirks

- `SafeLlamafileClient` needs `gguf_path` (stem = model name): use `/tmp/<model-name>.gguf`
- `recommended_sampling=False` — model not in Forge's registry

## Testing

```bash
pytest tests/unit/ -q              # All 233 tests
pytest tests/unit/ -q -k "loop"    # Specific rule
```

All 233 tests must pass before committing.

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
- All 233 unit tests must pass
- No hardcoded scenario-specific logic
- Block responses must return **text**, not empty tool calls
- Enforcement prompts must mention `respond()` as the exit tool
- No injection/modification for non-coding-agent requests

## Versioning & Releases

SemVer: `MAJOR.MINOR.PATCH`

- **PATCH**: Bug fixes, no new features
- **MINOR**: New features, new rules, backward-compatible
- **MAJOR**: Breaking API changes

### Release Process

1. Update `version` in `pyproject.toml`
2. Update README/CLAUDE.md if needed
3. `pytest tests/unit/ -q` — all 233 must pass
4. `git commit -m "vX.Y.Z: description" && git push`
5. `git tag vX.Y.Z && git push origin vX.Y.Z`
6. GitHub Actions: test → build → verify CLI → publish to PyPI

### CI Pipeline

- Push to `main`: tests + lint
- Tag `v*`: build → verify → publish to PyPI (trusted publishing, no tokens)
- Refresh local: `uv pip install -e ".[dev]" && coding-guardrails --version`
