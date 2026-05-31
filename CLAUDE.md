# Coding Guardrails

An LLM proxy with safety guardrails, built on [Forge](https://github.com/antoinezambelli/forge) (vendored at `.vendors/forge/`).

## Quick Start

```bash
# Activate venv
source .venv/bin/activate

# Run tests
pytest tests/unit/ -q          # 233 tests, ~0.1s

# Install editable
pip install -e .
```

## Architecture

Two-layer proxy sitting between agent and LLM:

```
Agent → :8081 (our proxy)
          ├── Layer 1: Forge (rescue, validate, retry, thinking capture)
          ├── Layer 2: Coding Guardrails (11 rules)
          └── → :8080 (llama-server / LLM backend)
```

### Key Modules

| File | Purpose |
|------|---------|
| `src/coding_guardrails/proxy/handler.py` | Request pipeline: preprocessing → L1 (Forge) → L2 (guardrails) → response |
| `src/coding_guardrails/proxy/layer1.py` | Instrumented wrapper around Forge's `run_inference()` with thinking capture |
| `src/coding_guardrails/proxy/client.py` | `SafeLlamafileClient` — thinking token capture, max_tokens forwarding |
| `src/coding_guardrails/proxy/server.py` | Asyncio HTTP server, OpenAI-compatible `/v1/chat/completions` |
| `src/coding_guardrails/middleware.py` | Composes all guardrail rules, provides `check()` / `record()` API |
| `src/coding_guardrails/cli.py` | `coding-guardrails serve` CLI with `--backend-url`, `--model`, `--port`, `--log-file` |
| `src/coding_guardrails/rules/` | 11 guardrail rules (see below) |

### Guardrail Rules (Layer 2)

| Rule | Action | What it does |
|------|--------|-------------|
| `prerequisites` | block | Enforce read-before-edit |
| `path_safety` | block | Path traversal blocking |
| `command_safety` | block | Destructive command blocking |
| `network` | block | Network access control |
| `sensitive_files` | block | Protect sensitive files |
| `secrets` | block/mask | Secret detection |
| `loop_detection` | nudge/block | Stagnation detection (≤2 unique tools over 14+ calls) |
| `session_budget` | nudge | File/command/read budget limits |
| `thoroughness` | block | Premature terminal submission (low tool exploration ratio) |
| `sequencing` | nudge | Test-after-change suggestions |
| `tool_resolution` | nudge | Empty/error result handling |

## Running the Proxy

### tmux Sessions

Four tmux sessions are used:

| Session | Purpose | Port |
|---------|---------|------|
| `llm-server` | `llama-server` with Qwen3.5-9B-UD-Q4_K_XL | :8080 |
| `guardrails` | `coding-guardrails serve` proxy | :8081 |
| `skills-master` | General workspace | — |
| `pi` | Pi coding agent | — |

```bash
# Start llama-server (in tmux: llm-server)
llama-server -m /path/to/Qwen3.5-9B-UD-Q4_K_XL.gguf --port 8080 --host 0.0.0.0

# Start proxy (in tmux: guardrails)
source .venv/bin/activate
coding-guardrails serve --backend-url http://localhost:8080 --model Qwen3.5-9B-UD-Q4_K_XL --port 8081 -v
```

### Important Client Quirks

- `SafeLlamafileClient` needs `gguf_path` for model identity (stem = model name), not `model` — file doesn't need to exist: use `/tmp/Qwen3.5-9B-UD-Q4_K_XL.gguf`
- `recommended_sampling=False` required — Qwen3.5-9B-UD-Q4_K_XL not in Forge's MODEL_SAMPLING_DEFAULTS registry

## Testing

```bash
source .venv/bin/activate
pytest tests/unit/ -q              # All 233 unit tests
pytest tests/unit/ -q -k "loop"    # Specific rule tests
```

All 233 tests must pass before committing.

## Eval

### Forge 30-Scenario Eval

```bash
source .venv/bin/activate

# Proxy mode (5 runs × 30 scenarios = 150 calls)
python eval/scripts/run_forge_eval.py --mode proxy --runs 5

# Direct mode (bypass proxy, LLM only)
python eval/scripts/run_forge_eval.py --mode direct --runs 5

# Both modes (direct vs proxy comparison)
python eval/scripts/run_forge_eval.py --mode both --runs 5

# Specific scenarios
python eval/scripts/run_forge_eval.py --mode proxy --scenario data_gap_recovery_extended argument_transformation

# Verbose (shows tool calls, thinking, results)
python eval/scripts/run_forge_eval.py --mode proxy --runs 1 -v
```

### Run Folder Structure

Each run creates `eval/runs/<timestamp>/`:

```
eval/runs/2026-05-31_204247Z/
├── run.json          # Run metadata (mode, model, timestamps)
├── eval.log          # Forge's verbose trace (tee'd from stdout)
├── proxy.log         # Proxy's logging module output
├── proxy.jsonl       # Per-scenario JSON results
├── proxy_summary.txt # Human-readable summary table
├── comparison.md     # Direct vs proxy comparison (both mode)
└── direct.jsonl      # Direct mode results (both mode)
```

### Layer 2 Guardrails Eval

```bash
python eval/scripts/run_layer2_eval.py    # 17 JSON scenarios
```

### Compare Direct vs Proxy

```bash
python eval/scripts/compare.py eval/runs/<timestamp>/direct.jsonl eval/runs/<timestamp>/proxy.jsonl
```

### Benchmark Results

Latest run: `eval/runs/2026-05-31_204247Z/` — **93% accuracy** (140/150), 100% completion.
Forge's published benchmark: ~84% accuracy. Our proxy: +9pp with zero regressions.

## Development Guidelines

- **Do NOT hack Forge source** — extend via public API, subclassing, or wrapping only
- Changes must survive upstream Forge updates (vendored at `.vendors/forge/`)
- All 233 unit tests must pass
- No hardcoded scenario-specific logic — guardrails must be general purpose
- Use `edit` for targeted changes, `write` for new files or complete rewrites
- Keep the proxy clean: no injection/annotation of messages for non-coding-agent requests
