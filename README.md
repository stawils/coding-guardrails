# coding-guardrails

> Safe, reliable local coding agent backend. Open-source, pip-installable.

**coding-guardrails** is a proxy that sits between your coding agent and a local LLM,
adding two layers of protection:

1. **Forge (Layer 1)** — Rescue parsing, retries, validation. Makes local models
   actually work for tool calling.
2. **Coding Guardrails (Layer 2)** — Read-before-edit, path safety, command blocking,
   secret masking, test-after-change suggestions.

One command to go from "I have a GPU" to "I have a safe local coding agent backend."

## Quick Start

```bash
# Install
pip install coding-guardrails

# Start llama-server (your local LLM backend)
llama-server -m model.gguf --jinja --fit on --flash-attn auto \
  --port 8080 -c 16384 --spec-type draft-mtp -np 1

# Start the proxy
coding-guardrails serve \
  --backend-url http://localhost:8080 \
  --model Qwen3.6-35B-A3B-UD-Q3_K_M \
  --port 8081

# Point your agent at http://localhost:8081/v1
```

That's it. Your agent sees a standard OpenAI-compatible API.

## What It Blocks

| Rule | Blocks | Example |
|---|---|---|
| **Path safety** | Reads/writes outside workspace | `read_file("/etc/passwd")` ❌ |
| **Command safety** | Destructive shell commands | `bash("rm -rf /")` ❌ |
| **Secret detection** | API keys, tokens, private keys | `bash("export AWS_SECRET_ACCESS_KEY=...")` ❌ |
| **Prerequisites** | Edit before read (soft nudge) | `edit_file()` without `read_file()` ⚠️ |
| **Sequencing** | Missing test runs (soft nudge) | Edit without `pytest` ⚠️ |
| **Tool resolution** | Empty/error results (soft nudge) | Tool returns `""` ⚠️ |

All rules are configurable. See [docs/rules.md](docs/rules.md).

## Supported Models

Optimized for the **Qwen 3.6** family with llama-server:

| Model | VRAM | Context | SWE-bench |
|---|---|---|---|
| **Qwen3.6-35B-A3B Q3_K_M** ⭐ | 21.6 GB | 16K | 73.4% |
| Qwen3.6-27B Q4_K_M | 22.0 GB | 4K | 77.2% |

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
  strength: hard  # hard = block, soft = warn

secrets:
  enabled: true
  strength: hard
  mask_value: "[REDACTED]"
```

Pass with `--config guardrail-config.yaml`.

## Architecture

```
Agent → coding-guardrails (:8081) → llama-server (:8080) → GPU
            │
            ├─ Layer 1 (Forge): rescue, validate, retry
            └─ Layer 2 (Guardrails): 6 safety rules
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

```bash
coding-guardrails eval --backend-url http://localhost:8081
```

Runs scenarios from `eval/scenarios/` and reports pass/fail by category.

## Development

```bash
git clone https://github.com/stawils/coding-guardrails.git
cd coding-guardrails
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# Run tests
pytest tests/unit/ -v

# Run against live backend
pytest tests/integration/ -v -m integration
```

## License

MIT
