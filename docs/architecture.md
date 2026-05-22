# Architecture

coding-guardrails is a two-layer proxy that sits between a coding agent and a
local LLM backend (llama-server).

## Request Flow

```
Agent
  │
  │ POST /v1/chat/completions (OpenAI format)
  ▼
coding-guardrails proxy (:8081)
  │
  ├── Layer 1: Forge
  │   ├── Convert: OpenAI messages → Forge Message objects
  │   ├── Inject: respond() tool for text fallback
  │   ├── Inference: send to llama-server
  │   ├── Validate: check response format
  │   ├── Rescue: parse malformed tool calls from text
  │   └── Retry: re-prompt on failures (up to max_retries)
  │
  ├── Layer 2: Coding Guardrails
  │   ├── path_safety: block reads/writes outside workspace
  │   ├── command_safety: block destructive shell commands
  │   ├── secrets: mask or block API keys, tokens, private keys
  │   ├── prerequisites: ensure files are read before editing
  │   ├── sequencing: suggest running tests after changes
  │   └── tool_resolution: warn on empty/error tool results
  │
  ▼
llama-server (:8080) → GPU inference
  │
  ▼
Response flows back through both layers to agent
```

## Layer 1: Forge (Reliability)

Forge handles the "make it actually work" layer:

- **Rescue parsing** — When the model outputs tool calls as text (not JSON),
  Forge parses them out and converts to proper tool call format
- **Validation** — Ensures tool call arguments are valid JSON and match
  the declared tool schema
- **Retry with nudge** — On failure, re-prompts with guidance (e.g.,
  "respond with valid JSON tool calls")
- **respond() injection** — Adds a `respond` tool so the model can produce
  plain text responses without the agent seeing tool call artifacts

## Layer 2: Coding Guardrails (Safety)

Our rules handle coding-specific concerns:

- **Hard blocks** — Immediately prevent dangerous actions (path traversal,
  destructive commands, secret exfiltration)
- **Soft nudges** — Suggest best practices (read before edit, run tests)
- **Stateful** — Prerequisites and sequencing track tool call history

## Configuration

Each rule is independently configurable via `guardrail-config.yaml`:

```yaml
path_safety:
  enabled: true
  blocked_prefixes: ["/etc/", "/sys/"]

command_safety:
  enabled: true
  strength: hard
```

Rules can be:
- **Disabled** — `enabled: false`
- **Softened** — `strength: soft` (nudge instead of block)
- **Customized** — add/remove blocked commands, paths, patterns

## Key Files

| File | Purpose |
|---|---|
| `proxy/server.py` | Asyncio HTTP server, routing, SSE streaming |
| `proxy/handler.py` | Layer 1 → Layer 2 pipeline |
| `middleware.py` | Rule composition, `check()` / `record()` API |
| `rules/*.py` | Individual rule implementations |
| `config.py` | YAML config loading with env var expansion |
| `models/profiles.py` | Model hardware and sampling characteristics |
| `eval.py` | Eval scenario runner |
