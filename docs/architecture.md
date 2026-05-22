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
  ├── Layer 1: Forge (reliability)
  │   ├── Convert: OpenAI messages → Forge Message objects
  │   ├── Inject: respond() tool for text fallback
  │   ├── Inference: send to llama-server (via SafeLlamafileClient)
  │   ├── Validate: check response format
  │   ├── Rescue: parse malformed tool calls from text
  │   └── Retry: re-prompt on failures (up to max_retries)
  │
  ├── max_tokens safety net
  │   ├── Forward agent's max_tokens to backend
  │   ├── Normalize max_completion_tokens → max_tokens
  │   └── Default 8192 token cap if none specified
  │
  ├── Layer 2: Coding Guardrails (10 rules)
  │   ├── path_safety      — block /etc/, /proc/, path traversal
  │   ├── command_safety   — block sudo, eval/curl, git destructive ops
  │   ├── network          — block file uploads, SSRF, metadata endpoints
  │   ├── sensitive_files  — block writes to .git/, .ssh/, CI pipelines
  │   ├── secrets          — detect/mask API keys, tokens, private keys
  │   ├── prerequisites    — ensure read-before-edit (prefix matching)
  │   ├── loop_detection   — detect and break stuck agent loops
  │   ├── session_budget   — cap file ops and commands per session
  │   ├── sequencing       — suggest running tests after changes
  │   └── tool_resolution  — warn on empty/error tool results
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
- **Retry with nudge** — On failure, re-prompts with guidance
- **respond() injection** — Adds a `respond` tool so the model can produce
  plain text responses

### SafeLlamafileClient

Our `SafeLlamafileClient` extends Forge's client without modifying Forge itself:

- Forwards `max_tokens` to the backend (Forge ignores this by default)
- Normalizes `max_completion_tokens` → `max_tokens` for llama-server
- Injects a default 8192 token cap to prevent runaway generation
- No modifications to Forge required — `pip install forge-guardrails` works as-is

## Layer 2: Coding Guardrails (Safety)

10 composable rules:

- **Hard blocks** — Immediately prevent dangerous actions (path traversal,
  destructive commands, secret exfiltration, network uploads)
- **Soft nudges** — Suggest best practices (read before edit, run tests,
  break loops, respect budgets)
- **Stateful** — Prerequisites, sequencing, loop detection, and session
  budget track tool call history

### Tool Name Matching

All rules use **prefix matching** for tool names. This means:
- `edit` matches `edit`, `edit_file`, `Edit`, `EDITOR`
- `read` matches `read`, `read_file`, `Read`, `READ_FILE`
- `bash` matches `bash`, `shell`, `Bash`

Works with Pi, Claude Code, Aider, OpenCode, Continue, and any other agent
without per-agent configuration.

## Configuration

Each rule is independently configurable via `guardrail-config.yaml`:

```yaml
path_safety:
  enabled: true
  blocked_prefixes: ["/etc/", "/sys/"]

command_safety:
  enabled: true

network:
  enabled: true
  block_uploads: true

sensitive_files:
  enabled: true

loop_detection:
  enabled: true
  nudge_threshold: 3
  block_threshold: 5

session_budget:
  enabled: true
  max_file_ops: 100
  max_commands: 200
```

Rules can be:
- **Disabled** — `enabled: false`
- **Softened** — change block to nudge
- **Customized** — add/remove blocked commands, paths, patterns, thresholds

## Key Files

| File | Purpose |
|---|---|
| `proxy/server.py` | Asyncio HTTP server, routing, SSE streaming |
| `proxy/handler.py` | Layer 1 → Layer 2 pipeline, structured logging |
| `proxy/client.py` | SafeLlamafileClient (max_tokens forwarding) |
| `middleware.py` | Rule composition, `check()` / `record()` API |
| `rules/*.py` | 10 individual rule implementations |
| `config.py` | YAML config loading with env var expansion |
| `models/profiles.py` | Model hardware and sampling characteristics |
| `eval.py` | Eval scenario runner |

## Logging

The proxy produces structured, readable logs:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
>> POST /v1/chat/completions
   msgs=30 tools=6 stream=True model=Qwen3.5-9B-UD-Q4_K_XL

──────────────────── ▸ LAYER 1 · Forge ◂ ────────────────────
🔧 Calling model (7 tools, 30 msgs, max 3 retries)
✅ Layer 1 complete (6.9s, 1 tool calls: edit(path=src/main.py))

────────────────── ▸ LAYER 2 · Guardrails ◂ ──────────────────
  ✅ edit — allowed
✅ Request PASSED (0ms)
```

Blocks show:
```
  🚫 read — BLOCKED [blocked prefix: /etc/passwd matches /etc/]
     ↳ Path '/etc/passwd' is outside the allowed workspace.
⛔ Request BLOCKED by Layer 2 (0ms)
```
