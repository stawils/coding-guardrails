# Agent Configuration

coding-guardrails is transparent — agents see a standard OpenAI-compatible API.
Just point the agent at the proxy instead of directly at llama-server.

## Quick Start

```bash
# Terminal 1: Start llama-server
llama-server -m model.gguf --jinja --fit on --flash-attn auto \
  --port 8080 -c 16384 --spec-type draft-mtp -np 1

# Terminal 2: Start proxy
coding-guardrails serve \
  --backend-url http://localhost:8080 \
  --model Qwen3.6-35B-A3B-UD-Q3_K_M \
  --port 8081

# Terminal 3: Start your agent pointing at :8081
```

## Pi

Set `api_base` to the proxy URL:

```yaml
# In Pi's config
model: "Qwen3.6-35B-A3B-UD-Q3_K_M"
api_base: "http://localhost:8081/v1"
```

## Aider

```bash
export OPENAI_API_BASE=http://localhost:8081/v1
export OPENAI_API_KEY=not-needed
aider --model openai/Qwen3.6-35B-A3B-UD-Q3_K_M
```

## Continue (VS Code)

In `~/.continue/config.json`:

```json
{
  "models": [{
    "title": "coding-guardrails",
    "provider": "openai",
    "model": "Qwen3.6-35B-A3B-UD-Q3_K_M",
    "apiBase": "http://localhost:8081/v1",
    "apiKey": "not-needed"
  }]
}
```

## Cline / Roo Code

In VS Code settings, set:
- API Base: `http://localhost:8081/v1`
- Model: `Qwen3.6-35B-A3B-UD-Q3_K_M`
- API Key: any non-empty string

## Claude Code (Anthropic CLI)

Point Claude Code at the proxy as a custom OpenAI-compatible provider:

```bash
# Set environment variables
export OPENAI_API_KEY=not-needed
export OPENAI_BASE_URL=http://localhost:8081/v1

# Run Claude Code with the custom provider
claude --model openai/Qwen3.6-35B-A3B-UD-Q3_K_M
```

Or in your project's `.claude/settings.json`:

```json
{
  "env": {
    "OPENAI_API_KEY": "not-needed",
    "OPENAI_BASE_URL": "http://localhost:8081/v1"
  }
}
```

Then use `claude --model openai/Qwen3.6-35B-A3B-UD-Q3_K_M`.

## OpenCode (Terminal AI)

Edit `~/.config/opencode/opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "model": "opencode-guardrails/qwen",
  "provider": {
    "opencode-guardrails": {
      "name": "Coding Guardrails",
      "npm": "@ai-sdk/openai-compatible",
      "options": {
        "baseURL": "http://localhost:8081/v1",
        "apiKey": "not-needed"
      },
      "models": {
        "qwen": {
          "name": "Qwen3.6-35B-A3B-UD-Q3_K_M",
          "_launch": true
        }
      }
    }
  }
}
```

Then run `opencode` — it will use the proxy automatically.

## Generic OpenAI-Compatible

Any agent that supports custom OpenAI API base URLs should work. Set:
- **API Base:** `http://localhost:8081/v1`
- **Model:** your model name
- **API Key:** any non-empty string (not validated)
