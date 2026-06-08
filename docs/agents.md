# Agent Configuration

coding-guardrails is transparent — agents see a standard OpenAI-compatible API.
Just point the agent at the proxy instead of directly at llama-server.

## Quick Start

```bash
# Terminal 1: Build + start cg's own llama-server (one-time build)
coding-guardrails server build
coding-guardrails server start --model gemma-4-26B-A4B-it-qat-UD-Q4_K_XL

# Terminal 2: Start proxy
coding-guardrails serve \
  --backend-url http://localhost:8080 \
  --model gemma-4-26B-A4B-it-qat-UD-Q4_K_XL \
  --port 8081

# Terminal 3: Start your agent pointing at :8081
```

See [server.md](server.md) for managing the cg-owned server (build, download,
start/stop, version).

## Pi

Set `api_base` to the proxy URL:

```yaml
# In Pi's config
model: "gemma-4-26B-A4B-it-qat-UD-Q4_K_XL"
api_base: "http://localhost:8081/v1"
```

## Aider

```bash
export OPENAI_API_BASE=http://localhost:8081/v1
export OPENAI_API_KEY=not-needed
aider --model openai/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL
```

## Continue (VS Code)

In `~/.continue/config.json`:

```json
{
  "models": [{
    "title": "coding-guardrails",
    "provider": "openai",
    "model": "gemma-4-26B-A4B-it-qat-UD-Q4_K_XL",
    "apiBase": "http://localhost:8081/v1",
    "apiKey": "not-needed"
  }]
}
```

## Cline / Roo Code

In VS Code settings, set:
- API Base: `http://localhost:8081/v1`
- Model: `gemma-4-26B-A4B-it-qat-UD-Q4_K_XL`
- API Key: any non-empty string

## Claude Code (Anthropic CLI)

Point Claude Code at the proxy as a custom OpenAI-compatible provider:

```bash
# Set environment variables
export OPENAI_API_KEY=not-needed
export OPENAI_BASE_URL=http://localhost:8081/v1

# Run Claude Code with the custom provider
claude --model openai/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL
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

Then use `claude --model openai/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL`.

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
          "name": "gemma-4-26B-A4B-it-qat-UD-Q4_K_XL",
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
