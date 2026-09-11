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

## Thinking mode (plain/text requests)

Tool-call requests already think by default. Plain (no-tool) requests answer
**directly without thinking** by default (`--auto-no-thinking` ON): reasoning
models like Qwen3.8-27B give instant clean answers instead of burning the
budget on reasoning that used to be silently discarded. That default is
unchanged.

To use thinking on a plain request, opt in per request (any one works):

```bash
# 1) Simplest — model-native thinking on
curl -s http://127.0.0.1:8081/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.8-27B-UD-Q3_K_XL",
       "messages":[{"role":"user","content":"Design a migration plan for X"}],
       "chat_template_kwargs":{"enable_thinking":true}}'

# 2) OpenAI-style effort level (none|minimal|low|medium|high|xhigh|max)
#    "none" disables thinking; levels are passed to the model's chat template.
# 3) Token cap on the reasoning block (llama.cpp thinking_budget_tokens)
#    — set your own when you need more/less than the 4096 server default.
```

With thinking on, the response carries the reasoning (verified live on
Qwen3.8-27B + MiniCPM5):

- Non-stream: `choices[0].message.reasoning_content` next to `content`.
- Stream: first a `delta.reasoning_content` chunk, then `content` chunks.

Safety guarantees (2026-09-11): the reasoning block is bounded
(`--thinking-budget-tokens`, default 4096, `0` disables) so it cannot eat
`max_tokens`; if a model still returns reasoning with an empty answer, the
proxy retries once with thinking off so you never receive an empty response.

Server-wide knobs:

| Flag | Effect |
|---|---|
| `--no-auto-no-thinking` | Thinking stays on by default for plain requests |
| `--thinking-budget-tokens N` | Reasoning token cap for plain thinking (default 4096) |
| `--reasoning-replay {full,keep-last,none}` | How reasoning is delivered: `keep-last` → `reasoning_content` field (default), `full` → merged into content, `none` → dropped |

When to use thinking: complex analysis/design/report tasks where a longer,
better-reasoned answer beats latency. Don't use it for routine turns (retrieval,
small edits) — thinking costs real time per token on local GPUs.

History hygiene: `reasoning_content` from your previous turn is forwarded as
context if you send it back in assistant messages. Either keep it (reasoning
helps it build on prior thoughts) or strip it before echo — full reasoning in
every turn inflates context and slows decoding.
