# Managing the cg-owned llama.cpp Server

coding-guardrails ships its **own** llama.cpp server so every user runs the
same, reproducible inference stack — no LM Studio, no manual flag guessing.
The pin is chosen to include critical fixes (notably the Gemma 4 tool-call
grammar fix, llama.cpp #21680, which older bundled binaries lack).

All artifacts live under one XDG-aware directory:

```
~/.local/share/coding-guardrails/
├── llama.cpp/                  pinned git checkout (commit afcda09d1)
│   └── build/bin/llama-server  the compiled binary
├── models/                     cg-owned GGUF cache (primary search path)
└── run/
    ├── llama-server.pid
    └── llama-server.log
```

## Commands

```bash
coding-guardrails server version    # pinned vs installed vs binary version
coding-guardrails server build      # clone (if needed) + cmake build
coding-guardrails server download <model>   # fetch GGUF into cg cache
coding-guardrails server start -m <model>   # launch llama-server (:8080)
coding-guardrails server status     # running? version? listening?
coding-guardrails server stop       # SIGTERM the running server
```

> Shorthand: these docs use `coding-guardrails server ...`. The installed
> binary is `coding-guardrails`; alias it (`alias cg=coding-guardrails`)
> if you prefer `cg server ...`.

## First-time setup

```bash
# 1. Build (CUDA auto-detected; ~20-40 min first time, incremental after)
coding-guardrails server build

# 2. Download a model (gemma-4-26B A4B QAT is the recommended default)
coding-guardrails server download gemma-4-26B-A4B-it-qat-UD-Q4_K_XL

# 3. Start the backend
coding-guardrails server start --model gemma-4-26B-A4B-it-qat-UD-Q4_K_XL

# 4. Start the proxy on top of it
coding-guardrails serve \
  --backend-url http://localhost:8080 \
  --model gemma-4-26B-A4B-it-qat-UD-Q4_K_XL \
  --port 8081
```

## Build options

```bash
coding-guardrails server build --cpu        # force CPU-only (skip CUDA)
coding-guardrails server build --cuda       # force CUDA on
coding-guardrails server build -j 8         # limit parallel jobs
```

The build auto-detects CUDA (`nvcc` or `/usr/local/cuda*`) and uses Ninja if
available. Re-running `build` updates to the pinned commit incrementally.

## Updating the pin

The pin lives in `src/coding_guardrails/server/version.py`
(`PINNED_COMMIT`). To move to a newer llama.cpp:

1. Bump `PINNED_COMMIT` to the new full SHA (and `PINNED_SHORT`).
2. Confirm the fix commits you need are ancestors of the new SHA.
3. `coding-guardrails server build` re-checks out and rebuilds.

The current pin (`afcda09d1`, build 9284) includes the Gemma 4 specialized
tool-call parser (#21418, #21704) that fixes complex/nested JSON tool-call
corruption.

## Start options

```bash
coding-guardrails server start -m <model> \
  --ctx 32768        # override context window (default: profile max)
  --ngl 99           # GPU layers (default 99 = all)
  --port 8080 \
  --wait             # block until /health responds
  --temp 0.7         # extra flags passed verbatim to llama-server
```

Flags come from the model's profile (`models/profiles.py` `boot_flags`), so
the q8 KV cache (`-ctk q8_0 -ctv q8_0`), `--jinja`, sampling, etc. are applied
automatically. Anything after `--` is forwarded verbatim.

The server runs detached (own process group) and survives the CLI exiting. Its
PID and combined stdout/stderr live under `run/`.

## Decoupling from LM Studio

The model registry (`models/registry.py`) searches the cg cache **first**;
LM Studio and HuggingFace caches are kept as **read-only fallbacks** so an
existing install keeps working. A clean cg user never needs them:

```python
# registry search order
1. ~/.local/share/coding-guardrails/models   # cg-owned (primary)
2. ~/.cache/lm-studio/models                 # fallback
3. ~/.cache/huggingface/hub                  # fallback
```

If a model is found in the cg cache, the fallbacks are ignored for it.

## Bringing your own llama-server

Already running a llama-server (or any OpenAI-compatible backend)? Skip
`server build/start` entirely and point the proxy at it:

```bash
coding-guardrails serve \
  --backend-url http://localhost:8080 \
  --model <your-model-name> \
  --port 8081
```

The proxy is backend-agnostic; it only needs an OpenAI `/v1` API. Note that
older llama.cpp builds may not include the Gemma 4 tool-call fix — use the
cg-owned build for Gemma models.
