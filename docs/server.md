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

# 2. Download a model (Qwen3.8-27B is the recommended default — measured
#    100% completion / 99% accuracy; vision-capable)
coding-guardrails server download Qwen3.8-27B-UD-Q3_K_XL

# 3. Start the backend
coding-guardrails server start --model Qwen3.8-27B-UD-Q3_K_XL

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

The pin lives in `src/coding_guardrails/server/version.py` (`PINNED_COMMIT`
+ `PINNED_SHORT`, and `PINNED_REPO` for the origin). To move to a newer
llama.cpp:

1. Bump `PINNED_COMMIT` to the new full SHA (and `PINNED_SHORT`).
2. Point `PINNED_REPO` at a repo that actually contains that SHA — `build`
   re-points the checkout's `origin` on every run, so moving between upstream
   and a fork needs no manual git work.
3. Confirm the fix commits you need are ancestors of the new SHA.
4. `coding-guardrails server build` re-checks out and rebuilds.

The current pin (`d493f84`, llama.cpp 0.9.0-dev) is **upstream master plus one
fork commit**: DSpine speculative decoding is not upstream yet, so `PINNED_REPO`
is `github.com/Ajay9o9/llama.cpp` and the commit is
`dflash : add DSpine draft model support` (new gguf tensor names, the code
bridge, the serial last-write refinement — no network/shell/file-write
additions). Everything earlier in this chain is included: the Gemma 4
tool-call parser (#21418, #21704), `LLM_ARCH_FLINT` (#30928) and the LFM3
tri-hybrid family (#31252/#31257/#31283).

Switch `PINNED_REPO` back to `https://github.com/ggml-org/llama.cpp.git` once
DSpine lands upstream; the drafter flags are stable either way.

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

### Speculative decoding drafters

A profile can name a draft model (`ModelProfile.drafter`, keyed into
`server/sources.py` `DRAFTERS`). The launcher then appends the full speculative
decoding argv automatically:

```
-md <drafter.gguf> --spec-type <type> --spec-draft-n-max <n>
--spec-draft-ngl 99 --spec-draft-type-k q8_0 --spec-draft-type-v q8_0
```

`cg server download <profile>` fetches the drafter alongside the target model.
If the drafter GGUF is missing, the launcher logs a warning and boots the
target **without** speculation rather than failing — the always-on tier must not
depend on an optional speed artifact.

Two VRAM details matter: the drafter's own allowance (`DRAFTERS[...].vram_gb`)
is added to the profile's gate, and a drafter costs roughly one copy of a hybrid
model's recurrent state per drafted token (Qwen3.8-27B: ~150 MB each), so a
long `--spec-draft-n-max` eats context budget. `spec_n_max` on the profile
overrides the drafter's default.

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
