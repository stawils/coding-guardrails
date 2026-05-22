# Supported Models

coding-guardrails works with any OpenAI-compatible backend. These profiles are
optimized for local inference with llama-server.

## Recommended: Qwen 3.6 Family

| Model | Quant | Size | VRAM | Context | SWE-bench | Arch |
|---|---|---|---|---|---|---|
| **Qwen3.6-35B-A3B Q3_K_M** | Q3_K_M | 15.9 GB | 21.6 GB | 16K | 73.4% | MoE |
| Qwen3.6-35B-A3B Q4_K_S | Q4_K_S | 19.7 GB | 24.5 GB | 8K | 73.4% | MoE |
| Qwen3.6-27B Q4_K_M | Q4_K_M | 17.6 GB | 22.0 GB | 4K | 77.2% | Dense |
| Qwen3.6-27B Q4_K_S | Q4_K_S | 15.8 GB | 20.0 GB | 8K | 77.2% | Dense |

### Why MoE over Dense?

The 35B-A3B MoE has only 3.5B active parameters, giving it a much smaller KV
cache than the 27B dense model. This means more context fits in VRAM — critical
for coding agents that need large tool schemas and conversation history.

| | 27B Dense | 35B-A3B MoE |
|---|---|---|
| Active params | 27B | 3.5B |
| Context w/ MTP | 4K | 16K |
| Quality | 77.2% | 73.4% |

For most coding tasks, 16K context at 73.4% beats 4K context at 77.2%.

## Boot Command

```bash
llama-server \
  -m <model>.gguf \
  --jinja --fit on --flash-attn auto \
  --port 8080 -c 16384 \
  --spec-type draft-mtp -np 1
```

**Key flags:**
- `--jinja` — enables native function calling
- `--fit on` — auto GPU/CPU layer split (essential for MoE)
- `--spec-type draft-mtp` — multi-token prediction for faster inference
- `--flash-attn auto` — FlashAttention when available
- `-np 1` — single inference slot (required for RTX 3090 Ti to fit 42 layers + 16K KV)

## Adding New Models

Add a profile in `src/coding_guardrails/models/profiles.py`:

```python
"my-model-name": ModelProfile(
    name="my-model-name",
    family="MyModel",
    quant="Q4_K_M",
    file_size_gb=14.0,
    vram_required_gb=18.0,
    context_tokens=8192,
    architecture="dense",
    active_params_b=14.0,
    swe_bench_verified=None,
    sampling={"temperature": 0.7, "top_p": 0.9},
    boot_flags=["--jinja", "--fit", "on"],
),
```
