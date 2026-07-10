"""CLI entry point for coding-guardrails."""

from __future__ import annotations

import asyncio
import logging
import signal
import sys

import click

from coding_guardrails.middleware import CodingGuardrails


@click.group()
@click.version_option(package_name="coding-guardrails")
def main() -> None:
    """coding-guardrails — Safe, reliable local coding agent backend.

    Layer 1: Forge (rescue parsing, retries, validation).
    Layer 2: Coding guardrails (read-before-edit, path safety, etc.).
    """


@main.command()
@click.option("--backend-url", required=True, help="URL of the llama-server backend (e.g. http://localhost:8080)")
@click.option("--model", required=True, help="Model name for sampling defaults (e.g. gemma-4-26B-A4B-it-qat-UD-Q4_K_XL)")
@click.option("--port", default=8081, type=int, help="Proxy listen port (default: 8081)")
@click.option("--host", default="127.0.0.1", help="Proxy listen host")
@click.option("--config", "config_path", help="Path to guardrail-config.yaml")
@click.option("--max-retries", default=3, type=int, help="Max Forge retries per request (default: 3)")
@click.option("--no-rescue", is_flag=True, help="Disable Forge rescue parsing")
@click.option("--no-guardrails", is_flag=True, help="Disable Layer 2 guardrails (Forge only)")
@click.option("--serialize", is_flag=True, help="Serialize requests (single-GPU)")
@click.option("--timeout", default=600, type=float, help="Backend request timeout in seconds (default: 600)")
@click.option("--verbose", "-v", is_flag=True, help="Verbose logging")
@click.option("--log-file", default=None, help="Also log to this file (for eval)")
@click.option("--manage-backend", is_flag=True,
              help="Manage the llama-server backend lifecycle: lazy load + idle-unload, with a VRAM-gate + queue. The proxy stays always-on; the GPU model loads on demand and frees VRAM when idle (shares the GPU with ComfyUI/Ollama).")
@click.option("--idle-timeout", default=90.0, type=float,
              help="Seconds idle before the managed backend unloads (frees VRAM). Default 90.")
@click.option("--queue-timeout", default=120.0, type=float,
              help="Seconds to wait for free VRAM before giving up (→ 503 → fleet L2 fallback). Default 120.")
def serve(
    backend_url: str,
    model: str,
    port: int,
    host: str,
    config_path: str | None,
    max_retries: int,
    no_rescue: bool,
    no_guardrails: bool,
    serialize: bool,
    timeout: float,
    verbose: bool,
    log_file: str | None,
    manage_backend: bool,
    idle_timeout: float,
    queue_timeout: float,
) -> None:
    """Start the coding-guardrails proxy server."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            datefmt="%H:%M:%S",
        ))
        logging.getLogger().addHandler(file_handler)
    # Silence httpx/httpcore noise
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    click.echo(f"Starting coding-guardrails proxy on {host}:{port}")
    click.echo(f"  Backend:    {backend_url}")
    click.echo(f"  Model:      {model}")
    click.echo(f"  Config:     {config_path or '(defaults)'}")
    click.echo(f"  Guardrails: {'disabled' if no_guardrails else 'enabled'}")

    try:
        asyncio.run(_run_proxy(
            backend_url=backend_url,
            model=model,
            port=port,
            host=host,
            config_path=config_path,
            max_retries=max_retries,
            rescue_enabled=not no_rescue,
            guardrails_enabled=not no_guardrails,
            serialize=serialize,
            timeout=timeout,
            manage_backend=manage_backend,
            idle_timeout=idle_timeout,
            queue_timeout=queue_timeout,
        ))
    except KeyboardInterrupt:
        click.echo("\nStopped.")


async def _run_proxy(
    backend_url: str,
    model: str,
    port: int,
    host: str,
    config_path: str | None,
    max_retries: int,
    rescue_enabled: bool,
    guardrails_enabled: bool,
    serialize: bool,
    timeout: float = 600.0,
    manage_backend: bool = False,
    idle_timeout: float = 90.0,
    queue_timeout: float = 120.0,
) -> None:
    """Async proxy startup and run loop."""
    from coding_guardrails.proxy.client import SafeLlamafileClient
    from forge.context.manager import ContextManager
    from forge.context.strategies import TieredCompact
    from coding_guardrails.proxy.server import GuardrailProxyServer
    from coding_guardrails.config import load_guardrail_config

    # ── Forge Layer 1 setup ──
    base = backend_url.rstrip("/")
    if not base.endswith("/v1"):
        base = base + "/v1"

    client = SafeLlamafileClient(
        gguf_path=model,
        base_url=base,
        mode="native",
        timeout=timeout,
        # Output cap for the backend. Must be large enough for the model to
        # emit a complete tool call (e.g. a full file in a `write` args JSON-
        # wrapped) WITHOUT mid-stream truncation. 8192 was too small for
        # verbose models (Gemma 4 26B) writing multi-KB files: the truncated
        # JSON args failed validation, the agent retried, and the same cap
        # truncated it again — an infinite loop. 16384 fits typical file
        # writes + JSON overhead and matches pi's maxTokens for these models.
        # pi can still override per-request; this is only the default.
        default_max_tokens=16384,
    )

    # Context budget: in managed mode the backend isn't up yet (lazy), so use the
    # model profile's context_tokens; otherwise auto-detect from the backend.
    if manage_backend:
        from coding_guardrails.models.profiles import get_profile
        prof = get_profile(model)
        budget = prof.context_tokens if prof else 8192
        logging.info("Context budget: %d tokens (from profile; backend lazy)", budget)
    else:
        ctx_len = await client.get_context_length()
        budget = ctx_len if ctx_len is not None else 8192
        logging.info("Context budget: %d tokens", budget)

    context_manager = ContextManager(
        strategy=TieredCompact(),
        budget_tokens=budget,
    )

    # ── Layer 2 guardrails setup ──
    if guardrails_enabled:
        guardrail_config = load_guardrail_config(config_path)
        guardrails = CodingGuardrails.from_config(guardrail_config)
        click.echo(f"  Rules:      {', '.join(r.name for r in guardrails._active_rules())}")
    else:
        guardrails = CodingGuardrails()  # No rules

    # ── Managed backend (lazy load + idle-unload, VRAM-gate + queue) ──
    backend_manager = None
    if manage_backend:
        from coding_guardrails.server.manager import BackendManager, BackendConfig
        backend_manager = BackendManager(BackendConfig(
            profile=model, idle_timeout=idle_timeout, queue_timeout=queue_timeout,
        ))
        click.echo(f"  Managed backend: lazy load + idle-unload ({idle_timeout:.0f}s), "
                   f"VRAM-queue ({queue_timeout:.0f}s → 503 → fleet L2)")

    # ── Start server ──
    server = GuardrailProxyServer(
        client=client,
        context_manager=context_manager,
        guardrails=guardrails,
        host=host,
        port=port,
        serialize_requests=serialize,
        max_retries=max_retries,
        rescue_enabled=rescue_enabled,
        model_name=model,
        backend_manager=backend_manager,
    )
    await server.start()
    click.echo(f"\n  Proxy ready at http://{host}:{port}")
    click.echo(f"  Point your agent at http://{host}:{port}/v1/chat/completions")

    # Block until interrupted
    try:
        while True:
            await asyncio.sleep(0.5)
    except asyncio.CancelledError:
        pass
    finally:
        await server.stop()


@main.command()
@click.option("--backend-url", required=True, help="URL to probe")
def probe(backend_url: str) -> None:
    """Probe model + backend compatibility."""
    import json

    click.echo(f"Probing {backend_url}...")

    try:
        import urllib.request
        base = backend_url.rstrip("/")
        if not base.endswith("/v1"):
            base = base + "/v1"

        # Check /v1/models
        resp = urllib.request.urlopen(f"{base}/models", timeout=10)
        data = json.loads(resp.read())
        models = data.get("data", [])
        if models:
            click.echo(f"  Models: {', '.join(m.get('id', '?') for m in models)}")
        else:
            click.echo("  Models: (none listed)")

        # Check /health if available
        try:
            resp = urllib.request.urlopen(f"{base.replace('/v1', '')}/health", timeout=5)
            click.echo(f"  Health: {resp.status} OK")
        except Exception:
            click.echo("  Health: (no /health endpoint)")

        # Check props
        try:
            resp = urllib.request.urlopen(f"{base.replace('/v1', '')}/props", timeout=5)
            props = json.loads(resp.read())
            ctx = props.get("default_generation_settings", {}).get("n_ctx", "?")
            click.echo(f"  Context: {ctx}")
        except Exception:
            click.echo("  Context: (couldn't detect)")

        click.echo("\n  Backend reachable")

    except Exception as exc:
        click.echo(f"  Error: {exc}", err=True)
        sys.exit(1)


@main.command("models")
def list_models() -> None:
    """Show supported model profiles."""
    from coding_guardrails.models.profiles import list_profiles

    profiles = list_profiles()
    click.echo("Supported models:\n")
    for p in profiles:
        arch = p.architecture.upper()
        swe = f"{p.swe_bench_verified}% SWE-bench" if p.swe_bench_verified else ""
        click.echo(f"  {p.name:<40s} ~{p.file_size_gb:.0f}GB  {swe}  ({arch})")

    click.echo("\nBoot command (primary):")
    primary = profiles[0] if profiles else None
    if primary:
        # Reconstruct flags with paired args
        boot_parts = []
        i = 0
        while i < len(primary.boot_flags):
            f = primary.boot_flags[i]
            if (f.startswith("--") or f.startswith("-")) and i + 1 < len(primary.boot_flags) and not primary.boot_flags[i + 1].startswith("-"):
                boot_parts.append(f"{f} {primary.boot_flags[i + 1]}")
                i += 2
            else:
                boot_parts.append(f)
                i += 1
        click.echo("  llama-server -m <model>.gguf \\")
        for part in boot_parts:
            click.echo(f"    {part} \\")


# Import and register the eval command
from coding_guardrails.eval import eval_cmd
main.add_command(eval_cmd, "eval")

# Register the server command group (cg-owned llama.cpp lifecycle)
from coding_guardrails.server.cli import server_cmd
main.add_command(server_cmd)


if __name__ == "__main__":
    main()
