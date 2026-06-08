"""``cg server`` command group — build, start, stop, status, download."""

from __future__ import annotations

import logging
import sys
import time

import click


@click.group("server")
def server_cmd() -> None:
    """Manage cg's own llama.cpp server (build, start, stop, download)."""


@server_cmd.command("version")
def _version() -> None:
    """Show the pinned, installed, and binary-reported versions."""
    from coding_guardrails.server.version import (
        PINNED_COMMIT,
        PINNED_SHORT,
        binary_version,
        installed_commit,
        is_up_to_date,
    )

    click.echo(f"Pinned commit:    {PINNED_SHORT} ({PINNED_COMMIT[:12]}...)")
    inst = installed_commit()
    click.echo(f"Checked out:      {inst[:12] if inst else '(not cloned)'}")
    if inst:
        flag = "OK" if is_up_to_date() else "MISMATCH"
        click.echo(f"  matches pin:    {flag}")
    bv = binary_version()
    click.echo(f"Binary version:   {bv or '(not built)'}")


@server_cmd.command("build")
@click.option(
    "--cpu",
    is_flag=True,
    help="Force a CPU-only build (skip CUDA even if detected).",
)
@click.option("--cuda", is_flag=True, help="Force CUDA on.")
@click.option(
    "-j",
    "--jobs",
    type=int,
    help="Parallel build jobs (default: CPU count).",
)
def _build(cpu: bool, cuda: bool, jobs: int | None) -> None:
    """Clone (if needed) and build the pinned llama.cpp."""
    from coding_guardrails.server.build import build

    if cpu and cuda:
        raise click.UsageError("--cpu and --cuda are mutually exclusive.")
    force = False if cpu else (True if cuda else None)
    try:
        path = build(force_cuda=force, jobs=jobs)
    except Exception as exc:  # noqa: BLE001 - surface as clean CLI error
        click.secho(f"Build failed: {exc}", fg="red", err=True)
        sys.exit(1)
    click.secho(f"Built: {path}", fg="green")


@server_cmd.command("start")
@click.option("-m", "--model", "model_name", required=True, help="Model profile name.")
@click.option("--ctx", type=int, help="Context window (default: profile max).")
@click.option("--ngl", type=int, default=99, show_default=True, help="GPU layers.")
@click.option("--host", default="0.0.0.0", show_default=True)
@click.option("--port", type=int, default=8080, show_default=True)
@click.option(
    "--wait/--no-wait",
    default=False,
    help="Wait for the server to respond before returning.",
)
@click.argument("extra_args", nargs=-1)
def _start(
    model_name: str,
    ctx: int | None,
    ngl: int,
    host: str,
    port: int,
    wait: bool,
    extra_args: tuple[str, ...],
) -> None:
    """Launch llama-server for a model profile.

    EXTRA_ARGS are passed verbatim to llama-server (e.g. --temp 0.7).
    """
    from coding_guardrails.server.launcher import start as launch

    try:
        pid = launch(
            model_name,
            ctx=ctx,
            ngl=ngl,
            host=host,
            port=port,
            extra=list(extra_args) if extra_args else None,
        )
    except (FileNotFoundError, RuntimeError, KeyError) as exc:
        click.secho(str(exc), fg="red", err=True)
        sys.exit(1)
    click.secho(f"Started pid {pid} on {host}:{port}", fg="green")

    if wait:
        _await_ready(host, port)


def _await_ready(host: str, port: int, timeout: float = 120.0) -> None:
    import urllib.error
    import urllib.request

    url = f"http://{host if host != '0.0.0.0' else '127.0.0.1'}:{port}/health"
    deadline = time.monotonic() + timeout
    click.echo(f"Waiting for {url} ...", nl=False)
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:  # noqa: S310
                if r.status == 200:
                    click.secho(" ready", fg="green")
                    return
        except (urllib.error.URLError, ConnectionError, OSError):
            time.sleep(1.0)
    click.secho(" timeout (server may still be loading the model)", fg="yellow")


@server_cmd.command("stop")
def _stop() -> None:
    """Stop the running llama-server."""
    from coding_guardrails.server.launcher import stop

    if stop():
        click.secho("Stopped.", fg="green")
    else:
        click.echo("Not running.")
    # show any stale leftover
    from coding_guardrails.server.launcher import is_running

    if is_running():
        click.secho("Still running after stop.", fg="yellow", err=True)


@server_cmd.command("status")
def _status() -> None:
    """Show server + binary state."""
    from coding_guardrails.server.launcher import status as snapshot

    s = snapshot()
    click.echo(f"Running:        {'yes' if s.running else 'no'}")
    if s.pid:
        click.echo(f"  pid:          {s.pid}")
    click.echo(f"Binary present: {'yes' if s.binary_present else 'no'}")
    if s.binary_version:
        click.echo(f"  version:      {s.binary_version}")
    if not s.binary_present:
        click.secho("  -> run: cg server build", fg="cyan")


@server_cmd.command("download")
@click.argument("model_name")
def _download(model_name: str) -> None:
    """Download a model GGUF into cg's own cache."""
    from coding_guardrails.server.download import download

    try:
        path = download(model_name)
    except KeyError as exc:
        click.secho(str(exc), fg="red", err=True)
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001
        click.secho(f"Download failed: {exc}", fg="red", err=True)
        sys.exit(1)
    click.secho(f"Downloaded: {path}", fg="green")


# silence the unused-import lint for logging (used once wiring grows)
_ = logging
