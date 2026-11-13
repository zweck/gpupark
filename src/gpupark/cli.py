"""gpupark command line."""

from __future__ import annotations

import contextlib
import json
import shutil
import socket
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import typer

from . import config as config_mod

app = typer.Typer(no_args_is_help=True, add_completion=False, help="Lease one GPU to many workloads.")


def _show_version(value: bool) -> None:
    if value:
        from . import __version__

        typer.echo(f"gpupark {__version__}")
        raise typer.Exit()


@app.callback()
def _root(
    version: bool = typer.Option(
        False, "--version", callback=_show_version, is_eager=True, help="Print the version and exit."
    ),
) -> None:
    """Lease one GPU to many workloads."""


# ------------------------------------------------------------------ client

def _socket_path() -> Path:
    override = None
    with contextlib.suppress(Exception):
        override = config_mod.load().socket_path
    return Path(override) if override else config_mod.default_socket_path()


def _call(payload: dict, stream: bool = False) -> Iterator[dict]:
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.connect(str(_socket_path()))
    except (FileNotFoundError, ConnectionRefusedError) as exc:
        raise typer.Exit(
            _fail(f"cannot reach the gpupark daemon ({exc}); start gpupark.service or 'gpupark daemon'")
        ) from None
    with sock.makefile("rwb") as handle:
        handle.write((json.dumps(payload) + "\n").encode())
        handle.flush()
        for line in handle:
            if not line.strip():
                continue
            event = json.loads(line)
            yield event
            if not stream:
                return


def _fail(message: str, code: int = 1) -> int:
    typer.echo(message, err=True)
    return code


# ----------------------------------------------------------------- daemon

@app.command()
def daemon() -> None:
    """Run the broker in the foreground (this is what the systemd unit runs)."""
    import asyncio

    from .daemon import Broker

    cfg = config_mod.load()
    broker = Broker(cfg)
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(broker.serve())


@app.command()
def install_service(force: bool = typer.Option(False, "--force")) -> None:
    """Install and start a systemd --user unit for the daemon (Linux)."""
    if sys.platform == "darwin":
        raise typer.Exit(
            _fail(
                "systemd is Linux-only; on macOS run 'gpupark daemon' in the foreground, "
                "or wrap it in a launchd plist"
            )
        )
    unit_dir = Path("~/.config/systemd/user").expanduser()
    unit_dir.mkdir(parents=True, exist_ok=True)
    unit = unit_dir / "gpupark.service"
    if unit.exists() and not force:
        raise typer.Exit(_fail(f"{unit} exists (pass --force to overwrite)"))
    unit.write_text(
        "[Unit]\n"
        "Description=gpupark GPU lease broker\n"
        "After=network.target\n\n"
        "[Service]\n"
        "Type=simple\n"
        f"ExecStart={sys.executable} -m gpupark.cli daemon\n"
        "Restart=on-failure\n"
        "RestartSec=2\n\n"
        "[Install]\n"
        "WantedBy=default.target\n"
    )
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    subprocess.run(["systemctl", "--user", "enable", "--now", "gpupark.service"], check=True)
    typer.echo(f"installed {unit}")


# -------------------------------------------------------------------- jobs

@app.command(context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
def run(
    ctx: typer.Context,
    need: float = typer.Option(0.0, "--need", help="VRAM the job needs, in GB"),
    priority: int = typer.Option(0, "--priority", help="higher runs first"),
    ttl: float | None = typer.Option(None, "--ttl", help="lease time limit, seconds"),
    quiet: bool = typer.Option(False, "--quiet", help="only stream the job's own output"),
) -> None:
    """Queue a command; it runs when the GPU is free."""
    argv: list[str] = [a for a in ctx.args if a != "--"]
    if not argv:
        raise typer.BadParameter("give a command after '--'")
    code = 0
    for event in _call(
        {"op": "submit", "argv": argv, "need_gb": need, "priority": priority, "ttl": ttl}, stream=True
    ):
        kind = event.get("ev")
        if kind == "output":
            typer.echo(event.get("line", ""))
        elif kind == "queued" and not quiet:
            typer.echo(f"queued {event['id']} (need {need:.1f} GB, priority {priority})", err=True)
        elif kind == "granted" and not quiet:
            typer.echo("granted", err=True)
        elif kind in ("done", "failed"):
            code = int(event.get("exit_code") or 0)
            if kind == "failed" and event.get("error"):
                typer.echo(event["error"], err=True)
                code = code or 1
            break
        elif kind in ("cancelled",):
            code = 130
            break
        elif "error" in event:
            raise typer.Exit(_fail(event["error"]))
    raise typer.Exit(code)


@app.command()
def lease(
    workload: str = typer.Argument(..., help="a workload name from workloads.yaml"),
    ttl: float | None = typer.Option(None, "--for", help="hold for N seconds, then release"),
) -> None:
    """Start a registered workload and hold the GPU until this command is interrupted."""
    for event in _call({"op": "submit", "workload": workload, "ttl": ttl, "hold": True}, stream=True):
        kind = event.get("ev")
        if kind == "queued":
            typer.echo(f"queued {event['id']}", err=True)
        elif kind == "granted":
            typer.echo(f"{workload} holds the GPU; Ctrl-C to release", err=True)
        elif kind in ("done", "failed", "cancelled"):
            break
        elif "error" in event:
            raise typer.Exit(_fail(event["error"]))


@app.command()
def cancel(job_id: str) -> None:
    """Cancel a queued job or stop a running one."""
    for event in _call({"op": "cancel", "id": job_id}):
        typer.echo("cancelled" if event.get("cancelled") else f"no such live job: {job_id}")
        raise typer.Exit(0 if event.get("cancelled") else 1)


def _current_job_id() -> str | None:
    for snapshot in _call({"op": "status"}):
        running = snapshot.get("running")
        if running:
            return running["id"]
        paused = snapshot.get("paused") or []
        if paused:
            return paused[0]["id"]
    return None


@app.command()
def pause(job_id: str = typer.Argument(None, help="job id (default: the current lease)")) -> None:
    """Give the card back temporarily; keep your place in the queue.

    A systemd workload is stopped so the baseline can run again. A command job is
    suspended, which does NOT free its VRAM - only a workload that can be stopped
    gives the card back.
    """
    target = job_id or _current_job_id()
    if not target:
        raise typer.Exit(_fail("nothing is running to pause"))
    for reply in _call({"op": "pause", "id": target}):
        if not reply.get("ok"):
            raise typer.Exit(_fail(reply.get("error", "pause failed")))
        typer.echo(f"paused {target} ({reply.get('state')}): {reply.get('note', '')}")


@app.command()
def resume(job_id: str = typer.Argument(None, help="job id (default: the paused lease)")) -> None:
    """Undo a pause: re-grant the lease (or return the job to the queue)."""
    target = job_id or _current_job_id()
    if not target:
        raise typer.Exit(_fail("no paused job to resume"))
    for reply in _call({"op": "resume", "id": target}):
        if not reply.get("ok"):
            raise typer.Exit(_fail(reply.get("error", "resume failed")))
        typer.echo(f"resumed {target} ({reply.get('state')}): {reply.get('note', '')}")


# ----------------------------------------------------------------- config

config_app = typer.Typer(help="Config file helpers.")
app.add_typer(config_app, name="config")


@config_app.command("init")
def config_init(
    force: bool = typer.Option(False, "--force", help="overwrite an existing file"),
    path: str | None = typer.Option(None, "--path", help="write somewhere other than the default"),
) -> None:
    """Write a starter workloads.yaml."""
    target = Path(path).expanduser() if path else config_mod.default_config_path()
    if target.exists() and not force:
        raise typer.Exit(_fail(f"{target} already exists (--force to overwrite)"))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(config_mod.EXAMPLE_YAML)
    typer.echo(f"wrote {target}")
    typer.echo("edit it, then: gpupark doctor && gpupark install-service")


# ------------------------------------------------------------------ views

@app.command()
def status(as_json: bool = typer.Option(False, "--json")) -> None:
    """Show the lease, the queue, and the GPU."""
    for snapshot in _call({"op": "status"}):
        if as_json:
            typer.echo(json.dumps(snapshot, indent=2))
            return
        running = snapshot.get("running")
        gpu_view = snapshot.get("gpu", {})
        meta = snapshot.get("baseline", {})
        if "free_mib" in gpu_view:
            typer.echo(
                f"gpu {gpu_view['index']}: {gpu_view['used_mib'] / 1024:.1f} GB used of "
                f"{gpu_view['total_mib'] / 1024:.1f} GB ({gpu_view['free_mib'] / 1024:.1f} GB free)"
            )
        else:
            typer.echo(f"gpu {gpu_view.get('index')}: {gpu_view.get('error', 'unknown')}")
        typer.echo(f"baseline: {meta.get('name') or '-'} ({meta.get('state')})")
        if running:
            what = running.get("workload") or " ".join(running.get("argv") or [])
            typer.echo(f"lease: {running['id']} {what} (need {running['need_gb']:.1f} GB)")
        else:
            typer.echo("lease: free")
        queue = snapshot.get("queue") or []
        typer.echo(f"queue: {len(queue)}")
        for job in queue:
            what = job.get("workload") or " ".join(job.get("argv") or [])
            typer.echo(f"  {job['id']}  {what}  (need {job['need_gb']:.1f} GB)")
        for job in snapshot.get("paused") or []:
            what = job.get("workload") or " ".join(job.get("argv") or [])
            typer.echo(f"  paused: {job['id']}  {what}  (need {job['need_gb']:.1f} GB) — 'gpupark resume'")


@app.command()
def workloads() -> None:
    """List the configured workloads."""
    cfg = config_mod.load()
    for name in sorted(cfg.workloads):
        item = cfg.workloads[name]
        target = item.target if isinstance(item.target, str) else " ".join(item.argv())
        flags = []
        if item.baseline:
            flags.append("baseline")
        if item.vram_gb:
            flags.append(f"{item.vram_gb:g} GB")
        typer.echo(f"{name:<16} {item.kind:<8} {target}  {', '.join(flags)}")


@app.command()
def doctor() -> None:
    """Check the things a lease depends on."""
    problems = 0
    cfg_path = config_mod.default_config_path()
    probe = "auto"
    if Path(cfg_path).exists():
        typer.echo(f"ok    config {cfg_path}")
        try:
            probe = config_mod.load().gpu_probe
        except Exception as exc:
            problems += 1
            typer.echo(f"FAIL  config unreadable: {exc}")
    else:
        problems += 1
        typer.echo(f"FAIL  config missing: {cfg_path} (run 'gpupark config init')")
    if probe == "none":
        typer.echo("ok    gpu probe disabled (gpu_probe: none) — the queue alone serialises work")
    elif shutil.which("nvidia-smi"):
        try:
            from . import gpu

            total, used = gpu.memory(0)
            typer.echo(f"ok    nvidia-smi: {used} / {total} MiB used")
        except Exception as exc:
            problems += 1
            typer.echo(f"FAIL  nvidia-smi: {exc}")
    else:
        problems += 1
        typer.echo("FAIL  nvidia-smi not on PATH")
    if shutil.which("systemctl"):
        result = subprocess.run(
            ["systemctl", "--user", "is-system-running"], capture_output=True, text=True, check=False
        )
        state = (result.stdout or result.stderr).strip() or "unknown"
        typer.echo(f"ok    systemd --user: {state}")
    else:
        problems += 1
        typer.echo("FAIL  systemctl not on PATH")
    try:
        for event in _call({"op": "ping"}):
            typer.echo("ok    daemon reachable" if event.get("pong") else "FAIL  daemon: bad reply")
    except typer.Exit:
        raise
    except Exception as exc:
        problems += 1
        typer.echo(f"FAIL  daemon unreachable: {exc}")
    raise typer.Exit(1 if problems else 0)


if __name__ == "__main__":
    app()
