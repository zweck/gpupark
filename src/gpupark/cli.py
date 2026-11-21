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


def _dur(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"


def _clock(job: dict) -> str:
    """', up 12m, ttl 28m left' — the clocks an agent would otherwise have to infer."""
    bits = []
    active = job.get("active_seconds")
    if active is not None:
        bits.append(f"up {_dur(active)}")
    left = job.get("ttl_remaining")
    if left is not None:
        bits.append(f"ttl {_dur(left)} left")
    elif job.get("started_at"):
        bits.append("no ttl")
    return (", " + ", ".join(bits)) if bits else ""


def _state_dir() -> Path:
    with contextlib.suppress(Exception):
        return config_mod.load().artifacts_dir()
    return config_mod.default_state_path().parent


def _tail(path: Path, lines: int) -> list[str]:
    """Last `lines` lines without reading a whole (possibly huge) log into memory."""
    if lines <= 0:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()
    block = 64 * 1024
    data = b""
    with path.open("rb") as handle:
        handle.seek(0, 2)
        position = handle.tell()
        while position > 0 and data.count(b"\n") <= lines:
            step = min(block, position)
            position -= step
            handle.seek(position)
            data = handle.read(step) + data
    return data.decode("utf-8", errors="replace").splitlines()[-lines:]


def _stream(events, *, quiet: bool = False, label: str = "run", grant_message: str | None = None) -> int:
    """Drive a submitted job's event stream and return its exit code.

    Shared by `run` and `lease`. `failed` carries the broker's own exit code (124 for a TTL
    expiry, 1 for a refusal) and a failure with no code still exits non-zero — the reason a
    `gpupark lease --for` that hit its deadline must not look like a success.
    """
    code = 0
    for event in events:
        kind = event.get("ev")
        if kind == "output":
            typer.echo(event.get("line", ""))
        elif kind == "queued" and not quiet:
            typer.echo(f"queued {event['id']} (need {float(event.get('need_gb') or 0):.1f} GB)", err=True)
        elif kind == "admitting" and not quiet:
            typer.echo("preparing (parking the baseline)", err=True)
        elif kind == "granted" and not quiet:
            typer.echo(grant_message or "granted", err=True)
        elif kind == "foreign" and not quiet:
            procs = event.get("processes") or []
            who = ", ".join(
                f"pid {item['pid']} {item['name']} {item['used_mib'] / 1024:.1f} GB" for item in procs
            )
            if event.get("blocked"):
                typer.echo(f"refusing to grant: foreign GPU use — {who}", err=True)
            elif event.get("unknown"):
                typer.echo("warning: GPU attribution unavailable; admission skipped", err=True)
            elif procs:
                typer.echo(f"warning: foreign GPU use — {who}", err=True)
        elif kind == "done":
            return int(event.get("exit_code") or 0)
        elif kind == "failed":
            code = int(event.get("exit_code") or 0) or 1
            typer.echo(event.get("error") or f"the {label} failed (exit {code})", err=True)
            return code
        elif kind == "cancelled":
            return 130
        elif "error" in event:
            typer.echo(str(event["error"]), err=True)
            return 1
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
    # Click drops every '--' it parses, so a job's own '--' would vanish (Godot's editor bake is
    # `godot --editor -- --lm-bake=x`). Rebuild the job argv from the real command line instead:
    # everything after the FIRST '--' belongs to the job. Requiring that separator is deliberate —
    # without it, a job's own '--' would be mistaken for ours and the command silently truncated.
    _raw = sys.argv[1:]
    if "--" not in _raw:
        raise typer.BadParameter(
            "put the job command after '--' (gpupark run --need 8 -- my-job --flag), so gpupark "
            "can tell its own flags from the job's"
        )
    argv = _raw[_raw.index("--") + 1:]
    if not argv:
        raise typer.BadParameter("give a command after '--'")
    # `run` takes no positional arguments of its own, so Click's leftover args must be exactly the
    # job command. When they differ, the `--` we split on was the JOB's own separator and the command
    # would be silently truncated — refuse it instead.
    if [token for token in argv if token != "--"] != list(ctx.args):
        raise typer.BadParameter(
            "gpupark run takes no positional arguments before '--': put the whole job command "
            "after it (gpupark run --need 8 -- my-job --flag)"
        )
    try:
        from click.core import ParameterSource

        need_given = ctx.get_parameter_source("need") is ParameterSource.COMMANDLINE
    except Exception:
        head = _raw[:_raw.index("--")]
        need_given = any(item == "--need" or item.startswith("--need=") for item in head)
    if not need_given and not quiet:
        typer.echo(
            "warning: no --need given — the baseline is not parked and the job runs beside the "
            "resident model; pass --need <GB> for GPU work",
            err=True,
        )
    raise typer.Exit(
        _stream(
            _call(
                {
                    "op": "submit",
                    "argv": argv,
                    "need_gb": need,
                    "priority": priority,
                    "ttl": ttl,
                },
                stream=True,
            ),
            quiet=quiet,
        )
    )


@app.command()
def lease(
    workload: str = typer.Argument(..., help="a workload name from workloads.yaml"),
    ttl: float | None = typer.Option(None, "--for", help="hold for N seconds, then release"),
) -> None:
    """Start a registered workload and hold the GPU until this command is interrupted."""
    raise typer.Exit(
        _stream(
            _call({"op": "submit", "workload": workload, "ttl": ttl, "hold": True}, stream=True),
            label="lease",
            grant_message=f"{workload} holds the GPU; Ctrl-C to release",
        )
    )


@app.command()
def cancel(job_id: str) -> None:
    """Cancel a queued job, or stop a running one (an id prefix is enough when it is unique)."""
    for event in _call({"op": "cancel", "id": job_id}):
        if event.get("cancelled"):
            typer.echo("cancelled")
            raise typer.Exit(0)
        raise typer.Exit(_fail(event.get("error") or f"no live job matching {job_id!r}"))


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
            if running.get("state") == "admitting":
                typer.echo(
                    f"lease: {running['id']} {what} (need {running['need_gb']:.1f} GB, preparing)"
                )
            else:
                pid = running.get("pid")
                seats = f", pid {pid}" if pid else ""
                typer.echo(
                    f"lease: {running['id']} {what} (need {running['need_gb']:.1f} GB"
                    f"{_clock(running)}{seats})"
                )
        else:
            typer.echo("lease: free")
        foreign = snapshot.get("foreign") or []
        if foreign:
            who = ", ".join(
                f"pid {p['pid']} {p['name']} {p['used_mib'] / 1024:.1f} GB" for p in foreign
            )
            typer.echo(f"foreign: {who}")
        elif snapshot.get("foreign_unknown"):
            typer.echo("foreign: unknown (probe or systemd could not attribute GPU memory)")
        queue = snapshot.get("queue") or []
        typer.echo(f"queue: {len(queue)}")
        for job in queue:
            what = job.get("workload") or " ".join(job.get("argv") or [])
            waiting = job.get("queued_for")
            waited = f", waiting {_dur(waiting)}" if waiting is not None else ""
            typer.echo(
                f"  {job.get('position', '?')}. {job['id']}  {what}  "
                f"(need {job['need_gb']:.1f} GB{waited})"
            )
        for job in snapshot.get("paused") or []:
            what = job.get("workload") or " ".join(job.get("argv") or [])
            typer.echo(f"  paused: {job['id']}  {what}  (need {job['need_gb']:.1f} GB) — 'gpupark resume'")


@app.command()
def history(
    as_json: bool = typer.Option(False, "--json"),
    limit: int = typer.Option(20, "--limit", help="how many of the most recent jobs to show"),
) -> None:
    """Show recently finished jobs: exit code, how long they waited and ran."""
    for snapshot in _call({"op": "history"}):
        rows = list(snapshot.get("recent") or [])[-limit:] if limit > 0 else []
        if as_json:
            typer.echo(json.dumps(rows, indent=2))
            return
        if not rows:
            typer.echo("no finished jobs yet")
            return
        for row in rows:
            what = row.get("workload") or " ".join(row.get("argv") or [])
            bits = []
            if row.get("exit_code") is not None:
                bits.append(f"exit {row['exit_code']}")
            if row.get("wait_seconds") is not None:
                bits.append(f"queued {_dur(row['wait_seconds'])}")
            if row.get("run_seconds") is not None:
                bits.append(f"ran {_dur(row['run_seconds'])}")
            typer.echo(f"{row['id']}  {str(row.get('state')):<9} {what}  ({', '.join(bits)})")
            if row.get("error"):
                typer.echo(f"    {row['error']}")


@app.command()
def logs(
    job_id: str = typer.Argument(..., help="job id, or a unique prefix"),
    lines: int = typer.Option(200, "--lines", "-n", help="how many trailing lines to print"),
    path_only: bool = typer.Option(False, "--path", help="print the log path instead of its text"),
) -> None:
    """Print a job's own output log, so a lost client does not lose the job's output."""
    token = job_id.strip().lower()
    if not token or any(bad in token for bad in ("/", "\\", "*", "?", "[", "]", "..")):
        raise typer.Exit(_fail(f"{job_id!r} is not a job id (or a unique prefix)"))
    log_dir = _state_dir() / "logs"
    matches = (
        sorted(item for item in log_dir.iterdir() if item.suffix == ".log" and item.stem.startswith(token))
        if log_dir.is_dir()
        else []
    )
    if not matches:
        raise typer.Exit(_fail(f"no log for job {job_id!r} in {log_dir}"))
    if len(matches) > 1:
        names = ", ".join(item.stem for item in matches[:6])
        raise typer.Exit(
            _fail(f"{job_id!r} matches {len(matches)} logs ({names}); give more characters")
        )
    target = matches[0]
    if path_only:
        typer.echo(str(target))
        return
    for line in _tail(target, lines):
        typer.echo(line)


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
