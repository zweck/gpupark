"""The broker: one queue, one lease, one baseline that yields the card."""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import json
import os
import signal
import socket
import stat as stat_mod
import struct
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import config as config_mod
from . import gpu, systemd

EVENTS_MAX = 4096  # bounded per-job buffer: a client that stops reading cannot OOM the broker


@dataclass
class Job:
    id: str
    need_gb: float
    priority: int
    submitted_at: float
    argv: list[str] | None = None
    workload_name: str | None = None
    ttl: float | None = None
    hold: bool = False
    state: str = "queued"  # queued|paused|granted|running|done|failed|cancelled
    exit_code: int | None = None
    error: str | None = None
    started_at: float | None = None
    ended_at: float | None = None
    paused_since: float | None = None  # set while a running job is paused
    paused_total: float = 0.0  # paused time that does not count against the TTL
    pause_request: bool = False
    suspended: bool = False  # command jobs: SIGSTOP sent
    proc: Any = None  # asyncio.subprocess.Process for command jobs
    pgid: int | None = None  # process group captured at spawn (never re-derived after exit)
    dropped: int = 0  # output lines dropped because nobody was reading
    events: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=EVENTS_MAX))
    cancel: asyncio.Event = field(default_factory=asyncio.Event)

    def emit(self, event: str, **fields: Any) -> None:
        try:
            self.events.put_nowait({"ev": event, "id": self.id, **fields})
        except asyncio.QueueFull:
            # lifecycle events matter, output lines do not: drop the line and count it
            self.dropped += 1


    def active_seconds(self, now: float | None = None) -> float:
        """Wall time this job has been running, excluding time spent paused."""
        now = time.time() if now is None else now
        paused = self.paused_total
        if self.paused_since is not None:
            paused += now - self.paused_since
        return max(0.0, now - (self.started_at or now) - paused)

    def public(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "state": self.state,
            "need_gb": self.need_gb,
            "priority": self.priority,
            "workload": self.workload_name,
            "argv": self.argv,
            "submitted_at": self.submitted_at,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "exit_code": self.exit_code,
            "error": self.error,
        }


class NoProbe(RuntimeError):
    """gpu_probe is "none": this box has no nvidia-smi to ask."""


class Host:
    """Every side effect the broker performs, in one place (fakeable in tests)."""

    def __init__(self, probe: str = "auto") -> None:
        self.probe = probe

    def gpu_memory(self, index: int) -> tuple[int, int]:
        if self.probe == "none":
            raise NoProbe("gpu probe disabled (gpu_probe: none)")
        return gpu.memory(index)

    def is_active(self, unit: str) -> bool:
        return systemd.is_active(unit)

    def start_unit(self, unit: str) -> None:
        systemd.start(unit)

    def stop_unit(self, unit: str) -> None:
        systemd.stop(unit)

    def wait_active(self, unit: str, timeout: float) -> bool:
        return systemd.wait_active(unit, timeout)

    def wait_inactive(self, unit: str, timeout: float) -> bool:
        return systemd.wait_inactive(unit, timeout)


class Broker:
    def __init__(self, cfg: config_mod.Config, host: Host | None = None) -> None:
        self.cfg = cfg
        self.host = host or Host(cfg.gpu_probe)
        self.jobs: dict[str, Job] = {}
        self.order: list[str] = []
        self.running: Job | None = None
        self.baseline_state = "unknown"
        self.scheduler_alive = False
        self.last_error: str | None = None
        self._wake = asyncio.Event()
        self._stopping = False
        self._exit = asyncio.Event()
        self._server: asyncio.AbstractServer | None = None
        self._lock_fd: int | None = None

    # ---------------------------------------------------------------- queue

    def submit(
        self,
        *,
        argv: list[str] | None = None,
        workload_name: str | None = None,
        need_gb: float | None = None,
        priority: int = 0,
        ttl: float | None = None,
        hold: bool = False,
    ) -> Job:
        workload = self.cfg.workload(workload_name) if workload_name else None
        if argv is None and workload is None:
            raise ValueError("a job needs either a workload or an argv")
        if need_gb is None:
            need_gb = workload.vram_gb if workload and workload.vram_gb is not None else 0.0
        if priority == 0 and workload is not None:
            priority = workload.priority
        job = Job(
            id=uuid.uuid4().hex[:8],
            need_gb=float(need_gb),
            priority=int(priority),
            submitted_at=time.time(),
            argv=argv,
            workload_name=workload.name if workload else None,
            ttl=ttl,
            hold=hold,
        )
        self.jobs[job.id] = job
        self.order.append(job.id)
        job.emit("queued", position=self._position(job))
        self._wake.set()
        return job

    def _position(self, job: Job) -> int:
        pending = [j for j in self.order if self.jobs[j].state == "queued"]
        try:
            return pending.index(job.id) + 1
        except ValueError:
            return 0

    def cancel(self, job_id: str) -> bool:
        job = self.jobs.get(job_id)
        if job is None:
            return False
        if job.state in ("done", "failed", "cancelled"):
            return False
        job.cancel.set()
        if job.state == "queued":
            job.state = "cancelled"
            job.ended_at = time.time()
            job.emit("cancelled")
        self._wake.set()
        return True

    def pause(self, job_id: str) -> dict[str, Any]:
        """Give the card back without losing your place in the queue.

        A queued job is held. A running systemd workload is stopped and the
        baseline comes back; a running command is SIGSTOPped, which does not free
        its VRAM (a suspended CUDA process keeps its allocations).
        """
        job = self.jobs.get(job_id)
        if job is None:
            return {"ok": False, "error": f"no such job {job_id!r}"}
        if job.state == "queued":
            job.state = "paused"
            return {"ok": True, "state": "paused", "note": "held in the queue; not granted until resumed"}
        if job.state in ("granted", "running") and not job.pause_request:
            job.pause_request = True
            job.paused_since = time.time()
            note = (
                "the process is suspended; its VRAM stays allocated, so the baseline cannot return"
                if job.workload_name is None
                else "stopping the workload and restoring the baseline"
            )
            job.emit("paused")
            return {"ok": True, "state": job.state, "note": note}
        return {"ok": False, "error": f"cannot pause a job that is {job.state}"}

    def resume(self, job_id: str) -> dict[str, Any]:
        job = self.jobs.get(job_id)
        if job is None:
            return {"ok": False, "error": f"no such job {job_id!r}"}
        if job.state == "paused":
            job.state = "queued"
            self._wake.set()
            return {"ok": True, "state": "queued", "note": "back in the queue at the same priority"}
        if job.pause_request and job.paused_since is not None:
            job.pause_request = False
            job.paused_total += time.time() - job.paused_since
            job.paused_since = None
            job.emit("resumed")
            return {"ok": True, "state": job.state, "note": "resuming"}
        return {"ok": False, "error": f"job {job_id} is not paused (state {job.state})"}

    def _forget(self, job: Job) -> None:
        """Drop a finished job and its buffered events: a long-lived broker must not grow."""
        self.jobs.pop(job.id, None)
        with contextlib.suppress(ValueError):
            self.order.remove(job.id)

    def _next_job(self) -> Job | None:
        pending = [self.jobs[j] for j in self.order if self.jobs[j].state == "queued"]
        if not pending:
            return None
        pending.sort(key=lambda j: (-j.priority, j.submitted_at))
        return pending[0]

    async def status(self) -> dict[str, Any]:
        try:
            # nvidia-smi can take seconds; never block the event loop for it
            total, used = await asyncio.to_thread(self.host.gpu_memory, self.cfg.gpu)
            gpu_view: dict[str, Any] = {
                "index": self.cfg.gpu,
                "total_mib": total,
                "used_mib": used,
                "free_mib": max(0, total - used),
            }
        except Exception as exc:  # nvidia-smi may be missing on a test box
            gpu_view = {"index": self.cfg.gpu, "error": str(exc)}
        baseline = self.cfg.baseline()
        return {
            "running": self.running.public() if self.running else None,
            "queue": [self.jobs[j].public() for j in self.order if self.jobs[j].state == "queued"],
            "paused": [
                self.jobs[j].public() for j in self.order if self.jobs[j].state == "paused"
            ]
            + ([self.running.public()] if self.running and self.running.pause_request else []),
            "baseline": {
                "name": baseline.name if baseline else None,
                "state": self.baseline_state,
            },
            "gpu": gpu_view,
            "scheduler": {"alive": self.scheduler_alive, "last_error": self.last_error},
        }

    # ------------------------------------------------------------ scheduling

    def _note_error(self, exc: BaseException) -> None:
        """Remember the last host-layer problem; status exposes it, the loop survives it."""
        self.last_error = f"{time.strftime('%Y-%m-%d %H:%M:%S')}: {exc}"
        print(f"gpupark: {exc}", flush=True)

    async def _scheduler(self) -> None:
        """Run the queue forever. A host-layer hiccup must never kill the loop."""
        self.scheduler_alive = True
        try:
            try:
                await self._restore_baseline()
            except Exception as exc:  # noqa: BLE001 - the loop must outlive its first restore
                self._note_error(exc)
            while not self._stopping:
                try:
                    job = self._next_job()
                    if job is None:
                        self._wake.clear()
                        with contextlib.suppress(asyncio.TimeoutError):
                            await asyncio.wait_for(self._wake.wait(), timeout=1.0)
                        continue
                    await self._run(job)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001 - systemctl timeout, stale pid, ...
                    self._note_error(exc)
                    await asyncio.sleep(1.0)
        finally:
            self.scheduler_alive = False

    async def _restore_baseline(self) -> None:
        baseline = self.cfg.baseline()
        if baseline is None:
            return
        if baseline.kind == "command":
            # a command baseline cannot be meaningfully kept by the broker yet
            self.baseline_state = "unsupported"
            return
        try:
            active = await asyncio.to_thread(self.host.is_active, str(baseline.target))
            if not active:
                await asyncio.to_thread(self.host.start_unit, str(baseline.target))
        except Exception as exc:
            self.baseline_state = f"error: {exc}"
            self._note_error(exc)
            return
        self.baseline_state = "running"

    async def _park_baseline(self) -> None:
        baseline = self.cfg.baseline()
        if baseline is None or baseline.kind != "unit":
            return
        unit = str(baseline.target)
        try:
            active = await asyncio.to_thread(self.host.is_active, unit)
            if active:
                await asyncio.to_thread(self.host.stop_unit, unit)
                await asyncio.to_thread(self.host.wait_inactive, unit, baseline.stop_timeout)
        except Exception as exc:
            self.baseline_state = f"error: {exc}"
            self._note_error(exc)
            return
        self.baseline_state = "parked"

    async def _wait_free(self, need_gb: float, timeout: float) -> bool:
        if self.cfg.gpu_probe == "none":
            return True  # nothing to measure; the queue alone serialises work
        need_mib = int(need_gb * 1024)

        def check() -> bool:
            total, used = self.host.gpu_memory(self.cfg.gpu)
            return (total - used) >= need_mib

        deadline = time.monotonic() + timeout
        while True:
            if await asyncio.to_thread(check):
                return True
            if time.monotonic() >= deadline:
                return False
            await asyncio.sleep(self.cfg.poll_seconds)

    async def _run(self, job: Job) -> None:
        job.state = "granted"
        job.started_at = time.time()
        self.running = job
        job.emit("granted")
        workload = self.cfg.workload(job.workload_name) if job.workload_name else None
        try:
            if job.need_gb > 0:
                await self._park_baseline()
                if not await self._wait_free(job.need_gb, timeout=job.ttl or self.cfg.grant_timeout):
                    raise RuntimeError(
                        f"the GPU never freed {job.need_gb:.1f} GB (still in use by something "
                        f"gpupark does not manage?)"
                    )
            job.state = "running"
            job.emit("running")
            code = await self._execute(job, workload)
            job.exit_code = code
            job.state = "done" if code == 0 else "failed"
        except asyncio.CancelledError:
            job.state = "cancelled"
        except Exception as exc:
            job.state = "failed"
            job.error = str(exc)
        finally:
            job.ended_at = time.time()
            self.running = None
            if job.state in ("done", "failed"):
                job.emit(job.state, exit_code=job.exit_code, error=job.error)
            else:
                job.emit("cancelled")
            self._forget(job)
            if workload is not None and workload.kind == "unit" and not job.hold:
                with contextlib.suppress(Exception):
                    await asyncio.to_thread(self.host.stop_unit, str(workload.target))
            await asyncio.sleep(self.cfg.cooldown_seconds)
            if self._next_job() is None:
                await self._restore_baseline()

    async def _execute(self, job: Job, workload: config_mod.Workload | None) -> int:
        if workload is not None and workload.kind == "unit":
            unit = str(workload.target)
            await asyncio.to_thread(self.host.start_unit, unit)
            if not await asyncio.to_thread(self.host.wait_active, unit, workload.ready_timeout):
                raise RuntimeError(f"{unit} did not become active within {workload.ready_timeout:.0f}s")
            while True:
                if job.cancel.is_set():
                    await asyncio.to_thread(self.host.stop_unit, unit)
                    raise asyncio.CancelledError()
                if job.pause_request and not job.suspended:
                    # give the card back: stop the workload, bring the baseline home
                    await asyncio.to_thread(self.host.stop_unit, unit)
                    await asyncio.to_thread(self.host.wait_inactive, unit, workload.stop_timeout)
                    job.suspended = True
                    await self._restore_baseline()
                    continue
                if not job.pause_request and job.suspended:
                    await self._park_baseline()
                    if not await self._wait_free(job.need_gb, timeout=self.cfg.grant_timeout):
                        raise RuntimeError(f"the GPU did not come back for {job.need_gb:.1f} GB after resume")
                    await asyncio.to_thread(self.host.start_unit, unit)
                    await asyncio.to_thread(self.host.wait_active, unit, workload.ready_timeout)
                    job.suspended = False
                    continue
                if job.suspended:
                    # paused: a stopped unit is expected, not the job finishing
                    await asyncio.sleep(0.5)
                    continue
                if not await asyncio.to_thread(self.host.is_active, unit):
                    return 0
                if job.ttl and job.active_seconds() >= job.ttl:
                    await asyncio.to_thread(self.host.stop_unit, unit)
                    return 0
                await asyncio.sleep(1.0)
        argv = job.argv or (workload.argv() if workload else [])
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env={**os.environ, "GPUPARK_JOB_ID": job.id},
            start_new_session=True,  # own process group, so pause reaches the whole tree
        )
        job.proc = proc
        with contextlib.suppress(ProcessLookupError):
            job.pgid = os.getpgid(proc.pid)

        async def pump() -> None:
            assert proc.stdout is not None
            async for line in proc.stdout:
                job.emit("output", line=line.decode(errors="replace").rstrip("\n"))

        async def watch_pause() -> None:
            """SIGSTOP/SIGCONT the job's process group. VRAM stays allocated."""
            while proc.returncode is None and job.pgid is not None:
                try:
                    if job.pause_request and not job.suspended:
                        os.killpg(job.pgid, signal.SIGSTOP)
                        job.suspended = True
                    elif not job.pause_request and job.suspended:
                        os.killpg(job.pgid, signal.SIGCONT)
                        job.suspended = False
                except (ProcessLookupError, PermissionError):
                    return
                await asyncio.sleep(0.2)

        async def watch_ttl() -> None:
            ttl = job.ttl or 0.0
            while proc.returncode is None and job.active_seconds() < ttl:
                await asyncio.sleep(0.5)

        pump_task = asyncio.create_task(pump())
        wait_task = asyncio.create_task(proc.wait())
        cancel_task = asyncio.create_task(job.cancel.wait())
        pause_task = asyncio.create_task(watch_pause())
        ttl_task = asyncio.create_task(watch_ttl()) if job.ttl else None

        async def finish_pump() -> None:
            """Let the reader drain, but never wait forever on a pipe a grandchild holds open."""
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(asyncio.shield(pump_task), timeout=2.0)
            pump_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await pump_task

        waiting = {wait_task, cancel_task} | ({ttl_task} if ttl_task else set())
        done, _ = await asyncio.wait(waiting, return_when=asyncio.FIRST_COMPLETED)
        pause_task.cancel()
        if ttl_task:
            ttl_task.cancel()
        if cancel_task in done and not wait_task.done():
            await self._stop_job(job, proc, signal.SIGTERM)
            await finish_pump()
            raise asyncio.CancelledError()
        if ttl_task is not None and ttl_task in done and not wait_task.done():
            # lease expired: stop the job tree, and say so instead of reporting success
            job.error = f"lease expired after {job.ttl:.0f}s (--ttl); the job was terminated"
            await self._stop_job(job, proc, signal.SIGTERM)
            await finish_pump()
            return 124
        await finish_pump()
        return proc.returncode or 0

    async def _stop_job(self, job: Job, proc: Any, sig: int) -> None:
        """Stop the whole process group (not just the direct child), then make sure it is gone."""

        def signal_group(how: int) -> None:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                if job.pgid is not None:
                    os.killpg(job.pgid, how)
                else:
                    proc.send_signal(how)

        if job.suspended:  # SIGTERM is not delivered to a stopped process
            signal_group(signal.SIGCONT)
            job.suspended = False
        signal_group(sig)
        with contextlib.suppress(Exception):
            await asyncio.wait_for(proc.wait(), timeout=10)
        if proc.returncode is None:
            signal_group(signal.SIGKILL)
            with contextlib.suppress(Exception):
                await asyncio.wait_for(proc.wait(), timeout=5)

    # -------------------------------------------------------------- service

    @staticmethod
    def _peer(writer: asyncio.StreamWriter) -> str:
        """Who is on the other end (uid/pid via SO_PEERCRED), for the request log."""
        try:
            sock = writer.get_extra_info("socket")
            creds = sock.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i"))
            pid, uid, _gid = struct.unpack("3i", creds)
            return f"uid={uid} pid={pid}"
        except Exception:
            return "uid=?"

    @staticmethod
    def _check_request(request: dict[str, Any]) -> None:
        """Reject wire types the broker cannot honour (a string ttl used to kill jobs)."""
        argv = request.get("argv")
        if argv is not None and (
            not isinstance(argv, list) or not all(isinstance(part, str) for part in argv)
        ):
            raise ValueError("argv must be a list of strings")
        for key in ("ttl", "need_gb"):
            value = request.get(key)
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
                raise ValueError(f"{key} must be a number")
        hold = request.get("hold", False)
        if not isinstance(hold, bool):
            raise ValueError("hold must be true or false")
        priority = request.get("priority", 0)
        if priority is not None and (isinstance(priority, bool) or not isinstance(priority, int)):
            raise ValueError("priority must be an integer")

    def _log(self, writer: asyncio.StreamWriter, op: str, request: dict[str, Any]) -> None:
        """One line per request, with the peer's uid: the broker keeps no other audit trail."""
        detail = (
            request.get("workload")
            or " ".join(request.get("argv") or [])
            or request.get("id")
            or "-"
        )
        print(f"gpupark: {self._peer(writer)} {op} {str(detail)[:160]}", flush=True)

    async def _watch_client(self, reader: asyncio.StreamReader, job: Job) -> None:
        """A lease whose client hangs up (Ctrl-C) is released instead of leaking the card."""
        try:
            while await reader.read(1024):
                pass
        except Exception:
            pass
        if job.state not in ("done", "failed", "cancelled"):
            self.cancel(job.id)

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        watcher: asyncio.Task | None = None
        try:
            line = await asyncio.wait_for(reader.readline(), timeout=30)
            request = json.loads(line or b"{}")
            if not isinstance(request, dict):
                raise ValueError("request must be a JSON object")
            op = str(request.get("op", "status"))
            self._log(writer, op, request)
            if op == "status":
                await self._reply(writer, await self.status())
            elif op == "cancel":
                await self._reply(writer, {"cancelled": self.cancel(str(request.get("id", "")))})
            elif op == "pause":
                await self._reply(writer, self.pause(str(request.get("id", ""))))
            elif op == "resume":
                await self._reply(writer, self.resume(str(request.get("id", ""))))
            elif op == "ping":
                await self._reply(writer, {"pong": True})
            elif op == "submit":
                self._check_request(request)
                job = self.submit(
                    argv=request.get("argv"),
                    workload_name=request.get("workload"),
                    need_gb=request.get("need_gb"),
                    priority=int(request.get("priority") or 0),
                    ttl=request.get("ttl"),
                    hold=bool(request.get("hold", False)),
                )
                # A lease is "hold the card while I am here": if the client goes away
                # (Ctrl-C), release it. A queued `run` job belongs to the broker and
                # keeps running; stop it with `gpupark cancel`.
                watcher = (
                    asyncio.create_task(self._watch_client(reader, job)) if job.hold else None
                )
                while True:
                    event = await job.events.get()
                    await self._reply(writer, event, newline=True)
                    if event["ev"] in ("done", "failed", "cancelled"):
                        break
            else:
                await self._reply(writer, {"error": f"unknown op {op!r}"})
        except Exception as exc:
            with contextlib.suppress(Exception):
                await self._reply(writer, {"error": str(exc)})
        finally:
            if watcher is not None:
                watcher.cancel()
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    async def _reply(self, writer: asyncio.StreamWriter, payload: dict, newline: bool = True) -> None:
        data = (json.dumps(payload) + "\n").encode()
        writer.write(data)
        await writer.drain()

    @staticmethod
    def _socket_is_live(sock: Path) -> bool:
        probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        probe.settimeout(0.5)
        try:
            probe.connect(str(sock))
        except OSError:
            return False
        finally:
            probe.close()
        return True

    def _prepare_runtime_dir(self, sock: Path) -> None:
        """Own the socket's directory: create it private, refuse one owned by somebody else."""
        parent = sock.parent
        parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        st = os.lstat(parent)
        if st.st_uid != os.getuid():
            raise SystemExit(
                f"{parent} is owned by uid {st.st_uid}, not this user; refusing to put a socket there"
            )
        if parent == Path(config_mod.runtime_dir()) and st.st_mode & 0o077:
            os.chmod(parent, 0o700)  # the default runtime directory must be private
        elif st.st_mode & 0o002:
            print(f"gpupark: warning: {parent} is world-writable", flush=True)

    def _claim_lock(self, sock: Path) -> None:
        """One broker per socket path: hold an exclusive lock for the daemon's lifetime."""
        path = sock.parent / "gpupark.lock"
        self._lock_fd = os.open(str(path), os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(self._lock_fd)
            self._lock_fd = None
            raise SystemExit(f"another gpupark broker holds {path}") from None

    def _clear_stale_socket(self, sock: Path) -> None:
        """Only a dead socket may go: never steal a live broker's socket, never delete a file."""
        if not (sock.exists() or sock.is_symlink()):
            return
        mode = os.lstat(sock).st_mode
        if not stat_mod.S_ISSOCK(mode):
            raise SystemExit(f"{sock} exists and is not a socket; remove it or fix socket_path")
        if self._socket_is_live(sock):
            raise SystemExit(f"another broker is already listening on {sock}")
        sock.unlink()

    async def serve(self) -> None:
        sock = Path(self.cfg.socket_path)
        self._prepare_runtime_dir(sock)
        self._claim_lock(sock)
        self._clear_stale_socket(sock)
        old_umask = os.umask(0o077)  # bind private: no window at the umask default
        try:
            self._server = await asyncio.start_unix_server(self.handle, path=str(sock))
        finally:
            os.umask(old_umask)
        os.chmod(sock, 0o600)
        task = asyncio.create_task(self._scheduler())
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(NotImplementedError):
                loop.add_signal_handler(sig, lambda: asyncio.ensure_future(self.shutdown()))
        print(f"gpupark listening on {sock}", flush=True)
        try:
            await self._exit.wait()
        finally:
            task.cancel()
            if self._server:
                self._server.close()
                await self._server.wait_closed()
            if self._lock_fd is not None:
                with contextlib.suppress(OSError):
                    os.close(self._lock_fd)
                self._lock_fd = None

    async def shutdown(self) -> None:
        """Stop scheduling and let `serve` return (SIGTERM/SIGINT land here)."""
        if self._stopping:
            return
        self._stopping = True
        if self.running is not None:
            self.cancel(self.running.id)
        if self._server:
            self._server.close()
        self._exit.set()
