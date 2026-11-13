"""The broker: one queue, one lease, one baseline that yields the card."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import config as config_mod
from . import gpu, systemd


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
    state: str = "queued"  # queued|granted|running|done|failed|cancelled
    exit_code: int | None = None
    error: str | None = None
    started_at: float | None = None
    ended_at: float | None = None
    events: asyncio.Queue = field(default_factory=asyncio.Queue)
    cancel: asyncio.Event = field(default_factory=asyncio.Event)

    def emit(self, event: str, **fields: Any) -> None:
        self.events.put_nowait({"ev": event, "id": self.id, **fields})

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


class Host:
    """Every side effect the broker performs, in one place (fakeable in tests)."""

    def gpu_memory(self, index: int) -> tuple[int, int]:
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
        self.host = host or Host()
        self.jobs: dict[str, Job] = {}
        self.order: list[str] = []
        self.running: Job | None = None
        self.baseline_state = "unknown"
        self._wake = asyncio.Event()
        self._stopping = False
        self._exit = asyncio.Event()
        self._server: asyncio.AbstractServer | None = None

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

    def _next_job(self) -> Job | None:
        pending = [self.jobs[j] for j in self.order if self.jobs[j].state == "queued"]
        if not pending:
            return None
        pending.sort(key=lambda j: (-j.priority, j.submitted_at))
        return pending[0]

    def status(self) -> dict[str, Any]:
        try:
            total, used = self.host.gpu_memory(self.cfg.gpu)
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
            "baseline": {
                "name": baseline.name if baseline else None,
                "state": self.baseline_state,
            },
            "gpu": gpu_view,
        }

    # ------------------------------------------------------------ scheduling

    async def _scheduler(self) -> None:
        await self._restore_baseline()
        while not self._stopping:
            job = self._next_job()
            if job is None:
                self._wake.clear()
                try:
                    await asyncio.wait_for(self._wake.wait(), timeout=1.0)
                except asyncio.TimeoutError:
                    pass
                continue
            await self._run(job)

    async def _restore_baseline(self) -> None:
        baseline = self.cfg.baseline()
        if baseline is None:
            return
        if baseline.kind == "command":
            # a command baseline cannot be meaningfully kept by the broker yet
            self.baseline_state = "unsupported"
            return
        if not await asyncio.to_thread(self.host.is_active, str(baseline.target)):
            try:
                await asyncio.to_thread(self.host.start_unit, str(baseline.target))
            except Exception as exc:
                self.baseline_state = f"error: {exc}"
                return
        self.baseline_state = "running"

    async def _park_baseline(self) -> None:
        baseline = self.cfg.baseline()
        if baseline is None or baseline.kind != "unit":
            return
        unit = str(baseline.target)
        if not await asyncio.to_thread(self.host.is_active, unit):
            self.baseline_state = "parked"
            return
        await asyncio.to_thread(self.host.stop_unit, unit)
        await asyncio.to_thread(self.host.wait_inactive, unit, baseline.stop_timeout)
        self.baseline_state = "parked"

    async def _wait_free(self, need_gb: float, timeout: float) -> bool:
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
            if workload is not None and workload.kind == "unit" and not job.hold:
                try:
                    await asyncio.to_thread(self.host.stop_unit, str(workload.target))
                except Exception:
                    pass
            await asyncio.sleep(self.cfg.cooldown_seconds)
            if self._next_job() is None:
                await self._restore_baseline()

    async def _execute(self, job: Job, workload: config_mod.Workload | None) -> int:
        if workload is not None and workload.kind == "unit":
            unit = str(workload.target)
            await asyncio.to_thread(self.host.start_unit, unit)
            if not await asyncio.to_thread(self.host.wait_active, unit, workload.ready_timeout):
                raise RuntimeError(f"{unit} did not become active within {workload.ready_timeout:.0f}s")
            deadline = time.monotonic() + job.ttl if job.ttl else None
            while True:
                if job.cancel.is_set():
                    await asyncio.to_thread(self.host.stop_unit, unit)
                    raise asyncio.CancelledError()
                if not await asyncio.to_thread(self.host.is_active, unit):
                    return 0
                if deadline and time.monotonic() >= deadline:
                    await asyncio.to_thread(self.host.stop_unit, unit)
                    return 0
                await asyncio.sleep(1.0)
        argv = job.argv or (workload.argv() if workload else [])
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env={**os.environ, "GPUPARK_JOB_ID": job.id},
        )

        async def pump() -> None:
            assert proc.stdout is not None
            async for line in proc.stdout:
                job.emit("output", line=line.decode(errors="replace").rstrip("\n"))

        pump_task = asyncio.create_task(pump())
        wait_task = asyncio.create_task(proc.wait())
        cancel_task = asyncio.create_task(job.cancel.wait())
        done, _ = await asyncio.wait(
            {wait_task, cancel_task}, return_when=asyncio.FIRST_COMPLETED
        )
        if cancel_task in done and not wait_task.done():
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=10)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
            await pump_task
            raise asyncio.CancelledError()
        await pump_task
        return proc.returncode or 0

    # -------------------------------------------------------------- service

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            line = await asyncio.wait_for(reader.readline(), timeout=30)
            request = json.loads(line or b"{}")
            op = request.get("op", "status")
            if op == "status":
                await self._reply(writer, self.status())
            elif op == "cancel":
                await self._reply(writer, {"cancelled": self.cancel(str(request.get("id", "")))})
            elif op == "ping":
                await self._reply(writer, {"pong": True})
            elif op == "submit":
                job = self.submit(
                    argv=request.get("argv"),
                    workload_name=request.get("workload"),
                    need_gb=request.get("need_gb"),
                    priority=int(request.get("priority", 0)),
                    ttl=request.get("ttl"),
                    hold=bool(request.get("hold", False)),
                )
                while True:
                    event = await job.events.get()
                    await self._reply(writer, event, newline=True)
                    if event["ev"] in ("done", "failed", "cancelled"):
                        break
            else:
                await self._reply(writer, {"error": f"unknown op {op!r}"})
        except Exception as exc:
            try:
                await self._reply(writer, {"error": str(exc)})
            except Exception:
                pass
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    async def _reply(self, writer: asyncio.StreamWriter, payload: dict, newline: bool = True) -> None:
        data = (json.dumps(payload) + "\n").encode()
        writer.write(data)
        await writer.drain()

    async def serve(self) -> None:
        sock = Path(self.cfg.socket_path)
        sock.parent.mkdir(parents=True, exist_ok=True)
        if sock.exists():
            sock.unlink()
        self._server = await asyncio.start_unix_server(self.handle, path=str(sock))
        os.chmod(sock, 0o600)
        task = asyncio.create_task(self._scheduler())
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, lambda: asyncio.ensure_future(self.shutdown()))
            except NotImplementedError:
                pass
        print(f"gpupark listening on {sock}", flush=True)
        try:
            await self._exit.wait()
        finally:
            task.cancel()
            if self._server:
                self._server.close()
                await self._server.wait_closed()

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
