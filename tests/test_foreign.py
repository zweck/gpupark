"""Lease integrity: VRAM taken by processes gpupark does not manage."""

import asyncio
import signal

from gpupark import config, gpu
from gpupark.daemon import Broker, Host


class FakeHost(Host):
    """A GPU with named foreign processes on it, and a systemd, in memory."""

    def __init__(self, total_mib=24576, idle_mib=300, procs=None):
        self.total_mib = total_mib
        self.idle_mib = idle_mib
        self.procs = list(procs or [])
        self.units = {}
        self.calls = []
        self.signals = []

    def gpu_memory(self, index):
        return self.total_mib, self.idle_mib + sum(p.used_mib for p in self.procs)

    def gpu_processes(self):
        return list(self.procs)

    def unit_pids(self, unit):
        return set()

    def signal_pid(self, pid, sig):
        self.signals.append((pid, sig))

    def is_active(self, unit):
        return bool(self.units.get(unit))

    def start_unit(self, unit):
        self.calls.append(("start", unit))
        self.units[unit] = True

    def stop_unit(self, unit):
        self.calls.append(("stop", unit))
        self.units[unit] = False

    def wait_active(self, unit, timeout):
        return bool(self.units.get(unit))

    def wait_inactive(self, unit, timeout):
        return not self.units.get(unit)


def cfg_for(workloads, **extra):
    raw = {"cooldown_seconds": 0, "poll_seconds": 0, "grant_timeout": 0.2, "workloads": workloads}
    raw.update(extra)
    return config.parse(raw)


def test_admission_refuses_when_a_foreign_process_holds_the_card():
    host = FakeHost(procs=[gpu.Proc(pid=4242, used_mib=4000, name="game")])
    broker = Broker(cfg_for({}), host)

    async def scenario():
        job = broker.submit(argv=["true"], need_gb=22)
        await broker._run(job)
        return job

    job = asyncio.run(scenario())
    assert job.state == "failed"
    assert "refusing to grant" in (job.error or "")
    assert "pid 4242" in (job.error or "") and "game" in (job.error or "")


def test_admission_tolerates_foreign_use_that_still_fits():
    host = FakeHost(procs=[gpu.Proc(pid=99, used_mib=1000, name="browser")])
    broker = Broker(cfg_for({}), host)

    async def scenario():
        job = broker.submit(argv=["true"], need_gb=1)
        await broker._run(job)
        events = []
        while not job.events.empty():
            events.append(job.events.get_nowait())
        return job, events

    job, events = asyncio.run(scenario())
    assert job.state == "done", job.error
    warns = [event for event in events if event["ev"] == "foreign"]
    assert warns and warns[0]["blocked"] is False
    assert host.signals == []  # default policy is warn: nothing is signalled


def test_kill_policy_targets_intruders_with_sigkill():
    host = FakeHost()
    broker = Broker(cfg_for({}, foreign_policy="kill"), host)
    broker._apply_foreign_policy([gpu.Proc(pid=7, used_mib=2048, name="intruder")])
    assert host.signals == [(7, signal.SIGKILL)]


def test_stop_policy_targets_intruders_with_sigstop():
    host = FakeHost()
    broker = Broker(cfg_for({}, foreign_policy="stop"), host)
    broker._apply_foreign_policy([gpu.Proc(pid=8, used_mib=1024, name="x")])
    assert host.signals == [(8, signal.SIGSTOP)]


def test_warn_policy_never_signals():
    host = FakeHost()
    broker = Broker(cfg_for({}), host)
    broker._apply_foreign_policy([gpu.Proc(pid=9, used_mib=2048, name="x")])
    assert host.signals == []


def test_a_failed_probe_is_unknown_not_empty(monkeypatch):
    def boom(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(gpu.subprocess, "run", boom)
    assert gpu.processes() == []


def test_processes_parses_nvidia_smi_rows(monkeypatch):
    class Result:
        stdout = "1234, 4096, python\n5678, 512, /usr/bin/game\nnot a row\n"

    monkeypatch.setattr(gpu.subprocess, "run", lambda *args, **kwargs: Result())
    procs = gpu.processes()
    assert [(p.pid, p.used_mib, p.name) for p in procs] == [
        (1234, 4096, "python"),
        (5678, 512, "/usr/bin/game"),
    ]


def test_status_reports_foreign_usage():
    host = FakeHost(procs=[gpu.Proc(pid=11, used_mib=3400, name="ApexGP")])
    broker = Broker(cfg_for({}), host)
    snapshot = asyncio.run(broker.status())
    assert snapshot["foreign"] and snapshot["foreign"][0]["pid"] == 11
    assert snapshot["foreign"][0]["used_mib"] == 3400


def test_foreign_policy_must_be_known():
    try:
        cfg_for({}, foreign_policy="nuke")
    except ValueError as exc:
        assert "foreign_policy" in str(exc)
    else:  # pragma: no cover - the config must reject it
        raise AssertionError("an unknown foreign_policy was accepted")
