import asyncio

from gpupark import config
from gpupark.daemon import Broker, Host


class FakeHost(Host):
    """A GPU and a systemd, in memory."""

    def __init__(self, total_mib=24576, baseline_mib=20000, idle_mib=300):
        self.total_mib = total_mib
        self.baseline_mib = baseline_mib
        self.idle_mib = idle_mib
        self.units = {}
        self.calls = []

    def gpu_memory(self, index):
        used = self.idle_mib + (self.baseline_mib if self.units.get("base.service") else 0)
        return self.total_mib, used

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


def cfg_for(workloads):
    return config.parse(
        {"cooldown_seconds": 0, "poll_seconds": 0, "grant_timeout": 0.2, "workloads": workloads}
    )


def test_priority_then_fifo():
    broker = Broker(cfg_for({"j": {"kind": "command", "target": ["true"], "vram_gb": 0}}))
    low = broker.submit(argv=["true"], priority=0)
    high = broker.submit(argv=["true"], priority=5)
    assert broker._next_job() is high
    broker.cancel(high.id)
    assert broker._next_job() is low


def test_baseline_is_parked_then_restored():
    cfg = cfg_for(
        {
            "base": {"kind": "unit", "target": "base.service", "vram_gb": 20, "baseline": True},
            "heavy": {"kind": "unit", "target": "heavy.service", "vram_gb": 20},
        }
    )
    host = FakeHost()
    broker = Broker(cfg, host)

    async def scenario():
        await broker._restore_baseline()
        assert host.units["base.service"] is True
        job = broker.submit(workload_name="heavy", need_gb=20, hold=False, ttl=0.5)
        await broker._run(job)
        return job

    job = asyncio.run(scenario())
    assert job.state == "done", job.error
    # parked the baseline before starting the heavy workload, then restored it
    assert ("stop", "base.service") in host.calls
    assert ("start", "heavy.service") in host.calls
    assert host.calls.index(("stop", "base.service")) < host.calls.index(("start", "heavy.service"))
    assert host.units["base.service"] is True
    assert broker.baseline_state == "running"


def test_a_job_that_never_fits_fails_with_a_useful_message():
    cfg = cfg_for(
        {
            "base": {"kind": "unit", "target": "base.service", "vram_gb": 20, "baseline": True},
        }
    )
    host = FakeHost(total_mib=24576, baseline_mib=20000)

    async def scenario():
        job = broker.submit(argv=["true"], need_gb=40)
        await broker._run(job)
        return job

    class StuckHost(FakeHost):
        def stop_unit(self, unit):
            # the baseline refuses to give its memory back: the classic zombie
            self.calls.append(("stop", unit))
            self.units[unit] = False
            self.baseline_mib = 20000

        def gpu_memory(self, index):
            return self.total_mib, self.idle_mib + self.baseline_mib

    broker = Broker(cfg, StuckHost())
    job = asyncio.run(scenario())
    assert job.state == "failed"
    assert "never freed" in (job.error or "")


def test_command_job_runs_and_streams():
    cfg = cfg_for({"j": {"kind": "command", "target": ["true"], "vram_gb": 0}})
    broker = Broker(cfg, FakeHost(total_mib=24576, baseline_mib=0))

    async def scenario():
        job = broker.submit(argv=["sh", "-c", "printf 'hello\\n'"], need_gb=0)
        await broker._run(job)
        lines = []
        while not job.events.empty():
            event = job.events.get_nowait()
            if event["ev"] == "output":
                lines.append(event["line"])
        return job, lines

    job, lines = asyncio.run(scenario())
    assert job.exit_code == 0
    assert "hello" in lines


def test_cancel_queued_job_marks_it_cancelled():
    broker = Broker(cfg_for({"j": {"kind": "command", "target": ["true"]}}))
    job = broker.submit(argv=["true"])
    assert broker.cancel(job.id) is True
    assert job.state == "cancelled"
    assert broker._next_job() is None
