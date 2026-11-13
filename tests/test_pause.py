import asyncio

from gpupark import config
from gpupark.daemon import Broker, Host


class FakeHost(Host):
    """A GPU and a systemd, in memory."""

    def __init__(self, total_mib=24576, baseline_mib=20000):
        self.total_mib = total_mib
        self.baseline_mib = baseline_mib
        self.units = {}
        self.calls = []

    def gpu_memory(self, index):
        used = 300 + (self.baseline_mib if self.units.get("base.service") else 0) + (
            self.baseline_mib if self.units.get("heavy.service") else 0
        )
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
        {"cooldown_seconds": 0, "poll_seconds": 0.05, "grant_timeout": 5, "workloads": workloads}
    )


def test_pausing_a_queued_job_holds_it_out_of_the_queue():
    broker = Broker(cfg_for({"j": {"kind": "command", "target": ["true"]}}))
    job = broker.submit(argv=["true"])
    assert broker.pause(job.id)["ok"] is True
    assert broker._next_job() is None
    assert broker.resume(job.id)["ok"] is True
    assert broker._next_job() is job


def test_pausing_a_unit_lease_hands_the_card_back_to_the_baseline():
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
        assert host.units.get("base.service") is True
        job = broker.submit(workload_name="heavy", ttl=60)
        run_task = asyncio.create_task(broker._run(job))
        await asyncio.sleep(0.4)
        assert host.units.get("heavy.service") is True
        assert not host.units.get("base.service")

        assert broker.pause(job.id)["ok"] is True
        await asyncio.sleep(1.6)  # the unit loop polls once a second
        assert host.units.get("heavy.service") is False, "the workload should be stopped"
        assert host.units.get("base.service") is True, "the baseline should be back"
        assert job.state == "running", "a paused lease is still the lease"

        assert broker.resume(job.id)["ok"] is True
        await asyncio.sleep(1.8)
        assert host.units.get("heavy.service") is True, "the workload should be back"
        assert host.units.get("base.service") is False

        broker.cancel(job.id)
        await asyncio.wait_for(run_task, timeout=5)
        return job

    job = asyncio.run(scenario())
    assert job.state == "cancelled"


def test_active_seconds_excludes_time_spent_paused():
    broker = Broker(cfg_for({"j": {"kind": "command", "target": ["true"]}}))
    job = broker.submit(argv=["true"])
    job.started_at = 1000.0
    job.paused_total = 30.0
    assert job.active_seconds(now=1100.0) == 70.0
    job.paused_since = 1090.0
    assert job.active_seconds(now=1100.0) == 60.0
