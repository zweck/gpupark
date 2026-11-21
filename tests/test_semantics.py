"""Semantics the broker must not get wrong: bounded waits, honest exits, and a job you can find again."""

import asyncio
import time

from gpupark import config
from gpupark.daemon import Broker, Host


class Rig(Host):
    """A GPU and a systemd, in memory, with the knobs each test needs."""

    def __init__(self, total_mib=24576, baseline_mib=20000, idle_mib=300):
        self.total_mib = total_mib
        self.baseline_mib = baseline_mib
        self.idle_mib = idle_mib
        self.units = {}
        self.calls = []
        self.alive_calls = 0
        self.dies_after: int | None = None  # None: units stay up; N: is_active goes False after N calls
        self.failure: str | None = None  # what unit_failure() reports once the unit is down

    def gpu_memory(self, index):
        used = self.idle_mib + (self.baseline_mib if self.units.get("base.service") else 0)
        return self.total_mib, used

    def is_active(self, unit):
        if self.dies_after is not None:
            self.alive_calls += 1
            return self.alive_calls <= self.dies_after
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

    def unit_failure(self, unit):
        return self.failure


def cfg_for(workloads=None, **extra):
    raw = {
        "cooldown_seconds": 0,
        "poll_seconds": 0.05,
        "grant_timeout": 0.4,
        "workloads": workloads or {},
    }
    raw.update(extra)
    return config.parse(raw)


def test_the_vram_wait_is_bounded_by_grant_timeout_not_the_lease_ttl():
    """A --ttl 600 job must not keep the baseline parked for ten minutes waiting for VRAM."""

    class Stuck(Rig):
        def stop_unit(self, unit):
            # the baseline gives up the unit but not its memory: the classic zombie
            self.calls.append(("stop", unit))
            self.units[unit] = False

        def gpu_memory(self, index):
            return self.total_mib, self.idle_mib + self.baseline_mib

    cfg = cfg_for(
        {"base": {"kind": "unit", "target": "base.service", "vram_gb": 20, "baseline": True}}
    )
    host = Stuck()
    broker = Broker(cfg, host)

    async def scenario():
        job = broker.submit(argv=["true"], need_gb=10, ttl=600)
        started = time.monotonic()
        await broker._run(job)
        return job, time.monotonic() - started

    job, elapsed = asyncio.run(scenario())
    assert job.state == "failed"
    assert "never freed" in (job.error or "")
    assert elapsed < 5, f"waited {elapsed:.1f}s for VRAM with ttl=600 and grant_timeout=0.4"


def test_unit_ttl_expiry_fails_with_124_like_a_command_job():
    cfg = cfg_for({"heavy": {"kind": "unit", "target": "heavy.service", "vram_gb": 0}})
    host = Rig()
    broker = Broker(cfg, host)

    async def scenario():
        job = broker.submit(workload_name="heavy", ttl=0.3)
        await broker._run(job)
        return job

    job = asyncio.run(scenario())
    assert job.state == "failed", job.state
    assert job.exit_code == 124, job.exit_code
    assert "expired" in (job.error or "")


def test_a_unit_that_dies_reports_its_systemd_result_instead_of_done():
    cfg = cfg_for({"heavy": {"kind": "unit", "target": "heavy.service", "vram_gb": 0}})
    host = Rig()
    host.dies_after = 0  # the unit is never actually up after start
    host.failure = "Result=exit-code ExecMainStatus=3"
    broker = Broker(cfg, host)

    async def scenario():
        job = broker.submit(workload_name="heavy", ttl=60)
        await broker._run(job)
        return job

    job = asyncio.run(scenario())
    assert job.state == "failed", job.state
    assert "stopped unexpectedly" in (job.error or "")
    assert "ExecMainStatus=3" in (job.error or "")


def test_a_job_larger_than_the_card_fails_before_parking_the_baseline():
    cfg = cfg_for(
        {"base": {"kind": "unit", "target": "base.service", "vram_gb": 20, "baseline": True}}
    )
    host = Rig(total_mib=24576)
    broker = Broker(cfg, host)

    async def scenario():
        job = broker.submit(argv=["true"], need_gb=100)
        await broker._run(job)
        return job

    job = asyncio.run(scenario())
    assert job.state == "failed", job.state
    assert "has only" in (job.error or "")
    assert ("stop", "base.service") not in host.calls, "the baseline was parked for a job that cannot fit"


def test_job_output_is_tee_d_to_a_log_file(tmp_path):
    cfg = cfg_for(state_path=str(tmp_path))
    broker = Broker(cfg, Rig())

    async def scenario():
        job = broker.submit(argv=["sh", "-c", "printf 'hello\\n'"], need_gb=0)
        await broker._run(job)
        return job

    job = asyncio.run(scenario())
    assert job.exit_code == 0
    assert job.log_path, "no log file was opened"
    text = (tmp_path / "logs" / f"{job.id}.log").read_text()
    assert "hello" in text
    assert "# done exit=0" in text
    assert (tmp_path / "logs" / f"{job.id}.log").exists()


def test_history_keeps_finished_jobs_with_their_clocks(tmp_path):
    cfg = cfg_for(state_path=str(tmp_path))
    broker = Broker(cfg, Rig())

    async def scenario():
        job = broker.submit(argv=["true"], need_gb=0)
        await broker._run(job)
        return job

    job = asyncio.run(scenario())
    assert len(broker.recent) == 1
    record = broker.recent[0]
    assert record["id"] == job.id
    assert record["state"] == "done"
    assert record["exit_code"] == 0
    assert record["wait_seconds"] is not None
    assert record["run_seconds"] is not None


def test_cancel_accepts_a_unique_id_prefix():
    broker = Broker(cfg_for(), Rig())
    first = broker.submit(argv=["true"])
    second = broker.submit(argv=["true"])
    prefix = first.id[:4]
    if second.id.startswith(prefix):
        # make the prefix unambiguous without depending on random ids
        second.id = "zz" + second.id
    assert broker.cancel(prefix) is True
    assert first.state == "cancelled"
    assert broker.jobs[second.id].state == "queued"
    assert broker.cancel(prefix) is False, "a prefix must not cancel twice"


def test_status_reports_queue_position_and_ttl_clocks():
    from gpupark.daemon import Job

    broker = Broker(cfg_for(), Rig())
    broker.submit(argv=["true"])
    snapshot = asyncio.run(broker.status())
    queued = snapshot["queue"][0]
    assert queued["position"] == 1
    assert queued["queued_for"] is not None
    assert snapshot["recent"] == []

    job = Job(
        id="abcd1234",
        need_gb=4,
        priority=0,
        submitted_at=time.time() - 50,
        started_at=time.time() - 10,
        ttl=60,
        state="running",
    )
    view = job.public()
    assert 9.0 <= view["active_seconds"] <= 11.0
    assert 49.0 <= view["ttl_remaining"] <= 51.0
    assert view["ttl"] == 60
