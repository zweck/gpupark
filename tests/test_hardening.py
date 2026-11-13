"""Regression tests for the hardening pass (queue bounds, socket hygiene, crash paths)."""

import asyncio
import json
import os
import stat

import pytest

from gpupark import config
from gpupark.daemon import EVENTS_MAX, Broker, Host


class QuietHost(Host):
    """No GPU, no systemd: the broker still runs command jobs."""

    def __init__(self) -> None:
        super().__init__("none")

    def gpu_memory(self, index):
        raise RuntimeError("no gpu")

    def is_active(self, unit):
        return False

    def start_unit(self, unit):
        pass

    def stop_unit(self, unit):
        pass

    def wait_active(self, unit, timeout):
        return True

    def wait_inactive(self, unit, timeout):
        return True


def cfg_with(sock, **extra):
    raw = {
        "socket_path": str(sock),
        "gpu_probe": "none",
        "poll_seconds": 0.05,
        "cooldown_seconds": 0,
        "grant_timeout": 1,
        "workloads": {},
    }
    raw.update(extra)
    return config.parse(raw)


def test_ttl_expiry_fails_a_command_job_rather_than_reporting_success():
    broker = Broker(cfg_with("/tmp/unused.sock"), QuietHost())

    async def scenario():
        job = broker.submit(argv=["sleep", "5"], need_gb=0, ttl=0.4)
        await broker._run(job)
        return job

    job = asyncio.run(scenario())
    assert job.state == "failed", job.state
    assert job.exit_code == 124
    assert "expired" in (job.error or "")


def test_output_queue_is_bounded_and_counts_drops():
    broker = Broker(cfg_with("/tmp/unused.sock"), QuietHost())
    job = broker.submit(argv=["true"])
    job.events.get_nowait()  # the "queued" event
    for i in range(EVENTS_MAX + 100):
        job.emit("output", line=str(i))
    assert job.events.qsize() == EVENTS_MAX
    assert job.dropped == 100


def test_scheduler_survives_a_host_layer_error():
    class FlakyHost(QuietHost):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def is_active(self, unit):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("systemctl --user is-active exploded")
            return False

    host = FlakyHost()
    workloads = {"base": {"kind": "unit", "target": "base.service", "baseline": True}}
    broker = Broker(cfg_with("/tmp/unused.sock", workloads=workloads), host)

    async def scenario():
        job = broker.submit(argv=["true"], need_gb=0)
        task = asyncio.create_task(broker._scheduler())
        for _ in range(40):
            if job.state in ("done", "failed"):
                break
            await asyncio.sleep(0.1)
        alive = broker.scheduler_alive
        error = broker.last_error
        broker._stopping = True
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return job, alive, error

    job, alive, error = asyncio.run(scenario())
    assert job.state == "done", job.error
    assert alive is True
    assert "exploded" in (error or "")


def test_second_broker_is_refused_instead_of_stealing_the_socket(tmp_path):
    sock = tmp_path / "g.sock"

    async def scenario():
        first = Broker(cfg_with(sock), QuietHost())
        serving = asyncio.create_task(first.serve())
        await asyncio.sleep(0.3)
        second = Broker(cfg_with(sock), QuietHost())
        with pytest.raises(SystemExit):
            await second.serve()
        await first.shutdown()
        await asyncio.wait_for(serving, timeout=3)

    asyncio.run(scenario())


def test_default_runtime_dir_is_made_private(tmp_path, monkeypatch):
    runtime = tmp_path / "run"
    runtime.mkdir()
    os.chmod(runtime, 0o755)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime))
    broker = Broker(cfg_with(runtime / "g.sock"), QuietHost())

    async def scenario():
        serving = asyncio.create_task(broker.serve())
        await asyncio.sleep(0.3)
        await broker.shutdown()
        await asyncio.wait_for(serving, timeout=3)

    asyncio.run(scenario())
    assert stat.S_IMODE(os.stat(runtime).st_mode) == 0o700


def test_client_hangup_releases_a_lease(tmp_path):
    sock = tmp_path / "g.sock"
    broker = Broker(cfg_with(sock), QuietHost())

    async def scenario():
        serving = asyncio.create_task(broker.serve())
        await asyncio.sleep(0.3)
        reader, writer = await asyncio.open_unix_connection(str(sock))
        payload = {"op": "submit", "argv": ["sleep", "30"], "need_gb": 0, "hold": True}
        writer.write((json.dumps(payload) + "\n").encode())
        await writer.drain()
        while True:
            event = json.loads(await asyncio.wait_for(reader.readline(), 5))
            if event["ev"] == "running":
                break
        writer.close()  # Ctrl-C on `gpupark lease`: the client goes away
        await writer.wait_closed()
        for _ in range(50):
            if broker.running is None:
                break
            await asyncio.sleep(0.1)
        still_running = broker.running
        await broker.shutdown()
        await asyncio.wait_for(serving, timeout=3)
        return still_running

    assert asyncio.run(scenario()) is None


def test_a_run_job_survives_its_client(tmp_path):
    """A queued command belongs to the broker: losing the watcher must not kill it."""
    sock = tmp_path / "g.sock"
    marker = tmp_path / "finished"
    broker = Broker(cfg_with(sock), QuietHost())

    async def scenario():
        serving = asyncio.create_task(broker.serve())
        await asyncio.sleep(0.3)
        reader, writer = await asyncio.open_unix_connection(str(sock))
        argv = ["sh", "-c", f"sleep 1; printf done > {marker}"]
        writer.write((json.dumps({"op": "submit", "argv": argv, "need_gb": 0}) + "\n").encode())
        await writer.drain()
        await asyncio.wait_for(reader.readline(), 5)  # "queued"
        writer.close()
        await writer.wait_closed()
        for _ in range(50):
            if marker.exists():
                break
            await asyncio.sleep(0.1)
        await broker.shutdown()
        await asyncio.wait_for(serving, timeout=3)

    asyncio.run(scenario())
    assert marker.read_text() == "done"


def test_cancel_kills_the_whole_process_group(tmp_path):
    sock = tmp_path / "g.sock"
    broker = Broker(cfg_with(sock), QuietHost())

    async def scenario():
        serving = asyncio.create_task(broker.serve())
        await asyncio.sleep(0.3)
        reader, writer = await asyncio.open_unix_connection(str(sock))
        argv = ["sh", "-c", "sleep 30 & sleep 30; wait"]
        writer.write((json.dumps({"op": "submit", "argv": argv, "need_gb": 0}) + "\n").encode())
        await writer.drain()
        while True:
            event = json.loads(await asyncio.wait_for(reader.readline(), 5))
            if event["ev"] == "running":
                break
        job = broker.running
        assert job is not None
        for _ in range(50):  # "running" is emitted just before the process is spawned
            if job.pgid is not None:
                break
            await asyncio.sleep(0.05)
        assert job.pgid is not None
        pgid = job.pgid
        cancel_reader, canceller = await asyncio.open_unix_connection(str(sock))  # one request per conn
        canceller.write((json.dumps({"op": "cancel", "id": job.id}) + "\n").encode())
        await canceller.drain()
        reply = json.loads(await asyncio.wait_for(cancel_reader.readline(), 5))
        assert reply.get("cancelled") is True
        canceller.close()
        await canceller.wait_closed()
        while True:
            event = json.loads(await asyncio.wait_for(reader.readline(), 5))
            if event["ev"] in ("cancelled", "failed", "done"):
                break
        await asyncio.sleep(0.3)
        writer.close()
        await writer.wait_closed()
        await broker.shutdown()
        await asyncio.wait_for(serving, timeout=3)
        return pgid

    pgid = asyncio.run(scenario())
    with pytest.raises(ProcessLookupError):
        os.killpg(pgid, 0)


def test_bad_request_types_are_rejected_over_the_socket(tmp_path):
    sock = tmp_path / "g.sock"
    broker = Broker(cfg_with(sock), QuietHost())

    async def ask(payload):
        """One request per connection: the daemon serves a single request then closes."""
        reader, writer = await asyncio.open_unix_connection(str(sock))
        writer.write((json.dumps(payload) + "\n").encode())
        await writer.drain()
        reply = json.loads(await asyncio.wait_for(reader.readline(), 5))
        writer.close()
        await writer.wait_closed()
        return reply

    async def scenario():
        serving = asyncio.create_task(broker.serve())
        await asyncio.sleep(0.3)
        replies = [
            await ask({"op": "submit", "argv": ["true"], "ttl": "60"}),
            await ask({"op": "submit", "argv": "true"}),
            await ask({"op": "submit", "argv": ["true"], "hold": "yes"}),
            await ask({"op": "ping"}),
        ]
        await broker.shutdown()
        await asyncio.wait_for(serving, timeout=3)
        return replies

    bad_ttl, bad_argv, bad_hold, ping = asyncio.run(scenario())
    assert "ttl" in bad_ttl["error"] and "number" in bad_ttl["error"]
    assert "argv" in bad_argv["error"] and "strings" in bad_argv["error"]
    assert "hold" in bad_hold["error"]
    assert ping.get("pong") is True
