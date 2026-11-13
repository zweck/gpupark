import asyncio

import pytest

from gpupark import config
from gpupark.daemon import Broker, Host


class BlindHost(Host):
    """A box with no nvidia-smi: every probe would fail."""

    def gpu_memory(self, index):
        raise RuntimeError("nvidia-smi: command not found")


def test_gpu_probe_none_queues_without_vram_gating():
    cfg = config.parse(
        {
            "gpu_probe": "none",
            "cooldown_seconds": 0,
            "poll_seconds": 0,
            "workloads": {"j": {"kind": "command", "target": ["true"], "vram_gb": 20}},
        }
    )
    broker = Broker(cfg, BlindHost())

    async def scenario():
        job = broker.submit(argv=["true"], need_gb=20)
        await broker._run(job)
        return job

    job = asyncio.run(scenario())
    assert job.state == "done", job.error


def test_gpu_probe_none_does_not_stop_the_baseline_it_cannot_measure():
    cfg = config.parse({"gpu_probe": "none", "workloads": {}})
    assert cfg.gpu_probe == "none"


def test_unknown_gpu_probe_is_refused():
    with pytest.raises(ValueError, match="gpu_probe must be"):
        config.parse({"gpu_probe": "maybe", "workloads": {}})
