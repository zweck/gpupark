import asyncio

from gpupark import config
from gpupark.daemon import Broker, Host


class QuietHost(Host):
    def gpu_memory(self, index):
        return 24576, 300

    def is_active(self, unit):
        return False

    def start_unit(self, unit):
        pass

    def stop_unit(self, unit):
        pass


def test_shutdown_lets_serve_return(tmp_path):
    cfg = config.parse(
        {
            "socket_path": str(tmp_path / "g.sock"),
            "cooldown_seconds": 0,
            "poll_seconds": 0,
            "workloads": {"j": {"kind": "command", "target": ["true"], "vram_gb": 0}},
        }
    )
    broker = Broker(cfg, QuietHost())

    async def scenario():
        serving = asyncio.create_task(broker.serve())
        await asyncio.sleep(0.2)
        await broker.shutdown()
        await asyncio.wait_for(serving, timeout=3)  # must return, not raise SystemExit

    asyncio.run(scenario())
