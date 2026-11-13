"""GPU memory probe, via nvidia-smi (no NVML bindings needed)."""

from __future__ import annotations

import subprocess
import time
from typing import Callable

MIB = 1024 * 1024


class GpuError(RuntimeError):
    pass


def memory(index: int = 0) -> tuple[int, int]:
    """(total_mib, used_mib) for one GPU."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                f"--id={index}",
                "--query-gpu=memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise GpuError(f"nvidia-smi failed: {exc}") from exc
    line = out.splitlines()[0]
    total, used = (int(part.strip()) for part in line.split(","))
    return total, used


def free_mib(index: int = 0) -> int:
    total, used = memory(index)
    return max(0, total - used)


def wait_for_free(
    need_mib: int,
    timeout: float,
    interval: float = 0.5,
    index: int = 0,
    probe: Callable[[int], tuple[int, int]] | None = None,
) -> bool:
    """Block until the GPU has at least `need_mib` free, or the timeout passes."""
    probe = probe or memory
    deadline = time.monotonic() + timeout
    while True:
        total, used = probe(index)
        if total - used >= need_mib:
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(interval)
