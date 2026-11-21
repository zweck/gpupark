"""GPU memory probe, via nvidia-smi (no NVML bindings needed)."""

from __future__ import annotations

import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

MIB = 1024 * 1024


class GpuError(RuntimeError):
    pass


@dataclass(frozen=True)
class Proc:
    """One process holding GPU memory."""

    pid: int
    used_mib: int
    name: str

    def public(self) -> dict:
        return {"pid": self.pid, "used_mib": self.used_mib, "name": self.name}


def processes(index: int = 0) -> list[Proc]:
    """Per-process GPU memory for ONE card, best effort.

    `nvidia-smi --query-compute-apps` lists CUDA apps (and, on current drivers, graphics apps
    that allocate GPU memory too). `--id` keeps the answer scoped to the configured card: on a
    multi-GPU host, processes on another GPU must never be attributed to this lease.
    A driver or container that cannot report gives [] — callers must treat "no answer" as
    "unknown", never as "nothing there".
    """
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                f"--id={index}",
                "--query-compute-apps=pid,used_memory,process_name",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return []
    procs: list[Proc] = []
    for line in out.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2 or not parts[0].isdigit():
            continue
        try:
            used = int(parts[1])
        except ValueError:
            continue
        name = parts[2] if len(parts) > 2 and parts[2] else "?"
        procs.append(Proc(pid=int(parts[0]), used_mib=used, name=name))
    return procs


def proc_table() -> dict[int, tuple[int, int]]:
    """`{pid: (ppid, pgrp)}` for every live process, from one pass over /proc.

    One scan serves both questions the broker asks: "who is in this tree?" and "who is in this
    process group?" (a job's children can be reparented or leave the tree but stay in the group).
    """
    table: dict[int, tuple[int, int]] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text()
            # fields after the last ')': state ppid pgrp ...
            fields = stat.rsplit(")", 1)[1].split()
            table[int(entry.name)] = (int(fields[1]), int(fields[2]))
        except (OSError, IndexError, ValueError):
            continue
    return table


def process_tree(root: int, table: dict[int, tuple[int, int]] | None = None) -> set[int]:
    """`root` and every live descendant, read from /proc. Empty when root is gone."""
    if root <= 0 or not Path(f"/proc/{root}").exists():
        return set()
    parent = {pid: ppid for pid, (ppid, _pgrp) in (table or proc_table()).items()}
    tree = {root}
    growing = True
    while growing:
        growing = False
        for pid, ppid in parent.items():
            if ppid in tree and pid not in tree:
                tree.add(pid)
                growing = True
    return tree


def process_group(pgid: int, table: dict[int, tuple[int, int]] | None = None) -> set[int]:
    """Every live pid whose process group is `pgid` — catches children that left the tree."""
    if pgid <= 0:
        return set()
    return {pid for pid, (_ppid, pgrp) in (table or proc_table()).items() if pgrp == pgid}


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
