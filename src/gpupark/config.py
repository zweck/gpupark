"""Configuration: workloads, paths, and defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def _xdg(var: str, fallback: str) -> Path:
    value = os.environ.get(var)
    return Path(value) if value else Path(fallback)


def config_home() -> Path:
    return _xdg("XDG_CONFIG_HOME", "~/.config").expanduser()


def runtime_dir() -> Path:
    return _xdg("XDG_RUNTIME_DIR", f"/tmp/gpupark-{os.getuid()}").expanduser()


def default_config_path() -> Path:
    override = os.environ.get("GPUPARK_CONFIG")
    return Path(override).expanduser() if override else config_home() / "gpupark" / "workloads.yaml"


def default_socket_path() -> Path:
    return runtime_dir() / "gpupark.sock"


def default_state_path() -> Path:
    return runtime_dir() / "gpupark" / "state.json"


@dataclass
class Workload:
    """Something that can hold the GPU: a systemd --user unit or a command."""

    name: str
    kind: str = "unit"  # "unit" | "command"
    target: Any = ""
    vram_gb: float | None = None
    baseline: bool = False
    priority: int = 0
    ready_timeout: float = 180.0
    stop_timeout: float = 30.0

    def argv(self) -> list[str]:
        if isinstance(self.target, str):
            return [self.target]
        return [str(part) for part in self.target]

    def check(self) -> None:
        if self.kind not in ("unit", "command"):
            raise ValueError(f"{self.name}: kind must be 'unit' or 'command', not {self.kind!r}")
        if self.kind == "unit" and not isinstance(self.target, str):
            raise ValueError(f"{self.name}: a unit workload needs a single unit name")
        if self.kind == "command" and not self.argv():
            raise ValueError(f"{self.name}: a command workload needs an argv")


@dataclass
class Config:
    gpu: int = 0
    poll_seconds: float = 0.5
    cooldown_seconds: float = 5.0
    grant_timeout: float = 120.0
    socket_path: str = ""
    state_path: str = ""
    workloads: dict[str, Workload] = field(default_factory=dict)

    def baseline(self) -> Workload | None:
        found = [w for w in self.workloads.values() if w.baseline]
        if len(found) > 1:
            names = ", ".join(w.name for w in found)
            raise ValueError(f"more than one baseline workload: {names}")
        return found[0] if found else None

    def workload(self, name: str) -> Workload:
        try:
            return self.workloads[name]
        except KeyError:
            known = ", ".join(sorted(self.workloads)) or "none"
            raise KeyError(f"unknown workload {name!r} (known: {known})") from None


def parse(raw: dict[str, Any]) -> Config:
    cfg = Config(
        gpu=int(raw.get("gpu", 0)),
        poll_seconds=float(raw.get("poll_seconds", 0.5)),
        cooldown_seconds=float(raw.get("cooldown_seconds", 5.0)),
        grant_timeout=float(raw.get("grant_timeout", 120.0)),
        socket_path=str(raw.get("socket_path") or default_socket_path()),
        state_path=str(raw.get("state_path") or default_state_path()),
    )
    for name, entry in (raw.get("workloads") or {}).items():
        entry = dict(entry or {})
        workload = Workload(
            name=name,
            kind=str(entry.get("kind", "unit")),
            target=entry.get("target", ""),
            vram_gb=entry.get("vram_gb"),
            baseline=bool(entry.get("baseline", False)),
            priority=int(entry.get("priority", 0)),
            ready_timeout=float(entry.get("ready_timeout", 180.0)),
            stop_timeout=float(entry.get("stop_timeout", 30.0)),
        )
        workload.check()
        cfg.workloads[name] = workload
    cfg.baseline()  # validates uniqueness
    return cfg


def load(path: Path | str | None = None) -> Config:
    path = Path(path).expanduser() if path else default_config_path()
    if not path.exists():
        raise FileNotFoundError(f"no gpupark config at {path} (see workloads.example.yaml)")
    with path.open() as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")
    return parse(raw)
