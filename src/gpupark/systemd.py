"""systemd --user helpers for unit workloads."""

from __future__ import annotations

import subprocess
import time


class SystemdError(RuntimeError):
    pass


def _systemctl(*args: str, timeout: float = 30) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            ["systemctl", "--user", *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SystemdError(f"systemctl --user {' '.join(args)} failed: {exc}") from exc


def is_active(unit: str) -> bool:
    result = _systemctl("is-active", "--", unit, timeout=10)
    return result.stdout.strip() in ("active", "activating", "reloading")


def start(unit: str) -> None:
    result = _systemctl("start", "--no-block", "--", unit)
    if result.returncode != 0:
        raise SystemdError(f"could not start {unit}: {result.stderr.strip() or result.stdout.strip()}")


def stop(unit: str) -> None:
    result = _systemctl("stop", "--no-block", "--", unit)
    if result.returncode != 0:
        # stopping an inactive unit is not an error worth failing a lease over
        message = (result.stderr or result.stdout).strip()
        if "not loaded" in message or "not found" in message:
            return
        raise SystemdError(f"could not stop {unit}: {message}")


def wait_active(unit: str, timeout: float, interval: float = 0.5) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if is_active(unit):
            return True
        time.sleep(interval)
    return False


def wait_inactive(unit: str, timeout: float, interval: float = 0.5) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not is_active(unit):
            return True
        time.sleep(interval)
    return False
