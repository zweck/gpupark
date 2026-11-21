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
    """True while the unit still occupies anything — including `deactivating`.

    Treating `deactivating` as inactive made `wait_inactive()` return immediately while the unit
    was still stopping and still holding its VRAM, so the broker could park the baseline and then
    refuse the job in the same second.
    """
    result = _systemctl("is-active", "--", unit, timeout=10)
    return result.stdout.strip() in ("active", "activating", "reloading", "deactivating")


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


def main_pid(unit: str) -> int:
    """The unit's MainPID, or 0 when it is not running. Used to attribute GPU memory.

    A failed `systemctl show` raises instead of answering 0: a silent 0 drops the unit's PIDs out
    of "ours", and the foreign-policy pass could then signal the very workload the lease owns.
    """
    result = _systemctl("show", "-p", "MainPID", "--value", "--", unit, timeout=10)
    if result.returncode != 0:
        raise SystemdError(
            f"systemctl show MainPID {unit} failed: {(result.stderr or result.stdout).strip()}"
        )
    try:
        return int(result.stdout.strip() or 0)
    except ValueError:
        return 0


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


def result(unit: str) -> tuple[str, int]:
    """(Result, ExecMainStatus) as systemd reports them, so a dead unit is not mistaken for a clean exit.

    `Result` is "success" for a clean exit, otherwise "exit-code", "signal", "timeout", ...;
    `ExecMainStatus` is the exit status or signal number that goes with it. Parsed as `key=value`
    because systemd orders the properties itself, not in the order `-p` was given.
    """
    shown = _systemctl("show", "-p", "Result", "-p", "ExecMainStatus", "--", unit, timeout=10)
    props: dict[str, str] = {}
    for line in (shown.stdout or "").splitlines():
        key, _, value = line.partition("=")
        if key.strip():
            props[key.strip()] = value.strip()
    name = props.get("Result", "")
    try:
        status = int(props.get("ExecMainStatus") or 0)
    except ValueError:
        status = 0
    return name, status
