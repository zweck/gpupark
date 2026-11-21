"""Keep the suite off the real state directory: a test must not write job logs into ~/.local/state."""

import pytest

from gpupark import config


@pytest.fixture(autouse=True)
def _isolated_state(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "default_state_path", lambda: tmp_path / "state" / "state.json")
