import pytest

from gpupark import config

RAW = {
    "gpu": 0,
    "cooldown_seconds": 2,
    "workloads": {
        "llm": {"kind": "unit", "target": "llm.service", "vram_gb": 21, "baseline": True},
        "comfy": {"kind": "unit", "target": "comfy.service", "vram_gb": 22},
        "job": {"kind": "command", "target": ["echo", "hi"], "vram_gb": 1},
    },
}


def test_parse_reads_workloads():
    cfg = config.parse(RAW)
    assert cfg.baseline().name == "llm"
    assert cfg.workload("comfy").vram_gb == 22
    assert cfg.workload("job").argv() == ["echo", "hi"]
    assert cfg.cooldown_seconds == 2


def test_two_baselines_are_refused():
    raw = {
        "workloads": {
            "a": {"kind": "unit", "target": "a.service", "baseline": True},
            "b": {"kind": "unit", "target": "b.service", "baseline": True},
        }
    }
    with pytest.raises(ValueError, match="more than one baseline"):
        config.parse(raw)


def test_unknown_workload_lists_known():
    cfg = config.parse(RAW)
    with pytest.raises(KeyError, match="known: comfy, job, llm"):
        cfg.workload("nope")


def test_unit_workload_needs_a_single_target():
    with pytest.raises(ValueError):
        config.parse({"workloads": {"x": {"kind": "unit", "target": ["a", "b"]}}})
