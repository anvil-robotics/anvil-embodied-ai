import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts/simulate_prediction_latencies.py"
SPEC = importlib.util.spec_from_file_location("simulate_prediction_latencies", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_report_covers_registered_policies_and_is_deterministic():
    baselines = json.loads(MODULE.DEFAULT_BASELINES.read_text())
    report_a = MODULE.build_report(baselines, samples=1000, seed=7, control_hz=30)
    report_b = MODULE.build_report(baselines, samples=1000, seed=7, control_hz=30)

    assert report_a["models"] == report_b["models"]
    assert set(report_a["models"]) == {
        "act", "diffusion", "smolvla", "pi0", "pi05", "molmoact2", "groot",
        "multi_task_dit", "evo1", "fastwam", "vla_jepa",
    }
    for profile in report_a["models"].values():
        assert profile["p50_ms"] < profile["p95_ms"] < profile["p99_ms"]
        assert profile["recommended_delay_steps"] >= 1


def test_percentile_interpolates():
    assert MODULE.percentile([0, 10], 0.5) == 5
