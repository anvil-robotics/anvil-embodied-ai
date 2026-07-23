import importlib.util
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
ACTION_LIMITER_PATH = (
    REPO_ROOT / "ros2" / "src" / "lerobot_control" / "lerobot_control" / "action_limiter.py"
)


def _action_limiter_class():
    spec = importlib.util.spec_from_file_location("action_limiter", ACTION_LIMITER_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.ActionLimiter


def test_null_max_delta_disables_software_limit() -> None:
    limiter = _action_limiter_class()(max_delta=None)
    current = np.array([0.0, 1.0])
    requested = np.array([2.0, -3.0])

    np.testing.assert_array_equal(limiter.process(requested, current), requested)
    assert limiter.get_clamped_joints(requested, current) == []
