import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / "configs" / "lerobot_control" / "inference_shirt_fold_pi05_shadow.yaml"
LIVE_CONFIG_PATH = REPO_ROOT / "configs" / "lerobot_control" / "inference_shirt_fold_pi05_live.yaml"
STRATEGY_PATH = (
    REPO_ROOT
    / "ros2"
    / "src"
    / "lerobot_control"
    / "lerobot_control"
    / "strategies"
    / "multi_process.py"
)
INFERENCE_NODE_PATH = (
    REPO_ROOT / "ros2" / "src" / "lerobot_control" / "lerobot_control" / "inference_node.py"
)


def test_shirt_fold_pi05_shadow_contract_is_right_first_and_disconnected() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text())

    assert config["model"]["task_description"] == "Fold the T-shirt properly"
    assert list(config["joint_names"]["arm_mapping"]) == ["r", "l"]
    assert config["joint_names"]["model_joint_order"] == [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "joint7",
        "finger_joint1",
    ]
    assert config["arms"]["right"]["action_start"] == 0
    assert config["arms"]["right"]["action_end"] == 8
    assert config["arms"]["left"]["action_start"] == 8
    assert config["arms"]["left"]["action_end"] == 16
    assert all(arm["command_topic"].startswith("/eval/") for arm in config["arms"].values())
    assert set(config["cameras"]["mapping"].values()) == {
        "base",
        "left_wrist",
        "right_wrist",
    }
    assert config["inference_tuning"]["rtc"]["queue_trigger_threshold"] == 30
    assert config["inference_tuning"]["rtc"]["execution_horizon"] == 20
    assert config["safety"] == {
        "max_position_delta": None,
        "min_position_delta": None,
    }

    live_config = yaml.safe_load(LIVE_CONFIG_PATH.read_text())
    assert live_config["safety"] == {
        "max_position_delta": None,
        "min_position_delta": None,
    }


def test_multi_process_strategy_honors_yaml_arm_insertion_order(monkeypatch) -> None:
    package = types.ModuleType("lerobot_control")
    package.__path__ = []
    strategies = types.ModuleType("lerobot_control.strategies")
    strategies.__path__ = []
    image_worker = types.ModuleType("lerobot_control.image_worker")
    image_worker.run_image_worker = lambda *_args, **_kwargs: None
    shared_buffer = types.ModuleType("lerobot_control.shared_image_buffer")
    shared_buffer.SharedImageBuffer = object
    qos = types.ModuleType("rclpy.qos")
    qos.HistoryPolicy = qos.QoSProfile = qos.ReliabilityPolicy = object
    sensor_msgs = types.ModuleType("sensor_msgs.msg")
    sensor_msgs.JointState = object

    for name, module in {
        "lerobot_control": package,
        "lerobot_control.strategies": strategies,
        "lerobot_control.image_worker": image_worker,
        "lerobot_control.shared_image_buffer": shared_buffer,
        "rclpy.qos": qos,
        "sensor_msgs.msg": sensor_msgs,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    spec = importlib.util.spec_from_file_location(
        "lerobot_control.strategies.multi_process",
        STRATEGY_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    strategy = module.MultiProcessStrategy()
    strategy._joint_names_config = {
        "observation_prefix": "follower",
        "separator": "_",
        "arm_mapping": {"r": "right", "l": "left"},
        "model_joint_order": ["joint1", "joint2"],
        "state_features": ["position"],
    }
    strategy._joint_positions = {
        "follower_r_joint1": 1.0,
        "follower_r_joint2": 2.0,
        "follower_l_joint1": 10.0,
        "follower_l_joint2": 20.0,
    }

    observation = strategy._build_observation({})

    np.testing.assert_array_equal(
        observation["observation.state"].numpy(),
        np.array([[1.0, 2.0, 10.0, 20.0]], dtype=np.float32),
    )


def test_rtc_acquires_and_preprocesses_observation_in_inference_thread() -> None:
    source = INFERENCE_NODE_PATH.read_text()

    worker_start = source.index("def _inference_loop")
    worker_end = source.index("def _setup_sync_prefetch")
    worker = source[worker_start:worker_end]
    acquire = "raw_obs = self.strategy.get_observation(self.camera_names)"
    preprocess = "obs = self._preprocess_policy_observation(raw_obs)"
    assert acquire in worker
    assert preprocess in worker
    assert worker.index(acquire) < worker.index(preprocess)

    timer_start = source.index("def _obs_update")
    timer_end = source.index("def _publish_loop")
    timer = source[timer_start:timer_end]
    rtc_return = "if self._uses_rtc_inference:\n            return"
    assert rtc_return in timer
    assert timer.index(rtc_return) < timer.index("self.strategy.get_observation")

    assert "self._latest_raw_obs" not in source
    assert "self._latest_obs" not in source


def test_sync_prefetch_acquires_observation_in_worker() -> None:
    source = INFERENCE_NODE_PATH.read_text()

    worker_start = source.index("def _sync_prefetch_loop")
    worker_end = source.index("def _preprocess_policy_observation")
    worker = source[worker_start:worker_end]
    assert "raw_obs = self.strategy.get_observation(self.camera_names)" in worker
    assert "self._sync_latest_raw_obs" not in source

    timer_start = source.index("def _obs_update")
    timer_end = source.index("def _publish_loop")
    timer = source[timer_start:timer_end]
    prefetch_return = "if self._uses_sync_prefetch:\n            return"
    assert prefetch_return in timer
    assert timer.index(prefetch_return) < timer.index("self.strategy.get_observation")


def test_live_config_uses_full_chunk_prefetch_without_rtc_overrun() -> None:
    config = yaml.safe_load(LIVE_CONFIG_PATH.read_text())
    sync = config["inference_tuning"]["sync"]
    rtc = config["inference_tuning"]["rtc"]

    assert rtc["enabled"] is False
    assert "allow_latency_overrun" not in rtc
    assert sync["async_prefetch"] is True
    assert sync["prefetch_threshold"] == 20
    assert sync["replace_pending_actions"] is True
