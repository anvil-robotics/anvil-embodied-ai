"""Unit tests for anvil-eval-ros components.

Coverage:
  1. anvil_eval_ros.cli — collect_mcap_files, build_episode_map, load_split_info, resolve_output_dir
  2. anvil_eval_ros.cli — main() end-to-end with --no-docker flag
  3. eval_recorder_node — _align_and_stack (pure numpy logic, tested via ROS2 mock)
  4. mcap_player_node — eval_plan loading (tested via ROS2 mock)

ROS2 node tests use sys.modules patching to mock rclpy so they can run outside a
ROS2 environment.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import numpy as np
import pytest

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[3]
MCAP_ROOT = REPO_ROOT / "data" / "raw" / "placing-block-r1"
ANVIL_EVAL_SRC = REPO_ROOT / "packages" / "anvil_eval" / "src"
ANVIL_EVAL_ROS_SRC = REPO_ROOT / "packages" / "anvil_eval_ros" / "src"
LEROBOT_CONTROL_SRC = REPO_ROOT / "ros2" / "src" / "lerobot_control"

# Make anvil_eval + anvil_eval_ros importable
for src in (ANVIL_EVAL_SRC, ANVIL_EVAL_ROS_SRC):
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

# Make lerobot_control importable (for ROS2 node modules)
if str(LEROBOT_CONTROL_SRC) not in sys.path:
    sys.path.insert(0, str(LEROBOT_CONTROL_SRC))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers: mock ROS2 modules so we can import the nodes without a ROS2 install
# ─────────────────────────────────────────────────────────────────────────────

def _install_ros2_mocks():
    """Patch all rclpy / std_msgs modules so ROS2 node files can be imported."""
    ros2_mocks = [
        "rclpy",
        "rclpy.node",
        "rclpy.qos",
        "rclpy.clock",
        "std_msgs",
        "std_msgs.msg",
        "sensor_msgs",
        "sensor_msgs.msg",
        "builtin_interfaces",
        "builtin_interfaces.msg",
    ]
    for mod in ros2_mocks:
        if mod not in sys.modules:
            sys.modules[mod] = MagicMock()

    # Ensure rclpy.node.Node is a real class so we can subclass it in tests
    mock_node_cls = type("Node", (), {
        "__init__": lambda self, *a, **kw: None,
        "get_logger": lambda self: MagicMock(),
        "create_publisher": lambda self, *a, **kw: MagicMock(),
        "create_subscription": lambda self, *a, **kw: MagicMock(),
        "create_timer": lambda self, *a, **kw: MagicMock(),
        "declare_parameter": lambda self, *a, **kw: None,
        "get_parameter": lambda self, name: MagicMock(
            get_parameter_value=lambda: MagicMock(
                string_value="", double_value=1.0,
                string_array_value=[], bool_value=False, integer_value=0,
            )
        ),
        "get_clock": lambda self: MagicMock(now=lambda: MagicMock(nanoseconds=0)),
        "destroy_node": lambda self: None,
    })
    sys.modules["rclpy.node"].Node = mock_node_cls
    sys.modules["rclpy"].ok = lambda: False
    sys.modules["rclpy"].shutdown = lambda: None
    sys.modules["rclpy"].init = lambda args=None: None
    sys.modules["rclpy"].spin = lambda node: None

    # std_msgs message stubs
    sys.modules["std_msgs.msg"].String = type("String", (), {"data": ""})
    sys.modules["std_msgs.msg"].Bool = type("Bool", (), {"data": False})
    sys.modules["std_msgs.msg"].Float64MultiArray = type(
        "Float64MultiArray", (), {"data": []}
    )


_install_ros2_mocks()


# ─────────────────────────────────────────────────────────────────────────────
# Section 1: anvil_eval_ros.cli — file collection and mapping
# ─────────────────────────────────────────────────────────────────────────────

from anvil_eval_ros.cli import (
    build_episode_map,
    collect_mcap_files,
    load_split_info,
    resolve_output_dir,
)


class TestCollectMcapFiles:
    def test_returns_sorted_mcap_paths(self):
        files = collect_mcap_files(MCAP_ROOT)
        assert len(files) > 0, "No MCAP files found — check data/raw/placing-block-r1/"
        # All paths end with .mcap
        assert all(str(p).endswith(".mcap") for p in files)
        # Must be sorted
        assert files == sorted(files)

    def test_count_matches_expected(self):
        files = collect_mcap_files(MCAP_ROOT)
        # placing-block-r1 has 250 episodes
        assert len(files) == 250

    def test_nonexistent_dir_returns_empty(self):
        files = collect_mcap_files(Path("/nonexistent/path/that/does/not/exist"))
        assert files == []

    def test_returns_path_objects(self):
        files = collect_mcap_files(MCAP_ROOT)
        assert all(isinstance(p, Path) for p in files)


class TestBuildEpisodeMap:
    def test_episode_0_is_first_sorted_mcap(self):
        ep_map = build_episode_map(MCAP_ROOT)
        files = collect_mcap_files(MCAP_ROOT)
        assert ep_map[0] == files[0]

    def test_episode_indices_are_sequential(self):
        ep_map = build_episode_map(MCAP_ROOT)
        assert sorted(ep_map.keys()) == list(range(len(ep_map)))

    def test_episode_count_matches_mcap_count(self):
        ep_map = build_episode_map(MCAP_ROOT)
        files = collect_mcap_files(MCAP_ROOT)
        assert len(ep_map) == len(files)

    def test_all_paths_exist(self):
        ep_map = build_episode_map(MCAP_ROOT)
        # Spot-check first 10
        for idx in range(min(10, len(ep_map))):
            assert ep_map[idx].exists(), f"MCAP for episode {idx} not found: {ep_map[idx]}"


class TestLoadSplitInfo:
    def test_loads_from_pretrained_model_subdir(self):
        """Priority A: pretrained_model/split_info.json"""
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "checkpoints" / "000010"
            pretrained = ckpt / "pretrained_model"
            pretrained.mkdir(parents=True)
            split_data = {
                "train_episodes": [0, 1, 2],
                "val_episodes": [3],
                "test_episodes": [4],
            }
            (pretrained / "split_info.json").write_text(json.dumps(split_data))

            result = load_split_info(ckpt)
            assert result["train"] == [0, 1, 2]
            assert result["val"] == [3]
            assert result["test"] == [4]

    def test_loads_from_job_root_fallback(self):
        """Priority B: job_root/split_info.json (older format)"""
        with tempfile.TemporaryDirectory() as tmp:
            job_root = Path(tmp) / "my_job"
            ckpt = job_root / "checkpoints" / "000010"
            ckpt.mkdir(parents=True)
            split_data = {
                "train_episodes": [10, 11],
                "val_episodes": [12],
                "test_episodes": [],
            }
            (job_root / "split_info.json").write_text(json.dumps(split_data))

            result = load_split_info(ckpt)
            assert result["train"] == [10, 11]
            assert result["val"] == [12]
            assert result["test"] == []

    def test_returns_empty_when_missing(self):
        """No split_info.json → returns empty dict without crashing."""
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "checkpoints" / "000010"
            ckpt.mkdir(parents=True)
            result = load_split_info(ckpt)
            assert result == {}

    def test_real_job_root_split_info(self):
        """Test against the real split_info.json at model_zoo."""
        split_json = REPO_ROOT / "model_zoo" / "placing-block-r1" / "diffusion_20260415_105906" / "split_info.json"
        if not split_json.exists():
            pytest.skip("Real split_info.json not available")

        # Use the job root as checkpoint path (no pretrained_model/ subdir)
        # load_split_info will fallback to parent.parent
        job_root = split_json.parent
        # Simulate: checkpoint = job_root / checkpoints / 000010
        fake_ckpt = job_root / "checkpoints" / "000010"
        fake_ckpt.mkdir(parents=True, exist_ok=True)

        result = load_split_info(fake_ckpt)
        assert len(result.get("train", [])) > 0
        assert len(result.get("val", [])) > 0
        total = len(result["train"]) + len(result["val"]) + len(result.get("test", []))
        assert total == 250  # placing-block-r1 has 250 episodes


class TestResolveOutputDir:
    def test_convention_with_checkpoints_parent(self):
        ckpt = Path("/path/to/job/checkpoints/000050")
        mcap_root = Path("/path/to/data/raw/placing-block-r1")
        out = resolve_output_dir(ckpt, mcap_root)
        assert out == Path("eval_results/placing-block-r1/job/000050/ros")

    def test_convention_without_checkpoints_parent(self):
        ckpt = Path("/path/to/job/000050")
        mcap_root = Path("/path/to/data/raw/my-dataset")
        out = resolve_output_dir(ckpt, mcap_root)
        # parent.name = "job" (not "checkpoints"), so job_name = "job"
        assert out == Path("eval_results/my-dataset/job/000050/ros")

    def test_output_dir_is_path(self):
        ckpt = Path("/a/b/checkpoints/last")
        mcap = Path("/c/d/dataset")
        out = resolve_output_dir(ckpt, mcap)
        assert isinstance(out, Path)


# ─────────────────────────────────────────────────────────────────────────────
# Section 2: anvil_eval_ros.cli — main() end-to-end with --no-docker
# ─────────────────────────────────────────────────────────────────────────────

class TestCliRosEvalMain:
    def _make_fake_checkpoint(self, tmp: Path, episodes: dict | None = None) -> Path:
        """Create minimal fake checkpoint structure with split_info.json."""
        ckpt = tmp / "job" / "checkpoints" / "000010"
        pretrained = ckpt / "pretrained_model"
        pretrained.mkdir(parents=True)

        split_data = episodes or {
            "split_ratio": [8, 1, 1],
            "total_episodes": 5,
            "train_episodes": [0, 1, 2],
            "val_episodes": [3],
            "test_episodes": [4],
        }
        (pretrained / "split_info.json").write_text(json.dumps(split_data))
        return ckpt

    def test_no_docker_generates_eval_plan(self):
        """--no-docker should generate eval_plan.json without launching Docker."""
        import subprocess

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ckpt = self._make_fake_checkpoint(tmp_path)
            output_dir = tmp_path / "eval_out"

            result = subprocess.run(
                [
                    "uv", "run", "anvil-eval-ros",
                    f"--checkpoint={ckpt}",
                    f"--mcap-root={MCAP_ROOT}",
                    f"--output-dir={output_dir}",
                    "--num-eps=2",
                    "--no-docker",
                ],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=str(REPO_ROOT),
            )

            if result.returncode != 0:
                pytest.fail(
                    f"anvil-eval-ros --no-docker failed:\n"
                    f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
                )

            # eval_plan.json must be generated
            plan_file = output_dir / "eval_plan.json"
            assert plan_file.exists(), f"eval_plan.json not found at {plan_file}"

            plan = json.loads(plan_file.read_text())
            assert "episodes" in plan
            assert len(plan["episodes"]) > 0

            # Check structure of each episode entry
            for ep in plan["episodes"]:
                assert "episode_idx" in ep
                assert "split_label" in ep
                assert "mcap_path" in ep
                assert Path(ep["mcap_path"]).exists(), f"MCAP path missing: {ep['mcap_path']}"

    def test_no_docker_respects_num_eps(self):
        """--num-eps N limits episodes per split."""
        import subprocess

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            # 250 episodes total in placing-block-r1
            ckpt = self._make_fake_checkpoint(tmp_path, episodes={
                "split_ratio": [8, 1, 1],
                "total_episodes": 250,
                "train_episodes": list(range(200)),
                "val_episodes": list(range(200, 225)),
                "test_episodes": list(range(225, 250)),
            })
            output_dir = tmp_path / "eval_out"

            result = subprocess.run(
                [
                    "uv", "run", "anvil-eval-ros",
                    f"--checkpoint={ckpt}",
                    f"--mcap-root={MCAP_ROOT}",
                    f"--output-dir={output_dir}",
                    "--num-eps=3",
                    "--no-docker",
                ],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=str(REPO_ROOT),
            )
            assert result.returncode == 0, result.stderr

            plan = json.loads((output_dir / "eval_plan.json").read_text())
            by_split: dict[str, list] = {}
            for ep in plan["episodes"]:
                by_split.setdefault(ep["split_label"], []).append(ep["episode_idx"])

            for split, eps in by_split.items():
                assert len(eps) <= 3, f"Split '{split}' has {len(eps)} episodes, expected ≤ 3"

    def test_no_docker_episode_mcap_mapping_consistent(self):
        """Episode index N must map to the N-th sorted MCAP file."""
        import subprocess

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ckpt = self._make_fake_checkpoint(tmp_path, episodes={
                "split_ratio": [1, 0, 0],
                "total_episodes": 5,
                "train_episodes": [0, 1, 2, 3, 4],
                "val_episodes": [],
                "test_episodes": [],
            })
            output_dir = tmp_path / "eval_out"

            result = subprocess.run(
                [
                    "uv", "run", "anvil-eval-ros",
                    f"--checkpoint={ckpt}",
                    f"--mcap-root={MCAP_ROOT}",
                    f"--output-dir={output_dir}",
                    "--num-eps=5",
                    "--no-docker",
                ],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=str(REPO_ROOT),
            )
            assert result.returncode == 0, result.stderr

            plan = json.loads((output_dir / "eval_plan.json").read_text())
            sorted_mcaps = sorted(collect_mcap_files(MCAP_ROOT))

            for ep in plan["episodes"]:
                idx = ep["episode_idx"]
                expected = str(sorted_mcaps[idx])
                assert ep["mcap_path"] == expected, (
                    f"Episode {idx}: expected {expected}, got {ep['mcap_path']}"
                )


# ─────────────────────────────────────────────────────────────────────────────
# Section 3: eval_recorder_node — _align_and_stack (pure numpy logic)
# ─────────────────────────────────────────────────────────────────────────────

# Import the module with rclpy mocked out
from lerobot_control.eval_recorder_node import EvalRecorderNode


class _MinimalRecorder:
    """Minimal stub of EvalRecorderNode for testing _align_and_stack."""
    _arm_names = ["left", "right"]
    _joint_names = [
        "left_joint1", "left_joint2", "left_joint3",
        "right_joint1", "right_joint2", "right_joint3",
    ]

    # Bind the method from the real class
    _align_and_stack = EvalRecorderNode._align_and_stack


class TestAlignAndStack:
    def _make_buf(self, arm: str, n: int, start_ts: int = 0, step_ns: int = 33_333_333) -> dict:
        """Build a single-arm buffer with n entries of 3-DoF actions."""
        entries = [(start_ts + i * step_ns, [float(i), float(i) + 0.1, float(i) + 0.2]) for i in range(n)]
        return {arm: entries}

    def test_single_arm_perfect_alignment(self):
        """Single arm, GT and pred have identical timestamps → perfect 1-to-1 alignment."""
        rec = _MinimalRecorder()
        rec._arm_names = ["left"]
        rec._joint_names = ["left_j1", "left_j2", "left_j3"]

        gt_buf = self._make_buf("left", 10)
        pred_buf = self._make_buf("left", 10)

        pred, gt = rec._align_and_stack(gt_buf, pred_buf)
        assert pred.shape == (10, 3)
        assert gt.shape == (10, 3)
        np.testing.assert_allclose(pred, gt, atol=1e-6)

    def test_two_arms_concatenated(self):
        """Two arms are concatenated in order: left first, right second."""
        rec = _MinimalRecorder()
        n = 5
        gt_buf = {
            "left":  [(i * 1_000_000, [1.0, 2.0, 3.0]) for i in range(n)],
            "right": [(i * 1_000_000, [4.0, 5.0, 6.0]) for i in range(n)],
        }
        pred_buf = {
            "left":  [(i * 1_000_000, [0.1, 0.2, 0.3]) for i in range(n)],
            "right": [(i * 1_000_000, [0.4, 0.5, 0.6]) for i in range(n)],
        }

        pred, gt = rec._align_and_stack(gt_buf, pred_buf)
        assert pred.shape == (n, 6)
        assert gt.shape == (n, 6)

        # First 3 dims = left arm GT values
        np.testing.assert_allclose(gt[:, :3], 1.0 * np.ones((n, 1)) + np.array([[0, 1, 2]]), atol=1e-6)
        # Last 3 dims = right arm GT values
        np.testing.assert_allclose(gt[:, 3:], 4.0 * np.ones((n, 1)) + np.array([[0, 1, 2]]), atol=1e-6)

    def test_nearest_neighbour_alignment(self):
        """Pred timestamps slightly offset from GT — nearest-neighbour match."""
        rec = _MinimalRecorder()
        rec._arm_names = ["left"]
        rec._joint_names = ["left_j1", "left_j2"]

        step = 33_000_000  # ~33ms
        gt_buf = {"left": [(i * step, [float(i), 0.0]) for i in range(5)]}
        # Pred is 5ms ahead of GT — should still match nearest
        offset = 5_000_000
        pred_buf = {"left": [(i * step + offset, [float(i) * 10, 0.0]) for i in range(5)]}

        pred, gt = rec._align_and_stack(gt_buf, pred_buf)
        assert pred.shape == (5, 2)

        # Each GT row should align to the pred row with closest timestamp
        for i in range(5):
            # gt index i → pred timestamp nearest to i*step → pred row i (offset is small)
            assert pred[i, 0] == pytest.approx(float(i) * 10, abs=1e-6)

    def test_empty_buffers_return_empty_arrays(self):
        rec = _MinimalRecorder()
        pred, gt = rec._align_and_stack({}, {})
        assert pred.shape[0] == 0
        assert gt.shape[0] == 0

    def test_missing_arm_in_pred_is_handled(self):
        """If pred is missing an arm entirely, we still get arrays (partial data)."""
        rec = _MinimalRecorder()
        rec._arm_names = ["left"]
        rec._joint_names = ["left_j1", "left_j2"]

        gt_buf = {"left": [(0, [1.0, 2.0]), (1_000_000, [3.0, 4.0])]}
        pred_buf = {}  # no data at all

        pred, gt = rec._align_and_stack(gt_buf, pred_buf)
        # No pred data → 0 rows
        assert pred.shape[0] == 0

    def test_output_dtype_is_float32(self):
        rec = _MinimalRecorder()
        rec._arm_names = ["left"]
        rec._joint_names = ["left_j1"]

        gt_buf = {"left": [(0, [1.0])]}
        pred_buf = {"left": [(0, [2.0])]}
        pred, gt = rec._align_and_stack(gt_buf, pred_buf)

        assert pred.dtype == np.float32
        assert gt.dtype == np.float32


# ─────────────────────────────────────────────────────────────────────────────
# Section 4: mcap_player_node — eval plan loading
# ─────────────────────────────────────────────────────────────────────────────

from lerobot_control.mcap_player_node import McapPlayerNode


class TestMcapPlayerNodePlanLoading:
    """Test McapPlayerNode's plan-loading logic without starting ROS2."""

    def _make_plan(self, tmp: Path, episodes: list[dict]) -> Path:
        plan = {
            "checkpoint_path": "/fake/checkpoint",
            "output_dir": str(tmp / "results"),
            "episodes": episodes,
        }
        plan_file = tmp / "eval_plan.json"
        plan_file.write_text(json.dumps(plan))
        return plan_file

    def test_plan_loaded_correctly(self):
        """McapPlayerNode reads eval_plan.json and stores episode list."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            episodes = [
                {"episode_idx": 0, "split_label": "train", "mcap_path": "/data/ep0.mcap"},
                {"episode_idx": 5, "split_label": "val",   "mcap_path": "/data/ep5.mcap"},
            ]
            plan_file = self._make_plan(tmp_path, episodes)

            # Patch declare/get_parameter to use our plan file
            original_get_param = McapPlayerNode.get_parameter

            def fake_get_param(self, name):
                mock = MagicMock()
                if name == "eval_plan_file":
                    mock.get_parameter_value.return_value.string_value = str(plan_file)
                elif name == "warmup_sec":
                    mock.get_parameter_value.return_value.double_value = 0.0
                elif name == "inter_episode_sec":
                    mock.get_parameter_value.return_value.double_value = 0.0
                elif name == "ack_timeout_sec":
                    mock.get_parameter_value.return_value.double_value = 1.0
                return mock

            McapPlayerNode.get_parameter = fake_get_param
            McapPlayerNode.declare_parameter = lambda self, *a, **kw: None

            try:
                node = McapPlayerNode.__new__(McapPlayerNode)
                node.get_logger = lambda: MagicMock()
                node.create_publisher = lambda *a, **kw: MagicMock()
                node.create_subscription = lambda *a, **kw: MagicMock()
                node.get_parameter = lambda name: fake_get_param(node, name)
                node.declare_parameter = lambda *a, **kw: None

                # Call __init__ manually but stop before starting threads
                import threading
                original_thread_start = threading.Thread.start

                started_threads = []

                def fake_start(self):
                    started_threads.append(self)

                threading.Thread.start = fake_start
                try:
                    McapPlayerNode.__init__(node)
                finally:
                    threading.Thread.start = original_thread_start

                # Verify plan loaded correctly
                assert node._plan["episodes"] == episodes
                assert len(node._plan["episodes"]) == 2
                assert node._plan["episodes"][0]["episode_idx"] == 0
                assert node._plan["episodes"][1]["split_label"] == "val"
            finally:
                McapPlayerNode.get_parameter = original_get_param

    def test_missing_plan_file_raises(self):
        """McapPlayerNode should raise FileNotFoundError for missing eval_plan.json."""
        def fake_get_param(self, name):
            mock = MagicMock()
            if name == "eval_plan_file":
                mock.get_parameter_value.return_value.string_value = "/nonexistent/plan.json"
            else:
                mock.get_parameter_value.return_value.double_value = 0.0
            return mock

        original_get_param = McapPlayerNode.get_parameter
        McapPlayerNode.get_parameter = fake_get_param
        McapPlayerNode.declare_parameter = lambda self, *a, **kw: None

        try:
            node = McapPlayerNode.__new__(McapPlayerNode)
            node.get_logger = lambda: MagicMock()
            node.create_publisher = lambda *a, **kw: MagicMock()
            node.create_subscription = lambda *a, **kw: MagicMock()
            node.get_parameter = lambda name: fake_get_param(node, name)
            node.declare_parameter = lambda *a, **kw: None

            with pytest.raises(FileNotFoundError):
                McapPlayerNode.__init__(node)
        finally:
            McapPlayerNode.get_parameter = original_get_param


# ─────────────────────────────────────────────────────────────────────────────
# Section 5: ensure_anvil_eval_importable (import path logic)
# ─────────────────────────────────────────────────────────────────────────────

class TestEnsureImportable:
    def test_anvil_eval_importable_after_call(self):
        """After _ensure_anvil_eval_importable, anvil_eval can be imported."""
        from lerobot_control.eval_recorder_node import _ensure_anvil_eval_importable
        _ensure_anvil_eval_importable()
        import anvil_eval.metrics  # Should not raise

    def test_path_not_duplicated(self):
        """Calling _ensure_anvil_eval_importable twice should not duplicate sys.path entries."""
        from lerobot_control.eval_recorder_node import _ensure_anvil_eval_importable
        _ensure_anvil_eval_importable()
        before = sys.path[:]
        _ensure_anvil_eval_importable()
        assert sys.path == before or sys.path.count(sys.path[0]) == 1
