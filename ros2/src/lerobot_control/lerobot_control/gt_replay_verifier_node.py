#!/usr/bin/env python3
"""GT-Replayer correctness verifier — live comparator for fake-hardware integration tests.

Seeds aside (handled by the fake-hardware mock's ``ee_seed_pose`` param, set by the
test driver from this same dataset's first recorded pose), this node independently
recomputes the expected published EE command from the dataset alone and compares it
against what ``dataset_gt_replayer_node`` actually publishes, live, over the whole
episode:

    published_cmd[t] (quat)  ==  dataset.observation.state[t+1] (converted to quat)

This holds for BOTH ``ee_abs`` (mcap_converter defines ``action[t]`` as a rot6d
re-encoding of ``observation.state[t+1]`` directly — a near-tautological check of the
passthrough + wire conversion) and ``ee_delta`` (the decoupled per-tick
``obs ∘ delta`` composition in ``inference_node._publish_loop`` — an algebraic
identity of ``ee_delta_forward``/``ee_delta_inverse`` being exact inverses, GIVEN the
mock's echoed obs at tick ``t`` equals ``observation.state[t]``, which the seed makes
true from ``t=0``). So one assertion validates mcap_converter's baked ``action``
column AND inference_node's restore/passthrough math together, not just the ROS-side
math in isolation. See claude_docs/gt-replayer-correctness-test-plan.md.

The last frame of the episode has no ``observation.state[t+1]`` to check against (it
was dropped at convert time — see mcap_converter's 1-frame-lookahead splice) and is
skipped, not compared.

Gripper is compared RAW (no ``gripper_factor``/clamp) — that scaling tunes live
model-inference feel and is orthogonal to converter/pipeline correctness. The test
driver launches the replay run with a config that neutralizes it
(``gripper_factor: 1.0``) so both sides stay in the same raw space.

Scope: this node is FAKE-HARDWARE-ONLY, full stop. Its tight tolerances
(``atol_pos_m=1e-4``, ``atol_rot_deg=0.5``) assume the mock's perfect,
instantaneous, dynamics-free echo — they only make sense against
``fake_hardware_node.py``. A real robot's physically-measured pose will
legitimately deviate from the recorded trajectory (actuation dynamics,
control-loop tracking error, physical latency), and that deviation is not a
bug this node should ever flag. Real-hardware evaluation uses a completely
different tool — a human operator watching and judging task success per
episode, not a numeric tolerance check — see ``scripts/gt_replay_human_eval.py``
and ``claude_docs/real-hardware-gt-replay-eval-plan.md``.

ROS2 parameters:
    dataset (str, required): path to the converted dataset (same one being replayed).
    episode (int): episode index (default 0).
    arms (str): comma-separated arm ids, in the SAME order as the dataset's
        observation_topics insertion order (default "left,right").
    atol_pos_m (float): position tolerance in meters (default 1e-4).
    atol_rot_deg (float): rotation tolerance in degrees, quaternion angle diff
        (default 0.5).
    atol_gripper_m (float): gripper tolerance in meters (default 1e-4).
    report_path (str): where to write the JSON pass/fail report
        (default "/workspace/reports/gt_replay_report.json").
    timeout_sec (float): safety timeout if replay stalls (default 60.0).
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node

from . import dataset_reader
from .ee_runtime import pose_arrival_error

_MAX_REPORTED_FAILURES = 5


class GtReplayVerifierNode(Node):
    """Live comparator: dataset's own observation.state trajectory vs. published commands."""

    def __init__(self):
        super().__init__("gt_replay_verifier")

        self.declare_parameter("dataset", "")
        self.declare_parameter("episode", 0)
        self.declare_parameter("arms", "left,right")
        self.declare_parameter("atol_pos_m", 1e-4)
        self.declare_parameter("atol_rot_deg", 0.5)
        self.declare_parameter("atol_gripper_m", 1e-4)
        self.declare_parameter("report_path", "/workspace/reports/gt_replay_report.json")
        self.declare_parameter("timeout_sec", 60.0)

        dataset = self.get_parameter("dataset").value
        if not dataset:
            raise ValueError("dataset parameter is required for gt_replay_verifier_node")
        self._dataset_path = Path(dataset)
        self._episode_idx = int(self.get_parameter("episode").value)
        self._arms: list[str] = [
            a.strip() for a in str(self.get_parameter("arms").value).split(",") if a.strip()
        ]
        self._atol_pos_m = float(self.get_parameter("atol_pos_m").value)
        self._atol_rot_deg = float(self.get_parameter("atol_rot_deg").value)
        self._atol_gripper_m = float(self.get_parameter("atol_gripper_m").value)
        self._report_path = Path(self.get_parameter("report_path").value)
        self._timeout_sec = float(self.get_parameter("timeout_sec").value)

        self._action_type = dataset_reader.resolve_action_type(
            self._dataset_path, logger=self.get_logger()
        )
        self._obs_quat = dataset_reader.load_episode_observations_quat(
            self._dataset_path, self._episode_idx
        )
        n_arms_in_dataset = self._obs_quat.shape[1] // 8
        if n_arms_in_dataset != len(self._arms):
            raise ValueError(
                f"arms param has {len(self._arms)} arms {self._arms}, but the dataset's "
                f"observation.state implies {n_arms_in_dataset} arms — pass a matching "
                "--ros-args -p arms:=... "
            )

        # Last frame has no observation.state[t+1] to compare against (dropped at
        # convert time) — the comparable range per arm is command indices 0..n_expected-1.
        self._n_expected = max(len(self._obs_quat) - 1, 0)

        self._lock = threading.Lock()
        self._cmd_count: dict[str, int] = {arm: 0 for arm in self._arms}
        self._pos_errs: dict[str, list[float]] = {arm: [] for arm in self._arms}
        self._rot_errs: dict[str, list[float]] = {arm: [] for arm in self._arms}
        self._gripper_errs: dict[str, list[float]] = {arm: [] for arm in self._arms}
        # _failures is capped at _MAX_REPORTED_FAILURES for the report; _fail_count is
        # the true, unbounded count used for the pass/fail decision.
        self._failures: dict[str, list[dict]] = {arm: [] for arm in self._arms}
        self._fail_count: dict[str, int] = {arm: 0 for arm in self._arms}
        self._seed_confirmed: dict[str, bool | None] = {arm: None for arm in self._arms}
        self._finalized = False
        self._start_time = time.monotonic()

        try:
            from anvil_msgs.msg import CommandedEEPose as _EEMsg
            from anvil_msgs.msg import MockEEPose as _MockEEMsg
        except ImportError as exc:
            self.get_logger().error(
                "[gt-replay-verify] anvil_msgs not found — requires colcon build with "
                "the anvil_msgs package. %s", exc
            )
            raise

        for arm in self._arms:
            # /ee_pose_{arm} is the MOCK's echo (MockEEPose, not CommandedEEPose) —
            # this node only ever runs against fake hardware (see module scope
            # note), so it always matches what fake_hardware_node.py publishes.
            self.create_subscription(
                _MockEEMsg, f"/ee_pose_{arm}",
                lambda msg, a=arm: self._on_obs(a, msg), 10,
            )
            self.create_subscription(
                _EEMsg, f"/commanded_ee_{arm}",
                lambda msg, a=arm: self._on_command(a, msg), 10,
            )

        self._timeout_timer = self.create_timer(1.0, self._check_timeout)

        self.get_logger().info(
            f"[gt-replay-verify] Ready. dataset={self._dataset_path} episode={self._episode_idx} "
            f"action_type={self._action_type} arms={self._arms} n_expected={self._n_expected} "
            f"(per arm)"
        )

    # ------------------------------------------------------------------ #
    # Subscriptions
    # ------------------------------------------------------------------ #

    @staticmethod
    def _msg_to_pos_quat_gripper(msg) -> tuple[np.ndarray, np.ndarray, float]:
        p = msg.pose.position
        o = msg.pose.orientation
        pos = np.array([p.x, p.y, p.z], dtype=np.float64)
        quat = np.array([o.x, o.y, o.z, o.w], dtype=np.float64)
        return pos, quat, float(msg.gripper)

    def _on_obs(self, arm: str, msg) -> None:
        """First message only: sanity-check the mock's seed against obs_quat[0].

        ``msg`` is a MockEEPose — unwrap ``.base`` to get the plain pose/gripper
        shape ``_msg_to_pos_quat_gripper`` expects (shared with ``_on_command``,
        which reads a real CommandedEEPose directly, no wrapping).
        """
        with self._lock:
            if self._seed_confirmed[arm] is not None:
                return
            pos, quat, gripper = self._msg_to_pos_quat_gripper(msg.base)
            arm_idx = self._arms.index(arm)
            s0 = arm_idx * 8
            expected = self._obs_quat[0, s0:s0 + 8]
            pos_err, rot_err = pose_arrival_error(np.concatenate([pos, quat]), expected)
            grip_err = abs(gripper - float(expected[7]))
            ok = (
                pos_err <= self._atol_pos_m
                and rot_err <= self._atol_rot_deg
                and grip_err <= self._atol_gripper_m
            )
            self._seed_confirmed[arm] = bool(ok)

        if ok:
            self.get_logger().info(f"[gt-replay-verify] arm={arm}: seed confirmed")
        else:
            self.get_logger().warn(
                f"[gt-replay-verify] arm={arm}: seed MISMATCH "
                f"(pos_err={pos_err:.6f}m rot_err={rot_err:.4f}deg grip_err={grip_err:.6f}m) "
                "— the mock's ee_seed_pose likely wasn't set/parsed correctly"
            )

    def _on_command(self, arm: str, msg) -> None:
        with self._lock:
            if self._finalized:
                return
            n = self._cmd_count[arm]
            self._cmd_count[arm] = n + 1
            if n >= self._n_expected:
                # Extra command beyond the comparable range (e.g. episode tail) —
                # nothing left to check against, not a failure.
                should_finalize = self._all_done_locked()
                if should_finalize:
                    self._finalize_locked()
                return

            pos, quat, gripper = self._msg_to_pos_quat_gripper(msg)
            arm_idx = self._arms.index(arm)
            s0 = arm_idx * 8
            expected = self._obs_quat[n + 1, s0:s0 + 8]

            pos_err, rot_err = pose_arrival_error(np.concatenate([pos, quat]), expected)
            grip_err = abs(gripper - float(expected[7]))

            self._pos_errs[arm].append(pos_err)
            self._rot_errs[arm].append(rot_err)
            self._gripper_errs[arm].append(grip_err)

            if (
                pos_err > self._atol_pos_m
                or rot_err > self._atol_rot_deg
                or grip_err > self._atol_gripper_m
            ):
                self._fail_count[arm] += 1
                if len(self._failures[arm]) < _MAX_REPORTED_FAILURES:
                    self._failures[arm].append({
                        "index": n,
                        "pos_err_m": pos_err,
                        "rot_err_deg": rot_err,
                        "gripper_err_m": grip_err,
                    })

            should_finalize = self._all_done_locked()
            if should_finalize:
                self._finalize_locked()

    def _all_done_locked(self) -> bool:
        return all(self._cmd_count[arm] >= self._n_expected for arm in self._arms)

    # ------------------------------------------------------------------ #
    # Completion
    # ------------------------------------------------------------------ #

    def _check_timeout(self) -> None:
        elapsed = time.monotonic() - self._start_time
        if elapsed <= self._timeout_sec:
            return
        with self._lock:
            if self._finalized:
                return
            self.get_logger().error(
                f"[gt-replay-verify] Timed out after {elapsed:.1f}s "
                f"(counts: { {a: self._cmd_count[a] for a in self._arms} }, "
                f"expected {self._n_expected} per arm)"
            )
            self._finalize_locked(timed_out=True)

    def _finalize_locked(self, timed_out: bool = False) -> None:
        """Write the JSON report and shut down. Caller must hold self._lock."""
        if self._finalized:
            return
        self._finalized = True

        per_arm = {}
        all_passed = not timed_out
        for arm in self._arms:
            n_compared = len(self._pos_errs[arm])
            n_failed = self._fail_count[arm]
            arm_passed = (
                not timed_out
                and n_failed == 0
                and n_compared == self._n_expected
                and bool(self._seed_confirmed[arm])
            )
            all_passed = all_passed and arm_passed
            per_arm[arm] = {
                "n_compared": n_compared,
                "n_expected": self._n_expected,
                "n_failed": n_failed,
                "max_pos_err_m": max(self._pos_errs[arm]) if self._pos_errs[arm] else 0.0,
                "max_rot_err_deg": max(self._rot_errs[arm]) if self._rot_errs[arm] else 0.0,
                "max_gripper_err_m": max(self._gripper_errs[arm]) if self._gripper_errs[arm] else 0.0,
                "seed_confirmed": bool(self._seed_confirmed[arm]) if self._seed_confirmed[arm] is not None else None,
                "first_failures": self._failures[arm],
            }

        report = {
            "all_passed": all_passed,
            "timed_out": timed_out,
            "action_type": self._action_type,
            "dataset": str(self._dataset_path),
            "episode": self._episode_idx,
            "arms": per_arm,
        }

        self._report_path.parent.mkdir(parents=True, exist_ok=True)
        self._report_path.write_text(json.dumps(report, indent=2))

        status = "PASS" if all_passed else "FAIL"
        self.get_logger().info(f"[gt-replay-verify] {status} — report written to {self._report_path}")
        for arm, data in per_arm.items():
            self.get_logger().info(
                f"[gt-replay-verify]   {arm}: n_compared={data['n_compared']}/{data['n_expected']} "
                f"max_pos_err={data['max_pos_err_m']:.6f}m "
                f"max_rot_err={data['max_rot_err_deg']:.4f}deg "
                f"max_grip_err={data['max_gripper_err_m']:.6f}m "
                f"seed_confirmed={data['seed_confirmed']}"
            )

        if rclpy.ok():
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = GtReplayVerifierNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        if node:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
