#!/usr/bin/env python3
"""Dedupe-based GT-replay verifier — companion to ``single_thread_gt_replayer_node.py``.

Unlike ``gt_replay_verifier_node.py`` (which compares each PUBLISHED command
against ``dataset.observation.state[t+1]``, counted by command-arrival order),
this node only ever subscribes to ``/ee_pose_<arm>`` — the mock's OBSERVED
echo — and compares the robot's actual reported trajectory against the
dataset's own recorded ``observation.state`` sequence directly. This is a more
end-to-end / black-box check: it validates what the (simulated) robot actually
reported reaching, not just what was sent out.

Mechanism: ``/ee_pose_<arm>`` republishes at a fixed high rate regardless of
whether the underlying pose changed (see ``fake_hardware_node.py``'s
``ee_pose_timer``), so most incoming messages are exact repeats of the last
one — these must be filtered out ("ignore same pose with high frequency").
The dedup key is ``MockEEPose.sequence``, NOT the pose value: sequence is
bumped by the mock exactly once per RECEIVED command
(``fake_hardware_node.py``'s ``_ee_command_callback``), regardless of whether
that command's value happens to equal the previous one — e.g. a bimanual
dataset row that moves only one arm still commands the OTHER arm's
(unchanged) target every tick, and that arm's sequence still advances.
Deduping on pose VALUE instead would silently collapse two genuinely distinct
dataset rows into one trace entry whenever they command an identical or
near-identical pose (a stationary/idle arm, or two frames that quantize to
the same float32 value) — permanently shifting that arm's trace index versus
the dataset row index from that point on, an easy-to-miss bug this node
specifically avoids by keying off sequence instead of value.

Because ``single_thread_gt_replayer_node`` publishes to every arm's topic on
every successful (non-held) tick, every arm's sequence advances in lockstep
with the replayer's cursor — so each arm's deduped trace entry ``n`` aligns
1:1 with ``dataset.observation.state[n]`` independently, no cross-arm
synchronization needed. Entry 0 is the seeded starting pose
(``observation.state[0]``, sequence 0, before any command); entry i
thereafter is the pose reached after the replayer's i-th published command.

Scope: FAKE-HARDWARE-ONLY (``MockEEPose.sequence`` doesn't exist on real
hardware's plain ``CommandedEEPose`` — see ``fake_hardware_node.py``'s module
docstring).

ROS2 parameters:
    dataset (str, required): path to the converted dataset (same one being replayed).
    episode (int): episode index (default 0).
    arms (str): comma-separated arm ids, in observation_topics order (default "left,right").
    atol_pos_m (float): pass/fail position tolerance in meters (default 1e-4).
    atol_rot_deg (float): pass/fail rotation tolerance in degrees (default 0.5).
    atol_gripper_m (float): pass/fail gripper tolerance in meters (default 1e-4).
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


class SingleThreadGtReplayVerifierNode(Node):
    """Dedupe /ee_pose_<arm> by MockEEPose.sequence; compare 1:1 vs dataset obs."""

    def __init__(self):
        super().__init__("single_thread_gt_replay_verifier")

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
            raise ValueError("dataset parameter is required for single_thread_gt_replay_verifier_node")
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
                f"observation.state implies {n_arms_in_dataset} arms"
            )
        # Every recorded observation row (including the seed at index 0) is expected
        # to appear as a distinct entry in the deduped trace — unlike the
        # command-count-based verifier, there's no "last frame has no t+1" dropped row.
        self._n_expected = len(self._obs_quat)

        self._lock = threading.Lock()
        self._last_seen_seq: dict[str, int | None] = {arm: None for arm in self._arms}
        self._trace_len: dict[str, int] = {arm: 0 for arm in self._arms}
        self._pos_errs: dict[str, list[float]] = {arm: [] for arm in self._arms}
        self._rot_errs: dict[str, list[float]] = {arm: [] for arm in self._arms}
        self._gripper_errs: dict[str, list[float]] = {arm: [] for arm in self._arms}
        self._failures: dict[str, list[dict]] = {arm: [] for arm in self._arms}
        self._fail_count: dict[str, int] = {arm: 0 for arm in self._arms}
        self._seed_confirmed: dict[str, bool | None] = {arm: None for arm in self._arms}
        self._finalized = False
        self._start_time = time.monotonic()

        try:
            from anvil_msgs.msg import MockEEPose as _MockEEMsg
        except ImportError as exc:
            self.get_logger().error(f"anvil_msgs not found — requires colcon build. {exc}")
            raise

        for arm in self._arms:
            self.create_subscription(
                _MockEEMsg, f"/ee_pose_{arm}",
                lambda msg, a=arm: self._on_obs(a, msg), 10,
            )

        self._timeout_timer = self.create_timer(1.0, self._check_timeout)

        self.get_logger().info(
            f"[single_thread_verify] Ready. dataset={self._dataset_path} episode={self._episode_idx} "
            f"action_type={self._action_type} arms={self._arms} n_expected={self._n_expected} (per arm)"
        )

    # ------------------------------------------------------------------ #
    # Subscription — dedupe by sequence, then compare
    # ------------------------------------------------------------------ #

    @staticmethod
    def _msg_to_pose8(base) -> np.ndarray:
        p, o = base.pose.position, base.pose.orientation
        return np.array([p.x, p.y, p.z, o.x, o.y, o.z, o.w, base.gripper], dtype=np.float64)

    def _on_obs(self, arm: str, msg) -> None:
        seq = int(msg.sequence)

        with self._lock:
            if self._finalized:
                return

            last_seq = self._last_seen_seq[arm]
            if last_seq is not None and seq == last_seq:
                return  # redundant periodic republish of an unchanged command — not a new entry
            self._last_seen_seq[arm] = seq

            n = self._trace_len[arm]
            self._trace_len[arm] = n + 1

            if n >= self._n_expected:
                # More distinct sequence values than the dataset has rows — nothing
                # left to compare against; not counted as pass or fail, just ignored.
                if self._all_done_locked():
                    self._finalize_locked()
                return

            pose8 = self._msg_to_pose8(msg.base)
            arm_idx = self._arms.index(arm)
            s0 = arm_idx * 8
            expected = self._obs_quat[n, s0:s0 + 8]
            pos_err, rot_err = pose_arrival_error(pose8, expected)
            grip_err = abs(pose8[7] - float(expected[7]))

            if n == 0:
                ok = (
                    pos_err <= self._atol_pos_m
                    and rot_err <= self._atol_rot_deg
                    and grip_err <= self._atol_gripper_m
                )
                self._seed_confirmed[arm] = bool(ok)
                if ok:
                    self.get_logger().info(f"[single_thread_verify] arm={arm}: seed confirmed")
                else:
                    self.get_logger().warn(
                        f"[single_thread_verify] arm={arm}: seed MISMATCH "
                        f"(pos_err={pos_err:.6f}m rot_err={rot_err:.4f}deg grip_err={grip_err:.6f}m)"
                    )

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

            if self._all_done_locked():
                self._finalize_locked()

    def _all_done_locked(self) -> bool:
        return all(self._trace_len[arm] >= self._n_expected for arm in self._arms)

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
                f"[single_thread_verify] Timed out after {elapsed:.1f}s "
                f"(trace_len: { {a: self._trace_len[a] for a in self._arms} }, "
                f"expected {self._n_expected} per arm)"
            )
            self._finalize_locked(timed_out=True)

    def _finalize_locked(self, timed_out: bool = False) -> None:
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
            "verifier": "single_thread_dedupe",
            "arms": per_arm,
        }

        self._report_path.parent.mkdir(parents=True, exist_ok=True)
        self._report_path.write_text(json.dumps(report, indent=2))

        status = "PASS" if all_passed else "FAIL"
        self.get_logger().info(f"[single_thread_verify] {status} — report written to {self._report_path}")
        for arm, data in per_arm.items():
            self.get_logger().info(
                f"[single_thread_verify]   {arm}: n_compared={data['n_compared']}/{data['n_expected']} "
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
        node = SingleThreadGtReplayVerifierNode()
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
