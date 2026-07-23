#!/usr/bin/env python3
"""Single-threaded GT-replayer — an experimental, structurally-simpler alternative
to ``dataset_gt_replayer_node.py`` for debugging the ee_delta residual flakiness.

Why this exists: the production replayer subclasses ``LeRobotInferenceNode``,
whose ``_obs_timer``/``_publish_timer`` run on separate threads/callback groups
and coordinate through two independent hold-gates (a fake-hardware-only
sequence guard, and a position-proximity guard) over a lock-protected shared
obs snapshot. That's a reasonable design for the general inference case
(obs/preprocess decoupled from publish timing), but for ee_delta GT-replay
specifically it's plausibly the SOURCE of the residual flakiness itself: two
independently-scheduled timers racing on a shared snapshot could plausibly
explain the observed failure signature (both arms diverging at the identical
tick — a single bad snapshot read would hit every arm's compose at once).

This node collapses obs-read / compose / publish into ONE per-tick step on
ONE timer (no locks, no separate obs thread, nothing to race). Structure:

  0. Seeding: this node does NOT home. It assumes the mock's initial pose has
     already been seeded to the dataset's own frame-0 observation (the
     existing ``ee_seed_pose`` mechanism on ``fake_hardware_node.py``,
     wired the same way ``gt_replay_correctness_test.py`` already does it) —
     so replay can start at row 0 immediately, no ramp-toward-home phase.
  1. Every tick: if the last published target hasn't been reached yet
     (``pose_arrival_error`` vs the freshest subscribed observation exceeds
     tolerance), HOLD — don't pop, don't publish, try again next tick. Once
     arrived, pop action[cursor], compose against the freshest observation
     (ee_delta) or pass through (ee_abs), publish, advance the cursor.
  2. The whole episode is driven by this single fixed-rate timer
     (``control_frequency``, default 30 Hz) — no separate obs timer.

Scope: EE modes only (ee_abs, ee_delta). No homing, no joint mode, no model
inference — a deliberately narrow debugging tool, not a replacement for
``dataset_gt_replayer_node.py``'s production real-hardware path.

ROS2 parameters:
    dataset (str, required): path to the converted dataset.
    episode (int): episode index (default 0).
    arms (str): comma-separated arm ids, in observation_topics order (default "left,right").
    control_frequency (float): step rate in Hz (default 30.0).
    anchor_atol_pos_m (float): wait-until-arrived position tolerance, m (default 0.025).
    anchor_atol_rot_deg (float): wait-until-arrived rotation tolerance, deg (default 6.0).
    hold_timeout_sec (float): give up (fail) if held this long without arriving (default 5.0).
    wait_until_arrived (bool): gate pop/publish on the previous target being reached
        (default True). False disables the gate entirely — every tick unconditionally
        pops/composes/publishes at the fixed control_frequency regardless of whether
        the prior target was reached (a straight timed loop, no arrival check at all).
        Exists to A/B this gate's effect on fake-hardware correctness/flakiness.
    loop (bool): restart from row 0 on completion instead of stopping (default False).
    hold_last (bool): keep the node alive (idle) after completion instead of shutting down (default True).
    mock_ee_pose_echo (bool): /ee_pose_<arm> carries MockEEPose (fake hardware) vs
        plain CommandedEEPose (real hardware) (default False).
    completion_signal_path (str): if set, JSON status file written on completion (default "").
    dry_run (bool): log intended actions without publishing or advancing cursor (default False).
    debug (bool): verbose numbered per-tick trace (default False).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node

from . import dataset_reader
from .ee_runtime import ee_delta_restore_step, ee_poses_from_chunk, pose_arrival_error


class SingleThreadGtReplayerNode(Node):
    """One timer, one thread: wait-until-arrived, then pop/compose/publish."""

    def __init__(self):
        super().__init__("single_thread_gt_replayer")

        self.declare_parameter("dataset", "")
        self.declare_parameter("episode", 0)
        self.declare_parameter("arms", "left,right")
        self.declare_parameter("control_frequency", 30.0)
        self.declare_parameter("anchor_atol_pos_m", 0.025)
        self.declare_parameter("anchor_atol_rot_deg", 6.0)
        self.declare_parameter("hold_timeout_sec", 5.0)
        self.declare_parameter("wait_until_arrived", True)
        self.declare_parameter("loop", False)
        self.declare_parameter("hold_last", True)
        self.declare_parameter("mock_ee_pose_echo", False)
        self.declare_parameter("completion_signal_path", "")
        self.declare_parameter("dry_run", False)
        self.declare_parameter("debug", False)

        dataset = self.get_parameter("dataset").value
        if not dataset:
            raise ValueError("dataset parameter is required for single_thread_gt_replayer_node")
        self._dataset_path = Path(dataset)
        self._episode_idx = int(self.get_parameter("episode").value)
        self._arms: list[str] = [
            a.strip() for a in str(self.get_parameter("arms").value).split(",") if a.strip()
        ]
        self._n_arms = len(self._arms)
        self._control_freq = float(self.get_parameter("control_frequency").value)
        self._atol_pos = float(self.get_parameter("anchor_atol_pos_m").value)
        self._atol_rot = float(self.get_parameter("anchor_atol_rot_deg").value)
        self._hold_timeout_sec = float(self.get_parameter("hold_timeout_sec").value)
        self._wait_until_arrived = bool(self.get_parameter("wait_until_arrived").value)
        self._loop = bool(self.get_parameter("loop").value)
        self._hold_last = bool(self.get_parameter("hold_last").value)
        self._mock_ee_pose_echo = bool(self.get_parameter("mock_ee_pose_echo").value)
        self._completion_signal_path = self.get_parameter("completion_signal_path").value
        self._dry_run = bool(self.get_parameter("dry_run").value)
        self._debug = bool(self.get_parameter("debug").value)

        self._action_type = dataset_reader.resolve_action_type(
            self._dataset_path, logger=self.get_logger()
        )
        if self._action_type not in ("ee_abs", "ee_delta"):
            raise ValueError(
                f"single_thread_gt_replayer_node only supports ee_abs/ee_delta, "
                f"got action_type={self._action_type!r}"
            )
        self._is_ee_delta = self._action_type == "ee_delta"

        actions = dataset_reader.load_episode_actions(self._dataset_path, self._episode_idx)
        if actions is None or len(actions) == 0:
            raise ValueError(
                f"No action rows found for episode {self._episode_idx} in {self._dataset_path}"
            )
        self._gt_actions = actions.astype(np.float32)

        # Replay state
        self._cursor = 0
        self._tick = 0
        self._latest_obs: np.ndarray | None = None  # (8*n_arms,) quat layout
        self._obs_received = {arm: False for arm in self._arms}
        self._last_commanded: np.ndarray | None = None  # (8*n_arms,) quat layout
        self._hold_start_time: float | None = None
        self._done = False
        self._signal_written = False

        try:
            from anvil_msgs.msg import CommandedEEPose, MockEEPose
        except ImportError as exc:
            self.get_logger().error(f"anvil_msgs not found — requires colcon build. {exc}")
            raise

        obs_msg_type = MockEEPose if self._mock_ee_pose_echo else CommandedEEPose
        self._cmd_pubs = {}
        for arm_idx, arm in enumerate(self._arms):
            self._cmd_pubs[arm] = self.create_publisher(CommandedEEPose, f"/commanded_ee_{arm}", 10)
            self.create_subscription(
                obs_msg_type, f"/ee_pose_{arm}",
                lambda msg, i=arm_idx, a=arm: self._on_obs(i, a, msg), 10,
            )

        self._timer = self.create_timer(1.0 / self._control_freq, self._step)

        self.get_logger().info(
            f"[single_thread_replay] Ready. dataset={self._dataset_path} "
            f"episode={self._episode_idx} action_type={self._action_type} arms={self._arms} "
            f"n_actions={len(self._gt_actions)} control_freq={self._control_freq}Hz "
            f"wait_until_arrived={self._wait_until_arrived} "
            f"anchor_atol=({self._atol_pos}m,{self._atol_rot}deg) mock_ee_pose_echo={self._mock_ee_pose_echo}"
        )

    # ------------------------------------------------------------------ #
    # Observation subscription — single-threaded, no lock needed: only one
    # rclpy callback (this one, or _step) ever executes at a time.
    # ------------------------------------------------------------------ #

    def _on_obs(self, arm_idx: int, arm: str, msg) -> None:
        base = msg.base if self._mock_ee_pose_echo else msg
        p, o = base.pose.position, base.pose.orientation
        if self._latest_obs is None:
            self._latest_obs = np.zeros(8 * self._n_arms, dtype=np.float64)
        s0 = arm_idx * 8
        self._latest_obs[s0:s0 + 8] = [p.x, p.y, p.z, o.x, o.y, o.z, o.w, base.gripper]
        self._obs_received[arm] = True

    # ------------------------------------------------------------------ #
    # Main control loop — one timer, one thread.
    # ------------------------------------------------------------------ #

    def _step(self) -> None:
        if self._done:
            return
        self._tick += 1

        if not all(self._obs_received.values()):
            if self._debug:
                self.get_logger().info(f"[single_thread_replay t={self._tick}] waiting for first obs from all arms")
            return

        # --- wait-until-arrived gate (skippable — see wait_until_arrived param) ---
        if self._wait_until_arrived and self._last_commanded is not None:
            worst_pos, worst_rot, worst_arm = 0.0, 0.0, None
            for i in range(self._n_arms):
                s0 = i * 8
                pos_err, rot_err = pose_arrival_error(
                    self._latest_obs[s0:s0 + 8], self._last_commanded[s0:s0 + 8]
                )
                if pos_err > worst_pos:
                    worst_pos, worst_arm = pos_err, i
                worst_rot = max(worst_rot, rot_err)
            held = worst_pos > self._atol_pos or worst_rot > self._atol_rot
            if held:
                if self._hold_start_time is None:
                    self._hold_start_time = time.monotonic()
                elapsed = time.monotonic() - self._hold_start_time
                if self._debug:
                    self.get_logger().info(
                        f"[single_thread_replay t={self._tick}] HELD: arm={worst_arm} "
                        f"pos_err={worst_pos:.4f}m rot_err={worst_rot:.2f}deg "
                        f"(tol pos<={self._atol_pos}m rot<={self._atol_rot}deg) held_for={elapsed:.2f}s"
                    )
                if elapsed > self._hold_timeout_sec:
                    self.get_logger().error(
                        f"[single_thread_replay] HOLD timeout after {elapsed:.1f}s at "
                        f"cursor={self._cursor}/{len(self._gt_actions)} — arm={worst_arm} never arrived"
                    )
                    self._write_signal({"status": "hold_timeout", "cursor": self._cursor})
                    self._done = True
                    if rclpy.ok():
                        rclpy.shutdown()
                    return
                return
            self._hold_start_time = None

        # --- pop / compose / publish ---
        if self._cursor >= len(self._gt_actions):
            self._finish()
            return

        row = self._gt_actions[self._cursor]
        abs_action = ee_delta_restore_step(row, self._latest_obs) if self._is_ee_delta else row

        if self._debug:
            self.get_logger().info(
                f"[single_thread_replay t={self._tick}] row={self._cursor}/{len(self._gt_actions)} "
                f"action[{self._cursor}]={row.round(4).tolist()} "
                f"obs={self._latest_obs.round(4).tolist()} "
                f"cmd_abs={np.asarray(abs_action).round(4).tolist()}"
            )

        if self._dry_run:
            self._cursor += 1
            return

        self._publish(abs_action)
        self._last_commanded = self._to_quat_layout(abs_action)
        self._cursor += 1

    def _finish(self) -> None:
        if self._loop:
            self._cursor = 0
            self._hold_start_time = None
            return
        self.get_logger().info(
            f"[single_thread_replay] episode complete — {len(self._gt_actions)} rows replayed"
        )
        self._write_signal({"status": "complete", "rows_replayed": len(self._gt_actions)})
        self._done = True
        if not self._hold_last and rclpy.ok():
            rclpy.shutdown()

    # ------------------------------------------------------------------ #
    # Publish / conversion helpers
    # ------------------------------------------------------------------ #

    def _publish(self, abs_action_rot6d: np.ndarray) -> None:
        from anvil_msgs.msg import CommandedEEPose
        from geometry_msgs.msg import Point, Pose, Quaternion
        from std_msgs.msg import Header

        now = self.get_clock().now().to_msg()
        poses = ee_poses_from_chunk(np.asarray(abs_action_rot6d)[np.newaxis, :], n_arms=self._n_arms)[0]
        for i, arm in enumerate(self._arms):
            pose_dict = poses[i]
            pos, quat_xyzw, gripper = pose_dict["pos"], pose_dict["quat_xyzw"], pose_dict["gripper"]
            msg = CommandedEEPose()
            msg.header = Header(stamp=now, frame_id="world")
            msg.pose = Pose(
                position=Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2])),
                orientation=Quaternion(
                    x=float(quat_xyzw[0]), y=float(quat_xyzw[1]),
                    z=float(quat_xyzw[2]), w=float(quat_xyzw[3]),
                ),
            )
            msg.gripper = float(gripper)
            self._cmd_pubs[arm].publish(msg)

    def _to_quat_layout(self, abs_action_rot6d: np.ndarray) -> np.ndarray:
        """Absolute rot6d action (10*n_arms) -> quat layout (8*n_arms), for the
        next tick's wait-until-arrived comparison against live obs (also quat layout).
        """
        poses = ee_poses_from_chunk(np.asarray(abs_action_rot6d)[np.newaxis, :], n_arms=self._n_arms)[0]
        out = np.zeros(8 * self._n_arms, dtype=np.float64)
        for i in range(self._n_arms):
            pos, quat_xyzw, gripper = poses[i]["pos"], poses[i]["quat_xyzw"], poses[i]["gripper"]
            out[i * 8:i * 8 + 8] = [*pos, *quat_xyzw, gripper]
        return out

    def _write_signal(self, extra: dict) -> None:
        if not self._completion_signal_path or self._signal_written:
            return
        self._signal_written = True
        path = Path(self._completion_signal_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"episode": self._episode_idx, **extra}))


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = SingleThreadGtReplayerNode()
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
