"""Eval Recorder Node — records GT + predicted actions, computes per-episode metrics.

Subscribes to:
  /eval/episode_start   — signals start of a new episode (JSON: {episode_idx, split_label})
  /follower_*/commands  — GT actions from MCAP replay
  /eval/follower_*/commands — predicted actions from inference node (eval config)
  /eval/eval_complete   — signals all episodes done

After each episode (detected by GT topic silence + inference drain):
  - Aligns GT and predicted by timestamp (nearest-neighbour)
  - Calls anvil_eval.metrics / plotting / reporting
  - Publishes /eval/episode_ack {episode_idx}

After /eval/eval_complete:
  - Writes summary metrics
  - Shuts down
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, Float64MultiArray, String

log = logging.getLogger(__name__)

# Must match mcap_player_node._EVAL_CTRL_QOS so DDS matches pub↔sub.
_EVAL_CTRL_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)


def _ensure_anvil_eval_importable() -> None:
    """Add anvil_eval to sys.path (repo-relative, no install required)."""
    env_path = os.environ.get("ANVIL_EVAL_PATH")
    if env_path:
        target = env_path
    else:
        # ros2/src/lerobot_control/lerobot_control/eval_recorder_node.py
        # → repo root is 4 levels up
        repo_root = Path(__file__).resolve().parents[4]
        target = str(repo_root / "packages" / "anvil_eval" / "src")

    if target not in sys.path:
        sys.path.insert(0, target)


class EvalRecorderNode(Node):
    """Records GT + predicted actions per episode and computes metrics."""

    def __init__(self):
        super().__init__("eval_recorder")

        # Parameters
        self.declare_parameter(
            "gt_topics",
            [
                "/follower_l_forward_position_controller/commands",
                "/follower_r_forward_position_controller/commands",
            ],
        )
        self.declare_parameter(
            "pred_topics",
            [
                "/eval/follower_l_forward_position_controller/commands",
                "/eval/follower_r_forward_position_controller/commands",
            ],
        )
        self.declare_parameter("arm_names", ["left", "right"])
        self.declare_parameter(
            "controller_joint_order",
            ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7", "finger_joint1"],
        )
        self.declare_parameter("output_dir", "/workspace/eval_results")
        self.declare_parameter("inference_drain_sec", 1.5)
        self.declare_parameter("silence_timeout_sec", 1.0)
        self.declare_parameter("silence_poll_sec", 0.05)

        gt_topics = self.get_parameter("gt_topics").get_parameter_value().string_array_value
        pred_topics = self.get_parameter("pred_topics").get_parameter_value().string_array_value
        self._arm_names: list[str] = list(
            self.get_parameter("arm_names").get_parameter_value().string_array_value
        )
        per_arm_joints: list[str] = list(
            self.get_parameter("controller_joint_order").get_parameter_value().string_array_value
        )
        self._output_dir = Path(
            self.get_parameter("output_dir").get_parameter_value().string_value
        )
        self._inference_drain_sec = (
            self.get_parameter("inference_drain_sec").get_parameter_value().double_value
        )
        self._silence_timeout_sec = (
            self.get_parameter("silence_timeout_sec").get_parameter_value().double_value
        )
        silence_poll_sec = (
            self.get_parameter("silence_poll_sec").get_parameter_value().double_value
        )

        # Build full joint name list: [left_joint1, ..., right_joint1, ...]
        self._joint_names: list[str] = []
        for arm in self._arm_names:
            for j in per_arm_joints:
                self._joint_names.append(f"{arm}_{j}")

        self._output_dir.mkdir(parents=True, exist_ok=True)
        plots_dir = self._output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        self._plots_dir = plots_dir

        # State
        self._lock = threading.Lock()
        self._recording = False
        self._current_ep_idx: int | None = None
        self._current_split: str = "replay"

        # Buffers: {arm_name: [(timestamp_ns, data_list), ...]}
        self._gt_buf: dict[str, list[tuple[int, list[float]]]] = defaultdict(list)
        self._pred_buf: dict[str, list[tuple[int, list[float]]]] = defaultdict(list)
        # Raw model output before postprocessing (from /monitor/raw_output, optional)
        self._raw_buf: list[tuple[int, list[float]]] = []

        # Silence tracking
        self._last_gt_time: float = 0.0
        self._last_pred_time: float = 0.0
        self._all_metrics: list = []
        # Track first-message arrival per arm for diagnostics
        self._gt_seen: set[str] = set()
        self._pred_seen: set[str] = set()
        # Rate-limit "still waiting for GT" warnings (monotonic seconds)
        self._last_waiting_warn: float = 0.0

        # Control topics (TRANSIENT_LOCAL — matches mcap_player_node QoS)
        self._ep_ack_pub = self.create_publisher(String, "/eval/episode_ack", _EVAL_CTRL_QOS)
        self.create_subscription(
            String, "/eval/episode_start", self._on_episode_start, _EVAL_CTRL_QOS
        )
        self.create_subscription(
            Bool, "/eval/eval_complete", self._on_eval_complete, _EVAL_CTRL_QOS
        )

        # GT action subscribers
        for i, topic in enumerate(gt_topics):
            arm = self._arm_names[i] if i < len(self._arm_names) else f"arm_{i}"
            self.create_subscription(
                Float64MultiArray,
                topic,
                lambda msg, a=arm: self._on_gt_action(a, msg),
                10,
            )

        # Predicted action subscribers
        for i, topic in enumerate(pred_topics):
            arm = self._arm_names[i] if i < len(self._arm_names) else f"arm_{i}"
            self.create_subscription(
                Float64MultiArray,
                topic,
                lambda msg, a=arm: self._on_pred_action(a, msg),
                10,
            )

        # Raw model output subscriber (optional — published by inference_monitor_node)
        self.create_subscription(
            Float64MultiArray,
            "/monitor/raw_output",
            self._on_raw_output,
            10,
        )

        # Silence detection timer (controls end-of-episode detection latency)
        self._silence_timer = self.create_timer(silence_poll_sec, self._check_silence)

        self.get_logger().info(
            f"[eval-recorder] Ready. Output: {self._output_dir}. "
            f"Joints ({len(self._joint_names)}): {self._joint_names}"
        )
        self.get_logger().info(
            f"[eval-recorder] GT topics:   {list(gt_topics)}"
        )
        self.get_logger().info(
            f"[eval-recorder] Pred topics: {list(pred_topics)}"
        )

    # ──────────────────────────────────────────────────────────────────────
    # Control callbacks
    # ──────────────────────────────────────────────────────────────────────

    def _on_episode_start(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
        except Exception:
            data = {}

        with self._lock:
            self._current_ep_idx = data.get("episode_idx", 0)
            self._current_split = data.get("split_label", "replay")
            self._gt_buf.clear()
            self._pred_buf.clear()
            self._raw_buf.clear()
            self._last_gt_time = 0.0
            self._last_pred_time = 0.0
            self._recording = True

        self.get_logger().info(
            f"[eval-recorder] Episode {self._current_ep_idx} ({self._current_split}) started"
        )

    def _on_eval_complete(self, msg: Bool) -> None:
        self.get_logger().info("[eval-recorder] Received eval_complete — writing summary")
        self._write_summary()
        rclpy.shutdown()

    # ──────────────────────────────────────────────────────────────────────
    # Action callbacks
    # ──────────────────────────────────────────────────────────────────────

    def _on_gt_action(self, arm: str, msg: Float64MultiArray) -> None:
        first_seen = False
        with self._lock:
            if arm not in self._gt_seen:
                self._gt_seen.add(arm)
                first_seen = True
            if not self._recording:
                return
            ts = self.get_clock().now().nanoseconds
            self._gt_buf[arm].append((ts, list(msg.data)))
            self._last_gt_time = time.monotonic()
        if first_seen:
            self.get_logger().info(
                f"[eval-recorder] First GT sample on arm '{arm}' (dim={len(msg.data)})"
            )

    def _on_pred_action(self, arm: str, msg: Float64MultiArray) -> None:
        first_seen = False
        with self._lock:
            if arm not in self._pred_seen:
                self._pred_seen.add(arm)
                first_seen = True
            if not self._recording:
                return
            ts = self.get_clock().now().nanoseconds
            self._pred_buf[arm].append((ts, list(msg.data)))
            self._last_pred_time = time.monotonic()
        if first_seen:
            self.get_logger().info(
                f"[eval-recorder] First pred sample on arm '{arm}' (dim={len(msg.data)})"
            )

    def _on_raw_output(self, msg: Float64MultiArray) -> None:
        with self._lock:
            if not self._recording:
                return
            ts = self.get_clock().now().nanoseconds
            self._raw_buf.append((ts, list(msg.data)))

    # ──────────────────────────────────────────────────────────────────────
    # Silence detection → episode finalization
    # ──────────────────────────────────────────────────────────────────────

    def _check_silence(self) -> None:
        now = time.monotonic()
        warn_no_gt = False
        with self._lock:
            if not self._recording:
                return
            if self._last_gt_time == 0.0:
                # Still waiting for the first GT sample — warn every 5s so a
                # topic-name / QoS mismatch surfaces fast instead of silently
                # hitting the ack timeout.
                if now - self._last_waiting_warn > 5.0:
                    self._last_waiting_warn = now
                    warn_no_gt = True

            gt_silent_for = now - self._last_gt_time
            pred_silent_for = now - self._last_pred_time

        if warn_no_gt:
            self.get_logger().warning(
                f"[eval-recorder] Ep {self._current_ep_idx}: still waiting for first GT sample — "
                "check that MCAP GT topic matches eval-recorder gt_topics parameter"
            )
            return
        if self._last_gt_time == 0.0:
            return

        # GT topic has gone silent (bag finished) AND inference has drained
        gt_done = gt_silent_for > self._silence_timeout_sec
        pred_done = pred_silent_for > (self._silence_timeout_sec + self._inference_drain_sec)

        if gt_done and pred_done:
            self._finalize_episode()

    # ──────────────────────────────────────────────────────────────────────
    # Episode finalization
    # ──────────────────────────────────────────────────────────────────────

    def _finalize_episode(self) -> None:
        with self._lock:
            if not self._recording:
                return
            self._recording = False
            ep_idx = self._current_ep_idx
            split_label = self._current_split
            gt_buf = dict(self._gt_buf)
            pred_buf = dict(self._pred_buf)
            raw_buf = list(self._raw_buf)

        self.get_logger().info(
            f"[eval-recorder] Finalizing ep {ep_idx}: "
            f"GT frames={sum(len(v) for v in gt_buf.values())}, "
            f"Pred frames={sum(len(v) for v in pred_buf.values())}"
        )

        try:
            predicted, ground_truth = self._align_and_stack(gt_buf, pred_buf)

            if predicted.shape[0] < 2:
                self.get_logger().warning(
                    f"[eval-recorder] Ep {ep_idx}: too few aligned frames ({predicted.shape[0]}), skipping"
                )
            else:
                raw_output = self._align_raw_to_gt(raw_buf, gt_buf) if raw_buf else None
                metrics = self._compute_and_save(
                    predicted, ground_truth, ep_idx, split_label, raw_output=raw_output
                )
                self._all_metrics.append(metrics)
                self.get_logger().info(
                    f"[eval-recorder] Ep {ep_idx} MAE={metrics.mae:.4f}"
                )
        except Exception as e:
            import traceback
            self.get_logger().error(f"[eval-recorder] Ep {ep_idx} finalization error: {e}")
            traceback.print_exc()

        # Acknowledge to mcap-player
        ack_msg = String()
        ack_msg.data = json.dumps({"episode_idx": ep_idx})
        self._ep_ack_pub.publish(ack_msg)
        self.get_logger().info(f"[eval-recorder] Ack sent for ep {ep_idx}")

    def _align_and_stack(
        self,
        gt_buf: dict[str, list[tuple[int, list[float]]]],
        pred_buf: dict[str, list[tuple[int, list[float]]]],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Align GT and predicted by timestamp, return (T, D) arrays."""
        # Merge arms in order
        def merge_arms(buf: dict) -> list[tuple[int, list[float]]]:
            """Merge arm buffers: align by timestamp, concatenate arm data."""
            if not buf:
                return []
            arms = self._arm_names
            arm_data = {a: buf.get(a, []) for a in arms}

            # Use first arm as reference timestamps
            ref_arm = arms[0]
            ref_entries = arm_data[ref_arm]
            if not ref_entries:
                return []

            merged = []
            for ts, ref_vals in ref_entries:
                row = list(ref_vals)
                for arm in arms[1:]:
                    entries = arm_data[arm]
                    if not entries:
                        continue
                    # Nearest-neighbour match
                    best = min(entries, key=lambda x: abs(x[0] - ts))
                    row.extend(best[1])
                merged.append((ts, row))
            return merged

        gt_merged = merge_arms(gt_buf)
        pred_merged = merge_arms(pred_buf)

        if not gt_merged or not pred_merged:
            return np.zeros((0, len(self._joint_names))), np.zeros((0, len(self._joint_names)))

        # Align pred to gt timestamps
        gt_ts = [t for t, _ in gt_merged]
        gt_data = [d for _, d in gt_merged]
        pred_ts = [t for t, _ in pred_merged]
        pred_data = [d for _, d in pred_merged]

        aligned_pred = []
        aligned_gt = []
        for i, ts in enumerate(gt_ts):
            # Find nearest pred timestamp
            j = min(range(len(pred_ts)), key=lambda k: abs(pred_ts[k] - ts))
            aligned_pred.append(pred_data[j])
            aligned_gt.append(gt_data[i])

        predicted = np.array(aligned_pred, dtype=np.float32)
        ground_truth = np.array(aligned_gt, dtype=np.float32)

        # Truncate to same dim (in case arm counts differ)
        d = min(predicted.shape[1], ground_truth.shape[1], len(self._joint_names))
        return predicted[:, :d], ground_truth[:, :d]

    def _align_raw_to_gt(
        self,
        raw_buf: list[tuple[int, list[float]]],
        gt_buf: dict[str, list[tuple[int, list[float]]]],
    ) -> np.ndarray | None:
        """Align /monitor/raw_output to GT timestamps; return (T, D) array or None.

        Uses GT timestamps as reference so the resulting array has the same
        number of rows as the predicted array from _align_and_stack.
        """
        if not raw_buf:
            return None
        ref_arm = self._arm_names[0] if self._arm_names else None
        gt_entries = gt_buf.get(ref_arm, []) if ref_arm else []
        if not gt_entries:
            return None

        raw_ts = [t for t, _ in raw_buf]
        raw_data = [d for _, d in raw_buf]

        aligned = []
        for gt_t, _ in gt_entries:
            j = min(range(len(raw_ts)), key=lambda k: abs(raw_ts[k] - gt_t))
            aligned.append(raw_data[j])

        return np.array(aligned, dtype=np.float32)

    def _compute_and_save(
        self,
        predicted: np.ndarray,
        ground_truth: np.ndarray,
        ep_idx: int,
        split_label: str,
        raw_output: np.ndarray | None = None,
    ):
        """Compute metrics and optionally save plot using anvil_eval."""
        _ensure_anvil_eval_importable()
        from anvil_eval.metrics import compute_episode_metrics

        joint_names = self._joint_names[: predicted.shape[1]]
        metrics = compute_episode_metrics(
            predicted, ground_truth, joint_names, ep_idx, split_label
        )

        # Plotting is optional — skip gracefully if matplotlib is unavailable
        try:
            from anvil_eval.plotting import plot_episode_joints
            plot_path = self._plots_dir / f"episode_{ep_idx:04d}_{split_label}.png"
            # Trim raw_output to same T and D as predicted
            raw_trimmed: np.ndarray | None = None
            if raw_output is not None and raw_output.shape[0] > 0:
                t = predicted.shape[0]
                d = predicted.shape[1]
                raw_trimmed = raw_output[:t, :d]
            plot_episode_joints(
                predicted, ground_truth, joint_names, metrics, plot_path,
                raw_output=raw_trimmed,
            )
        except ImportError:
            self.get_logger().warning(
                "[eval-recorder] matplotlib not available — skipping plot for ep %d", ep_idx
            )

        return metrics

    def _write_summary(self) -> None:
        """Write summary metrics after all episodes."""
        if not self._all_metrics:
            self.get_logger().warning("[eval-recorder] No metrics to summarize")
            return

        _ensure_anvil_eval_importable()
        from anvil_eval.plotting import plot_summary_box_plot
        from anvil_eval.reporting import write_metrics_csv, write_metrics_summary

        write_metrics_summary(self._all_metrics, self._output_dir / "metrics_summary.json")
        write_metrics_csv(self._all_metrics, self._output_dir / "metrics_per_episode.csv")

        joint_names = self._joint_names[: self._all_metrics[0].per_joint_mae.__len__()]
        plot_summary_box_plot(
            self._all_metrics, joint_names, self._plots_dir / "summary_per_joint_mae.png"
        )

        # Print summary to terminal
        all_mae = [m.mae for m in self._all_metrics]
        self.get_logger().info(
            f"[eval-recorder] Summary: {len(self._all_metrics)} episodes, "
            f"mean MAE={np.mean(all_mae):.4f} ± {np.std(all_mae):.4f}"
        )
        self.get_logger().info(f"[eval-recorder] Results saved to: {self._output_dir}")


def main(args=None):
    rclpy.init(args=args)
    node = EvalRecorderNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, Exception):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
