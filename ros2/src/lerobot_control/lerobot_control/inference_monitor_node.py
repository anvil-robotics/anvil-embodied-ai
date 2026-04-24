#!/usr/bin/env python3
"""Inference Monitor Node — records /monitor/* topics to CSV for offline analysis.

Subscribes to topics published by inference_node (requires
--ros-args -p monitor_enable:=true on the inference node) and writes
a CSV that can be plotted offline with scripts/plot_monitor_csv.py.

Usage:
    ros2 run lerobot_control inference_monitor_node \\
        --ros-args -p output_dir:=/tmp/monitor \\
                   -p use_delta_actions:=true \\
                   -p joint_names:=right_joint1,right_joint2,right_joint3,right_joint4,right_joint5,right_joint6,right_joint7,right_finger_joint1
"""

from __future__ import annotations

import csv
import signal
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


class InferenceMonitorNode(Node):
    """Subscribes to /monitor/* topics and writes a CSV for offline plotting."""

    def __init__(self):
        super().__init__("inference_monitor_node")

        self.declare_parameter("output_dir", "")
        self.declare_parameter("use_delta_actions", False)
        self.declare_parameter("joint_names", "")

        raw_output_dir = self.get_parameter("output_dir").value
        if not raw_output_dir:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            raw_output_dir = f"./inference_monitor_{ts}"
        self._output_dir = Path(raw_output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

        self._use_delta_actions: bool = self.get_parameter("use_delta_actions").value
        raw_joint_names: str = self.get_parameter("joint_names").value
        self._joint_names: list[str] = (
            [n.strip() for n in raw_joint_names.split(",") if n.strip()]
            if raw_joint_names else []
        )

        # CSV writer
        self._csv_path = self._output_dir / "inference_data.csv"
        self._csv_file = open(self._csv_path, "w", newline="")  # noqa: SIM115
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_header_written = False
        self._lock = threading.Lock()

        # Latest data buffers — written by callbacks, flushed by timer
        self._latest_obs: np.ndarray | None = None
        self._latest_raw: np.ndarray | None = None
        self._latest_cmd: np.ndarray | None = None
        self._latest_ts: float = 0.0

        self.create_subscription(Float64MultiArray, "/monitor/obs_state", self._on_obs, 10)
        self.create_subscription(Float64MultiArray, "/monitor/raw_output", self._on_raw, 10)
        self.create_subscription(Float64MultiArray, "/monitor/control_cmd", self._on_cmd, 10)

        # Timer-based flush: poll at ~30 Hz so all three callbacks have had a
        # chance to run before we log the step (avoids single-threaded executor
        # race where _on_obs fires before _on_raw/_on_cmd in the same cycle).
        self.create_timer(1.0 / 30.0, self._timer_flush)

        self.get_logger().info(
            f"[monitor] Listening on /monitor/{{obs_state,raw_output,control_cmd}}\n"
            f"[monitor] use_delta_actions: {self._use_delta_actions}\n"
            f"[monitor] joint_names: {self._joint_names or '(none, will use indices)'}\n"
            f"[monitor] Output: {self._output_dir}\n"
            f"[monitor] Plot:   uv run python scripts/plot_monitor_csv.py {self._csv_path}"
        )

    # ──────────────────────────────────────────────────────────────────────
    # Callbacks
    # ──────────────────────────────────────────────────────────────────────

    def _on_obs(self, msg: Float64MultiArray) -> None:
        with self._lock:
            self._latest_obs = np.array(msg.data, dtype=np.float32)
            self._latest_ts = time.monotonic()

    def _on_raw(self, msg: Float64MultiArray) -> None:
        with self._lock:
            self._latest_raw = np.array(msg.data, dtype=np.float32)

    def _on_cmd(self, msg: Float64MultiArray) -> None:
        with self._lock:
            self._latest_cmd = np.array(msg.data, dtype=np.float32)

    def _timer_flush(self) -> None:
        """Periodic flush at ~30 Hz — log only when all three buffers are ready."""
        with self._lock:
            obs = self._latest_obs
            raw = self._latest_raw
            cmd = self._latest_cmd
            ts = self._latest_ts
            if obs is None or raw is None or cmd is None:
                return
            # Consume raw/cmd so the next tick doesn't duplicate the same step
            self._latest_raw = None
            self._latest_cmd = None

        self._log_step(obs, raw, cmd, ts)

    # ──────────────────────────────────────────────────────────────────────
    # Logging
    # ──────────────────────────────────────────────────────────────────────

    def _log_step(self, obs: np.ndarray, raw: np.ndarray, cmd: np.ndarray, ts: float) -> None:
        if not self._csv_header_written:
            n = len(obs)
            # Write metadata comment lines before the CSV header so plot_monitor_csv.py
            # can auto-configure the plot layout without needing CLI flags.
            joint_names_str = ",".join(self._joint_names) if self._joint_names else ""
            self._csv_file.write(f"# use_delta_actions: {str(self._use_delta_actions).lower()}\n")
            self._csv_file.write(f"# joint_names: {joint_names_str}\n")

            header = (
                ["timestamp"]
                + [f"obs_state_{i}" for i in range(n)]
                + [f"raw_output_{i}" for i in range(len(raw))]
                + [f"control_cmd_{i}" for i in range(len(cmd))]
                + [f"delta_cmd_{i}" for i in range(len(cmd))]
            )
            self._csv_writer.writerow(header)
            self._csv_header_written = True

        delta_cmd = cmd - obs[:len(cmd)]
        row = [f"{ts:.6f}"] + obs.tolist() + raw.tolist() + cmd.tolist() + delta_cmd.tolist()
        self._csv_writer.writerow(row)
        self._csv_file.flush()

    # ──────────────────────────────────────────────────────────────────────
    # Shutdown
    # ──────────────────────────────────────────────────────────────────────

    def destroy_node(self) -> None:
        self._csv_file.close()
        self.get_logger().info(f"[monitor] CSV saved: {self._csv_path}")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = InferenceMonitorNode()

    def _sigint_handler(sig, frame):
        node.get_logger().info("[monitor] Shutting down...")
        rclpy.shutdown()

    signal.signal(signal.SIGINT, _sigint_handler)
    signal.signal(signal.SIGTERM, _sigint_handler)

    try:
        rclpy.spin(node)
    except Exception:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
