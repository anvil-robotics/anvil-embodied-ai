#!/usr/bin/env python3
"""One-shot latency check for the fake-hardware mock's echo loop.

Measures wall-clock time from publishing a ``/commanded_ee_<arm>`` command to
seeing it reflected on ``/ee_pose_<arm>``. Deliberately isolated from the full
control loop (no inference, no dataset replay, no hold-gates) — this only
characterizes the mock's OWN pub/sub + processing responsiveness, to rule it
in or out as a contributor to the ee_delta control-loop flakiness under
investigation elsewhere.

Usage (run inside a container on the same ROS/DDS domain as mock-robot, e.g.
via ``docker compose -f docker-compose.fake-hardware.yml run --rm replay bash``
and then, from that shell):
    ros2 run lerobot_control ee_pose_latency_check --arm left --trials 50
"""
from __future__ import annotations

import argparse
import statistics
import time

import numpy as np
import rclpy
from rclpy.node import Node


class EEPoseLatencyCheckNode(Node):
    """Publishes one CommandedEEPose at a time, timing until its MockEEPose echo lands."""

    def __init__(self, arm: str):
        super().__init__("ee_pose_latency_check")
        from anvil_msgs.msg import CommandedEEPose, MockEEPose

        self._CommandedEEPose = CommandedEEPose
        self._cmd_pub = self.create_publisher(CommandedEEPose, f"/commanded_ee_{arm}", 10)
        self.create_subscription(MockEEPose, f"/ee_pose_{arm}", self._on_ee_pose, 10)

        self._pending: np.ndarray | None = None
        self._t_send: float | None = None
        self._latency_sec: float | None = None

    def _on_ee_pose(self, msg) -> None:
        if self._pending is None:
            return
        got = np.array(
            [
                msg.base.pose.position.x, msg.base.pose.position.y, msg.base.pose.position.z,
                msg.base.pose.orientation.x, msg.base.pose.orientation.y,
                msg.base.pose.orientation.z, msg.base.pose.orientation.w,
                msg.base.gripper,
            ]
        )
        if np.allclose(got, self._pending, atol=1e-9):
            self._latency_sec = time.monotonic() - self._t_send
            self._pending = None

    def send_and_wait(self, pose: np.ndarray, timeout_sec: float) -> float | None:
        """Publish *pose* as a command; return echo latency in seconds, or None on timeout."""
        self._pending = pose
        self._latency_sec = None

        msg = self._CommandedEEPose()
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = (
            float(pose[0]), float(pose[1]), float(pose[2]),
        )
        msg.pose.orientation.x, msg.pose.orientation.y = float(pose[3]), float(pose[4])
        msg.pose.orientation.z, msg.pose.orientation.w = float(pose[5]), float(pose[6])
        msg.gripper = float(pose[7])

        self._t_send = time.monotonic()
        self._cmd_pub.publish(msg)

        deadline = time.monotonic() + timeout_sec
        while self._latency_sec is None and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.001)

        return self._latency_sec


def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", default="left")
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--timeout-sec", type=float, default=1.0)
    parsed, _ = parser.parse_known_args()

    rclpy.init(args=args)
    node = EEPoseLatencyCheckNode(parsed.arm)

    latencies_ms: list[float] = []
    misses = 0
    try:
        for i in range(parsed.trials):
            # Distinct position each trial so the echo match is unambiguous
            # even if a stale republish is still in flight.
            pose = np.array(
                [0.1 + 0.001 * i, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0, 0.01 + 0.0001 * i]
            )
            latency = node.send_and_wait(pose, timeout_sec=parsed.timeout_sec)
            if latency is None:
                misses += 1
                node.get_logger().warn(f"trial {i}: TIMEOUT waiting for echo")
                continue
            latencies_ms.append(latency * 1000.0)
            time.sleep(0.01)  # small gap between trials

        if latencies_ms:
            latencies_ms.sort()
            n = len(latencies_ms)
            p95 = latencies_ms[min(n - 1, int(n * 0.95))]
            print(f"\n=== ee_pose echo latency (arm={parsed.arm}, n={n}, {misses} misses) ===")
            print(f"  min    = {latencies_ms[0]:.3f} ms")
            print(f"  median = {statistics.median(latencies_ms):.3f} ms")
            print(f"  mean   = {statistics.mean(latencies_ms):.3f} ms")
            print(f"  p95    = {p95:.3f} ms")
            print(f"  max    = {latencies_ms[-1]:.3f} ms")
        else:
            print("No successful trials — all timed out.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
