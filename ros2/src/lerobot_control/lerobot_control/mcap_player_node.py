"""MCAP Player Node — replays MCAP files as ROS2 topics for offline eval.

Reads an eval_plan.json produced by `anvil-eval-ros`, then for each episode:
  1. Publishes /eval/episode_start with {episode_idx, split_label}
  2. Runs `ros2 bag play <mcap_path> --storage mcap` as a subprocess
  3. Waits for /eval/episode_ack from eval_recorder_node (episode processed)

After all episodes: publishes /eval/eval_complete and shuts down.
"""

from __future__ import annotations

import json
import logging
import subprocess
import threading
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, String

log = logging.getLogger(__name__)

# Coordination handshake topics must latch so late-joining peers (cross-container
# DDS discovery can lag the `depends_on: service_healthy` gate) still receive
# the most recent sample — otherwise episode_start/ack can be dropped on the floor.
_EVAL_CTRL_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)


class McapPlayerNode(Node):
    """Orchestrates MCAP replay for offline evaluation."""

    def __init__(self):
        super().__init__("mcap_player")

        # Parameters
        self.declare_parameter("eval_plan_file", "/workspace/eval_plan.json")
        self.declare_parameter("warmup_sec", 5.0)
        self.declare_parameter("inter_episode_sec", 1.0)
        self.declare_parameter("ack_timeout_sec", 120.0)
        self.declare_parameter("post_start_sleep_sec", 0.2)

        eval_plan_file = self.get_parameter("eval_plan_file").get_parameter_value().string_value
        self._warmup_sec = self.get_parameter("warmup_sec").get_parameter_value().double_value
        self._inter_episode_sec = self.get_parameter("inter_episode_sec").get_parameter_value().double_value
        self._ack_timeout_sec = self.get_parameter("ack_timeout_sec").get_parameter_value().double_value
        self._post_start_sleep_sec = (
            self.get_parameter("post_start_sleep_sec").get_parameter_value().double_value
        )

        # Load eval plan
        plan_path = Path(eval_plan_file)
        if not plan_path.exists():
            self.get_logger().error(f"[mcap-player] eval_plan_file not found: {plan_path}")
            raise FileNotFoundError(plan_path)

        self._plan = json.loads(plan_path.read_text())

        # Remap host MCAP paths to the container mount point /workspace/mcap_data.
        # eval_plan.json stores absolute host paths; inside Docker, MCAP_ROOT is
        # mounted at /workspace/mcap_data, so we strip the original mcap_root prefix.
        host_mcap_root = self._plan.get("mcap_root", "")
        container_mcap_root = Path("/workspace/mcap_data")
        if host_mcap_root and container_mcap_root.exists():
            for ep in self._plan.get("episodes", []):
                host_path = ep["mcap_path"]
                if host_path.startswith(host_mcap_root):
                    rel = Path(host_path).relative_to(host_mcap_root)
                    ep["mcap_path"] = str(container_mcap_root / rel)

        episodes = self._plan.get("episodes", [])
        self.get_logger().info(f"[mcap-player] Loaded eval plan: {len(episodes)} episodes")

        # Publishers (TRANSIENT_LOCAL so late subscribers still catch the latest sample)
        self._ep_start_pub = self.create_publisher(String, "/eval/episode_start", _EVAL_CTRL_QOS)
        self._ep_done_pub = self.create_publisher(String, "/eval/episode_done", _EVAL_CTRL_QOS)
        self._eval_complete_pub = self.create_publisher(Bool, "/eval/eval_complete", _EVAL_CTRL_QOS)

        # Ack subscriber
        self._ack_event = threading.Event()
        self._last_ack_idx: int | None = None
        self._ack_sub = self.create_subscription(
            String, "/eval/episode_ack", self._ack_callback, _EVAL_CTRL_QOS
        )

        # Start the playback loop in a background thread so rclpy can spin
        self._playback_thread = threading.Thread(target=self._playback_loop, daemon=True)
        self._playback_thread.start()

    # ──────────────────────────────────────────────────────────────────────
    # Callbacks
    # ──────────────────────────────────────────────────────────────────────

    def _on_joint_state(self, msg: JointState) -> None:
        """Synthesize GT commands: action[t] = observation[t+1].

        When the previous frame exists, extract its arm joint positions and
        publish them as the commanded positions for that timestep.
        """
        if self._prev_joint_state is not None and self._synth_pub is not None:
            prev_name_to_pos: dict[str, float] = dict(
                zip(self._prev_joint_state.name, self._prev_joint_state.position)
            )
            positions = [
                prev_name_to_pos.get(j, 0.0) for j in self._synthesize_arm_joint_names
            ]
            cmd = Float64MultiArray()
            cmd.data = positions
            self._synth_pub.publish(cmd)
        self._prev_joint_state = msg

    def _ack_callback(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
            self._last_ack_idx = data.get("episode_idx")
            self._ack_event.set()
        except Exception as e:
            self.get_logger().warning(f"[mcap-player] Bad ack message: {e}")

    # ──────────────────────────────────────────────────────────────────────
    # Playback loop (background thread)
    # ──────────────────────────────────────────────────────────────────────

    def _playback_loop(self) -> None:
        """Main evaluation loop — runs in background thread."""
        # Give inference node time to warm up
        self.get_logger().info(f"[mcap-player] Waiting {self._warmup_sec}s for inference warmup...")
        time.sleep(self._warmup_sec)

        episodes = self._plan.get("episodes", [])
        for i, episode in enumerate(episodes):
            ep_idx = episode["episode_idx"]
            split = episode["split_label"]
            mcap_path = episode["mcap_path"]

            self.get_logger().info(
                f"[mcap-player] Episode {i+1}/{len(episodes)}: "
                f"idx={ep_idx} split={split} mcap={Path(mcap_path).name}"
            )

            if not Path(mcap_path).exists():
                self.get_logger().error(f"[mcap-player] MCAP not found: {mcap_path}, skipping")
                continue

            self._play_episode(ep_idx, split, mcap_path)

            if i < len(episodes) - 1:
                time.sleep(self._inter_episode_sec)

        # All done
        self.get_logger().info("[mcap-player] All episodes complete — publishing eval_complete")
        self._eval_complete_pub.publish(Bool(data=True))
        time.sleep(1.0)  # Let the message propagate

        # Shutdown
        rclpy.shutdown()

    def _play_episode(self, ep_idx: int, split_label: str, mcap_path: str) -> None:
        """Play one MCAP episode and wait for eval-recorder ack."""
        # 1. Signal episode start
        start_msg = String()
        start_msg.data = json.dumps({"episode_idx": ep_idx, "split_label": split_label})
        self._ep_start_pub.publish(start_msg)
        self.get_logger().info(f"[mcap-player] Published episode_start for ep {ep_idx}")

        # Brief pause for recorder to reset
        time.sleep(self._post_start_sleep_sec)

        # 2. Run ros2 bag play
        self.get_logger().info(f"[mcap-player] Playing {mcap_path}")
        try:
            proc = subprocess.Popen(
                ["ros2", "bag", "play", mcap_path, "--storage", "mcap"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            proc.wait()
        except FileNotFoundError:
            self.get_logger().error("[mcap-player] `ros2` not found. Is ROS2 sourced?")
            return
        except Exception as e:
            self.get_logger().error(f"[mcap-player] ros2 bag play failed: {e}")
            return

        self.get_logger().info(f"[mcap-player] Bag play finished for ep {ep_idx}, waiting for ack...")

        # 3. Wait for eval-recorder ack
        self._ack_event.clear()
        deadline = time.time() + self._ack_timeout_sec
        while time.time() < deadline:
            if self._ack_event.wait(timeout=1.0):
                if self._last_ack_idx == ep_idx:
                    self.get_logger().info(f"[mcap-player] Received ack for ep {ep_idx}")
                    return
                # Different episode ack — reset and keep waiting
                self._ack_event.clear()

        self.get_logger().warning(
            f"[mcap-player] Ack timeout for ep {ep_idx} after {self._ack_timeout_sec}s, continuing"
        )


def main(args=None):
    rclpy.init(args=args)
    node = McapPlayerNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, Exception):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
