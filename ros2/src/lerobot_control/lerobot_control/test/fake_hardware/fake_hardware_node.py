#!/usr/bin/env python3
"""Mock controller node for integration testing of lerobot_control.

This node simulates a robot controller by:
- Publishing dummy CompressedImage (configurable resolution and FPS)
- Joint mode (default): publishing dummy joint states at 500Hz (matches real
  robot), subscribing to joint action commands and validating them.
- EE mode (``ee_mode:=true``): publishing CommandedEEPose observations on
  ``/ee_pose_<arm>``, and — critically — subscribing to
  ``/commanded_ee_<arm>`` and ECHOING each received command back as the next
  published observation (``next_ee_pose ≈ last_received_command``). This
  closed-loop echo is what actually exercises the decoupled delta-mode
  publish loop's self-correction (``absolute_target = obs_pose ∘ delta``,
  see claude_docs/ee-delta-flow-plan.md, Item 2b) — a static/random EE-pose
  publisher would only validate topic wiring, not the feedback loop itself.
  Joint-mode topics/timers are NOT started in EE mode (mirrors production's
  either-joint-or-ee exclusivity).

ROS2 parameters:
    timeout (float): Seconds before exit with failure (default 30.0)
    required_actions (int): Valid actions needed before exit success (default 10)
    camera_resolution (str): "480p", "720p", or "1080p" (default "480p")
    camera_fps (int): Camera publish rate in Hz (default 30)
    ee_mode (bool): Enable EE-space pub/sub instead of joint-space (default False)
    ee_arms (str): Comma-separated arm ids, e.g. "left,right" (default "left,right")
    ee_pose_fps (float): /ee_pose_<arm> publish rate in Hz (default 100.0)
    ee_seed_pose (str): Comma-separated flat floats, 8 values per arm in ee_arms
        order (quaternion layout: x,y,z,qx,qy,qz,qw,gripper), seeding the initial
        EE pose instead of the hardcoded default. Empty (default) keeps the
        hardcoded default; a malformed value logs a warning and falls back to it
        rather than crashing the mock. Used by the GT-replayer correctness test
        to seed the echo loop from a converted dataset's own first recorded pose.

The node exits with code 0 after receiving the required number of valid actions
(joint commands, or EE commands in ee_mode), or exits with code 1 on timeout or
invalid data. This exit-code contract is identical across both modes.

Explicitly out of scope (see claude_docs/ee-delta-flow-plan.md, Item 2b):
real actuation dynamics, physical velocity/latency limits, sensor latency —
the echo is instantaneous and perfect. This validates SOFTWARE timing and
composition correctness only, never physical behavior; it is not a
substitute for real-hardware validation (Item 6).
"""

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, JointState
from std_msgs.msg import Float64MultiArray


_RESOLUTION_MAP = {
    "480p": (480, 640),
    "720p": (720, 1280),
    "1080p": (1080, 1920),
}


class MockControllerNode(Node):
    """ROS2 node that simulates a robot controller for testing."""

    def __init__(self):
        super().__init__("mock_controller")

        # Declare parameters
        self.declare_parameter("timeout", 30.0)
        self.declare_parameter("required_actions", 10)
        self.declare_parameter("camera_resolution", "480p")
        self.declare_parameter("camera_fps", 30)
        self.declare_parameter("ee_mode", False)
        self.declare_parameter("ee_arms", "left,right")
        self.declare_parameter("ee_pose_fps", 100.0)
        self.declare_parameter("ee_seed_pose", "")

        # Get parameter values
        self._timeout = self.get_parameter("timeout").value
        self._required_actions = self.get_parameter("required_actions").value
        self._camera_res_label = self.get_parameter("camera_resolution").value
        self._camera_fps = self.get_parameter("camera_fps").value
        self._ee_mode = bool(self.get_parameter("ee_mode").value)
        self._ee_arms = [
            a.strip() for a in str(self.get_parameter("ee_arms").value).split(",") if a.strip()
        ]
        self._ee_pose_fps = float(self.get_parameter("ee_pose_fps").value)
        self._ee_seed_pose = str(self.get_parameter("ee_seed_pose").value)

        # Resolve resolution
        h, w = _RESOLUTION_MAP.get(self._camera_res_label, (480, 640))

        self.get_logger().info(
            f"MockControllerNode initialized: timeout={self._timeout}s, "
            f"required_actions={self._required_actions}, "
            f"resolution={w}x{h} ({self._camera_res_label}), camera_fps={self._camera_fps}, "
            f"ee_mode={self._ee_mode}" + (f", ee_arms={self._ee_arms}" if self._ee_mode else "")
        )

        # Publishers — 4 CompressedImage cameras matching production topics
        self._camera_topics = [
            "/cam_waist/image_raw/compressed",
            "/cam_wrist_r/image_raw/compressed",
            "/cam_chest/image_raw/compressed",
            "/cam_wrist_l/image_raw/compressed",
        ]
        self.image_pubs = [
            self.create_publisher(CompressedImage, topic, 10)
            for topic in self._camera_topics
        ]
        self.image_timer = self.create_timer(1.0 / self._camera_fps, self.publish_image)

        if self._ee_mode:
            self._setup_ee_mode()
        else:
            self._setup_joint_mode()

        # Timeout check timer (1Hz) — shared by both modes
        self.timeout_timer = self.create_timer(1.0, self.check_timeout)

        # State
        self.valid_actions_received = 0
        self.start_time = self.get_clock().now()

        # Random number generator
        self._rng = np.random.default_rng()

        # Pre-generate a dummy image and JPEG-encode once (reuse across frames)
        dummy_rgb = self._rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
        _, self._jpeg_data = cv2.imencode(".jpg", dummy_rgb, [cv2.IMWRITE_JPEG_QUALITY, 50])

    # ------------------------------------------------------------------ #
    # Joint mode (default, unchanged from before)
    # ------------------------------------------------------------------ #

    def _setup_joint_mode(self) -> None:
        self.joint_pub = self.create_publisher(JointState, "/joint_states", 10)

        # Subscribers — one per arm, matching inference_node publish topics
        self._action_subs = [
            self.create_subscription(Float64MultiArray, topic, self.action_callback, 10)
            for topic in [
                "/follower_l_forward_position_controller/commands",
                "/follower_r_forward_position_controller/commands",
            ]
        ]

        # Separate timer: 500Hz joint states (matches real robot)
        self.joint_timer = self.create_timer(1.0 / 500.0, self.publish_joint_state)

        # Joint names for 16-DOF robot (8 joints per arm: finger + 7 joints)
        # Naming matches production: follower_{l,r}_{joint_id}
        self.joint_names = [
            "follower_l_finger_joint1",
            "follower_l_joint1",
            "follower_l_joint2",
            "follower_l_joint3",
            "follower_l_joint4",
            "follower_l_joint5",
            "follower_l_joint6",
            "follower_l_joint7",
            "follower_r_finger_joint1",
            "follower_r_joint1",
            "follower_r_joint2",
            "follower_r_joint3",
            "follower_r_joint4",
            "follower_r_joint5",
            "follower_r_joint6",
            "follower_r_joint7",
        ]

    def publish_joint_state(self):
        """Publish dummy joint states at 500Hz."""
        joint_msg = JointState()
        joint_msg.header.stamp = self.get_clock().now().to_msg()
        joint_msg.header.frame_id = "base_link"
        joint_msg.name = self.joint_names
        joint_msg.position = (self._rng.random(16) * 2 * np.pi - np.pi).tolist()
        joint_msg.velocity = [0.0] * 16
        joint_msg.effort = [0.0] * 16
        self.joint_pub.publish(joint_msg)

    def action_callback(self, msg: Float64MultiArray):
        """Handle incoming joint action commands from per-arm controller topics."""
        for i, val in enumerate(msg.data):
            if not np.isfinite(val):
                self.get_logger().error(
                    f"Invalid action value at index {i}: {val} (must be finite)"
                )
                raise SystemExit(1)
        self._record_valid_action()

    # ------------------------------------------------------------------ #
    # EE mode — closed-loop echo (see module docstring)
    # ------------------------------------------------------------------ #

    def _parse_ee_seed_pose(self) -> dict[str, dict] | None:
        """Parse ``ee_seed_pose`` into per-arm ``{pos, quat, gripper}`` state dicts.

        Returns ``None`` (caller falls back to the hardcoded default) when the
        param is empty or malformed — a bad seed string must never crash the mock.
        """
        raw = self._ee_seed_pose.strip()
        if not raw:
            return None

        try:
            values = [float(v) for v in raw.split(",")]
        except ValueError:
            self.get_logger().warn(
                f"[ee_mode] ee_seed_pose could not be parsed as comma-separated "
                f"floats ({raw!r}) — using default initial pose"
            )
            return None

        expected_len = 8 * len(self._ee_arms)
        if len(values) != expected_len:
            self.get_logger().warn(
                f"[ee_mode] ee_seed_pose has {len(values)} values, expected "
                f"{expected_len} (8 per arm x {len(self._ee_arms)} arms) — using "
                "default initial pose"
            )
            return None

        seeded: dict[str, dict] = {}
        for i, arm in enumerate(self._ee_arms):
            chunk = values[i * 8:(i + 1) * 8]
            seeded[arm] = {
                "pos": np.array(chunk[0:3], dtype=np.float64),
                "quat": np.array(chunk[3:7], dtype=np.float64),
                "gripper": chunk[7],
            }
        return seeded

    def _setup_ee_mode(self) -> None:
        from anvil_msgs.msg import CommandedEEPose, MockEEPose

        # Per-arm current EE pose state. Seeded from ee_seed_pose when given
        # (e.g. the GT-replayer correctness test seeds this from a converted
        # dataset's own first recorded observation.state row); otherwise an
        # all-zero "never commanded yet" sentinel (see the else branch below).
        # Updated in-place by _ee_command_callback whenever a command arrives —
        # this IS the "echo/integrate" closed-loop behavior.
        self._ee_state: dict[str, dict]
        seeded_state = self._parse_ee_seed_pose()
        if seeded_state is not None:
            self._ee_state = seeded_state
            self.get_logger().info(
                f"[ee_mode] Seeded initial EE pose from ee_seed_pose param "
                f"for arms: {list(seeded_state.keys())}"
            )
        else:
            # All-zero sentinel (including a non-unit [0,0,0,0] quat) — deliberately
            # invalid as a rotation, so it's trivially distinguishable from any real
            # command's echo. Fine because nothing reads it as a rotation before the
            # first command overwrites it in-place.
            self._ee_state = {
                arm: {
                    "pos": np.array([0.0, 0.0, 0.0], dtype=np.float64),
                    "quat": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64),
                    "gripper": 0.0,
                }
                for arm in self._ee_arms
            }

        self._ee_pose_pubs: dict[str, object] = {}
        self._ee_command_subs = []
        # Per-arm monotonically-incrementing counter for MockEEPose.sequence.
        # Bumped in _ee_command_callback — i.e. once per actually-received command,
        # NOT once per publish_ee_poses() tick. This distinction is the whole
        # point: publish_ee_poses runs on its own fixed-rate timer and would
        # happily re-publish the same not-yet-updated _ee_state more than once
        # if a command is still in flight, which is exactly the stale-echo bug
        # this exists to let a consumer detect. Incrementing per publish instead
        # would advance even on a re-publish of unchanged state, defeating the
        # whole mechanism. See ee_obs_sequence_guard.py and MockEEPose.msg.
        self._ee_seq_by_arm: dict[str, int] = {arm: 0 for arm in self._ee_arms}
        for arm in self._ee_arms:
            obs_topic = f"/ee_pose_{arm}"
            cmd_topic = f"/commanded_ee_{arm}"
            # /ee_pose_{arm} publishes MockEEPose, NOT CommandedEEPose — this echo
            # topic is fake-hardware-only (see MockEEPose.msg's docstring); real
            # hardware's /ee_pose_{arm} is a plain CommandedEEPose and never carries
            # a sequence field.
            self._ee_pose_pubs[arm] = self.create_publisher(MockEEPose, obs_topic, 10)
            self._ee_command_subs.append(
                self.create_subscription(
                    CommandedEEPose, cmd_topic,
                    lambda msg, arm=arm: self._ee_command_callback(arm, msg),
                    10,
                )
            )
            self.get_logger().info(f"[ee_mode] arm={arm}: publishing {obs_topic}, subscribing {cmd_topic}")

        self.ee_pose_timer = self.create_timer(1.0 / self._ee_pose_fps, self.publish_ee_poses)

    def publish_ee_poses(self):
        """Publish every arm's current (possibly just-echoed) EE pose.

        Called by the fixed-rate ``ee_pose_timer``. See ``_publish_one_ee_pose``
        for the actual per-arm publish (also called directly, per-arm, from
        ``_ee_command_callback`` for a latency-minimizing immediate push).
        """
        for arm in self._ee_state:
            self._publish_one_ee_pose(arm)

    def _publish_one_ee_pose(self, arm: str) -> None:
        """Publish MockEEPose (base pose/gripper + sequence) for a single arm.

        Publishes MockEEPose, not plain CommandedEEPose — see MockEEPose.msg's
        docstring for why this needs to be a separate fake-hardware-only type.
        """
        from anvil_msgs.msg import CommandedEEPose, MockEEPose
        from geometry_msgs.msg import Point, Pose, Quaternion
        from std_msgs.msg import Header

        state = self._ee_state[arm]
        stamp = self.get_clock().now().to_msg()
        base = CommandedEEPose()
        base.header = Header(stamp=stamp, frame_id="world")
        pos, quat = state["pos"], state["quat"]
        base.pose = Pose(
            position=Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2])),
            orientation=Quaternion(
                x=float(quat[0]), y=float(quat[1]), z=float(quat[2]), w=float(quat[3])
            ),
        )
        base.gripper = float(state["gripper"])

        msg = MockEEPose()
        msg.base = base
        # Whatever _ee_command_callback last set — NOT incremented here (see
        # _ee_seq_by_arm's comment above for why).
        msg.sequence = self._ee_seq_by_arm[arm]
        self._ee_pose_pubs[arm].publish(msg)

    def _ee_command_callback(self, arm: str, msg) -> None:
        """Echo/integrate a received CommandedEEPose as the arm's next observed pose.

        This is the closed-loop feedback the decoupled delta-mode publish
        loop depends on: `next_ee_pose ≈ last_received_command`. A perfect,
        instantaneous echo (no dynamics, no latency) — deliberately, per the
        plan's stated scope: this validates software timing/composition
        correctness, not physical actuation behavior.
        """
        values = [
            msg.pose.position.x, msg.pose.position.y, msg.pose.position.z,
            msg.pose.orientation.x, msg.pose.orientation.y,
            msg.pose.orientation.z, msg.pose.orientation.w,
            msg.gripper,
        ]
        for i, val in enumerate(values):
            if not np.isfinite(val):
                self.get_logger().error(
                    f"[ee_mode] Invalid CommandedEEPose value for arm={arm} at index {i}: "
                    f"{val} (must be finite)"
                )
                raise SystemExit(1)

        self._ee_state[arm]["pos"] = np.array(values[0:3], dtype=np.float64)
        self._ee_state[arm]["quat"] = np.array(values[3:7], dtype=np.float64)
        self._ee_state[arm]["gripper"] = values[7]
        self._ee_seq_by_arm[arm] += 1

        # Immediate push, in addition to the periodic ee_pose_timer — minimizes
        # echo latency instead of waiting up to 1/ee_pose_fps for the next tick.
        self._publish_one_ee_pose(arm)

        self._record_valid_action()

    # ------------------------------------------------------------------ #
    # Shared: image publishing, timeout, exit-code bookkeeping
    # ------------------------------------------------------------------ #

    def publish_image(self):
        """Publish dummy CompressedImage on all 4 cameras at 30Hz."""
        stamp = self.get_clock().now().to_msg()
        data = self._jpeg_data.tobytes()
        for pub in self.image_pubs:
            msg = CompressedImage()
            msg.header.stamp = stamp
            msg.header.frame_id = "camera_link"
            msg.format = "jpeg"
            msg.data = data
            pub.publish(msg)

    def check_timeout(self):
        """Check if timeout has been exceeded."""
        elapsed = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
        if elapsed > self._timeout:
            self.get_logger().error(
                f"Timeout after {elapsed:.1f}s. "
                f"Received {self.valid_actions_received}/{self._required_actions} valid actions."
            )
            raise SystemExit(1)

    def _record_valid_action(self) -> None:
        """Shared valid-action counter/exit logic for both joint and EE modes."""
        self.valid_actions_received += 1
        self.get_logger().info(
            f"Valid action received [{self.valid_actions_received}/{self._required_actions}]"
        )

        if self.valid_actions_received >= self._required_actions:
            elapsed = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
            self.get_logger().info(
                f"Test PASSED! Received {self._required_actions} valid actions in {elapsed:.1f}s"
            )
            raise SystemExit(0)


def main(args=None):
    """Entry point for the mock controller node."""
    rclpy.init(args=args)
    node = MockControllerNode()

    exit_code = 0
    try:
        rclpy.spin(node)
    except SystemExit as e:
        exit_code = e.code if e.code is not None else 0
        node.get_logger().info(f"Node shutting down with exit code {exit_code}")
    finally:
        node.destroy_node()
        rclpy.shutdown()

    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
