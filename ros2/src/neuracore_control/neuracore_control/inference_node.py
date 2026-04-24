#!/usr/bin/env python3

"""ROS2 node: local Neuracore policy inference → arm position commands.

Designed to run on a GPU PC, discover /joint_states + camera topics from the
Robot PC over CycloneDDS, and publish 8-element Float64MultiArray (7 arm +
1 gripper) commands back to the Robot PC's forward_position_controllers.

Mirrors the logging schema used by anvil-workcell's neuracore_bridge data
collector: joint names, gripper normalization (0..0.05m → [0, 1]), camera
naming ('cam_wrist_l', etc.) all match, so a model trained from that
collector's data runs here unchanged.
"""

import csv
import os
import time
from typing import Dict, List, Optional, Tuple

import neuracore as nc
import numpy as np
import rclpy
import torch
from neuracore_types import DataType
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, JointState
from std_msgs.msg import Float64MultiArray

from .common import (
    CMD_L_TOPIC,
    DEFAULT_CAMERA_TOPICS,
    GRIPPER_HI,
    LEFT_ARM,
    LEFT_GRIPPER,
    camera_name_from_topic,
    decode_compressed_image,
    gripper_denormalize,
    gripper_normalize,
    header_time,
)


class NeuracoreInferenceNode(Node):
    """Runs a Neuracore policy locally and publishes arm commands."""

    def __init__(self):
        super().__init__("neuracore_inference_node")
        self.get_logger().info("[neura-infer] startup")

        self.declare_parameter("robot_name", "anvil_openarm")
        self.declare_parameter("urdf_path", "")
        self.declare_parameter("model_file", "")
        self.declare_parameter("train_run_name", "")
        self.declare_parameter("camera_topics", DEFAULT_CAMERA_TOPICS)
        self.declare_parameter("inference_rate_hz", 30.0)
        self.declare_parameter("debug", False)
        self.declare_parameter("max_joint_delta", 0.05)
        self.declare_parameter("predictions_log", "")

        self._debug = (
            self.get_parameter("debug").get_parameter_value().bool_value
        )
        self._max_joint_delta = float(
            self.get_parameter("max_joint_delta")
            .get_parameter_value()
            .double_value
        )

        self._predictions_file = None
        self._predictions_writer = None
        log_path = (
            self.get_parameter("predictions_log")
            .get_parameter_value()
            .string_value
        )
        if log_path:
            os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
            self._predictions_file = open(log_path, "w", newline="")
            self._predictions_writer = csv.writer(self._predictions_file)
            header = ["t"]
            header += [f"target_{n}" for n in LEFT_ARM]
            header += ["target_grip"]
            header += [f"current_{n}" for n in LEFT_ARM]
            header += ["current_grip"]
            self._predictions_writer.writerow(header)
            self._predictions_file.flush()
            self.get_logger().info(
                f"[neura-infer] writing predictions to {log_path}"
            )

        self._policy = None
        self._chunk: Optional[np.ndarray] = None
        self._chunk_idx = 0
        self._latest_joint_state: Optional[Tuple[float, JointState]] = None
        self._latest_frames: Dict[str, Tuple[float, np.ndarray]] = {}
        self._tick_count = 0

        self._camera_topics: List[str] = list(
            self.get_parameter("camera_topics")
            .get_parameter_value()
            .string_array_value
        )
        self._camera_names: List[str] = [
            camera_name_from_topic(t) for t in self._camera_topics
        ]

        self._arm_joints = LEFT_ARM
        self._gripper_joints = [LEFT_GRIPPER]

        # Embodiment descriptions — MUST match how the model was trained.
        # If your neuracore training run used different ordering, adjust here.
        self._input_desc = {
            DataType.JOINT_POSITIONS: {
                i: n for i, n in enumerate(self._arm_joints)
            },
            DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS: {
                i: n for i, n in enumerate(self._gripper_joints)
            },
            DataType.RGB_IMAGES: {
                i: n for i, n in enumerate(self._camera_names)
            },
        }
        self._output_desc = {
            DataType.JOINT_TARGET_POSITIONS: {
                i: n for i, n in enumerate(self._arm_joints)
            },
            DataType.PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS: {
                i: n for i, n in enumerate(self._gripper_joints)
            },
        }

        if not self._init_neuracore():
            raise RuntimeError("[neura-infer] neuracore init failed — aborting")

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(
            JointState, "/joint_states", self._on_joint_state, sensor_qos
        )
        for topic in self._camera_topics:
            self.create_subscription(
                CompressedImage,
                topic,
                lambda msg, t=topic: self._on_camera(msg, t),
                sensor_qos,
            )
            self.get_logger().info(f"[neura-infer] subscribed: {topic}")

        self._cmd_l_pub = self.create_publisher(Float64MultiArray, CMD_L_TOPIC, 10)

        model_file = (
            self.get_parameter("model_file").get_parameter_value().string_value
        )
        train_run_name = (
            self.get_parameter("train_run_name")
            .get_parameter_value()
            .string_value
        )
        err = self._load_policy(model_file, train_run_name)
        if err is not None:
            raise RuntimeError(f"[neura-infer] {err}")

        rate = float(
            self.get_parameter("inference_rate_hz")
            .get_parameter_value()
            .double_value
        )
        self.create_timer(1.0 / rate, self._tick)
        self.get_logger().info(f"[neura-infer] ready, ticking at {rate} Hz")

    # ----------------------------------------------------------------- init

    def _init_neuracore(self) -> bool:
        api_key = os.environ.get("NEURACORE_API_KEY", "")
        if not api_key:
            self.get_logger().error("[neura-infer] NEURACORE_API_KEY not set")
            return False
        try:
            nc.login(api_key=api_key)
        except Exception as e:
            self.get_logger().error(f"[neura-infer] login failed: {e}")
            return False

        robot_name = (
            self.get_parameter("robot_name").get_parameter_value().string_value
        )
        urdf_path = (
            self.get_parameter("urdf_path").get_parameter_value().string_value
        )
        self.get_logger().info(
            f"[neura-infer] connecting robot '{robot_name}' (urdf='{urdf_path}')"
        )
        try:
            if urdf_path:
                nc.connect_robot(robot_name, urdf_path=urdf_path)
            else:
                nc.connect_robot(robot_name)
        except Exception as e:
            self.get_logger().error(f"[neura-infer] connect_robot failed: {e}")
            return False
        return True

    def _load_policy(self, model_file: str, train_run_name: str) -> Optional[str]:
        if not model_file and not train_run_name:
            return "no model_file or train_run_name configured"

        kwargs = {
            "input_embodiment_description": self._input_desc,
            "output_embodiment_description": self._output_desc,
        }
        if model_file:
            kwargs["model_file"] = model_file
            src = f"model_file={model_file}"
        else:
            kwargs["train_run_name"] = train_run_name
            src = f"train_run_name={train_run_name}"

        self.get_logger().info(f"[neura-infer] loading policy ({src})")
        t0 = time.perf_counter()
        try:
            self._policy = nc.policy(**kwargs)
        except Exception as e:
            return f"policy load failed: {e}"
        self.get_logger().info(
            f"[neura-infer] policy loaded in {time.perf_counter() - t0:.2f}s"
        )
        return None

    # ------------------------------------------------------------ callbacks

    def _on_joint_state(self, msg: JointState) -> None:
        self._latest_joint_state = (header_time(msg), msg)

    def _on_camera(self, msg: CompressedImage, topic: str) -> None:
        cam_name = camera_name_from_topic(topic)
        try:
            rgb = decode_compressed_image(msg, size=(640, 480))
        except Exception as e:
            self.get_logger().warning(
                f"[neura-infer] decode {cam_name} failed: {e}",
                throttle_duration_sec=10.0,
            )
            return
        self._latest_frames[cam_name] = (header_time(msg), rgb)

    # ----------------------------------------------------------- inference

    def _tick(self) -> None:
        if self._policy is None:
            return
        if self._latest_joint_state is None:
            self.get_logger().warning(
                "[neura-infer] waiting for /joint_states",
                throttle_duration_sec=5.0,
            )
            return
        missing = [n for n in self._camera_names if n not in self._latest_frames]
        if missing:
            self.get_logger().warning(
                f"[neura-infer] waiting for cameras: {missing}",
                throttle_duration_sec=5.0,
            )
            return

        try:
            self._log_observations()
        except Exception as e:
            self.get_logger().warning(
                f"[neura-infer] log observations failed: {e}",
                throttle_duration_sec=5.0,
            )
            return

        if self._chunk is None or self._chunk_idx >= len(self._chunk):
            try:
                self._chunk = self._predict_chunk()
                self._chunk_idx = 0
            except Exception as e:
                self.get_logger().warning(
                    f"[neura-infer] predict failed: {e}",
                    throttle_duration_sec=5.0,
                )
                return

        action = self._chunk[self._chunk_idx]
        self._chunk_idx += 1

        self._publish_commands(action)
        self._tick_count += 1
        if self._tick_count % 300 == 1:
            self.get_logger().info(
                f"[neura-infer] {self._tick_count} ticks, "
                f"chunk_idx={self._chunk_idx}/{len(self._chunk)}"
            )

    def _log_observations(self) -> None:
        js_t, js = self._latest_joint_state
        positions: Dict[str, float] = {}
        grippers: Dict[str, float] = {}
        for i, name in enumerate(js.name):
            if i >= len(js.position):
                continue
            if name in self._gripper_joints:
                grippers[name] = float(js.position[i])
            elif name in self._arm_joints:
                positions[name] = float(js.position[i])

        if positions:
            nc.log_joint_positions(positions, timestamp=js_t)
        for name, raw in grippers.items():
            nc.log_parallel_gripper_open_amount(
                name, gripper_normalize(raw), timestamp=js_t
            )
        for cam_name, (t, rgb) in self._latest_frames.items():
            nc.log_rgb(cam_name, rgb, timestamp=t)

    def _predict_chunk(self) -> np.ndarray:
        """Run policy and return (horizon, n_arm + n_grippers) array."""
        t0 = time.perf_counter()
        preds = self._policy.predict(timeout=5)
        predict_ms = (time.perf_counter() - t0) * 1e3

        joint_preds = preds[DataType.JOINT_TARGET_POSITIONS]
        arm_tensors = [joint_preds[n].value for n in self._arm_joints]
        arm = torch.cat(arm_tensors, dim=2)[0]  # (horizon, n_arm)

        grip_preds = preds.get(DataType.PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS, {})
        if grip_preds:
            grip_tensors = [grip_preds[n].open_amount for n in self._gripper_joints]
            grip = torch.cat(grip_tensors, dim=2)[0]
            out = torch.cat([arm, grip], dim=1)
        else:
            out = arm

        chunk = out.detach().cpu().numpy().astype(np.float64)
        self.get_logger().info(
            f"[neura-infer] predict ok: {predict_ms:.1f}ms, horizon={chunk.shape[0]}"
        )
        return chunk

    def _current_arm_positions(self) -> Optional[np.ndarray]:
        if self._latest_joint_state is None:
            return None
        _, js = self._latest_joint_state
        lookup = {n: p for n, p in zip(js.name, js.position)}
        try:
            return np.array([lookup[n] for n in self._arm_joints], dtype=np.float64)
        except KeyError:
            return None

    def _current_gripper_position(self) -> float:
        if self._latest_joint_state is None or not self._gripper_joints:
            return float("nan")
        _, js = self._latest_joint_state
        for name, pos in zip(js.name, js.position):
            if name == self._gripper_joints[0]:
                return float(pos)
        return float("nan")

    def _publish_commands(self, action: np.ndarray) -> None:
        n_arm = len(self._arm_joints)

        target_arm = action[:n_arm]
        current_arm = self._current_arm_positions()
        if current_arm is not None and len(current_arm) == n_arm:
            raw_delta = target_arm - current_arm
            clamped = int(np.sum(np.abs(raw_delta) > self._max_joint_delta))
            limited = np.clip(
                raw_delta, -self._max_joint_delta, self._max_joint_delta
            )
            safe_arm = (current_arm + limited).tolist()
            if clamped:
                self.get_logger().warning(
                    f"[neura-infer] delta-clamped {clamped}/{n_arm} arm joints "
                    f"to ±{self._max_joint_delta:.3f} rad",
                    throttle_duration_sec=2.0,
                )
        else:
            safe_arm = target_arm.tolist()

        if action.shape[0] >= n_arm + 1:
            left_grip_raw = gripper_denormalize(float(action[n_arm]))
        else:
            left_grip_raw = GRIPPER_HI

        if self._predictions_writer is not None and self._predictions_file is not None:
            current_arm_row = (
                current_arm.tolist()
                if current_arm is not None
                else [float("nan")] * n_arm
            )
            row = [time.time()]
            row += target_arm.tolist()
            row += [left_grip_raw]
            row += current_arm_row
            row += [self._current_gripper_position()]
            self._predictions_writer.writerow(row)
            self._predictions_file.flush()

        if self._debug:
            self.get_logger().info(
                f"[neura-infer] DEBUG action | "
                f"L_arm={['%.3f' % v for v in safe_arm]} "
                f"L_grip={left_grip_raw:.3f}"
            )
            return

        msg_l = Float64MultiArray()
        msg_l.data = safe_arm + [left_grip_raw]
        self._cmd_l_pub.publish(msg_l)

    # ------------------------------------------------------------ shutdown

    def destroy_node(self) -> None:
        self.get_logger().info("[neura-infer] destroy_node")
        if self._policy is not None:
            try:
                self._policy.disconnect()
            except Exception as e:
                self.get_logger().warning(f"[neura-infer] disconnect failed: {e}")
        if self._predictions_file is not None:
            self._predictions_file.close()
        super().destroy_node()


def main() -> None:
    rclpy.init()
    node = NeuracoreInferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
