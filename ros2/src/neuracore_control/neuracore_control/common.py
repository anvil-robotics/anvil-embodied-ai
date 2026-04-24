"""Shared constants and helpers for neuracore_control.

Copied verbatim from anvil-workcell/ros2/src/neuracore_bridge/neuracore_bridge/
common.py so joint naming + gripper normalization stays consistent between
the data-collection side (on the robot PC) and the inference side (here).
If you edit one, edit the other.
"""

import time
from typing import Tuple

import cv2
import numpy as np

LEFT_ARM = [f"follower_l_joint{i}" for i in range(1, 8)]
RIGHT_ARM = [f"follower_r_joint{i}" for i in range(1, 8)]
LEFT_GRIPPER = "follower_l_finger_joint1"
RIGHT_GRIPPER = "follower_r_finger_joint1"
FINGER_JOINTS = {LEFT_GRIPPER, RIGHT_GRIPPER}

# URDF hard limits for finger_joint1 (prismatic, meters).
GRIPPER_LO = 0.0
GRIPPER_HI = 0.05

CMD_L_TOPIC = "/follower_l_forward_position_controller/commands"
CMD_R_TOPIC = "/follower_r_forward_position_controller/commands"

DEFAULT_CAMERA_TOPICS = [
    "/cam_wrist_l/image_raw/compressed",
    "/cam_chest/image_raw/compressed",
    "/cam_waist/image_raw/compressed",
]


def header_time(msg) -> float:
    """Publisher stamp if set, else wall clock."""
    try:
        s, n = msg.header.stamp.sec, msg.header.stamp.nanosec
        if s or n:
            return s + n * 1e-9
    except AttributeError:
        pass
    return time.time()


def gripper_normalize(value: float) -> float:
    """Raw finger_joint1 position (m) → [0, 1]."""
    return max(0.0, min(1.0, (value - GRIPPER_LO) / (GRIPPER_HI - GRIPPER_LO)))


def gripper_denormalize(value: float) -> float:
    """[0, 1] → raw finger_joint1 position (m), clamped to URDF limits."""
    clamped = max(0.0, min(1.0, value))
    return clamped * (GRIPPER_HI - GRIPPER_LO) + GRIPPER_LO


def decode_compressed_image(
    msg, size: Tuple[int, int] = (640, 480)
) -> np.ndarray:
    """CompressedImage (MJPEG/JPEG) → uint8 HxWx3 RGB, resized to (W, H)."""
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("cv2.imdecode returned None")
    bgr = cv2.resize(bgr, size, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def camera_name_from_topic(topic: str) -> str:
    """'/cam_wrist_l/image_raw/compressed' → 'cam_wrist_l'."""
    return topic.lstrip("/").split("/")[0]
