"""Self-contained mesh-free OpenArm kinematic models."""

from __future__ import annotations

import hashlib
import importlib
import json
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class LinkSpec:
    pos: tuple[float, float, float]
    quat: tuple[float, float, float, float]
    axis: tuple[float, float, float]
    joint_range: tuple[float, float]


@dataclass(frozen=True)
class ArmModelSpec:
    root_pos: tuple[float, float, float]
    root_quat: tuple[float, float, float, float]
    links: tuple[LinkSpec, ...]
    tcp_pos: tuple[float, float, float]
    tcp_quat: tuple[float, float, float, float]


@dataclass(frozen=True)
class RobotModelSpec:
    model_id: str
    provenance: str
    arms: dict[str, ArmModelSpec]


IDENTITY_QUAT = (1.0, 0.0, 0.0, 0.0)


def _link(
    pos: tuple[float, float, float],
    axis: tuple[float, float, float],
    joint_range: tuple[float, float],
    quat: tuple[float, float, float, float] = IDENTITY_QUAT,
) -> LinkSpec:
    return LinkSpec(pos=pos, quat=quat, axis=axis, joint_range=joint_range)


def _reference_arm(side: str) -> ArmModelSpec:
    mirror = -1.0 if side == "right" else 1.0
    root_quat = (
        0.7071054825112363,
        0.7071080798594735 * (-mirror),
        0.0,
        0.0,
    )
    joint1_range = (-1.396263, 3.490659) if side == "right" else (-3.490659, 1.396263)
    joint2_range = (
        (-0.17453267320510335, 3.3161253267948965)
        if side == "right"
        else (-3.3161253267948965, 0.17453267320510335)
    )
    link2_quat = (
        0.7071067811882787,
        0.7071067811848163 * (-mirror),
        0.0,
        0.0,
    )
    joint7_axis = (0.0, 1.0 if side == "right" else -1.0, 0.0)
    return ArmModelSpec(
        root_pos=(0.0, 0.031 * mirror, 0.0),
        root_quat=root_quat,
        links=(
            _link((0.0, 0.0, 0.0625), (0.0, 0.0, 1.0), joint1_range),
            _link(
                (-0.0301, 0.0, 0.06),
                (-1.0, 0.0, 0.0),
                joint2_range,
                link2_quat,
            ),
            _link((0.0301, 0.0, 0.06625), (0.0, 0.0, 1.0), (-1.570796, 1.570796)),
            # HF's J4_5cm_extended part increases the J3-to-J4 segment by 50 mm.
            _link((0.0, 0.0315, 0.20375), (0.0, 1.0, 0.0), (0.0, 2.443461)),
            _link((0.0, -0.0315, 0.0955), (0.0, 0.0, 1.0), (-1.570796, 1.570796)),
            _link((0.0375, 0.0, 0.1205), (1.0, 0.0, 0.0), (-0.785398, 0.785398)),
            _link((-0.0375, 0.0, 0.0), joint7_axis, (-1.570796, 1.570796)),
        ),
        # Fixed link8 + hand + original pinch-center TCP. HF custom jaws retain
        # the OpenArm hand mount; jaw aperture is calibrated separately.
        tcp_pos=(0.000001, -0.0045, 0.1801),
        tcp_quat=IDENTITY_QUAT,
    )


def _target_arm(side: str) -> ArmModelSpec:
    mirror = -1.0 if side == "right" else 1.0
    joint1_axis = (0.0, mirror, 0.0)
    joint2_range = (-0.17453, 3.3161) if side == "right" else (-3.3161, 0.17453)
    joint6_axis = (0.0, 1.0 if side == "right" else -1.0, 0.0)
    return ArmModelSpec(
        root_pos=(0.0, 0.031 * mirror, 0.0),
        root_quat=IDENTITY_QUAT,
        links=(
            _link((0.0, 0.0625 * mirror, 0.0), joint1_axis, (-2.3562, 2.3562)),
            _link((0.0, 0.06 * mirror, 0.0), (-1.0, 0.0, 0.0), joint2_range),
            _link((0.0, 0.0, -0.06625), (0.0, 0.0, -1.0), (-1.5708, 1.5708)),
            _link((0.0, 0.0, -0.15375), (0.0, -1.0, 0.0), (0.0, 2.4435)),
            _link((0.0, 0.0, -0.0955), (0.0, 0.0, -1.0), (-1.5708, 1.5708)),
            _link((0.0, 0.0, -0.1205), joint6_axis, (-0.7854, 1.2217)),
            _link((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (-1.5708, 1.5708)),
        ),
        tcp_pos=(-0.02193, 0.0, -0.138),
        tcp_quat=(0.70710678, 0.0, 0.70710678, 0.0),
    )


BUILTIN_MODELS: dict[str, RobotModelSpec] = {
    "hf_openarm_v1_extended": RobotModelSpec(
        model_id="hf_openarm_v1_extended",
        provenance=(
            "enactic/openarm v1 kinematics plus lerobot/openarms-hardware-modifications"
            "@5299ad9 J4_5cm_extended"
        ),
        arms={side: _reference_arm(side) for side in ("right", "left")},
    ),
    "anvil_openarm_v2": RobotModelSpec(
        model_id="anvil_openarm_v2",
        provenance="anvil-openarm-mujoco models/anvil_openarm_bimanual.xml",
        arms={side: _target_arm(side) for side in ("right", "left")},
    ),
}


def model_spec_hash(spec: RobotModelSpec) -> str:
    payload = json.dumps(asdict(spec), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def get_model_spec(model_id: str) -> RobotModelSpec:
    try:
        return BUILTIN_MODELS[model_id]
    except KeyError as exc:
        raise KeyError(f"unknown built-in kinematic model: {model_id}") from exc


def _values(values: tuple[float, ...]) -> str:
    return " ".join(f"{value:.12g}" for value in values)


def arm_mjcf(spec: RobotModelSpec, side: str) -> str:
    """Generate a standalone mesh-free MuJoCo arm model."""
    arm = spec.arms[side]
    lines = [
        f'<mujoco model="{spec.model_id}_{side}">',
        '  <compiler angle="radian" autolimits="true"/>',
        '  <option gravity="0 0 0"/>',
        "  <worldbody>",
        f'    <body name="root" pos="{_values(arm.root_pos)}" quat="{_values(arm.root_quat)}">',
    ]
    indent = "      "
    for index, link in enumerate(arm.links, start=1):
        lines.extend(
            [
                (
                    f'{indent}<body name="link{index}" pos="{_values(link.pos)}" '
                    f'quat="{_values(link.quat)}">'
                ),
                (
                    f'{indent}  <joint name="joint{index}" type="hinge" '
                    f'axis="{_values(link.axis)}" range="{_values(link.joint_range)}"/>'
                ),
                (
                    f'{indent}  <geom type="sphere" size="0.001" mass="0.001" '
                    'contype="0" conaffinity="0" rgba="0 0 0 0"/>'
                ),
            ]
        )
        indent += "  "
    lines.append(
        f'{indent}<site name="tcp" pos="{_values(arm.tcp_pos)}" '
        f'quat="{_values(arm.tcp_quat)}" size="0.001"/>'
    )
    for _ in arm.links:
        indent = indent[:-2]
        lines.append(f"{indent}</body>")
    lines.extend(["    </body>", "  </worldbody>", "</mujoco>"])
    return "\n".join(lines)


class MujocoArmKinematics:
    """FK and Jacobians for one built-in arm."""

    def __init__(self, spec: RobotModelSpec, side: str):
        try:
            mujoco: Any = importlib.import_module("mujoco")
        except ImportError as exc:
            raise RuntimeError(
                "MuJoCo is required for embodiment bridging; install anvil-embodiment"
            ) from exc
        self._mujoco = mujoco
        self.spec = spec
        self.side = side
        self.model = mujoco.MjModel.from_xml_string(arm_mjcf(spec, side))
        self.data = mujoco.MjData(self.model)
        self.site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
        self.qpos_ids = np.asarray(
            [
                self.model.jnt_qposadr[
                    mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}")
                ]
                for i in range(1, 8)
            ],
            dtype=np.int32,
        )
        self.dof_ids = np.asarray(
            [
                self.model.jnt_dofadr[
                    mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}")
                ]
                for i in range(1, 8)
            ],
            dtype=np.int32,
        )
        self.limits = np.asarray([link.joint_range for link in spec.arms[side].links])

    def pose(self, joints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        joints = np.asarray(joints, dtype=np.float64)
        if joints.shape != (7,) or not np.all(np.isfinite(joints)):
            raise ValueError("joints must be a finite 7-D vector")
        self.data.qpos[self.qpos_ids] = joints
        self._mujoco.mj_forward(self.model, self.data)
        position = self.data.site_xpos[self.site_id].copy()
        rotation = self.data.site_xmat[self.site_id].reshape(3, 3).copy()
        return position, rotation

    def jacobian(self, joints: np.ndarray) -> np.ndarray:
        self.pose(joints)
        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        jacr = np.zeros((3, self.model.nv), dtype=np.float64)
        self._mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.site_id)
        return np.concatenate([jacp[:, self.dof_ids], jacr[:, self.dof_ids]], axis=0)


def _quat_matrix_np(quat: tuple[float, float, float, float]) -> np.ndarray:
    w, x, y, z = np.asarray(quat, dtype=np.float64) / np.linalg.norm(quat)
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _torch_fixed_transform(
    pos: tuple[float, float, float],
    quat: tuple[float, float, float, float],
    *,
    dtype: Any,
    device: Any,
) -> Any:
    import torch

    transform = torch.eye(4, dtype=dtype, device=device)
    transform[:3, :3] = torch.as_tensor(_quat_matrix_np(quat), dtype=dtype, device=device)
    transform[:3, 3] = torch.as_tensor(pos, dtype=dtype, device=device)
    return transform


def _torch_axis_rotation(axis: tuple[float, float, float], angle: Any) -> Any:
    import torch

    unit = torch.as_tensor(axis, dtype=angle.dtype, device=angle.device)
    unit = unit / torch.linalg.vector_norm(unit)
    x, y, z = unit.unbind()
    zero = torch.zeros_like(x)
    skew = torch.stack(
        [
            torch.stack([zero, -z, y]),
            torch.stack([z, zero, -x]),
            torch.stack([-y, x, zero]),
        ]
    )
    eye = torch.eye(3, dtype=angle.dtype, device=angle.device)
    shape = angle.shape + (3, 3)
    skew = skew.expand(shape)
    eye = eye.expand(shape)
    outer = torch.outer(unit, unit).expand(shape)
    return (
        torch.cos(angle)[..., None, None] * eye
        + (1 - torch.cos(angle))[..., None, None] * outer
        + torch.sin(angle)[..., None, None] * skew
    )


def torch_forward_kinematics(spec: RobotModelSpec, side: str, joints: Any) -> tuple[Any, Any]:
    """Differentiable TCP FK for tensors with a final dimension of seven."""
    import torch

    if joints.shape[-1] != 7:
        raise ValueError("joints must have a final dimension of 7")
    arm = spec.arms[side]
    batch_shape = joints.shape[:-1]
    transform = (
        _torch_fixed_transform(
            arm.root_pos, arm.root_quat, dtype=joints.dtype, device=joints.device
        )
        .expand(batch_shape + (4, 4))
        .clone()
    )
    for index, link in enumerate(arm.links):
        fixed = _torch_fixed_transform(
            link.pos, link.quat, dtype=joints.dtype, device=joints.device
        ).expand(batch_shape + (4, 4))
        rotation = (
            torch.eye(4, dtype=joints.dtype, device=joints.device)
            .expand(batch_shape + (4, 4))
            .clone()
        )
        rotation[..., :3, :3] = _torch_axis_rotation(link.axis, joints[..., index])
        transform = transform @ fixed @ rotation
    tcp = _torch_fixed_transform(
        arm.tcp_pos, arm.tcp_quat, dtype=joints.dtype, device=joints.device
    ).expand(batch_shape + (4, 4))
    transform = transform @ tcp
    return transform[..., :3, 3], transform[..., :3, :3]
