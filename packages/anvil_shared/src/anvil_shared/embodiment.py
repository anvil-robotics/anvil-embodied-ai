"""Policy-neutral embodiment and zero/one-shot experiment contracts."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class EmbodimentError(ValueError):
    """An embodiment, policy binding, or experiment is unsafe or inconsistent."""


def _require_finite_sequence(value: Any, *, length: int, label: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise EmbodimentError(f"{label} must contain {length} values")
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise EmbodimentError(f"{label} must contain only numeric values") from exc
    if not all(math.isfinite(item) for item in result):
        raise EmbodimentError(f"{label} must contain only finite values")
    return result


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise EmbodimentError(f"missing JSON file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise EmbodimentError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EmbodimentError(f"expected an object in {path}")
    return value


@dataclass(frozen=True)
class ArmContract:
    side: str
    command_topic: str
    command_order: tuple[str, ...]
    tcp_site: str


@dataclass(frozen=True)
class EmbodimentContract:
    """Robot semantics generated from the canonical MuJoCo model."""

    path: Path
    robot: str
    control_hz: float
    arms: dict[str, ArmContract]
    action_surfaces: frozenset[str]
    joint_ranges: dict[str, tuple[float, float]]

    @classmethod
    def load(cls, path: str | Path) -> EmbodimentContract:
        source = Path(path)
        raw = _read_json(source)
        if raw.get("schema_version") != 1:
            raise EmbodimentError("embodiment schema_version must be 1")
        arms = {}
        for side, arm in raw.get("arms", {}).items():
            order = tuple(str(value) for value in arm.get("command_order", []))
            if len(order) != len(set(order)) or not order:
                raise EmbodimentError(f"arm {side} command_order must be non-empty and unique")
            arms[side] = ArmContract(
                side=side,
                command_topic=str(arm.get("command_topic", "")),
                command_order=order,
                tcp_site=str(arm.get("tcp_site", "")),
            )
        if not arms:
            raise EmbodimentError("embodiment must define at least one arm")
        ranges = {}
        for name, values in raw.get("joint_ranges", {}).items():
            if not isinstance(values, list) or len(values) != 2:
                raise EmbodimentError(f"joint range {name} must be [min, max]")
            low, high = (float(value) for value in values)
            if not (math.isfinite(low) and math.isfinite(high) and low < high):
                raise EmbodimentError(f"joint range {name} is invalid")
            ranges[name] = (low, high)
        return cls(
            path=source,
            robot=str(raw.get("robot", "")),
            control_hz=float(raw.get("control_hz", 0.0)),
            arms=arms,
            action_surfaces=frozenset(str(value) for value in raw.get("action_surfaces", [])),
            joint_ranges=ranges,
        )

    def active_joint_ranges(self, side: str) -> list[tuple[float, float] | None]:
        try:
            order = self.arms[side].command_order
        except KeyError as exc:
            raise EmbodimentError(f"unknown arm side: {side}") from exc
        result = []
        for name in order:
            if name == "finger_joint1":
                result.append(None)
                continue
            side_key = f"openarm_{'left' if side == 'l' else 'right'}_{name}"
            result.append(self.joint_ranges.get(side_key, self.joint_ranges.get(name)))
        return result


@dataclass(frozen=True)
class PolicyBinding:
    model_type: str
    arm: str
    action_surface: str
    normalization_source: str
    camera_roles: dict[str, str]
    active_state_dim: int
    active_action_dim: int
    padded_capacity: int | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> PolicyBinding:
        return cls(
            model_type=str(value.get("model_type", "")),
            arm=str(value.get("arm", "")),
            action_surface=str(value.get("action_surface", "")),
            normalization_source=str(value.get("normalization_source", "")),
            camera_roles={str(k): str(v) for k, v in value.get("camera_roles", {}).items()},
            active_state_dim=int(value.get("active_state_dim", 0)),
            active_action_dim=int(value.get("active_action_dim", 0)),
            padded_capacity=(
                int(value["padded_capacity"]) if value.get("padded_capacity") is not None else None
            ),
        )

    def validate(self, embodiment: EmbodimentContract) -> None:
        if self.model_type not in {"pi05", "groot", "smolvla", "act", "vla_jepa"}:
            raise EmbodimentError(f"unsupported experiment model: {self.model_type}")
        if self.arm not in embodiment.arms:
            raise EmbodimentError(f"binding references unknown arm: {self.arm}")
        if self.action_surface not in embodiment.action_surfaces:
            raise EmbodimentError(f"unsupported action surface: {self.action_surface}")
        if self.normalization_source not in {"embodiment_limits", "checkpoint", "demonstration"}:
            raise EmbodimentError(f"unsupported normalization source: {self.normalization_source}")
        if not self.camera_roles or any(not value for value in self.camera_roles.values()):
            raise EmbodimentError("camera_roles must map every policy role to a scene camera")
        arm_dim = len(embodiment.arms[self.arm].command_order)
        if self.action_surface == "joint_position" and self.active_action_dim != arm_dim:
            raise EmbodimentError(
                f"joint-position action dimension {self.active_action_dim} does not match arm {arm_dim}"
            )
        if self.active_state_dim <= 0 or self.active_action_dim <= 0:
            raise EmbodimentError("active dimensions must be positive")
        if self.padded_capacity is not None and (
            self.active_state_dim > self.padded_capacity
            or self.active_action_dim > self.padded_capacity
        ):
            raise EmbodimentError("active dimensions exceed padded policy capacity")
        if self.model_type == "pi05" and self.padded_capacity != 32:
            raise EmbodimentError("Pi0.5 binding must declare its 32-D padded capacity")
        if self.model_type in {"act", "vla_jepa"} and self.normalization_source != "checkpoint":
            raise EmbodimentError(f"{self.model_type} baseline requires checkpoint normalization")


def normalize_from_limits(
    values: list[float], ranges: list[tuple[float, float] | None]
) -> list[float]:
    """Map physical joint values to [-1, 1]; pass unbounded gripper slots through."""
    if len(values) != len(ranges):
        raise EmbodimentError("value/range dimensions differ")
    result = []
    for value, bounds in zip(values, ranges, strict=True):
        if not math.isfinite(value):
            raise EmbodimentError("joint values must be finite")
        if bounds is None:
            result.append(value)
            continue
        low, high = bounds
        result.append(max(-1.0, min(1.0, 2.0 * (value - low) / (high - low) - 1.0)))
    return result


@dataclass(frozen=True)
class AdapterVectorContract:
    """Ordered vector contract on one side of an embodiment adapter."""

    names: tuple[str, ...]
    arm_order: tuple[str, ...]
    joints_per_arm: int
    joint_unit: str

    @classmethod
    def from_dict(cls, value: dict[str, Any], *, label: str) -> AdapterVectorContract:
        names = tuple(str(name) for name in value.get("names", []))
        arm_order = tuple(str(side) for side in value.get("arm_order", []))
        joints_per_arm = int(value.get("joints_per_arm", 0))
        joint_unit = str(value.get("joint_unit", ""))
        if not names or len(names) != len(set(names)):
            raise EmbodimentError(f"{label}.names must be non-empty and unique")
        if arm_order != ("right", "left"):
            raise EmbodimentError(f"{label}.arm_order must be ['right', 'left']")
        if joints_per_arm != 8 or len(names) != len(arm_order) * joints_per_arm:
            raise EmbodimentError(f"{label} must describe a right-then-left 16-D vector")
        if joint_unit not in {"radian", "degree"}:
            raise EmbodimentError(f"{label}.joint_unit must be radian or degree")
        return cls(
            names=names,
            arm_order=arm_order,
            joints_per_arm=joints_per_arm,
            joint_unit=joint_unit,
        )


@dataclass(frozen=True)
class KinematicModelContract:
    """Pinned built-in kinematic model used by an adapter."""

    model_id: str
    sha256: str
    provenance: str

    @classmethod
    def from_dict(cls, value: dict[str, Any], *, label: str) -> KinematicModelContract:
        model_id = str(value.get("model_id", ""))
        sha256 = str(value.get("sha256", ""))
        provenance = str(value.get("provenance", ""))
        if not model_id or not provenance:
            raise EmbodimentError(f"{label} requires model_id and provenance")
        if len(sha256) != 64 or any(char not in "0123456789abcdef" for char in sha256):
            raise EmbodimentError(f"{label}.sha256 must be a lowercase SHA-256 digest")
        return cls(model_id=model_id, sha256=sha256, provenance=provenance)


@dataclass(frozen=True)
class GripperCalibration:
    """Endpoint calibration represented as a monotonic open fraction."""

    reference_closed: float
    reference_open: float
    target_closed: float
    target_open: float

    @classmethod
    def from_dict(cls, value: dict[str, Any], *, side: str) -> GripperCalibration:
        values = _require_finite_sequence(
            [
                value.get("reference_closed"),
                value.get("reference_open"),
                value.get("target_closed"),
                value.get("target_open"),
            ],
            length=4,
            label=f"grippers.{side}",
        )
        calibration = cls(*values)
        if calibration.reference_closed == calibration.reference_open:
            raise EmbodimentError(f"grippers.{side} reference endpoints must differ")
        if calibration.target_closed == calibration.target_open:
            raise EmbodimentError(f"grippers.{side} target endpoints must differ")
        return calibration


@dataclass(frozen=True)
class IKContract:
    max_iterations: int
    restart_count: int
    damping: float
    continuity_weight: float
    position_tolerance_m: float
    orientation_tolerance_rad: float
    joint_limit_margin_rad: float
    max_step_rad: float

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> IKContract:
        result = cls(
            max_iterations=int(value.get("max_iterations", 80)),
            restart_count=int(value.get("restart_count", 8)),
            damping=float(value.get("damping", 0.03)),
            continuity_weight=float(value.get("continuity_weight", 0.002)),
            position_tolerance_m=float(value.get("position_tolerance_m", 0.001)),
            orientation_tolerance_rad=float(value.get("orientation_tolerance_rad", 0.0174533)),
            joint_limit_margin_rad=float(value.get("joint_limit_margin_rad", 0.001)),
            max_step_rad=float(value.get("max_step_rad", 0.2)),
        )
        numeric = (
            result.damping,
            result.continuity_weight,
            result.position_tolerance_m,
            result.orientation_tolerance_rad,
            result.joint_limit_margin_rad,
            result.max_step_rad,
        )
        if (
            result.max_iterations < 1
            or result.restart_count < 0
            or not all(math.isfinite(item) and item >= 0 for item in numeric)
        ):
            raise EmbodimentError("ik values must be finite and non-negative")
        if result.position_tolerance_m == 0 or result.orientation_tolerance_rad == 0:
            raise EmbodimentError("ik tolerances must be positive")
        return result


@dataclass(frozen=True)
class ResidualAdapterContract:
    chunk_size: int
    hidden_size: int
    num_layers: int
    max_joint_correction_rad: float
    max_joint_range_fraction: float
    correct_grippers: bool

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ResidualAdapterContract:
        result = cls(
            chunk_size=int(value.get("chunk_size", 30)),
            hidden_size=int(value.get("hidden_size", 128)),
            num_layers=int(value.get("num_layers", 2)),
            max_joint_correction_rad=float(value.get("max_joint_correction_rad", 0.15)),
            max_joint_range_fraction=float(value.get("max_joint_range_fraction", 0.1)),
            correct_grippers=bool(value.get("correct_grippers", False)),
        )
        if result.chunk_size < 1 or result.hidden_size < 1 or result.num_layers < 1:
            raise EmbodimentError("residual dimensions must be positive")
        if not 0 < result.max_joint_correction_rad <= math.pi:
            raise EmbodimentError("residual max_joint_correction_rad must be in (0, pi]")
        if not 0 < result.max_joint_range_fraction <= 1:
            raise EmbodimentError("residual max_joint_range_fraction must be in (0, 1]")
        if result.correct_grippers:
            raise EmbodimentError("schema v1 keeps grippers deterministic")
        return result


@dataclass(frozen=True)
class EmbodimentAdapterSpec:
    """Complete, fail-closed contract for a frozen-policy embodiment adapter."""

    path: Path
    adapter_id: str
    deployment_status: str
    base_policy_repo: str
    base_policy_revision: str
    base_policy_processor_sha256: dict[str, str]
    reference_vector: AdapterVectorContract
    target_vector: AdapterVectorContract
    reference_model: KinematicModelContract
    target_model: KinematicModelContract
    grippers: dict[str, GripperCalibration]
    ik: IKContract
    residual: ResidualAdapterContract

    @classmethod
    def load(cls, path: str | Path) -> EmbodimentAdapterSpec:
        source = Path(path)
        raw = _read_json(source)
        if raw.get("schema_version") != 1:
            raise EmbodimentError("adapter schema_version must be 1")
        policy = raw.get("base_policy", {})
        hashes = {
            str(name): str(digest) for name, digest in policy.get("processor_sha256", {}).items()
        }
        required_hashes = {
            "policy_preprocessor.json",
            "policy_postprocessor.json",
            "policy_preprocessor_step_3_normalizer_processor.safetensors",
            "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
        }
        if set(hashes) != required_hashes or any(
            len(value) != 64 or any(char not in "0123456789abcdef" for char in value)
            for value in hashes.values()
        ):
            raise EmbodimentError(
                "base_policy.processor_sha256 must pin processor JSON and state files"
            )
        models = raw.get("models", {})
        grippers = {
            side: GripperCalibration.from_dict(value, side=side)
            for side, value in raw.get("grippers", {}).items()
        }
        if set(grippers) != {"right", "left"}:
            raise EmbodimentError("grippers must define right and left calibrations")
        result = cls(
            path=source,
            adapter_id=str(raw.get("adapter_id", "")),
            deployment_status=str(raw.get("deployment_status", "")),
            base_policy_repo=str(policy.get("repo_id", "")),
            base_policy_revision=str(policy.get("revision", "")),
            base_policy_processor_sha256=hashes,
            reference_vector=AdapterVectorContract.from_dict(
                raw.get("reference_vector", {}), label="reference_vector"
            ),
            target_vector=AdapterVectorContract.from_dict(
                raw.get("target_vector", {}), label="target_vector"
            ),
            reference_model=KinematicModelContract.from_dict(
                models.get("reference", {}), label="models.reference"
            ),
            target_model=KinematicModelContract.from_dict(
                models.get("target", {}), label="models.target"
            ),
            grippers=grippers,
            ik=IKContract.from_dict(raw.get("ik", {})),
            residual=ResidualAdapterContract.from_dict(raw.get("residual", {})),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if not self.adapter_id:
            raise EmbodimentError("adapter_id is required")
        if self.deployment_status not in {"offline_only", "live_approved"}:
            raise EmbodimentError("deployment_status must be offline_only or live_approved")
        if not self.base_policy_repo or len(self.base_policy_revision) != 40:
            raise EmbodimentError("base_policy must pin a repo_id and 40-character revision")
        if self.reference_vector.names != self.target_vector.names:
            raise EmbodimentError("reference and target feature names must match exactly")

    def require_live_approved(self) -> None:
        if self.deployment_status != "live_approved":
            raise EmbodimentError(
                f"adapter {self.adapter_id} is {self.deployment_status}; refusing live execution"
            )


@dataclass(frozen=True)
class ExperimentContract:
    scene_bundle: Path
    instruction: str
    success_threshold: float
    seeds: tuple[int, ...]
    one_shot_demo: Path | None
    bindings: tuple[PolicyBinding, ...]
    observation_renderer: dict[str, Any]

    @classmethod
    def from_dict(cls, value: dict[str, Any], base: Path) -> ExperimentContract:
        one = value.get("one_shot", {})
        demo = one.get("demonstration")
        return cls(
            scene_bundle=(base / str(value.get("scene_bundle", ""))).resolve(),
            instruction=str(value.get("instruction", "")).strip(),
            success_threshold=float(value.get("success_threshold", 0.8)),
            seeds=tuple(int(seed) for seed in value.get("seeds", [])),
            one_shot_demo=(base / str(demo)).resolve() if demo else None,
            bindings=tuple(PolicyBinding.from_dict(item) for item in value.get("models", [])),
            observation_renderer=dict(value.get("observation_renderer", {})),
        )

    def validate(self, *, mode: str) -> EmbodimentContract:
        if mode not in {"zero", "one"}:
            raise EmbodimentError("mode must be zero or one")
        if not self.instruction:
            raise EmbodimentError("experiment instruction is required")
        if not 0 < self.success_threshold <= 1:
            raise EmbodimentError("success_threshold must be in (0, 1]")
        if len(self.seeds) < 20 or len(self.seeds) != len(set(self.seeds)):
            raise EmbodimentError("experiment requires at least 20 unique held-out seeds")
        scene = _read_json(self.scene_bundle / "scene_manifest.json")
        embodiment_path = self.scene_bundle / str(scene.get("embodiment_manifest", ""))
        embodiment = EmbodimentContract.load(embodiment_path)
        if not self.bindings:
            raise EmbodimentError("experiment requires model bindings")
        for binding in self.bindings:
            binding.validate(embodiment)
        renderer = self.observation_renderer
        if renderer:
            if renderer.get("type") != "gaussian_mujoco_hybrid":
                raise EmbodimentError("unsupported observation renderer type")
            if renderer.get("gaussian", {}).get("render_mode") != "RGB+ED":
                raise EmbodimentError("hybrid Gaussian renderer must use RGB+ED")
            cameras = set(renderer.get("cameras", []))
            required = {
                camera for binding in self.bindings for camera in binding.camera_roles.values()
            }
            missing = required - cameras
            if missing:
                raise EmbodimentError(
                    f"observation renderer is missing policy cameras: {sorted(missing)}"
                )
        if mode == "one" and (self.one_shot_demo is None or not self.one_shot_demo.exists()):
            raise EmbodimentError("one-shot mode requires exactly one existing demonstration")
        return embodiment
