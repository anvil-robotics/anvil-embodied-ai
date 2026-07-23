"""Bidirectional, fail-closed kinematic embodiment bridge."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from anvil_shared.embodiment import EmbodimentAdapterSpec, GripperCalibration

from .kinematics import (
    MujocoArmKinematics,
    RobotModelSpec,
    get_model_spec,
    model_spec_hash,
)

# The v1 pinch-center site and v2 follower TCP describe the same physical tool
# with different local axes.  R_reference = R_target @ this fixed rotation.
TARGET_TO_REFERENCE_TCP_ROTATION = np.asarray(
    [[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]],
    dtype=np.float64,
)


class BridgeError(RuntimeError):
    """A pose could not be mapped safely between embodiments."""


@dataclass(frozen=True)
class IKDiagnostic:
    side: str
    direction: str
    converged: bool
    iterations: int
    position_error_m: float
    orientation_error_rad: float
    min_joint_limit_margin_rad: float


@dataclass(frozen=True)
class BridgeResult:
    values: np.ndarray
    diagnostics: tuple[IKDiagnostic, ...]

    @property
    def valid(self) -> bool:
        return all(item.converged for item in self.diagnostics)


def _orientation_error(desired: np.ndarray, current: np.ndarray) -> np.ndarray:
    relative = desired @ current.T
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    vee = np.asarray(
        [
            relative[2, 1] - relative[1, 2],
            relative[0, 2] - relative[2, 0],
            relative[1, 0] - relative[0, 1],
        ]
    )
    if angle < 1e-8:
        return 0.5 * vee
    sine = float(np.sin(angle))
    if abs(sine) < 1e-8:
        eigenvalues, eigenvectors = np.linalg.eigh(relative)
        axis = eigenvectors[:, int(np.argmin(np.abs(eigenvalues - 1.0)))]
        return angle * axis / max(np.linalg.norm(axis), 1e-12)
    return angle * vee / (2.0 * sine)


def _gripper_fraction(value: float, closed: float, opened: float) -> float:
    return float(np.clip((value - closed) / (opened - closed), 0.0, 1.0))


def _target_to_reference_gripper(value: float, calibration: GripperCalibration) -> float:
    fraction = _gripper_fraction(value, calibration.target_closed, calibration.target_open)
    return calibration.reference_closed + fraction * (
        calibration.reference_open - calibration.reference_closed
    )


def _reference_to_target_gripper(value: float, calibration: GripperCalibration) -> float:
    fraction = _gripper_fraction(value, calibration.reference_closed, calibration.reference_open)
    return calibration.target_closed + fraction * (
        calibration.target_open - calibration.target_closed
    )


class KinematicEmbodimentBridge:
    """Map 16-D bimanual states/actions through TCP pose rather than joint labels."""

    @staticmethod
    def _to_radians(values: np.ndarray, unit: str) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        return np.deg2rad(values) if unit == "degree" else values.copy()

    @staticmethod
    def _from_radians(values: np.ndarray, unit: str) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        return np.rad2deg(values) if unit == "degree" else values.copy()

    def __init__(self, spec: EmbodimentAdapterSpec):
        self.spec = spec
        self.reference_spec = get_model_spec(spec.reference_model.model_id)
        self.target_spec = get_model_spec(spec.target_model.model_id)
        self._verify_model(self.reference_spec, spec.reference_model.sha256)
        self._verify_model(self.target_spec, spec.target_model.sha256)
        self.reference = {
            side: MujocoArmKinematics(self.reference_spec, side) for side in ("right", "left")
        }
        self.target = {
            side: MujocoArmKinematics(self.target_spec, side) for side in ("right", "left")
        }

    @staticmethod
    def _verify_model(model: RobotModelSpec, expected_hash: str) -> None:
        actual = model_spec_hash(model)
        if actual != expected_hash:
            raise BridgeError(
                f"kinematic model {model.model_id} hash mismatch: expected {expected_hash}, got {actual}"
            )

    @property
    def target_joint_ranges(self) -> np.ndarray:
        rows: list[tuple[float, float]] = []
        for side in ("right", "left"):
            limits = self._from_radians(
                self.target[side].limits, self.spec.target_vector.joint_unit
            )
            rows.extend(tuple(row) for row in limits)
            calibration = self.spec.grippers[side]
            rows.append(
                (
                    min(calibration.target_closed, calibration.target_open),
                    max(calibration.target_closed, calibration.target_open),
                )
            )
        return np.asarray(rows, dtype=np.float64)

    @property
    def residual_bounds(self) -> np.ndarray:
        ranges = self.target_joint_ranges
        width = ranges[:, 1] - ranges[:, 0]
        max_correction = self._from_radians(
            np.asarray([self.spec.residual.max_joint_correction_rad]),
            self.spec.target_vector.joint_unit,
        )[0]
        bounds = np.minimum(
            max_correction,
            self.spec.residual.max_joint_range_fraction * width,
        )
        bounds[[7, 15]] = 0.0
        return bounds

    @staticmethod
    def _validate_vector(value: np.ndarray, *, matrix: bool = False) -> np.ndarray:
        result = np.asarray(value, dtype=np.float64)
        expected = 2 if matrix else 1
        if result.ndim != expected or result.shape[-1] != 16 or not np.all(np.isfinite(result)):
            shape = "[T, 16]" if matrix else "[16]"
            raise BridgeError(f"expected a finite {shape} value, got {result.shape}")
        return result

    def target_state_to_policy(
        self, state: np.ndarray, previous_policy_state: np.ndarray | None = None
    ) -> BridgeResult:
        state = self._validate_vector(state)
        previous = (
            self._validate_vector(previous_policy_state)
            if previous_policy_state is not None
            else None
        )
        output = np.zeros(16, dtype=np.float64)
        diagnostics: list[IKDiagnostic] = []
        for side_index, side in enumerate(("right", "left")):
            start = side_index * 8
            source = self._to_radians(state[start : start + 7], self.spec.target_vector.joint_unit)
            desired_position, desired_rotation = self.target[side].pose(source)
            seed = (
                self._to_radians(
                    previous[start : start + 7],
                    self.spec.reference_vector.joint_unit,
                )
                if previous is not None
                else self._semantic_seed(source, self.reference[side].limits)
            )
            solved, diagnostic = self._solve_with_restarts(
                destination=self.reference[side],
                desired_position=desired_position,
                desired_rotation=desired_rotation @ TARGET_TO_REFERENCE_TCP_ROTATION,
                seed=seed,
                side=side,
                direction="target_to_reference",
            )
            output[start : start + 7] = self._from_radians(
                solved, self.spec.reference_vector.joint_unit
            )
            output[start + 7] = _target_to_reference_gripper(
                state[start + 7], self.spec.grippers[side]
            )
            diagnostics.append(diagnostic)
        result = BridgeResult(output, tuple(diagnostics))
        self._require_valid(result)
        return result

    def target_chunk_to_policy(
        self, chunk: np.ndarray, current_policy_state: np.ndarray | None = None
    ) -> BridgeResult:
        chunk = self._validate_vector(chunk, matrix=True)
        previous = current_policy_state
        mapped = []
        diagnostics: list[IKDiagnostic] = []
        for row in chunk:
            result = self.target_state_to_policy(row, previous)
            mapped.append(result.values)
            diagnostics.extend(result.diagnostics)
            previous = result.values
        return BridgeResult(np.stack(mapped), tuple(diagnostics))

    def policy_chunk_to_target(
        self, chunk: np.ndarray, current_target_state: np.ndarray
    ) -> BridgeResult:
        chunk = self._validate_vector(chunk, matrix=True)
        previous = self._validate_vector(current_target_state)
        mapped: list[np.ndarray] = []
        diagnostics: list[IKDiagnostic] = []
        for row in chunk:
            output = np.zeros(16, dtype=np.float64)
            for side_index, side in enumerate(("right", "left")):
                start = side_index * 8
                reference_joints = self._to_radians(
                    row[start : start + 7], self.spec.reference_vector.joint_unit
                )
                desired_position, desired_rotation = self.reference[side].pose(reference_joints)
                solved, diagnostic = self._solve_with_restarts(
                    destination=self.target[side],
                    desired_position=desired_position,
                    desired_rotation=desired_rotation @ TARGET_TO_REFERENCE_TCP_ROTATION.T,
                    seed=self._to_radians(
                        previous[start : start + 7],
                        self.spec.target_vector.joint_unit,
                    ),
                    side=side,
                    direction="reference_to_target",
                )
                output[start : start + 7] = self._from_radians(
                    solved, self.spec.target_vector.joint_unit
                )
                output[start + 7] = _reference_to_target_gripper(
                    row[start + 7], self.spec.grippers[side]
                )
                diagnostics.append(diagnostic)
            mapped.append(output)
            previous = output
        result = BridgeResult(np.stack(mapped), tuple(diagnostics))
        self._require_valid(result)
        return result

    @staticmethod
    def _semantic_seed(source: np.ndarray, limits: np.ndarray) -> np.ndarray:
        seed = np.asarray(source, dtype=np.float64).copy()
        seed[5], seed[6] = source[6], source[5]
        return np.clip(seed, limits[:, 0], limits[:, 1])

    def _solve_with_restarts(
        self,
        *,
        destination: MujocoArmKinematics,
        desired_position: np.ndarray,
        desired_rotation: np.ndarray,
        seed: np.ndarray,
        side: str,
        direction: str,
    ) -> tuple[np.ndarray, IKDiagnostic]:
        """Use continuity first, then bounded deterministic restarts if needed."""
        original_seed = np.asarray(seed, dtype=np.float64)
        solved, diagnostic = self._solve(
            destination=destination,
            desired_position=desired_position,
            desired_rotation=desired_rotation,
            seed=original_seed,
            side=side,
            direction=direction,
        )
        if diagnostic.converged or self.spec.ik.restart_count == 0:
            return solved, diagnostic

        lower = destination.limits[:, 0] + self.spec.ik.joint_limit_margin_rad
        upper = destination.limits[:, 1] - self.spec.ik.joint_limit_margin_rad
        rng_seed = 20260717 + (0 if side == "right" else 1)
        rng = np.random.default_rng(rng_seed)
        best = (solved, diagnostic)
        best_score = self._ik_score(diagnostic, solved, original_seed, destination.limits)
        for candidate_seed in rng.uniform(lower, upper, size=(self.spec.ik.restart_count, 7)):
            candidate = self._solve(
                destination=destination,
                desired_position=desired_position,
                desired_rotation=desired_rotation,
                seed=candidate_seed,
                side=side,
                direction=direction,
            )
            candidate_score = self._ik_score(
                candidate[1], candidate[0], original_seed, destination.limits
            )
            if candidate_score < best_score:
                best = candidate
                best_score = candidate_score
        return best

    def _ik_score(
        self,
        diagnostic: IKDiagnostic,
        solved: np.ndarray,
        seed: np.ndarray,
        limits: np.ndarray,
    ) -> tuple[float, float]:
        if diagnostic.converged:
            width = np.maximum(limits[:, 1] - limits[:, 0], 1e-6)
            continuity = float(np.linalg.norm((solved - seed) / width))
            return (0.0, continuity)
        error = (
            diagnostic.position_error_m / self.spec.ik.position_tolerance_m
            + diagnostic.orientation_error_rad / self.spec.ik.orientation_tolerance_rad
        )
        return (1.0, float(error))

    def _solve(
        self,
        *,
        destination: MujocoArmKinematics,
        desired_position: np.ndarray,
        desired_rotation: np.ndarray,
        seed: np.ndarray,
        side: str,
        direction: str,
    ) -> tuple[np.ndarray, IKDiagnostic]:
        config = self.spec.ik
        lower = destination.limits[:, 0] + config.joint_limit_margin_rad
        upper = destination.limits[:, 1] - config.joint_limit_margin_rad
        q = np.clip(np.asarray(seed, dtype=np.float64), lower, upper)
        seed = q.copy()
        position_error = float("inf")
        orientation_error = float("inf")
        converged = False
        iteration = 0

        for iteration_index in range(1, config.max_iterations + 1):
            iteration = iteration_index
            current_position, current_rotation = destination.pose(q)
            position_delta = desired_position - current_position
            orientation_delta = _orientation_error(desired_rotation, current_rotation)
            position_error = float(np.linalg.norm(position_delta))
            orientation_error = float(np.linalg.norm(orientation_delta))
            if (
                position_error <= config.position_tolerance_m
                and orientation_error <= config.orientation_tolerance_rad
            ):
                converged = True
                break

            error = np.concatenate([position_delta, orientation_delta])
            jacobian = destination.jacobian(q)
            damped = jacobian @ jacobian.T
            damped += config.damping**2 * np.eye(6, dtype=np.float64)
            try:
                task_step = jacobian.T @ np.linalg.solve(damped, error)
                task_projector = jacobian.T @ np.linalg.solve(damped, jacobian)
            except np.linalg.LinAlgError:
                break
            nullspace = np.eye(7, dtype=np.float64) - task_projector
            posture_step = -config.continuity_weight * nullspace @ (q - seed)
            step = task_step + posture_step
            step = np.clip(step, -config.max_step_rad, config.max_step_rad)
            q = np.clip(q + step, lower, upper)

        min_margin = float(
            np.min(np.minimum(q - destination.limits[:, 0], destination.limits[:, 1] - q))
        )
        diagnostic = IKDiagnostic(
            side=side,
            direction=direction,
            converged=converged,
            iterations=iteration,
            position_error_m=position_error,
            orientation_error_rad=orientation_error,
            min_joint_limit_margin_rad=min_margin,
        )
        return q, diagnostic

    @staticmethod
    def _require_valid(result: BridgeResult) -> None:
        failed = [item for item in result.diagnostics if not item.converged]
        if not failed:
            return
        first = failed[0]
        raise BridgeError(
            f"{first.direction} IK failed for {first.side}: "
            f"position error={first.position_error_m:.4f} m, "
            f"orientation error={first.orientation_error_rad:.4f} rad"
        )
