"""Action limiter for safe robot control.

Applies delta limiting and joint reordering to actions before publishing
to ROS2 controllers.
"""

import numpy as np


class ActionLimiter:
    """
    Applies delta limiting and joint reordering before publishing actions.

    This class handles:
    1. Reordering actions from model joint order to controller joint order
    2. Applying delta limiting to prevent large joint movements
    3. Converting delta actions to absolute positions if needed
    """

    def __init__(
        self,
        max_delta: float = 0.1,
        min_delta_threshold: float | None = None,
        model_joint_order: list[str] | None = None,
        controller_joint_order: list[str] | None = None,
        delta_exclude_joints: list[str] | None = None,
        logger=None,
    ):
        """
        Initialize action limiter.

        Args:
            max_delta: Maximum position change per step (radians)
            min_delta_threshold: Minimum per-joint change to publish a new command.
                When set, commands are held at the last published value until the
                cumulative change exceeds this threshold. Helps overcome motor
                friction when model deltas are too small to move the joint.
            model_joint_order: Order the ML model outputs actions
            controller_joint_order: Order the ROS2 controller expects
            delta_exclude_joints: Joint names kept in absolute space (not delta-restored)
            logger: Optional ROS2 logger
        """
        self.max_delta = max_delta
        self.min_delta_threshold = min_delta_threshold
        self._last_published: np.ndarray | None = None
        self.model_joint_order = model_joint_order or []
        self.controller_joint_order = controller_joint_order or []
        self.logger = logger

        # Build reorder indices
        self.reorder_indices = self._build_reorder_indices()

        # Resolve excluded joint names → indices in controller order (post-reorder space).
        # Fall back to model order when controller order is not configured.
        ref_order = self.controller_joint_order or self.model_joint_order
        self._delta_exclude_indices: set[int] = {
            ref_order.index(name)
            for name in (delta_exclude_joints or [])
            if name in ref_order
        }

    def reset(self) -> None:
        """Reset deadband state (call between episodes or on model reload)."""
        self._last_published = None


    def _log(self, level: str, msg: str):
        """Log message using ROS2 logger or print."""
        if self.logger:
            getattr(self.logger, level)(msg)
        else:
            print(f"[{level.upper()}] {msg}")

    def _build_reorder_indices(self) -> list[int]:
        """
        Build index mapping from model joint order to controller joint order.

        Returns:
            List where reorder_indices[model_idx] = controller_idx
            Empty list if no reordering needed
        """
        if not self.model_joint_order or not self.controller_joint_order:
            return []

        if len(self.model_joint_order) != len(self.controller_joint_order):
            self._log(
                "warn",
                f"Joint order lengths differ: model={len(self.model_joint_order)}, controller={len(self.controller_joint_order)}",
            )
            return []

        if self.model_joint_order == self.controller_joint_order:
            return []

        reorder_indices = []
        for model_joint in self.model_joint_order:
            if model_joint in self.controller_joint_order:
                ctrl_idx = self.controller_joint_order.index(model_joint)
                reorder_indices.append(ctrl_idx)
            else:
                self._log("error", f"Joint '{model_joint}' not found in controller_joint_order")
                return []

        return reorder_indices

    def reorder(self, action: np.ndarray) -> np.ndarray:
        """
        Reorder action from model joint order to controller joint order.

        Args:
            action: Action array in model joint order

        Returns:
            Action array in controller joint order
        """
        if not self.reorder_indices:
            return action

        if len(action) != len(self.reorder_indices):
            return action

        reordered = np.zeros_like(action)
        for model_idx, ctrl_idx in enumerate(self.reorder_indices):
            reordered[ctrl_idx] = action[model_idx]
        return reordered

    def apply_delta_limit(self, action: np.ndarray, current_positions: np.ndarray) -> np.ndarray:
        """
        Apply delta limiting to prevent large joint movements.

        Args:
            action: Target action (absolute positions)
            current_positions: Current joint positions

        Returns:
            Delta-limited action
        """
        if current_positions is None or len(current_positions) != len(action):
            return action

        delta = action - current_positions
        clamped_delta = np.clip(delta, -self.max_delta, self.max_delta)
        return current_positions + clamped_delta

    def process(
        self,
        action: np.ndarray,
        current_positions: np.ndarray | None = None,
        joint_order: list[str] | None = None,
        ref_state: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Process action: reorder and apply delta limiting.

        Delta restore is handled upstream (in inference_node) before actions reach here.
        This method receives absolute actions and only applies reordering + safety limiting.

        Args:
            action: Absolute action (in model joint order), already delta-restored upstream
            current_positions: Current joint positions (in controller order)
            joint_order: Unused, kept for backward compatibility
            ref_state: Unused, kept for backward compatibility

        Returns:
            Processed action ready for publishing (in controller order, delta-limited)
        """
        # Make a copy to avoid modifying original
        action = action.copy()

        # Reorder from model order to controller order
        action = self.reorder(action)

        # Apply delta limiting (always against current position for safety)
        if current_positions is not None:
            action = self.apply_delta_limit(action, current_positions)

        # Deadband: suppress commands smaller than min_delta_threshold to overcome
        # motor friction when model outputs are consistently tiny.
        if self.min_delta_threshold is not None:
            if self._last_published is None:
                self._last_published = action.copy()
            else:
                mask = np.abs(action - self._last_published) >= self.min_delta_threshold
                action = np.where(mask, action, self._last_published)
                self._last_published = action.copy()

        return action

    def get_clamped_joints(self, action: np.ndarray, current_positions: np.ndarray) -> list[int]:
        """
        Get indices of joints that would be clamped.

        Args:
            action: Target action (absolute positions)
            current_positions: Current joint positions

        Returns:
            List of joint indices that exceed max_delta
        """
        if current_positions is None or len(current_positions) != len(action):
            return []

        delta = np.abs(action - current_positions)
        return list(np.where(delta > self.max_delta)[0])
