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
        model_joint_order: list[str] | None = None,
        controller_joint_order: list[str] | None = None,
        use_delta_actions: bool = False,
        delta_exclude_joints: list[str] | None = None,
        logger=None,
    ):
        """
        Initialize action limiter.

        Args:
            max_delta: Maximum position change per step (radians)
            model_joint_order: Order the ML model outputs actions
            controller_joint_order: Order the ROS2 controller expects
            use_delta_actions: If True, model outputs deltas to add to current position
            delta_exclude_joints: Joint names kept in absolute space during training
                (these joints must NOT have current_position added at inference)
            logger: Optional ROS2 logger
        """
        self.max_delta = max_delta
        self.model_joint_order = model_joint_order or []
        self.controller_joint_order = controller_joint_order or []
        self.use_delta_actions = use_delta_actions
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

        Args:
            action: Raw action from model (in model joint order)
            current_positions: Current joint positions (in controller order)
            joint_order: Joint order for delta action conversion
            ref_state: Reference joint positions from when the action chunk was
                generated (in controller order).  When provided, delta restoration
                uses ref_state as the baseline instead of current_positions, so
                that all queued steps in a chunk share the same reference.  Falls
                back to current_positions when None (backward-compatible).

        Returns:
            Processed action ready for publishing (in controller order, delta-limited)
        """
        # Make a copy to avoid modifying original
        action = action.copy()

        # Reorder from model order to controller order
        action = self.reorder(action)

        # Convert delta actions to absolute if needed.
        # Joints in _delta_exclude_indices were trained as absolute — keep their
        # model output as-is and only add the baseline for the remaining joints.
        # Use ref_state (state at chunk generation) when available so that all
        # queued steps share the same reference; fall back to current_positions.
        if self.use_delta_actions:
            _base = ref_state if ref_state is not None else current_positions
            if _base is not None:
                restored = _base + action
                for idx in self._delta_exclude_indices:
                    restored[idx] = action[idx]
                action = restored

        # Apply delta limiting (always against current position for safety)
        if current_positions is not None:
            action = self.apply_delta_limit(action, current_positions)

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
