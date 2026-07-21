#!/usr/bin/env python3
"""Dataset GT-Replayer Node

Replays a converted dataset's recorded ground-truth ``action`` rows through the
real inference pipeline (``LeRobotInferenceNode``), injected at the exact seam
where a model's predicted action would normally appear (``_produce_action``).
Everything downstream of that seam — the classic action deque, the decoupled
delta-mode publish loop's per-tick ``obs ∘ delta`` composition, absolute
restoration, message building, and publishing — runs completely unmodified.
This validates mcap_converter's output against the real inference pipeline's
own consumption of it end-to-end, not just the transform math in isolation
(see claude_docs/dataset-gt-replayer-plan.md for the full design).

v1 scope: classic (non-VLA) action_types only — joint_abs, ee_abs, ee_delta.
ee_relative is not a dataset-native encoding (mcap_converter's "relative" is
reserved/unimplemented) and VLA predictions use a separate background-thread
RTC seam; both are out of scope.

Usage:
    ros2 launch lerobot_control dataset_gt_replay.launch.py \\
        dataset:=/path/to/converted/dataset config_file:=/path/to/config.yaml
"""

import json
import signal
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor

from . import dataset_reader
from .inference_node import LeRobotInferenceNode


class DatasetGtReplayerNode(LeRobotInferenceNode):
    """Replays a dataset's recorded GT ``action`` rows as if they were model output.

    Overrides exactly the seams ``LeRobotInferenceNode`` exposes for this:
      - ``_validate_required_params``: requires ``dataset`` instead of ``model_path``.
      - ``_load_run_metadata``: derives the same meta-dict shape from the dataset's
        ``conversion_config.yaml`` + ``meta/info.json`` instead of a checkpoint's
        ``config.json``/``anvil_config.json``.
      - ``_setup_model``: no model to load — loads the target episode's raw
        ``action`` rows (native on-disk encoding, physical units) instead.
      - ``_produce_action``: returns the next recorded row instead of running the
        model, with backpressure so rows are never dropped by the deque's ``maxlen``.

    Everything else — obs reading, the shared EE-conversion head in
    ``_obs_update``, the classic action deque, ``_publish_loop`` (including the
    ee_delta decoupled composition), publishers, and shutdown/hold-position — is
    inherited unchanged.

    Two more capabilities layered on top, both for
    ``scripts/gt_replay_human_eval.py`` (see claude_docs/real-hardware-gt-replay-
    eval-plan.md):

    - ``completion_signal_path``: when set, a small JSON sentinel is written the
      moment this episode's replay finishes (``{"status": "complete", ...}``),
      is interrupted by SIGTERM (``"interrupted"``), or homing fails
      (``"homing_failed"``) — see ``_write_signal``. Lets a host-side wrapper
      poll a file instead of screen-scraping logs, mirroring
      ``gt_replay_verifier_node``'s ``report_path`` convention.
    - Pre-replay homing (EE mode only, ``home_before_replay`` default true):
      before GT playback begins, the robot is commanded to this episode's
      frame-0 recorded pose and held there — via small hooks in the inherited
      ``_obs_update``/``_publish_loop`` (see ``_check_homing_arrival``/
      ``_publish_home_target``) — until the live observation confirms arrival
      within tolerance, or ``homing_timeout_sec`` elapses. This is NOT the same
      check as ``GtReplayVerifierNode``'s tight trajectory-match tolerances —
      it's a coarser "did the robot get close enough to start" gate, using
      real-hardware-appropriate defaults.
    """

    def _setup_config(self) -> None:
        # New params must be declared (and read) before super()._setup_config()
        # runs, since it calls _validate_required_params()/_load_run_metadata()
        # (overridden below) which need them.
        self.declare_parameter("dataset", "")
        self.declare_parameter("episode", 0)
        self.declare_parameter("loop", False)
        self.declare_parameter("hold_last", True)
        self.declare_parameter("dry_run", False)
        self.declare_parameter("completion_signal_path", "")
        self.declare_parameter("home_before_replay", True)
        # Looser than anvil_eval's real-hardware EE pass/fail threshold (0.02m/
        # 5.0deg, metrics.py) — a real robot's physical controller was observed
        # live, across several attempts, plateauing anywhere from ~0.022m/5.9deg
        # up to ~0.039m/9.1deg without fully closing the gap within a realistic
        # homing_timeout_sec. Set comfortably above the worst observed value
        # (not just barely past it) so homing isn't perpetually on the edge of
        # this same failure as conditions vary run to run.
        self.declare_parameter("home_atol_pos_m", 0.05)
        self.declare_parameter("home_atol_rot_deg", 10.0)
        self.declare_parameter("homing_timeout_sec", 30.0)
        self.declare_parameter("home_max_pos_delta_m", 0.01)
        self.declare_parameter("home_max_rot_delta_deg", 2.0)

        self.episode_idx: int = self.get_parameter("episode").value
        self.loop_replay: bool = self.get_parameter("loop").value
        self.hold_last: bool = self.get_parameter("hold_last").value
        self.dry_run: bool = self.get_parameter("dry_run").value
        self.completion_signal_path: str = self.get_parameter("completion_signal_path").value
        self.home_before_replay: bool = self.get_parameter("home_before_replay").value
        self.home_atol_pos_m: float = float(self.get_parameter("home_atol_pos_m").value)
        self.home_atol_rot_deg: float = float(self.get_parameter("home_atol_rot_deg").value)
        self.homing_timeout_sec: float = float(self.get_parameter("homing_timeout_sec").value)
        self.home_max_pos_delta_m: float = float(self.get_parameter("home_max_pos_delta_m").value)
        self.home_max_rot_delta_deg: float = float(self.get_parameter("home_max_rot_delta_deg").value)
        self._signal_written: bool = False

        super()._setup_config()

    def _validate_required_params(self) -> None:
        """Require ``dataset`` instead of the base class's ``model_path``."""
        dataset = self.get_parameter("dataset").value
        if not dataset:
            raise ValueError("dataset parameter is required for dataset_gt_replayer_node")
        self.dataset_path = Path(dataset)
        if not self.dataset_path.exists():
            raise ValueError(f"dataset path does not exist: {self.dataset_path}")
        # Cosmetic only: makes _log_startup's "Model:" line show something useful.
        self.model_path = str(self.dataset_path)

    def _load_run_metadata(self) -> dict:
        """Derive the meta-dict shape from the dataset instead of a checkpoint.

        ``image_shape``/``obs_state_dim`` come from ``meta/info.json``'s
        ``features`` dict (the dataset's own feature-shape record). ``action_type``
        is resolved from ``conversion_config.yaml``'s ``data_space``/``action_encoding``
        via a plain lenient YAML read — not through mcap_converter's ``ConfigLoader``,
        which isn't shipped in the inference Docker image; every dataset this tool
        targets was converted with the current (v1.1) schema, so the two field
        names are read directly rather than pulling in the migration-aware loader
        for a case that can't occur here.
        """
        info_path = self.dataset_path / "meta" / "info.json"
        if not info_path.exists():
            raise RuntimeError(f"meta/info.json not found in {self.dataset_path}")
        info = json.loads(info_path.read_text())
        features: dict = info.get("features", {})

        obs_state = features.get("observation.state", {})
        obs_state_dim = (obs_state.get("shape") or [None])[0]

        image_shape = (480, 640, 3)
        for feat in features.values():
            if feat.get("dtype") == "video":
                c, h, w = feat["shape"]
                image_shape = (h, w, c)
                break

        self._total_episodes = info.get("total_episodes", 0)
        if self.episode_idx >= self._total_episodes:
            raise ValueError(
                f"episode {self.episode_idx} out of range "
                f"(dataset has {self._total_episodes} episodes)"
            )

        return {
            "image_shape": image_shape,
            "model_type": "gt_replay",
            "obs_state_dim": obs_state_dim,
            "action_type": self._resolve_action_type(),
            "task_description": info.get("task_description", ""),
        }

    def _resolve_action_type(self) -> str:
        """Resolve action_type from conversion_config.yaml (see dataset_reader.py)."""
        return dataset_reader.resolve_action_type(self.dataset_path, logger=self.get_logger())

    def _setup_model(self) -> None:
        """No model to load — load this episode's recorded action rows instead."""
        actions = self._load_episode_actions(self.episode_idx)
        if actions is None or len(actions) == 0:
            raise RuntimeError(
                f"no frames found for episode {self.episode_idx} in {self.dataset_path}"
            )
        self._gt_actions = actions
        self._replay_cursor = 0
        self._episode_done_logged = False
        self.get_logger().info(
            f"[gt-replay] Loaded episode {self.episode_idx}: {len(actions)} action rows "
            f"(action_type={self.action_type})"
        )
        self._setup_homing()

    def _load_episode_actions(self, episode_idx: int) -> np.ndarray | None:
        """Read the ``action`` column for one episode (see dataset_reader.py)."""
        return dataset_reader.load_episode_actions(self.dataset_path, episode_idx)

    # ------------------------------------------------------------------ #
    # Pre-replay homing (EE mode only) — see class docstring
    # ------------------------------------------------------------------ #

    def _setup_homing(self) -> None:
        """Prepare (or skip) the pre-replay homing phase.

        Disabled outright for joint mode (not designed/tested there) and when
        ``home_before_replay`` is false (e.g. the fake-hardware wrapper's default
        for ``--target fake``, since the mock's ``ee_seed_pose`` already provides
        an instant, exact seed — homing against it would just be redundant).
        When active, ``_homing_confirmed`` starts false; the inherited
        ``_obs_update``/``_publish_loop`` hold on it (see inference_node.py) until
        ``_check_homing_arrival`` flips it.
        """
        if not self.is_ee or not self.home_before_replay:
            self._homing_confirmed = True
            self._homing_status: str | None = "skipped"
            return

        obs_quat = dataset_reader.load_episode_observations_quat(
            self.dataset_path, self.episode_idx
        )
        self._home_target_quat = obs_quat[0]
        self._homing_confirmed = False
        self._homing_status = None
        self._homing_start_time = time.monotonic()
        self.get_logger().info(
            f"[gt-replay] homing to episode {self.episode_idx}'s frame-0 pose before "
            f"replay (atol_pos={self.home_atol_pos_m}m atol_rot={self.home_atol_rot_deg}deg "
            f"timeout={self.homing_timeout_sec}s, ramped at "
            f"{self.home_max_pos_delta_m}m/{self.home_max_rot_delta_deg}deg per tick)"
        )

    def _check_homing_arrival(self) -> None:
        """Called from ``_obs_update`` every tick while homing is unconfirmed.

        Compares the freshest observed pose (``_last_raw_ee_obs_np``, quat
        layout) against ``_home_target_quat``, per arm, and confirms once EVERY
        arm is within tolerance. Gives up (writes the ``homing_failed`` sentinel
        and shuts down — nothing downstream should start GT playback against an
        unconfirmed start pose) once ``homing_timeout_sec`` elapses.
        """
        if not hasattr(self, "_last_raw_ee_obs_np"):
            return  # no observation yet — nothing to compare against

        from .ee_runtime import pose_arrival_error

        n_arms = len(self._home_target_quat) // 8
        max_pos_err = 0.0
        max_rot_err = 0.0
        for i in range(n_arms):
            s0 = i * 8
            pos_err, rot_err = pose_arrival_error(
                self._last_raw_ee_obs_np[s0:s0 + 8], self._home_target_quat[s0:s0 + 8]
            )
            max_pos_err = max(max_pos_err, pos_err)
            max_rot_err = max(max_rot_err, rot_err)

        if max_pos_err <= self.home_atol_pos_m and max_rot_err <= self.home_atol_rot_deg:
            self._homing_confirmed = True
            self._homing_status = "confirmed"
            self.get_logger().info(
                f"[gt-replay] homing confirmed (pos_err={max_pos_err:.4f}m "
                f"rot_err={max_rot_err:.2f}deg)"
            )
            return

        elapsed = time.monotonic() - self._homing_start_time
        # TEMP DEBUG (real-hardware homing-plateau investigation): throttled to
        # once/sec, not every tick, to see whether pos_err/rot_err is actually
        # still closing on the target or has already plateaued well before
        # homing_timeout_sec — distinguishes "just needs more time" from a
        # genuine steady-state tracking-error ceiling on the real controller.
        if getattr(self, "_debug", False):
            last_logged = getattr(self, "_last_homing_debug_log_sec", -1)
            elapsed_sec = int(elapsed)
            if elapsed_sec != last_logged:
                self._last_homing_debug_log_sec = elapsed_sec
                self.get_logger().info(
                    f"[DEBUG-HOMING] t={elapsed:.1f}s pos_err={max_pos_err:.4f}m "
                    f"rot_err={max_rot_err:.2f}deg (tolerance: pos<={self.home_atol_pos_m}m "
                    f"rot<={self.home_atol_rot_deg}deg)"
                )
        if elapsed > self.homing_timeout_sec:
            self._homing_status = "failed"
            self.get_logger().error(
                f"[gt-replay] homing FAILED after {elapsed:.1f}s (pos_err={max_pos_err:.4f}m "
                f"rot_err={max_rot_err:.2f}deg, tolerance: pos<={self.home_atol_pos_m}m "
                f"rot<={self.home_atol_rot_deg}deg) — GT playback will NOT start this episode"
            )
            self._write_signal({"status": "homing_failed"})
            if rclpy.ok():
                rclpy.shutdown()

    def _publish_home_target(self) -> None:
        """Called from ``_publish_loop`` every tick while homing is unconfirmed.

        Publishes a RAMPED step toward the frame-0 target — at most
        ``home_max_pos_delta_m``/``home_max_rot_delta_deg`` of motion this
        tick (see ``ee_runtime.ramp_toward_pose``) — rather than jumping
        straight there in one shot. ``inference_node.py``'s ``action_limiter``
        (the joint-space per-tick delta-limiting safety net) is explicitly not
        applied in EE mode, so without this, homing would command the robot
        directly to the target regardless of how far its actual current pose
        is from it. Reuses ``_publish_ee_action`` — no new message-construction
        code. If no observation has arrived yet, skips this tick rather than
        ramping from an unknown starting point.
        """
        from anvil_shared.ee_transform import ee_obs_abs_forward

        from .ee_runtime import ramp_toward_pose

        current = getattr(self, "_last_raw_ee_obs_np", None)
        if current is None:
            return

        n_arms = len(self._home_target_quat) // 8
        ramped = np.concatenate([
            ramp_toward_pose(
                current[i * 8:(i + 1) * 8],
                self._home_target_quat[i * 8:(i + 1) * 8],
                self.home_max_pos_delta_m,
                self.home_max_rot_delta_deg,
            )
            for i in range(n_arms)
        ])
        home_rot6d = ee_obs_abs_forward(ramped)
        self._publish_ee_action(home_rot6d)

    def _write_signal(self, extra: dict) -> None:
        """Write (once) the completion-signal sentinel, if ``completion_signal_path`` is set.

        Guarded to fire at most once per run: whichever terminal status is
        reached first (``complete``/``homing_failed``/``interrupted``) wins —
        e.g. a SIGTERM arriving after a normal completion must never overwrite
        that report with a less informative ``interrupted``.
        """
        if not self.completion_signal_path or self._signal_written:
            return
        self._signal_written = True
        payload = {
            "episode": self.episode_idx,
            "homing_status": getattr(self, "_homing_status", "skipped"),
            **extra,
        }
        path = Path(self.completion_signal_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))

    def _produce_action(
        self, observation: dict, ee_obs_window_rel: np.ndarray | None
    ) -> np.ndarray | None:
        """Return the next recorded GT action row instead of a model prediction.

        Backpressure: returns ``None`` (produces nothing this tick) while the
        deque is nearly full, so a replayed row — which can't be regenerated
        like a model prediction — is never silently dropped by ``maxlen``.
        """
        if len(self._classic_action_deque) >= self._classic_action_deque.maxlen - 1:
            return None

        if self._replay_cursor >= len(self._gt_actions):
            if self.loop_replay:
                self._replay_cursor = 0
                self._episode_done_logged = False
            else:
                if not self._episode_done_logged:
                    suffix = "" if self.hold_last else " — shutting down"
                    self.get_logger().info(
                        f"[gt-replay] episode {self.episode_idx} complete "
                        f"({len(self._gt_actions)} rows replayed){suffix}"
                    )
                    self._episode_done_logged = True
                    self._write_signal({
                        "status": "complete",
                        "rows_replayed": len(self._gt_actions),
                    })
                    if not self.hold_last and rclpy.ok():
                        rclpy.shutdown()
                return None

        action = self._gt_actions[self._replay_cursor].astype(np.float32)

        if self.dry_run:
            self.get_logger().info(
                f"[gt-replay] [dry-run] row {self._replay_cursor}: "
                f"[{', '.join(f'{v:.4f}' for v in action)}]"
            )
            self._replay_cursor += 1
            return None

        self._replay_cursor += 1
        self.metrics.record_inference()
        return action


def main(args=None):
    """Main entry point, mirroring inference_node.main()'s SIGTERM handling."""
    rclpy.init(args=args)
    node = None
    executor = None

    # See inference_node.main() for why SIGTERM is handled explicitly (Docker
    # sends SIGTERM on stop; the default action skips destroy_node()/cleanup).
    def _sigterm_handler(signum, frame):
        if node is not None:
            node.get_logger().info("[shutdown] SIGTERM received, shutting down...")
            # No-op if a terminal status (complete/homing_failed) was already
            # written — see _write_signal's once-only guard.
            node._write_signal({"status": "interrupted"})
        else:
            print("[shutdown] SIGTERM received during startup, shutting down...")
        if rclpy.ok():
            rclpy.shutdown()

    signal.signal(signal.SIGTERM, _sigterm_handler)

    try:
        node = DatasetGtReplayerNode()
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(node)
        node.get_logger().info("Starting dataset GT replay...")
        executor.spin()
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        if executor:
            executor.shutdown()
        if node:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
