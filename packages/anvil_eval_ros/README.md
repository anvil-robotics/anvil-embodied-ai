# anvil-eval-ros

ROS2 MCAP replay evaluation — orchestrates `docker-compose` and ROS2 nodes to
replay real teleoperation MCAP recordings through the full inference stack.

This package contains the **host-side CLI only**.  The ROS2 nodes it launches
(`mcap_player_node`, `eval_recorder_node`, `inference_node`) live in
`ros2/src/lerobot_control/` and are built by `colcon`.

## Install

```bash
uv sync --all-packages
```

## Usage

```bash
uv run anvil-eval-ros \
    --checkpoint model_zoo/<dataset>/<job>/checkpoints/<step> \
    --mcap-root data/raw/<dataset>
```

See the root [README — ROS2 MCAP Replay section](../../README.md#ros2-mcap-replay-anvil-eval-ros)
for the full flag reference and Docker prerequisites.

## Why a separate package?

`anvil_eval` (offline evaluation) should stay installable without ROS2 or
Docker.  This package carries the orchestration CLI and its dependencies so
offline-only users don't pay for them.
