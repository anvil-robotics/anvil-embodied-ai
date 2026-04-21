# anvil-shared

Pure-Python utilities shared across `anvil_trainer`, `anvil_eval`, and
`anvil_eval_ros`.

No torch / lerobot / ROS2 dependencies — safe to import from any context
including offline tooling and ROS2 nodes.

## Modules

- `anvil_shared.splits` — dataset episode-split helpers:
    - `compute_split_episodes(total_episodes, ratio, seed)` — deterministic
      random 3-way split.
    - `load_split_info(path)` — read `split_info.json` from a checkpoint dir.
    - `save_split_info(path, split_info)` — write `split_info.json`.

Both `anvil_trainer` and `anvil_eval` consume these helpers so split logic
stays in one place.
