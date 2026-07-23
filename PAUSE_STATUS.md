# Pause Status

Development on this repository is paused as of **2026-07-23**. This file indexes what's
still open so a future session can resume without re-deriving context. It intentionally
covers only unfinished/undone branches — branches that were fully wrapped up or retired
during the pause are not listed here.

## Branches with unfinished work

### `patrick/implement-ee-space`
End-effector (EE) Cartesian-space support across the pipeline + a GT-replay
ground-truth-replay verification tool. Most active branch at pause time.
Full status, the open `ee_delta` GT-replay async-failure investigation, and concrete
next steps: see `claude_docs/RESUME-implement-ee-space.md` on that branch.

### `patrick/sim-valid-dev`
LIBERO-benchmark simulation-validity study for the Anvil pipeline. Full status,
findings so far, and next steps: see `claude_docs/RESUME-sim-valid-dev.md` on that branch.

## Branches left untouched (not ours to finalize)

These remote branches belong to other contributors and were left exactly as they were —
no changes, no resume notes added:

- `origin/andre/eval-plots` (Andre Thomas) — anvil-eval horizon + gripper-phase diagnostics
- `origin/daniel/config_extractor_cleanup` (Daniel Pino) — config extractor cleanup
- `origin/daniel/integration` (Daniel Pino) — integration work

## Resuming

1. `git fetch --all` to see current branch state.
2. Pick a branch from the list above, read its `claude_docs/RESUME-*.md`.
3. `uv sync --all-packages` to set up the environment; see each resume note for
   branch-specific test/run commands.
