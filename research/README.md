# research/

One folder per **research topic** — the results and write-ups produced by the `anvil_sim`
simulation validation harness (a developer tool; see [`docs/simulation.md`](../docs/simulation.md)).

Each `research/<topic>/` is self-describing and laid out the same way:

| path | tracked? | what |
|---|---|---|
| `report.md`, `diary.md`, `README.md` | ✅ git | the write-ups + a reading guide |
| `ledger/` | ✗ ignored | machine-generated results table (`RESULTS.md` / `results.json`) |
| `experiments/<name>/` | ✗ ignored | raw per-experiment output (stage logs, gt-replay, eval, smoke) |
| `analysis/` | ✗ ignored | generated figures / JSON |
| `replay/` | ✗ ignored | cached GT-replay baseline + debug traces |
| `logs/` | ✗ ignored | multi-run driver logs |

Only the markdown write-ups are version-controlled; everything heavy is regenerable by re-running
the harness (`.gitignore` ignores `research/**/{ledger,experiments,analysis,replay,logs}/`). The topic
name equals the harness `study` name, so a new study `foo` automatically lands in `research/foo/`.
Model checkpoints (large binaries) live in a parallel `model_zoo/research/<topic>/<name>/`.

Current topics:
- [`libero_ee/`](libero_ee/README.md) — EE-space action-representation study on LIBERO.
