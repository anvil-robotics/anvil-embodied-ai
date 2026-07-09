# Simulation Validation Harness (`anvil_sim`)

A **gated LIBERO simulation validation flow** for cheaply testing a new policy idea, data
treatment, action representation, or model — *before* spending a full training run on it.

> **Why it exists.** Closed-loop success depends on the eval path being correct, and eval-path
> bugs are invisible to training loss and to synthetic-value unit tests. This project burned four
> full training sweeps on broken eval paths before this harness existed. The harness makes that
> structurally impossible: every cheap check runs **before** training, and a **GT-replay gate**
> replays the dataset's own ground-truth actions through your exact eval path (no policy) — if the
> ground truth can't succeed through your eval path, no checkpoint ever will, so the run aborts in
> minutes with a per-step trace instead of failing silently after 35 minutes of training.

This is a **developer tool**, separate from the main data → train → infer pipeline.

---

## Architecture: harness + study plugin

- **Harness** (`packages/anvil_sim/src/anvil_sim/`, study-agnostic, reusable): the gated pipeline
  (`bench_runner.py`), the spec schema (`bench_spec.py`), the GT-replay engine (`eval_replay.py`),
  and the `Study` plugin interface (`study.py`).
- **Study** (`studies/libero_ee/`, this project's action-representation study registered as a
  plugin): the dataset builders (`libero_convert.py`), eval decoders (`libero_processor.py`,
  `eval_libero_ee.py`), math validators (`math_validators.py`), replay adapter
  (`replay_adapter.py`), the treatment specs (`configs/`), and the write-up (`report.md`).

A spec selects its study with a top-level `study:` field (default `libero_ee`). To validate a
brand-new domain you register a new `Study`; to add a treatment to the existing LIBERO study you
edit `studies/libero_ee/` (see [Testing your idea](#testing-your-idea)).

The EE action-representation study that this harness was built for — its full results, conclusions,
and a diary of every experiment (including the wrong turns) — is the worked example:
[`studies/libero_ee/report.md`](../packages/anvil_sim/src/anvil_sim/studies/libero_ee/report.md).

---

## Setup

```bash
uv sync --package anvil-sim --extra dev

# LIBERO's first import shows an interactive prompt that EOF-crashes in non-interactive
# contexts. Pre-seed its config once to skip it:
mkdir -p ~/.libero && python3 -c "
import os, yaml, importlib.util
pkg = os.path.dirname(importlib.util.find_spec('libero.libero').origin)
yaml.dump({
    'benchmark_root': pkg,
    'bddl_files': os.path.join(pkg, './bddl_files'),
    'init_states': os.path.join(pkg, './init_files'),
    'datasets': os.path.join(pkg, '../datasets'),
    'assets': os.path.join(pkg, './assets'),
}, open(os.path.expanduser('~/.libero/config.yaml'), 'w'))
"
```

Set `HF_HOME=/some/writable/cache` if you don't want the default `~/.cache/huggingface`.

**Build the datasets** (one-time, ~5–6 min; add `--max-episodes=2` for a fast smoke run):

```bash
uv run --package anvil-sim anvil-libero-convert
```

This writes the study's dataset groups under `data/datasets/ee-space/libero-task{N}-<group>/`.

---

## Quick start — run an existing treatment

```bash
uv run --package anvil-sim anvil-sim-bench run \
  packages/anvil_sim/src/anvil_sim/studies/libero_ee/configs/task10_native_act.yaml
```

The pipeline runs eight idempotent stages and appends results to the ledger. Check anytime:

```bash
uv run --package anvil-sim anvil-sim-bench status      # prints outputs/bench/RESULTS.md
```

Add `--dry-run` to validate a spec without running it.

---

## The pipeline and its gates

```
convert → validate-math → dataset-validate → gt-replay → smoke → train → eval → record
```

| stage | what it does |
|---|---|
| `convert` | ensure the spec's dataset group (and the baseline group) exist; build if missing |
| `validate-math` | real-data identity check for the treatment's target math (round-trip to float precision) |
| `dataset-validate` | dataset schema/stats validation |
| **`gt-replay`** | **HARD GATE** — replay the dataset's ground-truth actions through the exact eval path (no policy) |
| `smoke` | 20-step train + 1-episode eval — no crash |
| `train` | full training (skipped when reusing a checkpoint) |
| `eval` | closed-loop eval → `eval_info.json` |
| `record` | upsert `outputs/bench/results.json` + regenerate `outputs/bench/RESULTS.md` |

**The GT-replay gate.** The treatment's ground truth is replayed at `n_action_steps=1` and must
land within `gates.gt_replay_margin` (default **15** pc points) of the study's native replay
baseline (computed once per task, cached at `outputs/bench/replay/baseline-task{N}`). If it falls
below that floor the run aborts:

```
GT-replay gate FAILED: treatment X% < native baseline Y% - margin 15 —
the eval path cannot execute even the ground truth; fix it before training (trace: ...)
```

The baseline spec skips self-gating. **Every cheap stage runs before `train`, so a broken eval
path costs minutes, not a training run.**

**Idempotency & control.** Each run writes `outputs/bench/runs/<name>/stage_status.json`; passed
stages are skipped on re-run (`record` always re-runs). Override with:

- `--from-stage <stage>` — start mid-pipeline.
- `--force-stage <stage>` (repeatable) — re-run a passed stage (also clears its stale
  `eval_info.json`, so `--force-stage eval` truly re-evaluates).
- `gates: {skip: [<stage>, ...]}` in the spec — mark a stage skipped-by-spec.

---

## Writing a spec

A treatment = one YAML file. Minimal (native baseline):

```yaml
# studies/libero_ee/configs/task10_native_act.yaml
name: task10-native-act
task_index: 10
env_suite: libero_goal
env_task_id: 8
dataset_group: native
train:
  trainer: lerobot-train        # or anvil-trainer
  policy_type: act              # or diffusion
  steps: 50000
eval:
  action_type: native
  control_mode: relative        # illegal deliver↔control_mode pairings are rejected at load time
  n_episodes: 50
```

A study-treatment example (formal-goal target trained via anvil-trainer):

```yaml
# studies/libero_ee/configs/task10_goal_abs_act.yaml
name: task10-goal-abs-act
task_index: 10
env_suite: libero_goal
env_task_id: 8
dataset_group: goalabs
train: {trainer: anvil-trainer, action_type: ee_abs, policy_type: act, steps: 50000}
eval:  {action_type: zerocal_goal_abs, control_mode: relative, n_episodes: 50}
```

**Fields** (`bench_spec.py`): top-level `name`, `task_index`, `env_suite`, `env_task_id`,
`dataset_group`, `train`, `eval`, `gates`, `study` (default `libero_ee`).
`TrainSpec`: `trainer` (`anvil-trainer`|`lerobot-train`), `action_type` (required for anvil-trainer,
forbidden for lerobot-train), `policy_type` (`act`|`diffusion`), `steps` (50000), `batch_size` (16),
`reuse_checkpoint`, `output_dir`. `EvalSpec`: `action_type`, `control_mode`
(`relative`|`absolute`), `n_episodes` (10). `GateSpec`: `gt_replay_margin` (15.0), `skip` ([]).

`load_spec` **rejects unknown keys** (top-level and per-section — typos fail loud) and runs
`validate()` (enum checks, trainer↔action_type rules, and the study's legality rules such as
dataset-group membership and deliver↔control_mode pairing).

---

## Testing your idea

### Add a treatment to the existing LIBERO study

Edit under `packages/anvil_sim/src/anvil_sim/studies/libero_ee/`:

1. **Dataset** — add your group to `libero_convert.py`: a name in `ALL_DATASET_GROUPS` + a
   `convert_episode_<yours>_actions` builder (follow `convert_episode_goal_abs_actions`).
2. **Eval decode** — add an eval `action_type` entry to the registry in `eval_libero_ee.py`
   (`_ZERO_CAL_ACTION_TYPES` / `_ZERO_CAL_GOAL_ACTION_TYPES`) as a 4-tuple
   `(obs_action_type, mode, deliver, gripper_mode)`, and the decode in `libero_processor.py`
   (`ZeroCalActionProcessorStep` mode or a `native_action_from_*` helper).
3. **Math validator** — add a `_validate_<yours>` and register it in `math_validators.py`
   (`MATH_VALIDATORS`, keyed by dataset group). This is what the `validate-math` gate runs.
4. **(If novel)** update `study.py` legality rules and `replay_adapter.py`
   `provider_mode`/`action_encoding` for the new action_type.
5. **Spec** — write a YAML in `configs/`.
6. **Run** — `anvil-sim-bench run configs/<yours>.yaml`.

`dataset_groups` and `eval_action_types` on the study are derived from those modules, so steps
1–2 usually make the new treatment loadable automatically.

> **Tip:** run the standalone GT-replay (below) on your new eval `action_type` *before* training —
> if it can't reproduce the ground truth, fix the decode math first.

### Register a brand-new study (new env / new domain)

Implement a `Study` (`anvil_sim/study.py`, a frozen dataclass) and register it. The 13 hooks:

| hook | returns / meaning |
|---|---|
| `name` | study name (the `study:` YAML value) |
| `dataset_groups` | `frozenset[str]` of legal dataset groups |
| `baseline_group` | the GT-replay baseline dataset group |
| `eval_action_types` | `tuple[str, ...]` of valid `eval.action_type` values |
| `dataset_root(group, task_index)` | `Path` to a dataset |
| `math_validators` | `{group: fn(spec) -> dict}` (raises on failure) — the `validate-math` gate |
| `legality(spec)` | `list[str]` of errors ([] == legal) — study-specific spec rules |
| `convert_command(task, groups)` | argv list to build datasets |
| `train_command(spec, *, steps, batch_size, output_dir)` | argv list |
| `eval_command(spec, checkpoint, output_dir, n_episodes)` | argv list |
| `dataset_validate_skip(spec)` | `bool` |
| `gt_replay` | `GtReplayConfig` (baseline action_type/control_mode/n_action_steps + `is_baseline(spec)`) |
| `replay_adapter` | `ReplayAdapter` (make_processors / provider_mode / action_encoding / codec — so the harness's replay engine stays study-agnostic) |

Register via `register_study("<name>", build_fn)` (see `studies/libero_ee/study.py::build_libero_ee_study`
for the worked example). The harness stays generic; only your study knows about your env,
representations, and math.

---

## Debugging with GT-replay

Run the dataset's ground truth through any eval path, no policy involved:

```bash
uv run --package anvil-sim anvil-libero-replay \
  --action-type native_n0 \
  --dataset-root data/datasets/ee-space/libero-task10-native-n0 \
  --control-mode relative --task libero_goal --task-id 8 \
  --n-episodes 10 --n-action-steps 100 \
  --output-dir outputs/bench/replay/debug
```

- A **healthy** eval path scores near the native baseline; a broken one scores ~0%.
- **`--n-action-steps=100`** is important: the gate runs at `1` (every step is a chunk start), but
  chunk-anchor bugs only bite at the policy's real horizon (>1). Running the replay at 100 is how a
  cross-episode chunk-anchor leak was caught here (see the study's Diary, bug #5).
- The per-step trace (`<output-dir>/trace.jsonl`) shows stored vs provided vs native command.

For a per-step trace of a trained policy: `anvil-eval-libero --action-type=<t> --trace-dir=<dir> ...`.

---

## Reading results

The **ledger** is machine-generated by the `record` stage — never hand-edited:

- `outputs/bench/RESULTS.md` — a table (name, task, dataset, policy, eval type, mode, n,
  replay GT/base, pc_success, recorded).
- `outputs/bench/results.json` — the same rows as JSON (upserted by name).

Per-run artifacts live under `outputs/bench/runs/<name>/` (stage status + logs) and
`outputs/eval/bench-<name>/eval_info.json` (+ videos).

---

## CLI reference

| command | purpose |
|---|---|
| `anvil-sim-bench run <spec.yaml> [--from-stage S] [--force-stage S ...] [--dry-run]` | run the gated pipeline |
| `anvil-sim-bench status` | print the ledger |
| `anvil-libero-replay --action-type … --dataset-root … --control-mode … --task … --task-id … --n-episodes … --n-action-steps … --output-dir …` | GT-replay diagnostic (no policy) |
| `anvil-libero-convert [--task-index N] [--only g1,g2] [--max-episodes N]` | build the study's datasets |
| `anvil-eval-libero --action-type=<t> [--trace-dir=<d>] --policy.path … --env.type=libero --env.task … --env.task_ids='[N]' --env.control_mode … --eval.n_episodes … --output_dir …` | ad-hoc closed-loop eval of a checkpoint |

(`anvil-eval-native-rot6d` is the equivalent eval driver for the study's `native_rot6d` arm.)

---

## See also

- [`studies/libero_ee/report.md`](../packages/anvil_sim/src/anvil_sim/studies/libero_ee/report.md)
  — the EE action-representation study: Summary (solid conclusions), Diary (timeline incl. the five
  eval-path bugs and the wrong turns), and the technical analysis. Read it as a concrete example of
  what running this harness end-to-end looks like.
