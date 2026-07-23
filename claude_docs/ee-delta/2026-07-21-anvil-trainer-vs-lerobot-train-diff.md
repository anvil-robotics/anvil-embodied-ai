# anvil_trainer vs plain lerobot-train — full diff (LIBERO vs production `ee_delta`)

Date: 2026-07-21
Branch: `patrick/implement-ee-space`

## Context

The production `ee_delta` checkpoint (176,913 real frames, bimanual, rot6d observation
encoding, Diffusion, early-stopped at 70,000/100,000 steps) shows near-zero motion at
real-hardware inference — see `2026-07-17-libero-vs-production-diagnosis.md` for the
earlier, separate `ee_rel` jitter diagnosis. LIBERO's validated `native`/`native_n0`
conditions (branch `patrick/sim-valid-dev`) hit 98–100% success on Diffusion with a
~39x smaller dataset (task10: 49 episodes, ~4,557 frames), which raised the question:
is `anvil_trainer` (the custom wrapper around `lerobot-train` used for the production
run) the critical difference from LIBERO's validated recipe (plain, unmodified
`lerobot-train`)? This report enumerates every concrete difference, verified against
source in both branches plus the installed `lerobot` package's own stock defaults.

## Headline finding

**LIBERO's 98–100% result never validated `anvil_trainer`'s `ee_delta` pipeline at all.**
`native`/`native_n0` run through *unmodified* `lerobot-train` — zero anvil_trainer
involvement (confirmed: `TrainSpec.__post_init__` in `anvil_sim/bench_spec.py`
auto-selects `trainer="lerobot-train"` whenever no `action_type` is set, and native-family
YAMLs never set it; `lerobot-train` + an `action_type` is explicitly rejected as illegal).
The one LIBERO experiment that would have been the true analog — a physical-unit
per-frame delta trained through `anvil_trainer` with a matching relative/additive restore
(`goal-{world,hand}-seq` + relative delivery) — was **gate-rejected before training even
started** (0% at the GT-replay math-check stage, a delivery-scaling bug: relative restore
double-applied to an already-scaled target) and was never fixed or retried (no follow-up
in `research/libero_ee/diary.md`). So: the *abstract mechanism* (world-frame, per-frame
anchor, Diffusion architecture) is well-validated at small scale; **this specific code
path — `anvil_trainer`'s `EEDeltaTransform` + `_compute_ee_delta_stats` + its injected
hyperparameter defaults — has never been validated at any scale, successfully or not.**

## Side-by-side: every concrete difference found

| Dimension | LIBERO `native`/`native_n0` (plain `lerobot-train`) | Production `ee_delta` (`anvil_trainer`) |
|---|---|---|
| **Dataset size** | 49 episodes, ~4,557 frames (task10) | 301 episodes, 176,913 frames |
| **Delta computed where** | `native`: not computed at all (LIBERO's own raw recorded action, untouched). `native_n0`: baked at convert time via `anvil_shared.ee_transform.ee_rel_world_forward` (per-frame anchor = live obs) | Baked at convert time via `mcap_converter`'s `_finalize_pending_action` → `ee_delta_forward` (world-frame, per-frame anchor = live obs) — same math family |
| **Rotation encoding** | 7-dim axis-angle action | 10-dim rot6d action (always, regardless of `observation_encoding`) |
| **Observation** | Raw 8-dim native state, untouched | Quat 8n on disk → converted live to rot6d 10n by `EEDeltaTransform.apply()` every `__getitem__` call |
| **Training entry point** | Stock `lerobot-train` CLI directly, **zero monkeypatches** | `anvil-trainer` → `train.py` wraps `lerobot_train()` inside `patched_lerobot(config)`, which installs **9 monkeypatches** (below) |
| **Dataset stats (action + obs.state)** | Stock lerobot's own default per-column mean/std/min/max | Fully replaced by `_compute_ee_delta_stats` (reads raw baked column directly, epsilon-floors std at 1e-6, force-clamps rot6d dims' min/max to ±1 so MIN_MAX passes them through unchanged) |
| **Train/val/test split** | None — trains on 100% of data | Random 8/1/1 episode split injected via a `make_dataset` monkeypatch |
| **Noise scheduler** | Stock default: **DDPM, 100 train timesteps** | Anvil-injected default: **DDIM, 50 train timesteps** (confirmed in this checkpoint's own `train_config.json`) |
| **EMA** | Off (not a stock lerobot feature at all for LIBERO's fork) | **On by default** (UMI-style `EMAModel`, power=0.75, max_value=0.9999) — published checkpoint weights are EMA-swapped at save time |
| **DDPM-IP (input perturbation)** | Off | **On by default**, alpha=0.1 |
| **GroupNorm** | Stock default: `use_group_norm=True` | Anvil-injected default: `use_group_norm=False` (confirmed in checkpoint) |
| **Image crop** | None (LIBERO images 256×256, no crop injected; stock `crop_shape=None`) | `crop_shape=[243, 432]` (confirmed in checkpoint) — real-world images are cropped, LIBERO's are not |
| **horizon / n_action_steps / n_obs_steps** | Stock defaults: 16 / 8 / 2 | **Identical: 16 / 8 / 2** — NOT a differentiator, contrary to earlier assumption |
| **normalization_mapping** | Stock: `{VISUAL: MEAN_STD, STATE: MIN_MAX, ACTION: MIN_MAX}` | **Identical** — confirmed in checkpoint. The mapping *mode* is the same; what differs is the stats *feeding* that mapping (row above) |
| **batch_size** | 16 | 48 |
| **Training steps** | 30,000 (full run) | 100,000 planned, 70,000 actual (early-stopped) |
| **Vision backbone** | Stock default resnet18 | Anvil-injected resnet18 (same, not a differentiator) |
| **Delivery/restore at eval** | `native`: none needed (model output fed straight to sim). `native_n0`: `ee_rel_world_inverse`-equivalent, current-state-anchored, confirmed exact to 1e-4 by a dedicated validator | `ee_delta_inverse`, current-state-anchored — same structural design, never independently gate-tested this way for this exact pipeline |

## The single most concrete, quantified difference: epochs of exposure

- LIBERO: 4,557 frames × 16 batch × 30,000 steps = 480,000 samples seen ≈ **~105 epochs**.
- Production (at the point it was stopped): 176,913 frames × 48 batch × 70,000 steps =
  3,360,000 samples ≈ **~19 epochs**.
- Even at the *full planned* 100,000 steps, production would only reach ≈ **27 epochs** —
  still ~4x fewer passes over the data than LIBERO's validated run, before even accounting
  for production's data being harder (real noise, more behavioral diversity, real vision).

This alone is a strong, quantified reason to expect production's model is meaningfully less
converged than LIBERO's, independent of any pipeline difference.

## Monkeypatches `anvil_trainer` installs (stock `lerobot-train` has none of these)

All installed via `TransformRunner`/`patched_lerobot` in `anvil_trainer/patches.py`:

1. `dataset_to_policy_features` (`lerobot.datasets.feature_utils` + `lerobot.policies.factory`)
   — reports obs.state shape as 10·n_arms so the policy is built with the post-transform dim
   (`transforms.py:212-214`, `_patch_obs_state_shape_8n_to_10n`).
2. `LeRobotDataset.__getitem__` (`patches.py:699`, `apply_dataset_patches`) — runs the
   `Transform` pipeline (e.g. `EEDeltaTransform`) per-sample, plus an absolute→relative
   index remap for the split.
3. `make_dataset` (`lerobot.datasets.factory` + `lerobot.scripts.lerobot_train`,
   `patches.py:837-838`) — builds the random 8/1/1 episode split and injects
   `_compute_ee_delta_stats`'s output into `train_dataset.meta.stats`.
4. `make_pre_post_processors` (`patches.py:849-850`) — captures a processor reference for
   later val/test loss computation.
5. `save_checkpoint` (`lerobot.utils.train_utils` + `lerobot.scripts.lerobot_train`,
   `patches.py:946-947`) — computes val/test loss (raw + EMA), EMA-swaps published
   `pretrained_model/` weights, writes `anvil_config.json`/`split_info.json`.
6. `update_policy` val-loss hook (`patches.py:994`) — periodic held-out loss logging.
7. `DiffusionModel.compute_loss` (`patches.py:1071`) — DDPM-IP input perturbation
   (`eps_perturbed = eps + alpha*randn_like(eps)`).
8. `update_policy` EMA hook (`patches.py:1173`) — steps the EMA shadow model every update
   (chained on top of the val-loss hook — both wrap the same function sequentially).

`EEDeltaTransform.apply()` (`transforms.py:318-345`) itself never touches `action`
(confirmed — only reshapes `observation.state` quat 8n → rot6d 10n, absolute, live at
`__getitem__` time); the delta *values* are exactly what `mcap_converter` baked ahead of
time at conversion, not recomputed during training.

## Ranked assessment — most to least likely to matter for "barely moving"

1. **Epochs of exposure (quantified above).** ~19 actual vs ~105 validated — the single
   largest, most concrete gap. Directly explains regression-to-mean/near-zero-delta output
   as a symptom of undertraining, independent of any bug.
2. **This exact pipeline has never been validated, even at small scale.** Not evidence of a
   bug, but real evidence of absence: nobody has confirmed `anvil_trainer`'s `EEDeltaTransform`
   + `_compute_ee_delta_stats` + DDIM/50 + EMA + DDPM-IP combination produces a working policy
   on ANY dataset, small or large.
3. **Scheduler/EMA/DDPM-IP/GroupNorm differences** are all UMI-derived choices layered on top
   of stock diffusion, none independently validated against this specific `ee_delta` baking
   scheme — plausible interaction effects, not confirmed causal.
4. **Image crop `[243, 432]`** is real-world-specific (LIBERO has no equivalent); could affect
   visual grounding difficulty but unrelated to the delta-output-scale symptom specifically.
5. **rot6d vs axis-angle, batch size, dataset scale itself** — LIBERO's own ablations show
   rot6d costs ~20pp on ACT but is neutral on Diffusion; batch size/dataset scale differences
   don't have a documented causal story, just correlate with more real-world difficulty.

## Recommended next steps (unchanged from the earlier discussion, not yet executed)

1. Check the wandb/training loss curve for this run — still decreasing at step 70k, or
   plateaued? Fastest, zero compute cost.
2. Evaluate 2–3 earlier checkpoints (010000/030000/050000) against the same eval episode and
   check whether predicted-delta variance trends upward — a real trend line vs one snapshot.
3. Run a small LIBERO-scale smoke test of `ee_delta` through the *actual* `anvil_trainer`
   pipeline (not LIBERO's plain lerobot-train) — the closest thing to the validation this
   pipeline has never received. Would isolate "pipeline bug" from "just needs more real data
   and steps" within hours rather than continuing to guess from the large run alone.

## Sources

- `research/libero_ee/report.md`, `research/libero_ee/diary.md` (branch `patrick/sim-valid-dev`)
- `packages/anvil_sim/src/anvil_sim/studies/libero_ee/{libero_convert,libero_processor,study,math_validators}.py`, `bench_spec.py` (branch `patrick/sim-valid-dev`)
- `packages/anvil_trainer/src/anvil_trainer/{config,transforms,patches}.py` (this branch)
- `packages/mcap_converter/src/mcap_converter/core/extractor.py`, `packages/anvil_shared/src/anvil_shared/ee_transform.py` (this branch)
- Installed `lerobot.policies.diffusion.configuration_diffusion.DiffusionConfig` defaults (queried directly from this worktree's `.venv`)
- `model_zoo/ee-space/pbib-standard-env-merged/ee_delta_pbib_standard_env_merged/checkpoints/last/pretrained_model/train_config.json` (this run's actual recorded hyperparameters)
