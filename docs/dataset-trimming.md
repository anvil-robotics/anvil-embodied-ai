# Motion-aware dataset trimming

`dataset-trim` removes passive setup and final idle time from a LeRobot dataset
while keeping action, state, task labels, timestamps, frame indices, and every
camera stream aligned. It always creates a new dataset and refuses to overwrite
an existing directory.

## Start alignment modes

- `motion` detects sustained arm departure from the initial median pose and
  retains 10 frames of pre-motion context by default. Use this when inference
  starts from the same home pose.
- `displacement` starts once any arm joint is 0.10 rad from the initial pose.
  This removes the home state from training, so inference must begin near the
  start-state distribution recorded in the plan.
- `gripper` starts 15 frames before the first sustained 0.01 gripper change.
  This aligns demonstrations by first interaction, but removes the initial reach
  and generally produces a wider joint-space start distribution.

The end detector compares arm and gripper actions with the median final pose,
rejects transients shorter than five frames, and keeps 10 post-settle frames.
All thresholds and offsets are configurable.

For cross-embodiment alignment, first make a `gripper` plan for the reference
dataset. Then use `--reference-plan` on the target dataset. The target keeps the
reference dataset's median amount of motion before first gripper interaction.
This aligns task phase without assuming that different arm lengths should share
joint angles or base-frame TCP coordinates.


## Analyze before writing video

```bash
uv run dataset-trim datasets/shirt-fold/lerobot \
  --start-mode displacement \
  --dry-run \
  --manifest eval_results/shirt-fold/trim-plans/displacement.json
```

Reference-timed first-interaction alignment:

```bash
uv run dataset-trim datasets/shirt-fold/lerobot \
  --start-mode gripper \
  --reference-plan eval_results/shirt-fold/trim-plans/hf-gripper.json \
  --dry-run \
  --manifest eval_results/shirt-fold/trim-plans/hf-phase-aligned.json
```


The JSON plan contains source hashes, every half-open frame window, detected
events, removed-frame counts, start state/action vectors, start-pose dispersion,
and a representative (medoid) start episode. Review it before materializing.

To override individual windows, create a JSON file such as:

```json
{
  "0": {"start": 120, "end": 1160},
  "7": {"start": 104}
}
```

Then pass `--overrides overrides.json` when producing a new plan.

## Materialize an approved plan

```bash
uv run dataset-trim datasets/shirt-fold/lerobot \
  --output datasets/shirt-fold/lerobot-displacement-trimmed \
  --apply-plan eval_results/shirt-fold/trim-plans/displacement.json
```

The source fingerprint is checked before writing. Numeric features are copied
exactly, images are decoded in timestamp batches and re-encoded using the source
codec, generated indices/timestamps are reset per LeRobot conventions, and
statistics and episode metadata are recomputed. The applied plan is also stored
as `meta/trim_manifest.json` in the output dataset.

## Download from Hugging Face

Use a dataset repo ID as `SOURCE` and choose an explicit local root:

```bash
uv run dataset-trim USER/DATASET \
  --download-root datasets/shirt-fold/hf-upload \
  --start-mode displacement \
  --dry-run \
  --manifest eval_results/shirt-fold/trim-plans/hf-upload-displacement.json
```

For a private dataset, authenticate with `hf auth login --force` first. Do not
paste a Hugging Face token into a command, plan, or chat transcript.

## Inference contract

Trimming past home changes the policy's initial-state contract. A policy trained
on a `displacement` or `gripper` dataset should not be started from home unless
separate demonstrations teach the transition. Stage the arms near a demonstrated
`start_state` (the plan identifies a representative episode) and arrange the
shirt consistently with the corresponding camera view.
