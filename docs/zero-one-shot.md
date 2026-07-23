# Zero- and one-shot scene experiments

Anvil separates the robot embodiment from each policy checkpoint. The compiled
`.anvilscene` produced by `anvil-openarm-mujoco` contains an
`embodiment_manifest.json` with joint order, limits, command topics, TCP sites,
and supported action surfaces. Policy bindings map those semantics into Pi0.5,
GR00T, or SmolVLA inputs without pretending the run is imitation learning.

## Build the scene

Capture the table with the guided iPhone Pro app, then run in the MuJoCo repo:

```bash
uv run python scripts/anvil_scene.py validate capture.anvilscan
uv run python scripts/anvil_scene.py build capture.anvilscan \
  --output generated/lego_in_cup.anvilscene
```

Update `scene_bundle` in
`configs/experiments/lego_in_cup_zero_one_shot.yaml`, then validate the
zero-shot bindings:

```bash
uv run python scripts/validate_zero_one_shot.py \
  configs/experiments/lego_in_cup_zero_one_shot.yaml --mode zero
```

The initial model matrix is Pi0.5, GR00T, and SmolVLA. ACT and VLA-JEPA remain
checkpoint-normalized trained baselines rather than zero-shot candidates.

## Normalization

Bindings declare one normalization source:

- `embodiment_limits` derives bounded state/action scaling from the canonical
  MuJoCo joint limits and is available to explicitly compatible zero-shot
  adapters.
- `checkpoint` requires serialized training statistics.
- `demonstration` is reserved for the one-real-demonstration adaptation path.

Empty or implicit normalization statistics are never treated as valid.

## One-shot gate

Do not populate `one_shot.demonstration` until the zero-shot evaluation fails
its advancement threshold. Record exactly one successful real episode, convert
it to a LeRobot dataset, and set that path. One-shot validation then becomes:

```bash
uv run python scripts/validate_zero_one_shot.py \
  configs/experiments/lego_in_cup_zero_one_shot.yaml --mode one
```

The contract requires at least 20 unique held-out simulation seeds. If the
