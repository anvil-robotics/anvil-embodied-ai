# v1.0 — legacy, pre-unification schema

These configs use the schema as it exists on the `main` branch: singular `robot_state_topic`,
topic-keyed `action_topics` (`{topic: {arm, joint_order}}`), and — for `openarm_single_quest_afo.yaml`
— `action_from_observation`/`action_from_observation_n`. No `data_space`, no `observation_topics`,
no EE support of any kind.

**These are currently not directly usable with `mcap-convert`** — the current loader's strict
mode (the default) rejects every unrecognized top-level key these files use, exactly the
"legacy formats ... no longer accepted" behavior documented in
`mcap_converter/config/loader.py`. This isn't a regression introduced by moving them here;
they were already unusable in their original location (see
`claude_docs/ee-delta-architecture-report.md`, "Known bugs, gaps, and rough edges" #9).

To bring one of these up to date, run it through `dataset-config-migrate` (see
`claude_docs/mcap-converter-encoding-refactor-plan.md` Part 0b) — it applies the exact
same v1.0 → v1.1 restructuring these files need (deriving `observation_topics` from
`robot_state_topic` + `joint_names.arms`, inverting `action_topics` to arm-keyed, dropping
`action_from_observation*`), verified against this exact file content in
`tests/unit/mcap_converter/test_versioning.py::TestRealMainBranchLegacyShapes`.

See `configs/mcap_converter/v1.1/` for the current, directly-usable configs.
