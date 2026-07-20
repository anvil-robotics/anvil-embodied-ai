# v1.1 — current unified joint+EE schema

The schema at the tip of `implement-ee-space`: `data_space` (`"joint"` | `"ee"`),
arm-keyed `observation_topics`/`action_topics`, EE support (`action_encoding`,
`observation_encoding`), and `schema_version` itself. These configs are directly usable
with `mcap-convert` today.

**New to this schema?** See `TEMPLATE_all_fields.yaml` first — every field, both
`data_space` variants, with comments on how each behaves in each situation. The configs
below are smaller, focused, directly-usable examples; the template is the comprehensive
reference.

| Config | data_space | Arms | Notes |
|---|---|---|---|
| `openarm_joint_bimanual.yaml` | `joint` | left + right | migrated replacement for `v1.0/openarm_bimanual_quest.yaml` |
| `openarm_ee_bimanual.yaml` | `ee` | left + right | primary EE default, `action_encoding: absolute` |
| `openarm_ee_bimanual_16x9.yaml` | `ee` | left + right | 16:9 camera variant of the above |
| `openarm_ee_left.yaml` | `ee` | left only | single-arm EE |
| `openarm_ee_delta_debug.yaml` | `ee` | right only | `action_encoding: delta` debug template, matches `tests/smoke/fixtures/ee-session/` |
| `openarm_ee_rot6d_obs_debug.yaml` | `ee` | right only | `observation_encoding: rot6d` debug template, same fixtures |
| `openarm_ee_bimanual_delta_debug.yaml` | `ee` | left + right | `action_encoding: delta` debug template, matches real recorded sessions (e.g. `data/raw_sessions/ee-space-testing`) |
| `openarm_ee_bimanual_rot6d_obs_debug.yaml` | `ee` | left + right | `observation_encoding: rot6d` debug template, same real-session layout |

See `configs/mcap_converter/v1.0/` for the legacy pre-unification configs this schema
replaced, and `claude_docs/mcap-converter-encoding-refactor-plan.md` for the full
versioning/migration design.
