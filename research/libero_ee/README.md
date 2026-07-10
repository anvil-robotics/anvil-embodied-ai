# libero_ee — EE-space action-representation study (LIBERO)

Which action representation should a real EE-space policy use, and how do we train relative-position
so it succeeds on **both ACT and Diffusion**? Answered in sim via the gated `anvil_sim` harness.

**Start here:** [`report.md`](report.md) — conclusions + the production recipe. Timeline and wrong
turns: [`diary.md`](diary.md). Harness itself: [`../../docs/simulation.md`](../../docs/simulation.md).

## Layout

```
report.md   diary.md   README.md          # tracked write-ups (this guide)
ledger/     RESULTS.md, results.json       # every run's headline numbers (regenerable)
experiments/<name>/                        # one dir per treatment (spec name):
    stage_status.json, <stage>.log,
    gt-replay/{trace.jsonl,replay_info.json},
    eval/{eval_info.json, videos/}, smoke_eval/ (+ smoke-*.log)
analysis/   mechanism.{json,png}, collapse.png, closed_loop_collapse.json, traces/
replay/     baseline-task<N>/, native-baseline/, ...   # cached GT-replay traces
logs/       g1_task11.log, g2_traces.log, ...          # multi-run driver logs
```
Everything except the `.md` write-ups is git-ignored and regenerable. Checkpoints:
`model_zoo/research/libero_ee/<name>/`. The study **code** (specs, converters, eval, analysis) lives
in the package: `packages/anvil_sim/src/anvil_sim/studies/libero_ee/`.

## How to read a result

1. **Conclusion** → [`report.md`](report.md) Part 1 (Summary) and §8 (the recipe).
2. **All the numbers** → [`ledger/RESULTS.md`](ledger/RESULTS.md) — one row per run
   (pc_success + GT-replay gate). This is the source of every table in the report.
3. **One run in detail** → `experiments/<name>/` — `stage_status.json` (per-stage pass/fail),
   `<stage>.log` (raw subprocess output), `gt-replay/trace.jsonl` (the gate), `eval/eval_info.json`
   (+ rollout videos).
4. **The mechanism study** → `analysis/` figures + JSON (see the mapping below).

## What backs each claim (report/diary ↔ files)

| Claim (report §) | Backing files |
|---|---|
| §3.0 native-family single-flip matrix (task10 ACT) | `ledger/RESULTS.md` rows `task10-native*` |
| §3.6 ACT+Diffusion matrix + **G1** generalization (task11) | `ledger/RESULTS.md` rows `task10-native*-{act,diffusion}`, `task11-native*` |
| §3.7 **G2** mechanism — chunk-anchor collapses Diffusion | `analysis/mechanism.{json,png}` (refuted magnitude story) + `analysis/collapse.png` & `analysis/closed_loop_collapse.json` (mode-collapse evidence) |
| §7 rotation robustness (task10/14/11) | `ledger/RESULTS.md` rows across `task10/11/14` |
| GT-replay gate caught 5 eval-path bugs (diary) | `experiments/<name>/gt-replay/`, `replay/baseline-task<N>/` |
| a diary `[exp]` entry | `experiments/<matching-name>/` |

## Reproduce

```bash
# one treatment through the gated pipeline (writes experiments/<name>/, updates ledger/)
uv run --package anvil-sim anvil-sim-bench run \
  packages/anvil_sim/src/anvil_sim/studies/libero_ee/configs/task10_native_n0_diffusion.yaml
uv run --package anvil-sim anvil-sim-bench status --study libero_ee   # print the ledger

# regenerate the G2 mechanism analysis (into analysis/)
uv run --package anvil-sim python -m anvil_sim.studies.libero_ee.analysis.mechanism_analysis
```
