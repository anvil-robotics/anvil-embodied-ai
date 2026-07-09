# anvil-sim

A reusable, gated **LIBERO simulation validation harness** — test a new policy idea, data
treatment, action representation, or model cheaply, *before* spending a full training run on it.
Every cheap check (including a GT-replay gate that catches eval-path bugs) runs before training.

- **Usage guide:** [`docs/simulation.md`](../../docs/simulation.md)
- **Worked example / results** (the EE action-representation study this was built for):
  [`src/anvil_sim/studies/libero_ee/report.md`](src/anvil_sim/studies/libero_ee/report.md)

The package is a study-agnostic harness (`bench_runner`, `bench_spec`, `eval_replay`, `study`) plus
the LIBERO EE study registered as a plugin under `src/anvil_sim/studies/libero_ee/`.
