"""Tests for the bench runner's pure orchestration logic (stage gating,
idempotency, failure handling, ledger). Subprocess/env stages are exercised
by the live pipeline, not unit-tested here."""

from __future__ import annotations

import json

import pytest

from anvil_sim import bench_runner
from anvil_sim.bench_runner import _ledger_dir, run_spec, stage_record
from anvil_sim.bench_spec import BenchSpec, EvalSpec, GateSpec, TrainSpec


def _spec(**gate_kwargs) -> BenchSpec:
    return BenchSpec(
        name="unit-spec",
        task_index=10,
        env_suite="libero_goal",
        env_task_id=8,
        dataset_group="goalabs",
        train=TrainSpec(action_type="ee_abs"),
        eval=EvalSpec(action_type="zerocal_goal_abs", control_mode="relative"),
        gates=GateSpec(**gate_kwargs),
        source_path="unit-spec.yaml",
    )


@pytest.fixture()
def fake_stages(monkeypatch, tmp_path):
    """Replace every stage fn with a call recorder; chdir to tmp so all the
    runner's relative paths (research/<study>/...) land in the sandbox."""
    monkeypatch.chdir(tmp_path)
    calls: list[str] = []

    def make(name, fail=False):
        def fn(spec):
            calls.append(name)
            if fail:
                raise RuntimeError(f"{name} exploded")
            return {"ok": name}
        return fn

    for stage in bench_runner.STAGES:
        monkeypatch.setitem(bench_runner._STAGE_FNS, stage, make(stage))
    return calls, make, monkeypatch


def test_all_stages_run_in_order(fake_stages):
    calls, _, _ = fake_stages
    run_spec(_spec())
    assert calls == list(bench_runner.STAGES)


def test_passed_stages_skip_on_rerun_except_record(fake_stages):
    calls, _, _ = fake_stages
    spec = _spec()
    run_spec(spec)
    calls.clear()
    run_spec(spec)
    assert calls == ["record"]  # everything else already passed


def test_failure_stops_pipeline_and_persists_status(fake_stages):
    calls, make, monkeypatch = fake_stages
    monkeypatch.setitem(bench_runner._STAGE_FNS, "gt-replay", make("gt-replay", fail=True))
    spec = _spec()
    with pytest.raises(SystemExit):
        run_spec(spec)
    assert calls == ["convert", "validate-math", "dataset-validate", "gt-replay"]
    status = json.loads((spec.run_dir / "stage_status.json").read_text())
    assert status["gt-replay"]["status"] == "failed"
    assert "exploded" in status["gt-replay"]["error"]
    assert "smoke" not in status  # never reached


def test_failed_stage_reruns_next_time(fake_stages):
    calls, make, monkeypatch = fake_stages
    spec = _spec()
    monkeypatch.setitem(bench_runner._STAGE_FNS, "smoke", make("smoke", fail=True))
    with pytest.raises(SystemExit):
        run_spec(spec)
    monkeypatch.setitem(bench_runner._STAGE_FNS, "smoke", make("smoke"))
    calls.clear()
    run_spec(spec)
    # passed stages skipped, failed stage retried, remainder continues
    assert calls == ["smoke", "train", "eval", "record"]


def test_gates_skip_honored(fake_stages):
    calls, _, _ = fake_stages
    run_spec(_spec(skip=["smoke", "gt-replay"]))
    assert "smoke" not in calls and "gt-replay" not in calls
    assert "train" in calls


def test_from_stage_starts_midway(fake_stages):
    calls, _, _ = fake_stages
    run_spec(_spec(), from_stage="eval")
    assert calls == ["eval", "record"]


def test_dry_run_executes_nothing(fake_stages):
    calls, _, _ = fake_stages
    run_spec(_spec(), dry_run=True)
    assert calls == []


def test_force_stage_clears_eval_cache_so_it_reruns(fake_stages):
    """--force-stage eval must drop the cached eval_info.json; otherwise
    stage_eval reuses the stale number and the "rerun" is a silent no-op."""
    calls, _, _ = fake_stages
    spec = _spec()
    run_spec(spec)  # first pass marks eval passed
    stale = spec.eval_output_dir / "eval_info.json"
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_text(json.dumps({"overall": {"pc_success": 10.0}}))
    calls.clear()

    run_spec(spec, from_stage="eval", force_stages=["eval"])
    assert calls == ["eval", "record"]  # eval actually re-ran
    assert not stale.exists()  # stale cache was cleared before the rerun


def test_clear_stage_cache_only_touches_forced_eval(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    spec = _spec()
    spec.eval_output_dir.mkdir(parents=True)
    (spec.eval_output_dir / "eval_info.json").write_text("{}")

    bench_runner._clear_stage_cache("train", spec)  # non-eval: no-op
    assert spec.eval_output_dir.exists()

    bench_runner._clear_stage_cache("eval", spec)  # eval: removed
    assert not spec.eval_output_dir.exists()


def test_replay_baseline_cache_key_includes_n_episodes(monkeypatch, tmp_path):
    """Bug fixed 2026-07-13: the old cache key was task_index-only, so a
    baseline computed at one n_episodes silently gated every later condition
    run at a different n_episodes. Two specs differing only in n_episodes
    must each compute (and cache under) their OWN baseline."""
    monkeypatch.chdir(tmp_path)
    calls: list[int] = []

    def fake_replay(**kwargs):
        calls.append(kwargs["n_episodes"])
        summary = {"pc_success": 100.0, "n_episodes": kwargs["n_episodes"]}
        # Mirror eval_replay.replay()'s own side effect (creates output_dir and
        # writes replay_info.json there) so the cache-reuse path under test
        # (_replay_baseline checking info_path.exists()) behaves realistically.
        output_dir = kwargs["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "replay_info.json").write_text(json.dumps(summary))
        return summary

    monkeypatch.setattr("anvil_sim.eval_replay.replay", fake_replay)

    spec_n10 = _spec()
    spec_n10.eval.n_episodes = 10
    spec_n49 = _spec()
    spec_n49.eval.n_episodes = 49

    baseline_10 = bench_runner._replay_baseline(spec_n10)
    baseline_49 = bench_runner._replay_baseline(spec_n49)

    assert baseline_10["n_episodes"] == 10
    assert baseline_49["n_episodes"] == 49
    assert calls == [10, 49]  # both actually computed; neither reused the other's cache

    dir10 = bench_runner.topic_root(spec_n10.study_name) / "replay" / f"baseline-task{spec_n10.task_index}-n10"
    dir49 = bench_runner.topic_root(spec_n49.study_name) / "replay" / f"baseline-task{spec_n49.task_index}-n49"
    assert dir10.exists() and dir49.exists()

    # Re-requesting either n_episodes now hits its OWN cache (no recompute).
    calls.clear()
    bench_runner._replay_baseline(spec_n10)
    assert calls == []


def test_clear_stage_cache_gt_replay_removes_matching_baseline_dir(monkeypatch, tmp_path):
    """--force-stage gt-replay must drop the cached baseline for THIS spec's
    n_episodes, or _replay_baseline keeps returning the stale number."""
    monkeypatch.chdir(tmp_path)
    spec = _spec()
    baseline_dir = (
        bench_runner.topic_root(spec.study_name) / "replay"
        / f"baseline-task{spec.task_index}-n{spec.eval.n_episodes}"
    )
    baseline_dir.mkdir(parents=True)
    (baseline_dir / "replay_info.json").write_text("{}")

    bench_runner._clear_stage_cache("train", spec)  # unaffected stage: no-op
    assert baseline_dir.exists()

    bench_runner._clear_stage_cache("gt-replay", spec)
    assert not baseline_dir.exists()


def test_record_surfaces_condition_status_in_ledger(monkeypatch, tmp_path):
    """The ledger's `status` column comes from the study's own classification
    (e.g. "⚠ provisional") — bench_runner just surfaces whatever the study
    returns, never hardcodes it. `_spec()`'s eval.action_type
    ("zerocal_goal_abs") is one of libero_ee's flagged conditions."""
    monkeypatch.chdir(tmp_path)
    spec = _spec()
    spec.run_dir.mkdir(parents=True)
    (spec.run_dir / "stage_status.json").write_text(json.dumps({
        "eval": {"status": "passed", "info": {"pc_success": 100.0}},
    }))

    stage_record(spec)
    ledger = _ledger_dir(spec.study_name)
    results = json.loads((ledger / "results.json").read_text())
    assert results[0]["status"] == "⚠ provisional"
    assert "⚠ provisional" in (ledger / "RESULTS.md").read_text()


def test_record_upserts_ledger_and_regenerates_md(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    spec = _spec()
    spec.run_dir.mkdir(parents=True)
    (spec.run_dir / "stage_status.json").write_text(json.dumps({
        "eval": {"status": "passed", "info": {"pc_success": 100.0}},
        "gt-replay": {"status": "passed", "info": {"pc_success": 80.0, "baseline": 60.0}},
    }))

    stage_record(spec)
    ledger = _ledger_dir(spec.study_name)
    results = json.loads((ledger / "results.json").read_text())
    assert len(results) == 1
    assert results[0]["pc_success"] == 100.0
    assert results[0]["gt_replay"]["baseline"] == 60.0
    assert "unit-spec" in (ledger / "RESULTS.md").read_text()

    # Upsert: same name replaces, not duplicates.
    stage_record(spec)
    assert len(json.loads((ledger / "results.json").read_text())) == 1
