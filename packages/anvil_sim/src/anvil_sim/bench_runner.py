"""``anvil-sim-bench`` — the gated validation pipeline for one treatment.

Runs a :mod:`anvil_sim.bench_spec` YAML through eight stages, ordered so
every cheap check runs BEFORE any expensive training:

1.  ``convert``           dataset group exists (create if missing)
2.  ``validate-math``     real-data math identities for this target family
3.  ``dataset-validate``  mcap_converter's schema/stats validation
4.  ``gt-replay``         HARD GATE: dataset ground truth must succeed
                          through the exact eval path, within
                          ``gates.gt_replay_margin`` pc points of the native
                          replay baseline on the same episodes (see
                          ``eval_replay.py`` — the tool that caught bug #4)
5.  ``smoke``             20-step train + 1-episode eval, no crash
6.  ``train``             full training (skipped when reusing a checkpoint)
7.  ``eval``              closed-loop eval, standard ``eval_info.json``
8.  ``record``            upsert ``research/<study>/ledger/results.json`` and
                          regenerate ``research/<study>/ledger/RESULTS.md``

All results for one study/topic live under ``research/<study>/`` (see
``bench_spec.topic_root``): per-experiment raw output in
``research/<study>/experiments/<name>/`` and the write-ups alongside.
Stages are idempotent: each writes its status to
``research/<study>/experiments/<name>/stage_status.json`` and re-running a spec
skips already-passed stages (``--force-stage`` / ``--from-stage`` override).
Subprocess stages log to ``research/<study>/experiments/<name>/<stage>.log``.

Usage::

    anvil-sim-bench run packages/anvil_sim/src/anvil_sim/studies/libero_ee/configs/task10_goal_abs_act.yaml
    anvil-sim-bench run spec.yaml --from-stage eval
    anvil-sim-bench run spec.yaml --dry-run
    anvil-sim-bench status
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from anvil_sim.bench_spec import RESEARCH_ROOT, BenchSpec, load_spec, topic_root

log = logging.getLogger(__name__)

STAGES = (
    "convert",
    "validate-math",
    "dataset-validate",
    "gt-replay",
    "smoke",
    "train",
    "eval",
    "record",
)

def _ledger_dir(study_name: str) -> Path:
    """Per-topic ledger dir: ``research/<study>/ledger/`` (RESULTS.md + results.json)."""
    return topic_root(study_name) / "ledger"


# --------------------------------------------------------------------------- #
# Small infrastructure                                                         #
# --------------------------------------------------------------------------- #


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_status(spec: BenchSpec) -> dict:
    path = spec.run_dir / "stage_status.json"
    if path.exists():
        return json.loads(path.read_text())
    return {}


def _save_status(spec: BenchSpec, status: dict) -> None:
    spec.run_dir.mkdir(parents=True, exist_ok=True)
    (spec.run_dir / "stage_status.json").write_text(json.dumps(status, indent=2))


def _run_logged(cmd: list[str], log_path: Path) -> None:
    """Run *cmd*, teeing output to *log_path*; raise on nonzero exit."""
    log.info("$ %s  (log: %s)", " ".join(cmd), log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False)
    if proc.returncode != 0:
        tail = "".join(log_path.read_text().splitlines(keepends=True)[-15:])
        raise RuntimeError(f"command failed (exit {proc.returncode}): {' '.join(cmd)}\n--- log tail ---\n{tail}")


# --------------------------------------------------------------------------- #
# Stage implementations — each returns an info dict on success, raises on fail #
# --------------------------------------------------------------------------- #


def stage_convert(spec: BenchSpec) -> dict:
    """Ensure the spec's dataset group AND the study's baseline group (needed
    by validate-math and the gt-replay baseline) exist for this task."""
    study = spec.study
    candidates = (
        (spec.dataset_group, spec.dataset_root),
        (study.baseline_group, study.dataset_root(study.baseline_group, spec.task_index)),
    )
    missing = sorted({g for g, root in candidates if not root.exists()})
    if not missing:
        return {"skipped": "datasets already exist"}
    _run_logged(study.convert_command(spec.task_index, missing), spec.run_dir / "convert.log")
    return {"created": missing}


def stage_validate_math(spec: BenchSpec) -> dict:
    validator = spec.study.math_validators.get(spec.dataset_group)
    if validator is None:
        return {"skipped": f"no math validator registered for group {spec.dataset_group!r}"}
    return validator(spec)


def stage_dataset_validate(spec: BenchSpec) -> dict:
    if spec.study.dataset_validate_skip(spec):
        return {"skipped": "dataset-validate targets the Anvil EE writer schema"}
    cmd = [
        "uv", "run", "--package", "mcap_converter", "dataset-validate",
        f"--root={spec.dataset_root}", "--repo-id=local",
    ]
    _run_logged(cmd, spec.run_dir / "dataset-validate.log")
    return {"validated": str(spec.dataset_root)}


def _replay_baseline(spec: BenchSpec) -> dict:
    """Study baseline GT-replay for this task — computed once, cached."""
    baseline_dir = topic_root(spec.study_name) / "replay" / f"baseline-task{spec.task_index}"
    info_path = baseline_dir / "replay_info.json"
    if info_path.exists():
        return json.loads(info_path.read_text())
    from anvil_sim.eval_replay import replay

    study = spec.study
    gr = study.gt_replay
    return replay(
        action_type=gr.baseline_action_type,
        dataset_root=study.dataset_root(study.baseline_group, spec.task_index),
        control_mode=gr.baseline_control_mode,
        task=spec.env_suite,
        task_id=spec.env_task_id,
        n_episodes=spec.eval.n_episodes,
        n_action_steps=gr.n_action_steps,
        output_dir=baseline_dir,
        adapter=study.replay_adapter,
    )


def stage_gt_replay(spec: BenchSpec) -> dict:
    from anvil_sim.eval_replay import replay

    baseline = _replay_baseline(spec)
    if spec.study.gt_replay.is_baseline(spec):
        return {"skipped": "treatment IS the native baseline", "baseline": baseline["pc_success"]}

    result = replay(
        action_type=spec.eval.action_type,
        dataset_root=spec.dataset_root,
        control_mode=spec.eval.control_mode,
        task=spec.env_suite,
        task_id=spec.env_task_id,
        n_episodes=spec.eval.n_episodes,
        n_action_steps=1,
        output_dir=spec.run_dir / "gt-replay",
        adapter=spec.study.replay_adapter,
    )
    floor = baseline["pc_success"] - spec.gates.gt_replay_margin
    if result["pc_success"] < floor:
        raise RuntimeError(
            f"GT-replay gate FAILED: treatment {result['pc_success']:.0f}% < "
            f"native baseline {baseline['pc_success']:.0f}% - margin "
            f"{spec.gates.gt_replay_margin:.0f} — the eval path cannot execute even the "
            f"ground truth; fix it before training (trace: {result['trace']})"
        )
    return {
        "pc_success": result["pc_success"],
        "baseline": baseline["pc_success"],
        "margin": spec.gates.gt_replay_margin,
    }


def stage_smoke(spec: BenchSpec) -> dict:
    if spec.train.reuse_checkpoint:
        return {"skipped": "reusing an existing checkpoint; nothing to smoke-train"}
    if spec.checkpoint.exists():
        return {"skipped": f"full checkpoint already exists ({spec.checkpoint}); smoke is moot"}
    smoke_dir = spec.run_dir / "smoke_model"
    if smoke_dir.exists():
        shutil.rmtree(smoke_dir)
    _run_logged(
        spec.study.train_command(spec, steps=20, batch_size=8, output_dir=smoke_dir),
        spec.run_dir / "smoke-train.log",
    )
    smoke_ckpt = smoke_dir / "checkpoints" / "last" / "pretrained_model"
    _run_logged(
        spec.study.eval_command(spec, smoke_ckpt, spec.run_dir / "smoke_eval", n_episodes=1),
        spec.run_dir / "smoke-eval.log",
    )
    shutil.rmtree(smoke_dir)  # ~hundreds of MB; the signal was "no crash"
    return {"train_steps": 20, "eval_episodes": 1}


def stage_train(spec: BenchSpec) -> dict:
    if spec.train.reuse_checkpoint:
        ckpt = Path(spec.train.reuse_checkpoint)
        if not ckpt.exists():
            raise RuntimeError(f"reuse_checkpoint does not exist: {ckpt}")
        return {"reused": str(ckpt)}
    if spec.checkpoint.exists():
        return {"skipped": f"checkpoint already exists: {spec.checkpoint}"}
    _run_logged(
        spec.study.train_command(spec, steps=spec.train.steps,
                                 batch_size=spec.train.batch_size, output_dir=spec.output_dir),
        spec.run_dir / "train.log",
    )
    if not spec.checkpoint.exists():
        raise RuntimeError(f"training finished but checkpoint missing: {spec.checkpoint}")
    return {"checkpoint": str(spec.checkpoint), "steps": spec.train.steps}


def stage_eval(spec: BenchSpec) -> dict:
    info_path = spec.eval_output_dir / "eval_info.json"
    if not info_path.exists():
        _run_logged(
            spec.study.eval_command(spec, spec.checkpoint, spec.eval_output_dir,
                                    spec.eval.n_episodes),
            spec.run_dir / "eval.log",
        )
    info = json.loads(info_path.read_text())
    return {"pc_success": info["overall"]["pc_success"], "eval_info": str(info_path)}


def stage_record(spec: BenchSpec) -> dict:
    status = _load_status(spec)
    entry = {
        "name": spec.name,
        "spec": spec.source_path,
        "task_index": spec.task_index,
        "dataset_group": spec.dataset_group,
        "trainer": spec.train.trainer,
        "train_action_type": spec.train.action_type,
        "policy_type": spec.train.policy_type,
        "eval_action_type": spec.eval.action_type,
        "control_mode": spec.eval.control_mode,
        "n_episodes": spec.eval.n_episodes,
        "pc_success": status.get("eval", {}).get("info", {}).get("pc_success"),
        "gt_replay": status.get("gt-replay", {}).get("info", {}),
        "checkpoint": str(spec.checkpoint),
        "recorded_at": _now(),
    }
    ledger_dir = _ledger_dir(spec.study_name)
    results_json = ledger_dir / "results.json"
    ledger_dir.mkdir(parents=True, exist_ok=True)
    results = json.loads(results_json.read_text()) if results_json.exists() else []
    results = [r for r in results if r["name"] != spec.name] + [entry]
    results.sort(key=lambda r: (r["task_index"], r["name"]))
    results_json.write_text(json.dumps(results, indent=2))
    _write_results_md(results, ledger_dir / "RESULTS.md")
    return {"ledger": str(results_json)}


def _write_results_md(results: list[dict], md_path: Path) -> None:
    lines = [
        "# LIBERO validation-harness results",
        "",
        "_Machine-generated by `anvil-sim-bench` (stage `record`). Do not edit by hand._",
        "",
        "| name | task | dataset | policy | eval type | mode | n | replay (GT/base) | pc_success | recorded |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        gr = r.get("gt_replay") or {}
        replay_txt = (
            f"{gr.get('pc_success', '—')}/{gr.get('baseline', '—')}"
            if gr and "pc_success" in gr
            else "—"
        )
        lines.append(
            f"| {r['name']} | {r['task_index']} | {r['dataset_group']} | {r['policy_type']} "
            f"| {r['eval_action_type']} | {r['control_mode']} | {r.get('n_episodes', '—')} | {replay_txt} "
            f"| **{r['pc_success']}** | {r['recorded_at']} |"
        )
    md_path.write_text("\n".join(lines) + "\n")


_STAGE_FNS = {
    "convert": stage_convert,
    "validate-math": stage_validate_math,
    "dataset-validate": stage_dataset_validate,
    "gt-replay": stage_gt_replay,
    "smoke": stage_smoke,
    "train": stage_train,
    "eval": stage_eval,
    "record": stage_record,
}


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #


def _clear_stage_cache(stage: str, spec: BenchSpec) -> None:
    """Drop a stage's cached artifact so ``--force-stage`` actually re-runs it.

    Several stages short-circuit on an existing artifact for the normal
    idempotent path — ``stage_eval`` reuses ``eval_info.json`` if present. That
    optimization silently defeats ``--force-stage eval`` (the stage function
    reruns but returns the stale number), so when a stage is explicitly forced
    we remove its cache first."""
    if stage == "eval" and spec.eval_output_dir.exists():
        shutil.rmtree(spec.eval_output_dir)


def run_spec(spec: BenchSpec, from_stage: str | None = None, dry_run: bool = False,
             force_stages: list[str] | None = None) -> dict:
    status = _load_status(spec)
    start = STAGES.index(from_stage) if from_stage else 0
    force = set(force_stages or [])
    for stage in STAGES[start:]:
        if stage in spec.gates.skip:
            status[stage] = {"status": "skipped-by-spec", "at": _now()}
            _save_status(spec, status)
            log.info("[%s] %s: SKIPPED (spec.gates.skip)", spec.name, stage)
            continue
        already = status.get(stage, {})
        if already.get("status") == "passed" and stage not in force and stage != "record":
            log.info("[%s] %s: already passed, skipping (use --force-stage to rerun)", spec.name, stage)
            continue
        log.info("[%s] %s: running ...", spec.name, stage)
        if dry_run:
            log.info("[%s] %s: DRY-RUN (not executed)", spec.name, stage)
            continue
        if stage in force:
            _clear_stage_cache(stage, spec)
        try:
            info = _STAGE_FNS[stage](spec)
        except Exception as exc:
            status[stage] = {"status": "failed", "at": _now(), "error": str(exc)}
            _save_status(spec, status)
            log.error("[%s] %s: FAILED — %s", spec.name, stage, exc)
            raise SystemExit(2) from exc
        status[stage] = {"status": "passed", "at": _now(), "info": info}
        _save_status(spec, status)
        log.info("[%s] %s: passed %s", spec.name, stage, json.dumps(info, default=str)[:200])
    return status


def cmd_status(study: str | None = None) -> None:
    """Print the ledger for one topic (``--study``) or every topic under research/."""
    if study:
        md_paths = [_ledger_dir(study) / "RESULTS.md"]
    else:
        md_paths = sorted(RESEARCH_ROOT.glob("*/ledger/RESULTS.md"))
    md_paths = [m for m in md_paths if m.exists()]
    if not md_paths:
        print("No results recorded yet.")
        return
    for md in md_paths:
        print(md.read_text())


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="run a spec through the gated pipeline")
    run_p.add_argument("spec", type=Path)
    run_p.add_argument("--from-stage", choices=STAGES, default=None)
    run_p.add_argument("--force-stage", action="append", choices=STAGES, default=None,
                       help="rerun this stage even if already passed (repeatable)")
    run_p.add_argument("--dry-run", action="store_true")

    status_p = sub.add_parser("status", help="print the results ledger")
    status_p.add_argument("--study", default=None,
                          help="one topic (default: every topic under research/)")

    args = parser.parse_args()
    if args.command == "status":
        cmd_status(args.study)
        return
    spec = load_spec(args.spec)
    run_spec(spec, from_stage=args.from_stage, dry_run=args.dry_run,
             force_stages=args.force_stage)
    print(f"\n[{spec.name}] pipeline complete. Ledger: {_ledger_dir(spec.study_name) / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
