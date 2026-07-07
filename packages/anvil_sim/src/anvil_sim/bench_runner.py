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
8.  ``record``            upsert ``outputs/bench/results.json`` and
                          regenerate ``outputs/bench/RESULTS.md``

Stages are idempotent: each writes its status to
``outputs/bench/runs/<name>/stage_status.json`` and re-running a spec skips
already-passed stages (``--force-stage`` / ``--from-stage`` override).
Subprocess stages log to ``outputs/bench/runs/<name>/<stage>.log``.

Usage::

    anvil-sim-bench run configs/libero_bench/task10_goal_abs_act.yaml
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

import numpy as np

from anvil_sim.bench_spec import BenchSpec, load_spec

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

BENCH_ROOT = Path("outputs/bench")
RESULTS_JSON = BENCH_ROOT / "results.json"
RESULTS_MD = BENCH_ROOT / "RESULTS.md"

_MATH_TOLERANCE = 1e-4  # max abs error allowed by validate-math identities


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


def _native_dataset_root(spec: BenchSpec) -> Path:
    return Path(f"data/datasets/ee-space/libero-task{spec.task_index}-native")


# --------------------------------------------------------------------------- #
# Stage implementations — each returns an info dict on success, raises on fail #
# --------------------------------------------------------------------------- #


def stage_convert(spec: BenchSpec) -> dict:
    """Ensure the spec's dataset group AND the native group (needed by
    validate-math and the gt-replay baseline) exist for this task."""
    missing = [g for g, root in
               ((spec.dataset_group, spec.dataset_root), ("native", _native_dataset_root(spec)))
               if not root.exists()]
    missing = sorted(set(missing))
    if not missing:
        return {"skipped": "datasets already exist"}
    cmd = [
        "uv", "run", "--package", "anvil-sim", "anvil-libero-convert",
        f"--task-index={spec.task_index}",
        f"--only={','.join(missing)}",
    ]
    _run_logged(cmd, spec.run_dir / "convert.log")
    return {"created": missing}


def _load_local_episode(root: Path, episode: int = 0) -> dict[str, np.ndarray]:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id="local", root=str(root))
    hf = ds.hf_dataset.select_columns(["episode_index", "action", "observation.state"]).with_format(None)
    actions, states = [], []
    for ep, act, sta in zip(hf["episode_index"], hf["action"], hf["observation.state"], strict=True):
        if int(ep) == episode:
            actions.append(np.asarray(act, dtype=np.float64))
            states.append(np.asarray(sta, dtype=np.float64))
    return {"action": np.stack(actions), "state": np.stack(states)}


def _validate_goalabs(spec: BenchSpec) -> dict:
    """goalabs family identity: recovering a native-delta from the stored
    formal goal against the SAME state must reproduce the native dataset's
    own command — pos/rot to float precision, gripper via native_cmd
    passthrough (the dimension bug #4 hid in; validated explicitly now)."""
    from anvil_sim.libero_processor import recovered_delta_native_action

    goal = _load_local_episode(spec.dataset_root)
    native = _load_local_episode(_native_dataset_root(spec))
    n = min(len(goal["action"]), len(native["action"]))
    max_err = 0.0
    for t in range(n):
        act10, state8 = goal["action"][t], goal["state"][t]
        recovered = recovered_delta_native_action(
            reconstructed_pos=act10[:3],
            reconstructed_rot6d=act10[3:9],
            reconstructed_gripper=float(act10[9]),
            current_state=state8.astype(np.float32),
            current_gripper=float(state8[7]),
            gripper_mode="native_cmd",
        )
        expected = np.clip(native["action"][t], -1.0, 1.0)
        max_err = max(max_err, float(np.abs(recovered - expected).max()))
    if max_err > _MATH_TOLERANCE:
        raise RuntimeError(f"goalabs identity failed: max_err={max_err:.2e} > {_MATH_TOLERANCE}")
    return {"identity": "goalabs->native command", "frames": n, "max_err": max_err}


def _validate_seq(spec: BenchSpec) -> dict:
    """delta/delta_hand family identity: accumulating the stored per-step
    deltas from the episode's first state must reproduce the achieved state
    trajectory (the check that exposed Experiment 7's v2 anchor bug)."""
    from anvil_shared.rotation import matrix_to_quat, quat_to_matrix, rot6d_to_matrix

    data = _load_local_episode(spec.dataset_root)
    hand_frame = spec.dataset_group == "delta_hand"
    running = data["state"][0].copy()
    max_err = 0.0
    n = len(data["action"]) - 1
    for t in range(n):
        act10 = data["action"][t]
        R_running = quat_to_matrix(running[3:7])
        R_delta = rot6d_to_matrix(act10[3:9])
        if hand_frame:
            new_pos = running[:3] + R_running @ act10[:3]
            new_r = R_running @ R_delta
        else:
            new_pos = running[:3] + act10[:3]
            new_r = R_delta @ R_running
        running = np.concatenate([new_pos, matrix_to_quat(new_r), [act10[9]]])
        max_err = max(max_err, float(np.abs(running[:3] - data["state"][t + 1][:3]).max()))
    if max_err > _MATH_TOLERANCE:
        raise RuntimeError(f"seq accumulation failed: max_err={max_err:.2e} > {_MATH_TOLERANCE}")
    return {"identity": "seq accumulation -> state trajectory", "frames": n, "max_err": max_err}


def _validate_act_from_obs(spec: BenchSpec) -> dict:
    """abs/rel family definitional check: action[t] encodes state[t+1]
    (act-from-obs); catches double-relativization-style dataset corruption
    (real bug #2) at the data level."""
    from anvil_shared.rotation import matrix_to_rot6d, quat_to_matrix

    data = _load_local_episode(spec.dataset_root)
    n = len(data["action"]) - 1
    max_err = 0.0
    for t in range(n):
        nxt = data["state"][t + 1]
        expected = np.concatenate(
            [nxt[:3], matrix_to_rot6d(quat_to_matrix(nxt[3:7])), [nxt[7]]]
        )
        max_err = max(max_err, float(np.abs(data["action"][t] - expected).max()))
    if max_err > _MATH_TOLERANCE:
        raise RuntimeError(f"act-from-obs identity failed: max_err={max_err:.2e} > {_MATH_TOLERANCE}")
    return {"identity": "action[t] == encode(state[t+1])", "frames": n, "max_err": max_err}


_MATH_VALIDATORS = {
    "goalabs": _validate_goalabs,
    "delta": _validate_seq,
    "delta_hand": _validate_seq,
    "abs": _validate_act_from_obs,
    "rel": _validate_act_from_obs,  # rel stores the same absolute column (relativized at load time)
}


def stage_validate_math(spec: BenchSpec) -> dict:
    validator = _MATH_VALIDATORS.get(spec.dataset_group)
    if validator is None:
        return {"skipped": f"no math validator registered for group {spec.dataset_group!r}"}
    return validator(spec)


def stage_dataset_validate(spec: BenchSpec) -> dict:
    if spec.dataset_group in ("native", "native_rot6d"):
        return {"skipped": "dataset-validate targets the Anvil EE writer schema"}
    cmd = [
        "uv", "run", "--package", "mcap_converter", "dataset-validate",
        f"--root={spec.dataset_root}", "--repo-id=local",
    ]
    _run_logged(cmd, spec.run_dir / "dataset-validate.log")
    return {"validated": str(spec.dataset_root)}


def _replay_baseline(spec: BenchSpec) -> dict:
    """Native GT-replay baseline for this task — computed once, cached."""
    baseline_dir = BENCH_ROOT / "replay" / f"baseline-task{spec.task_index}"
    info_path = baseline_dir / "replay_info.json"
    if info_path.exists():
        return json.loads(info_path.read_text())
    from anvil_sim.eval_replay import replay

    return replay(
        action_type="native",
        dataset_root=_native_dataset_root(spec),
        control_mode="relative",
        task=spec.env_suite,
        task_id=spec.env_task_id,
        n_episodes=spec.eval.n_episodes,
        n_action_steps=1,
        output_dir=baseline_dir,
    )


def stage_gt_replay(spec: BenchSpec) -> dict:
    from anvil_sim.eval_replay import replay

    baseline = _replay_baseline(spec)
    if spec.eval.action_type in ("native", "native_rot6d") and spec.dataset_group == "native":
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


def _train_cmd(spec: BenchSpec, *, steps: int, batch_size: int, output_dir: Path) -> list[str]:
    if spec.train.trainer == "anvil-trainer":
        return [
            "uv", "run", "--package", "anvil-trainer", "anvil-trainer",
            f"--dataset.root={spec.dataset_root}",
            f"--policy.type={spec.train.policy_type}",
            f"--action-type={spec.train.action_type}",
            f"--output_dir={output_dir}",
            f"--job_name={spec.name}",
            f"--batch_size={batch_size}",
            f"--steps={steps}",
            "--policy.device=cuda",
            "--wandb.enable=false",
        ]
    return [  # lerobot-train (native family)
        "uv", "run", "--package", "anvil-sim", "lerobot-train",
        "--dataset.repo_id=local",
        f"--dataset.root={spec.dataset_root}",
        f"--policy.type={spec.train.policy_type}",
        "--policy.push_to_hub=false",
        f"--output_dir={output_dir}",
        f"--job_name={spec.name}",
        f"--batch_size={batch_size}",
        f"--steps={steps}",
        "--policy.device=cuda",
        "--wandb.enable=false",
    ]


def _eval_cmd(spec: BenchSpec, checkpoint: Path, output_dir: Path, n_episodes: int) -> list[str]:
    if spec.eval.action_type == "native":
        entry = ["uv", "run", "--package", "anvil-sim", "lerobot-eval"]
        extra: list[str] = []
    elif spec.eval.action_type == "native_rot6d":
        entry = ["uv", "run", "--package", "anvil-sim", "anvil-eval-native-rot6d"]
        extra = []
    else:
        entry = ["uv", "run", "--package", "anvil-sim", "anvil-eval-libero"]
        extra = [f"--action-type={spec.eval.action_type}"]
    return [
        *entry, *extra,
        f"--policy.path={checkpoint}",
        "--env.type=libero",
        f"--env.task={spec.env_suite}",
        f"--env.task_ids=[{spec.env_task_id}]",
        f"--env.control_mode={spec.eval.control_mode}",
        f"--eval.n_episodes={n_episodes}",
        "--eval.batch_size=1",
        f"--output_dir={output_dir}",
        "--policy.device=cuda",
    ]


def stage_smoke(spec: BenchSpec) -> dict:
    if spec.train.reuse_checkpoint:
        return {"skipped": "reusing an existing checkpoint; nothing to smoke-train"}
    if spec.checkpoint.exists():
        return {"skipped": f"full checkpoint already exists ({spec.checkpoint}); smoke is moot"}
    smoke_dir = spec.run_dir / "smoke_model"
    if smoke_dir.exists():
        shutil.rmtree(smoke_dir)
    _run_logged(
        _train_cmd(spec, steps=20, batch_size=8, output_dir=smoke_dir),
        spec.run_dir / "smoke-train.log",
    )
    smoke_ckpt = smoke_dir / "checkpoints" / "last" / "pretrained_model"
    _run_logged(
        _eval_cmd(spec, smoke_ckpt, spec.run_dir / "smoke_eval", n_episodes=1),
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
        _train_cmd(spec, steps=spec.train.steps, batch_size=spec.train.batch_size,
                   output_dir=spec.output_dir),
        spec.run_dir / "train.log",
    )
    if not spec.checkpoint.exists():
        raise RuntimeError(f"training finished but checkpoint missing: {spec.checkpoint}")
    return {"checkpoint": str(spec.checkpoint), "steps": spec.train.steps}


def stage_eval(spec: BenchSpec) -> dict:
    info_path = spec.eval_output_dir / "eval_info.json"
    if not info_path.exists():
        _run_logged(
            _eval_cmd(spec, spec.checkpoint, spec.eval_output_dir, spec.eval.n_episodes),
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
    BENCH_ROOT.mkdir(parents=True, exist_ok=True)
    results = json.loads(RESULTS_JSON.read_text()) if RESULTS_JSON.exists() else []
    results = [r for r in results if r["name"] != spec.name] + [entry]
    results.sort(key=lambda r: (r["task_index"], r["name"]))
    RESULTS_JSON.write_text(json.dumps(results, indent=2))
    _write_results_md(results)
    return {"ledger": str(RESULTS_JSON)}


def _write_results_md(results: list[dict]) -> None:
    lines = [
        "# LIBERO validation-harness results",
        "",
        "_Machine-generated by `anvil-sim-bench` (stage `record`). Do not edit by hand._",
        "",
        "| name | task | dataset | policy | eval type | mode | replay (GT/base) | pc_success | recorded |",
        "|---|---|---|---|---|---|---|---|---|",
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
            f"| {r['eval_action_type']} | {r['control_mode']} | {replay_txt} "
            f"| **{r['pc_success']}** | {r['recorded_at']} |"
        )
    RESULTS_MD.write_text("\n".join(lines) + "\n")


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


def cmd_status() -> None:
    if not RESULTS_JSON.exists():
        print("No results recorded yet.")
        return
    print(RESULTS_MD.read_text() if RESULTS_MD.exists() else RESULTS_JSON.read_text())


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

    sub.add_parser("status", help="print the results ledger")

    args = parser.parse_args()
    if args.command == "status":
        cmd_status()
        return
    spec = load_spec(args.spec)
    run_spec(spec, from_stage=args.from_stage, dry_run=args.dry_run,
             force_stages=args.force_stage)
    print(f"\n[{spec.name}] pipeline complete. Ledger: {RESULTS_MD}")


if __name__ == "__main__":
    main()
