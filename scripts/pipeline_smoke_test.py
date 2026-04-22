#!/usr/bin/env python3
"""End-to-end CLI smoke test for the anvil training / eval stack.

Exercises mcap-convert → anvil-trainer → anvil-eval → anvil-eval-ros against the
fixture at data/raw/test-session (5 stub MCAPs, single right arm, uses
action_from_observation fallback in openarm_single_quest.yaml).

Usage:
  uv run python scripts/pipeline_smoke_test.py               # run all 4 steps (step 4 launches Docker)
  uv run python scripts/pipeline_smoke_test.py --no-docker   # step 4 only generates eval_plan.json
  uv run python scripts/pipeline_smoke_test.py --select 3,4  # run only 3 and 4
  uv run python scripts/pipeline_smoke_test.py --select 1 --force   # wipe + rerun
  uv run python scripts/pipeline_smoke_test.py --keep-going         # don't stop on failure

Each step reads its inputs from stable artifact paths produced by earlier steps,
so you can rerun a subset after fixing a later stage without redoing the whole
pipeline.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# ── Stable artifact paths ────────────────────────────────────────────────────
MCAP_ROOT = REPO / "data" / "raw" / "test-session"
DATASET_DIR = REPO / "data" / "datasets" / "test-session"
TRAIN_OUT = REPO / "model_zoo" / "test-session" / "smoke"
CHECKPOINT = TRAIN_OUT / "checkpoints" / "000010"
PRETRAINED = CHECKPOINT / "pretrained_model"
EVAL_OUT = REPO / "eval_results" / "test-session" / "smoke" / "raw"
EVAL_ROS_OUT = REPO / "eval_results" / "test-session" / "smoke" / "ros"
CONVERT_CONFIG = REPO / "configs" / "mcap_converter" / "openarm_single_quest.yaml"


# ── Step result ──────────────────────────────────────────────────────────────
@dataclass
class StepResult:
    ok: bool
    duration_s: float
    artifact: Path
    notes: str = ""


def _run(cmd: list[str], env_extra: dict | None = None) -> int:
    """Run a subprocess, streaming its stdout/stderr directly to the console.

    Returns the exit code. Full CLI output is shown live; the summary at the
    end only prints PASS/FAIL per step since the user already saw the details.
    """
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    print(f"  $ {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=REPO, env=env)
    return proc.returncode


def _rmtree(path: Path) -> None:
    """Remove a directory tree, falling back to sudo for Docker-owned (root) files."""
    try:
        shutil.rmtree(path)
    except PermissionError:
        subprocess.run(["sudo", "rm", "-rf", str(path)], check=True)


def _missing(path: Path) -> StepResult:
    return StepResult(ok=False, duration_s=0.0, artifact=path, notes=f"missing: {path.relative_to(REPO)}")


# ── Step 1: mcap-convert ─────────────────────────────────────────────────────
def run_step_convert(force: bool) -> StepResult:
    if not any(MCAP_ROOT.rglob("*.mcap")):
        return _missing(MCAP_ROOT)
    expected = DATASET_DIR / "meta" / "info.json"
    if force and DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)
    if expected.exists() and not force:
        return StepResult(ok=True, duration_s=0.0, artifact=DATASET_DIR, notes="cached")

    t0 = time.monotonic()
    rc = _run([
        "uv", "run", "mcap-convert",
        "-i", str(MCAP_ROOT),
        "-o", str(DATASET_DIR.parent),
        "--config", str(CONVERT_CONFIG),
        "--robot-type", "anvil_openarm",
    ])
    dt = time.monotonic() - t0

    if rc != 0:
        return StepResult(ok=False, duration_s=dt, artifact=DATASET_DIR, notes=f"exit {rc}")
    if not expected.exists():
        return StepResult(ok=False, duration_s=dt, artifact=DATASET_DIR,
                          notes=f"missing {expected.relative_to(REPO)}")
    return StepResult(ok=True, duration_s=dt, artifact=DATASET_DIR)


# ── Step 2: anvil-trainer ────────────────────────────────────────────────────
def run_step_train(force: bool, steps_override: int) -> StepResult:
    if not (DATASET_DIR / "meta" / "info.json").exists():
        return _missing(DATASET_DIR / "meta" / "info.json")

    # Checkpoint directory name depends on --steps; we always save at that step.
    ckpt_name = f"{steps_override:06d}"
    ckpt_dir = TRAIN_OUT / "checkpoints" / ckpt_name
    expected = ckpt_dir / "pretrained_model" / "model.safetensors"

    if force and TRAIN_OUT.exists():
        shutil.rmtree(TRAIN_OUT)
    if expected.exists() and not force:
        return StepResult(ok=True, duration_s=0.0, artifact=ckpt_dir, notes="cached")

    t0 = time.monotonic()
    rc = _run(
        [
            "uv", "run", "anvil-trainer",
            f"--dataset.root={DATASET_DIR}",
            "--dataset.repo_id=local",
            "--policy.type=diffusion",
            "--policy.push_to_hub=false",
            "--split-ratio=3,1,1",
            f"--steps={steps_override}",
            f"--save_freq={steps_override}",
            "--log_freq=5",
            "--batch_size=1",
            "--num_workers=0",
            "--eval_freq=0",
            f"--output_dir={TRAIN_OUT}",
            "--job_name=smoke",
        ],
        env_extra={"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
    )
    dt = time.monotonic() - t0

    if rc != 0:
        return StepResult(ok=False, duration_s=dt, artifact=ckpt_dir, notes=f"exit {rc}")
    if not expected.exists():
        return StepResult(ok=False, duration_s=dt, artifact=ckpt_dir,
                          notes=f"missing {expected.relative_to(REPO)}")
    return StepResult(ok=True, duration_s=dt, artifact=ckpt_dir)


# ── Step 3: anvil-eval ───────────────────────────────────────────────────────
def run_step_eval(force: bool, steps_override: int) -> StepResult:
    ckpt_dir = TRAIN_OUT / "checkpoints" / f"{steps_override:06d}"
    if not (ckpt_dir / "pretrained_model" / "config.json").exists():
        return _missing(ckpt_dir / "pretrained_model" / "config.json")

    expected = EVAL_OUT / "metrics_summary.json"
    if force and EVAL_OUT.exists():
        shutil.rmtree(EVAL_OUT)
    if expected.exists() and not force:
        return StepResult(ok=True, duration_s=0.0, artifact=expected, notes="cached")

    t0 = time.monotonic()
    rc = _run([
        "uv", "run", "anvil-eval",
        "--checkpoint", str(ckpt_dir),
        "--dataset", str(DATASET_DIR),
        "--num-eps", "1",
        "--output-dir", str(EVAL_OUT),
    ])
    dt = time.monotonic() - t0

    if rc != 0:
        return StepResult(ok=False, duration_s=dt, artifact=expected, notes=f"exit {rc}")
    if not expected.exists():
        return StepResult(ok=False, duration_s=dt, artifact=expected,
                          notes="missing metrics_summary.json")
    return StepResult(ok=True, duration_s=dt, artifact=expected)


def _mcap_has_commands_topic(mcap_path: Path) -> bool:
    """Return True if the MCAP file contains a .../commands topic (any arm).

    Used to fast-fail step 4 docker mode on observation-only bags (which have
    no action GT to replay, so eval-recorder would time-out on every episode).
    """
    try:
        from mcap.reader import make_reader
        with mcap_path.open("rb") as f:
            reader = make_reader(f)
            for _, channel in reader.get_summary().channels.items():
                if channel.topic.endswith("/commands"):
                    return True
    except Exception:
        pass
    return False


# ── Step 4: anvil-eval-ros (full Docker stack by default, --no-docker to skip)
def run_step_eval_ros(force: bool, steps_override: int, with_docker: bool) -> StepResult:
    ckpt_dir = TRAIN_OUT / "checkpoints" / f"{steps_override:06d}"
    if not (ckpt_dir / "pretrained_model" / "config.json").exists():
        return _missing(ckpt_dir / "pretrained_model" / "config.json")

    # Pre-flight: docker mode needs a GT `.../commands` topic in the replay
    # bags. Observation-only recordings cause eval-recorder to hit ack_timeout
    # on every episode (see diagnostic log "still waiting for first GT sample").
    if with_docker:
        sample = next(MCAP_ROOT.rglob("*.mcap"), None)
        if sample is not None and not _mcap_has_commands_topic(sample):
            return StepResult(
                ok=False, duration_s=0.0, artifact=EVAL_ROS_OUT,
                notes=(
                    f"MCAP fixture has no `.../commands` GT topic "
                    f"(observation-only recording). Use --no-docker, or "
                    f"record new MCAPs that include the action-command topic."
                ),
            )

    # Final artifact differs between the two modes:
    #   --no-docker (default): just eval_plan.json
    #   --with-docker: full run produces metrics_summary.json
    expected = (EVAL_ROS_OUT / "metrics_summary.json") if with_docker else (EVAL_ROS_OUT / "eval_plan.json")
    if force and EVAL_ROS_OUT.exists():
        _rmtree(EVAL_ROS_OUT)
    if expected.exists() and not force:
        return StepResult(ok=True, duration_s=0.0, artifact=expected, notes="cached")

    cmd = [
        "uv", "run", "anvil-eval-ros",
        "--checkpoint", str(ckpt_dir),
        "--mcap-root", str(MCAP_ROOT),
        "--num-eps", "1",
        "--output-dir", str(EVAL_ROS_OUT),
    ]
    if not with_docker:
        cmd.append("--no-docker")
    else:
        # Remove any stale containers from a previous interrupted run so
        # docker-compose can recreate them. Container names are hardcoded in
        # docker-compose.eval.yml, so compose env-var validation is bypassed.
        subprocess.run(
            ["docker", "rm", "-f",
             "lerobot-eval-inference", "lerobot-eval-player", "lerobot-eval-recorder"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

    t0 = time.monotonic()
    rc = _run(cmd)
    dt = time.monotonic() - t0

    if rc != 0:
        return StepResult(ok=False, duration_s=dt, artifact=expected, notes=f"exit {rc}")
    if not expected.exists():
        return StepResult(ok=False, duration_s=dt, artifact=expected,
                          notes=f"missing {expected.name}")

    notes_bits: list[str] = []
    if with_docker and expected.name == "metrics_summary.json":
        summary = json.loads(expected.read_text())
        overall = summary.get("overall", {})
        notes_bits.append(f"mean MAE={overall.get('mean_mae', float('nan')):.4f}")
    else:
        plan = json.loads(expected.read_text())
        notes_bits.append(f"{len(plan.get('episodes', []))} eps")
    return StepResult(ok=True, duration_s=dt, artifact=expected, notes=", ".join(notes_bits))


# ── Driver ───────────────────────────────────────────────────────────────────
STEPS: dict[int, tuple[str, callable]] = {
    1: ("mcap-convert",  run_step_convert),
    2: ("anvil-trainer", run_step_train),
    3: ("anvil-eval",    run_step_eval),
    4: ("anvil-eval-ros",run_step_eval_ros),
}


def parse_select(raw: str) -> list[int]:
    if raw == "all":
        return list(STEPS)
    out = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        n = int(chunk)
        if n not in STEPS:
            raise SystemExit(f"invalid step: {n}; valid: {list(STEPS)}")
        out.append(n)
    return out


def format_row(i: int, total: int, name: str, res: StepResult) -> str:
    status = "PASS" if res.ok else "FAIL"
    rel_art = res.artifact.relative_to(REPO) if res.artifact.is_relative_to(REPO) else res.artifact
    tail = f"  [{res.notes}]" if res.notes else ""
    return f"[{i}/{total}] {name:<15} ... {status:<4} ({res.duration_s:5.1f}s)  → {rel_art}{tail}"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--select", default="all", help="comma-separated step numbers or 'all' (default: all)")
    p.add_argument("--force", action="store_true", help="delete existing artifacts before each selected step")
    p.add_argument("--keep-going", action="store_true", help="don't stop on first failure")
    p.add_argument("--steps-override", type=int, default=10, help="training --steps value (default: 10)")
    p.add_argument("--no-docker", action="store_true",
                   help="step 4 skips the Docker stack and only generates eval_plan.json. "
                        "Default is to run the full inference + mcap-player + eval-recorder stack "
                        "(requires GPU + ~3min).")
    args = p.parse_args()

    selected = parse_select(args.select)
    total = len(selected)
    results: list[tuple[int, str, StepResult]] = []

    overall_t0 = time.monotonic()
    for pos, step_no in enumerate(selected, start=1):
        name, fn = STEPS[step_no]
        print(f"\n═══ [{pos}/{total}] Step {step_no}: {name} ═══", flush=True)
        if step_no == 1:
            res = fn(args.force)
        elif step_no == 4:
            res = fn(args.force, args.steps_override, with_docker=not args.no_docker)
        else:
            res = fn(args.force, args.steps_override)

        print(format_row(pos, total, name, res), flush=True)
        results.append((pos, name, res))
        if not res.ok and not args.keep_going:
            break

    passed = sum(1 for _, _, r in results if r.ok)
    failed = len(results) - passed
    dt = time.monotonic() - overall_t0
    print()
    print(f"{passed} passed, {failed} failed in {dt:.1f}s")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
