#!/usr/bin/env python3
"""End-to-end CLI smoke test for the anvil training / eval stack.

Eight scenarios across two fixture sessions:

Joint-space (tests/smoke/fixtures/test-session, 5 stub MCAPs, single right arm):
  joint_abs_afo              — action_from_observation=true,  action_type=joint_abs
  joint_abs                  — action_from_observation=false, action_type=joint_abs
  joint_abs_delta_obs_t      — shares joint_abs's converted dataset, action_type=delta_obs_t
  joint_abs_delta_sequential — shares joint_abs's converted dataset, action_type=delta_sequential

EE Cartesian (tests/smoke/fixtures/ee-session, 5 stub MCAPs, single right arm):
  ee_abs         — action_type=ee_abs    (EE absolute rot6d)
  ee_rel         — action_type=ee_rel    (EE SE(3) relative; shares converted dataset with ee_abs)
  ee_delta       — action_type=ee_delta  (EE per-frame Delta(n->n+1), world-frame,
                   observation_encoding="quaternion" — the schema default). Own converted
                   dataset — action_encoding="delta" bakes a different action column than
                   ee_abs/ee_rel's "absolute" encoding, so it cannot share their dataset.
  ee_delta_rot6d — same as ee_delta but observation_encoding="rot6d" (10 dims/arm instead
                   of quaternion's 8) — exercises the observation_encoding-aware path in
                   anvil_shared/ee_transform.py (fixed 2026-07-19, see
                   claude_docs/ee-delta-training-flow-gaps-fix-plan.md). Own converted
                   dataset (different observation.state shape than ee_delta's).

EE space is inherently AFO — /ee_pose_right is both observation and action source.
ee_rel step 1 shows "cached" when the shared EE dataset already exists from ee_abs.
Same for joint_abs_delta_* against joint_abs. ee_delta/ee_delta_rot6d each always
convert their own dataset.

Each scenario runs all 4 steps: mcap-convert → anvil-trainer → anvil-eval → anvil-eval-ros,
EXCEPT ee_delta/ee_delta_rot6d, which today only support steps 1–2 (mcap-convert,
anvil-trainer) — steps 3–4 need eval-side ee_delta support (anvil_eval / anvil_eval_ros
action-type branches) that hasn't landed yet. Run them with `--select 1,2` until it does.

Usage:
  uv run python tests/smoke/scripts/pipeline_smoke_test.py                               # all scenarios, all 4 steps
  uv run python tests/smoke/scripts/pipeline_smoke_test.py --scenario joint_abs_afo      # AFO only
  uv run python tests/smoke/scripts/pipeline_smoke_test.py --scenario joint_abs,joint_abs_afo  # joint only
  uv run python tests/smoke/scripts/pipeline_smoke_test.py --scenario ee_abs,ee_rel      # EE only
  uv run python tests/smoke/scripts/pipeline_smoke_test.py --scenario ee_delta,ee_delta_rot6d --select 1,2  # ee_delta convert+train only, both encodings
  uv run python tests/smoke/scripts/pipeline_smoke_test.py --select 1,2                 # steps 1+2 for all scenarios
  uv run python tests/smoke/scripts/pipeline_smoke_test.py --force                      # wipe + rerun
  uv run python tests/smoke/scripts/pipeline_smoke_test.py --no-docker                  # step 4 skips Docker
  uv run python tests/smoke/scripts/pipeline_smoke_test.py --keep-going                 # don't stop on failure
  uv run python tests/smoke/scripts/pipeline_smoke_test.py --resume                     # test step 2 resume path

Each step reads its inputs from stable artifact paths produced by earlier steps,
so you can rerun a subset after fixing a later stage without redoing the whole
pipeline.

EE fixture generation:
  If tests/smoke/fixtures/ee-session/ is missing, run:
    uv run python tests/smoke/fixtures/scripts/generate_ee_fixtures.py
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

REPO = Path(__file__).resolve().parents[3]   # tests/smoke/scripts/ → repo root

# ── Smoke test root ───────────────────────────────────────────────────────────
# Layout under tests/smoke/:
#
#   tests/smoke/
#     scripts/
#       pipeline_smoke_test.py  ← this file
#     fixtures/
#       test-session/                          ← stub MCAP recordings (committed)
#       configs/
#         mcap-converter-smoke-test-afo.yaml   ← AFO test config (committed)
#         mcap-converter-smoke-test-cmd.yaml   ← CMD test config (committed)
#     outputs/                  ← gitignored generated artifacts
#       datasets/afo/   datasets/cmd/
#       model_zoo/afo/  model_zoo/cmd/
#       eval_results/afo/  eval_results/cmd/

SMOKE_ROOT = Path(__file__).resolve().parents[1]   # tests/smoke/
FIXTURES = SMOKE_ROOT / "fixtures"
OUTPUTS = SMOKE_ROOT / "outputs"

MCAP_ROOT    = FIXTURES / "test-session"
EE_MCAP_ROOT = FIXTURES / "ee-session"


def _mcap_input_copy(root: Path) -> Path:
    """A persistent, gitignored copy of a committed fixture session under
    outputs/, reused (not recreated) across scenarios and reruns.

    mcap-valid's default report now writes inside its `-i` input's own
    resolved location, not cwd — so Step 1 pointing `-i` at a tracked fixture
    directly (MCAP_ROOT or EE_MCAP_ROOT) would write mcap_valid_reports/ into
    the tracked fixture tree on every smoke test run. All scenarios/steps use
    this copy instead so mcap-valid and mcap-convert see the same episode
    paths (needed for --quality-report path matching in resolve_quality_skip_paths).
    """
    dest = OUTPUTS / "mcap_input" / root.name
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(root, dest)
    return dest


# ── Scenario definition ──────────────────────────────────────────────────────

@dataclass
class Scenario:
    key: str
    label: str
    mcap_root: Path
    dataset_dir: Path
    train_out: Path
    eval_out: Path
    eval_ros_out: Path
    convert_config: Path
    action_type: str = "joint_abs"     # joint_abs | ee_abs | ee_rel | ee_delta
    # Override the inference base config for step 4 (None = joint default)
    inference_config: Path | None = None

    @property
    def checkpoint(self) -> Path:
        return self.train_out / "checkpoints"

    def ckpt_dir(self, steps: int) -> Path:
        return self.train_out / "checkpoints" / f"{steps:06d}"


# mcap-convert appends "<data_space>-<encoding>/<input-dir-name>" to the output path,
# so the dataset ends up at <output_base>/joint-abs/<name>/ or ee-abs/<name>/.
# Use scenario-specific parent dirs under outputs/ to keep artifacts separate.
# ee_abs and ee_rel point to the SAME dataset_dir — step 1 for ee_rel shows
# "cached" when ee_abs has already converted, and re-converts if forced.
# joint_abs_delta_* scenarios share the SAME dataset as joint_abs (classic
# delta action-type variants trained/evaluated on the same converted data) —
# step 1 for those is "cached" once joint_abs has already converted.
_MCAP_NAME    = MCAP_ROOT.name    # "test-session"
_EE_MCAP_NAME = EE_MCAP_ROOT.name  # "ee-session"
# gitignored copies of the tracked fixtures — see _mcap_input_copy docstring
# (mcap-valid's report would otherwise land inside the tracked fixture tree)
_MCAP_INPUT    = _mcap_input_copy(MCAP_ROOT)
_EE_MCAP_INPUT = _mcap_input_copy(EE_MCAP_ROOT)

_EE_INFERENCE_CFG = FIXTURES / "configs" / "inference-eval-smoke-test-ee.yaml"

SCENARIOS: dict[str, Scenario] = {
    "joint_abs_afo": Scenario(
        key="joint_abs_afo",
        label="joint_abs AFO",
        mcap_root=_MCAP_INPUT,
        dataset_dir=OUTPUTS / "datasets" / "joint_abs_afo" / "joint-abs" / _MCAP_NAME,
        train_out=OUTPUTS / "model_zoo" / "joint_abs_afo" / "smoke",
        eval_out=OUTPUTS / "eval_results" / "joint_abs_afo" / "raw",
        eval_ros_out=OUTPUTS / "eval_results" / "joint_abs_afo" / "ros",
        convert_config=FIXTURES / "configs" / "mcap-converter-smoke-test-afo.yaml",
    ),
    "joint_abs": Scenario(
        key="joint_abs",
        label="joint_abs CMD",
        mcap_root=_MCAP_INPUT,
        dataset_dir=OUTPUTS / "datasets" / "joint_abs" / "joint-abs" / _MCAP_NAME,
        train_out=OUTPUTS / "model_zoo" / "joint_abs" / "smoke",
        eval_out=OUTPUTS / "eval_results" / "joint_abs" / "raw",
        eval_ros_out=OUTPUTS / "eval_results" / "joint_abs" / "ros",
        convert_config=FIXTURES / "configs" / "mcap-converter-smoke-test-cmd.yaml",
    ),
    "joint_abs_delta_obs_t": Scenario(
        key="joint_abs_delta_obs_t",
        label="joint_abs CMD delta_obs_t",
        mcap_root=_MCAP_INPUT,
        # shared with joint_abs — step 1 is "cached" once joint_abs has converted
        dataset_dir=OUTPUTS / "datasets" / "joint_abs" / "joint-abs" / _MCAP_NAME,
        train_out=OUTPUTS / "model_zoo" / "joint_abs_delta_obs_t" / "smoke",
        eval_out=OUTPUTS / "eval_results" / "joint_abs_delta_obs_t" / "raw",
        eval_ros_out=OUTPUTS / "eval_results" / "joint_abs_delta_obs_t" / "ros",
        convert_config=FIXTURES / "configs" / "mcap-converter-smoke-test-cmd.yaml",
        action_type="delta_obs_t",
    ),
    "joint_abs_delta_sequential": Scenario(
        key="joint_abs_delta_sequential",
        label="joint_abs CMD delta_sequential",
        mcap_root=_MCAP_INPUT,
        # shared with joint_abs — step 1 is "cached" once joint_abs has converted
        dataset_dir=OUTPUTS / "datasets" / "joint_abs" / "joint-abs" / _MCAP_NAME,
        train_out=OUTPUTS / "model_zoo" / "joint_abs_delta_sequential" / "smoke",
        eval_out=OUTPUTS / "eval_results" / "joint_abs_delta_sequential" / "raw",
        eval_ros_out=OUTPUTS / "eval_results" / "joint_abs_delta_sequential" / "ros",
        convert_config=FIXTURES / "configs" / "mcap-converter-smoke-test-cmd.yaml",
        action_type="delta_sequential",
    ),
    "ee_abs": Scenario(
        key="ee_abs",
        label="ee_abs",
        mcap_root=_EE_MCAP_INPUT,
        dataset_dir=OUTPUTS / "datasets" / "ee" / "ee-abs" / _EE_MCAP_NAME,
        train_out=OUTPUTS / "model_zoo" / "ee_abs" / "smoke",
        eval_out=OUTPUTS / "eval_results" / "ee_abs" / "raw",
        eval_ros_out=OUTPUTS / "eval_results" / "ee_abs" / "ros",
        convert_config=FIXTURES / "configs" / "mcap-converter-smoke-test-ee.yaml",
        action_type="ee_abs",
        inference_config=_EE_INFERENCE_CFG,
    ),
    "ee_rel": Scenario(
        key="ee_rel",
        label="ee_rel",
        mcap_root=_EE_MCAP_INPUT,
        # Same dataset as ee_abs — step 1 is "cached" when ee_abs already converted.
        dataset_dir=OUTPUTS / "datasets" / "ee" / "ee-abs" / _EE_MCAP_NAME,
        train_out=OUTPUTS / "model_zoo" / "ee_rel" / "smoke",
        eval_out=OUTPUTS / "eval_results" / "ee_rel" / "raw",
        eval_ros_out=OUTPUTS / "eval_results" / "ee_rel" / "ros",
        convert_config=FIXTURES / "configs" / "mcap-converter-smoke-test-ee.yaml",
        action_type="ee_rel",
        inference_config=_EE_INFERENCE_CFG,
    ),
    "ee_delta": Scenario(
        key="ee_delta",
        label="ee_delta (quaternion obs)",
        mcap_root=_EE_MCAP_INPUT,
        # Own dataset — delta is baked differently from ee_abs/ee_rel (per-frame
        # Delta(n->n+1) action, not the next frame's absolute pose), so it cannot
        # share their converted dataset. mcap-convert appends <data_space>-<encoding>
        # = "ee-delta" to the output path (see the path-suffixing note above).
        dataset_dir=OUTPUTS / "datasets" / "ee_delta" / "ee-delta" / _EE_MCAP_NAME,
        train_out=OUTPUTS / "model_zoo" / "ee_delta" / "smoke",
        eval_out=OUTPUTS / "eval_results" / "ee_delta" / "raw",
        eval_ros_out=OUTPUTS / "eval_results" / "ee_delta" / "ros",
        convert_config=FIXTURES / "configs" / "mcap-converter-smoke-test-ee-delta.yaml",
        action_type="ee_delta",
        inference_config=_EE_INFERENCE_CFG,
    ),
    "ee_delta_rot6d": Scenario(
        key="ee_delta_rot6d",
        label="ee_delta (rot6d obs)",
        mcap_root=_EE_MCAP_INPUT,
        # Same action_encoding="delta" as ee_delta above, but observation_encoding=
        # "rot6d" instead of the default "quaternion" — a different observation.state
        # shape (10/arm vs 8/arm), so its own dataset dir. Exercises the
        # observation_encoding-aware path in anvil_shared/ee_transform.py fixed
        # 2026-07-19 (see claude_docs/ee-delta-training-flow-gaps-fix-plan.md).
        dataset_dir=OUTPUTS / "datasets" / "ee_delta_rot6d" / "ee-delta" / _EE_MCAP_NAME,
        train_out=OUTPUTS / "model_zoo" / "ee_delta_rot6d" / "smoke",
        eval_out=OUTPUTS / "eval_results" / "ee_delta_rot6d" / "raw",
        eval_ros_out=OUTPUTS / "eval_results" / "ee_delta_rot6d" / "ros",
        convert_config=FIXTURES / "configs" / "mcap-converter-smoke-test-ee-delta-rot6d.yaml",
        action_type="ee_delta",
        inference_config=_EE_INFERENCE_CFG,
    ),
}


# ── Step result ──────────────────────────────────────────────────────────────

@dataclass
class StepResult:
    ok: bool
    duration_s: float
    artifact: Path
    notes: str = ""


def _run(cmd: list[str], env_extra: dict | None = None) -> int:
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    print(f"  $ {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=REPO, env=env)
    return proc.returncode


def _save_docker_logs(output_dir: Path) -> None:
    """Dump logs from eval Docker containers into output_dir/docker_logs/ for post-mortem."""
    containers = [
        "lerobot-eval-inference",
        "lerobot-eval-player",
        "lerobot-eval-recorder",
        "lerobot-eval-monitor",
    ]
    log_dir = output_dir / "docker_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    for name in containers:
        result = subprocess.run(
            ["docker", "logs", "--timestamps", name],
            capture_output=True, text=True,
        )
        if result.returncode == 0 or result.stdout or result.stderr:
            combined = result.stdout + result.stderr
            (log_dir / f"{name}.log").write_text(combined)


def _rmtree(path: Path) -> None:
    try:
        shutil.rmtree(path)
    except PermissionError:
        # Docker containers write as root; use an alpine container to remove
        # instead of sudo (which requires a terminal / password in CI).
        subprocess.run(
            ["docker", "run", "--rm",
             "-v", f"{path.parent}:/data",
             "alpine", "rm", "-rf", f"/data/{path.name}"],
            check=True,
        )


def _missing(path: Path) -> StepResult:
    return StepResult(ok=False, duration_s=0.0, artifact=path,
                      notes=f"missing: {path.relative_to(REPO)}")


# ── Step 1: mcap-convert ─────────────────────────────────────────────────────

def run_step_convert(sc: Scenario, force: bool) -> StepResult:
    if not any(sc.mcap_root.rglob("*.mcap")):
        return _missing(sc.mcap_root)
    expected = sc.dataset_dir / "meta" / "info.json"
    if force and sc.dataset_dir.exists():
        shutil.rmtree(sc.dataset_dir)
    if expected.exists() and not force:
        return StepResult(ok=True, duration_s=0.0, artifact=sc.dataset_dir, notes="cached")

    # mcap-convert now requires a mcap-valid quality report to exist first.
    # All scenarios share the same mcap_root, so one report is generated once
    # and reused across scenarios/reruns.
    report_path = OUTPUTS / "mcap_valid_reports" / f"{sc.mcap_root.name}.json"
    if force or not report_path.exists():
        report_path.parent.mkdir(parents=True, exist_ok=True)
        valid_rc = _run([
            "uv", "run", "mcap-valid",
            "-i", str(sc.mcap_root),
            "--format", "json",
            "--output", str(report_path),
        ])
        if valid_rc != 0:
            return StepResult(ok=False, duration_s=0.0, artifact=report_path,
                              notes=f"mcap-valid exit {valid_rc}")

    t0 = time.monotonic()
    # mcap-convert appends <data_space>-space/<input-name>/ to the given -o dir,
    # so we pass dataset_dir.parent.parent (the base above the space-subdir).
    rc = _run([
        "uv", "run", "mcap-convert",
        "-i", str(sc.mcap_root),
        "-o", str(sc.dataset_dir.parent.parent),
        "--config", str(sc.convert_config),
        "--robot-type", "anvil_openarm",
        "--quality-report", str(report_path),
    ])
    dt = time.monotonic() - t0

    if rc != 0:
        return StepResult(ok=False, duration_s=dt, artifact=sc.dataset_dir, notes=f"exit {rc}")
    if not expected.exists():
        return StepResult(ok=False, duration_s=dt, artifact=sc.dataset_dir,
                          notes=f"missing {expected.relative_to(REPO)}")
    return StepResult(ok=True, duration_s=dt, artifact=sc.dataset_dir)


# ── Step 2: anvil-trainer ────────────────────────────────────────────────────

def _build_train_cmd(sc: Scenario, job_name: str, steps: int, save_freq: int,
                     exclude_observs: str = "") -> list[str]:
    cmd = [
        "uv", "run", "anvil-trainer",
        f"--dataset.root={sc.dataset_dir}",
        "--dataset.repo_id=local",
        "--policy.type=diffusion",
        "--policy.push_to_hub=false",
        "--split-ratio=3,1,1",
        f"--steps={steps}",
        f"--save_freq={save_freq}",
        "--log_freq=5",
        "--batch_size=1",
        "--num_workers=0",
        "--eval_freq=0",
        f"--output_dir={sc.train_out}",
        f"--job_name={job_name}",
    ]
    if sc.action_type != "joint_abs":
        cmd.append(f"--action-type={sc.action_type}")
    if exclude_observs:
        cmd.append(f"--exclude-observs={exclude_observs}")
    return cmd


def run_step_train(sc: Scenario, force: bool, steps_override: int,
                   exclude_observs: str = "", do_resume: bool = False) -> StepResult:
    if not (sc.dataset_dir / "meta" / "info.json").exists():
        return _missing(sc.dataset_dir / "meta" / "info.json")

    ckpt_dir = sc.ckpt_dir(steps_override)
    expected = ckpt_dir / "pretrained_model" / "model.safetensors"

    if force and sc.train_out.exists():
        shutil.rmtree(sc.train_out)
    if expected.exists() and not force:
        return StepResult(ok=True, duration_s=0.0, artifact=ckpt_dir, notes="cached")

    # job_name incorporates the exclude tag so WandB / logging rows stay distinct
    excl_tag = ("_excl_" + exclude_observs.replace(",", "_").replace(".", "_")
                if exclude_observs else "")
    job_name = f"smoke{excl_tag}"
    _env = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}

    if do_resume:
        # ── Resume path ─────────────────────────────────────────────────────
        # Phase 1: train to halfway to create the checkpoint we will resume from.
        # Phase 2: resume=sc.train_out with --steps=steps_override to complete.
        half = max(1, steps_override // 2)
        half_ckpt = sc.ckpt_dir(half)
        half_expected = half_ckpt / "pretrained_model" / "model.safetensors"

        phase1_dt = 0.0
        if not half_expected.exists():
            t0 = time.monotonic()
            rc = _run(_build_train_cmd(sc, job_name, half, half, exclude_observs), env_extra=_env)
            phase1_dt = time.monotonic() - t0
            if rc != 0:
                return StepResult(ok=False, duration_s=phase1_dt, artifact=half_ckpt,
                                  notes=f"phase1 exit {rc}")
            if not half_expected.exists():
                return StepResult(ok=False, duration_s=phase1_dt, artifact=half_ckpt,
                                  notes=f"phase1 missing {half_expected.relative_to(REPO)}")

        # Phase 2: resume from the partial checkpoint
        t1 = time.monotonic()
        resume_cmd = [
            "uv", "run", "anvil-trainer",
            f"--resume={sc.train_out}",
            f"--steps={steps_override}",
        ]
        if sc.action_type != "joint_abs":
            resume_cmd.append(f"--action-type={sc.action_type}")
        rc = _run(resume_cmd, env_extra=_env)
        total_dt = phase1_dt + (time.monotonic() - t1)

        if rc != 0:
            return StepResult(ok=False, duration_s=total_dt, artifact=ckpt_dir,
                              notes=f"phase2 exit {rc}")
        if not expected.exists():
            return StepResult(ok=False, duration_s=total_dt, artifact=ckpt_dir,
                              notes=f"missing {expected.relative_to(REPO)}")
        return StepResult(ok=True, duration_s=total_dt, artifact=ckpt_dir,
                          notes=f"resumed from step {half}")
    else:
        # ── Normal single-phase training ─────────────────────────────────────
        t0 = time.monotonic()
        rc = _run(_build_train_cmd(sc, job_name, steps_override, steps_override, exclude_observs),
                  env_extra=_env)
        dt = time.monotonic() - t0

        if rc != 0:
            return StepResult(ok=False, duration_s=dt, artifact=ckpt_dir, notes=f"exit {rc}")
        if not expected.exists():
            return StepResult(ok=False, duration_s=dt, artifact=ckpt_dir,
                              notes=f"missing {expected.relative_to(REPO)}")
        return StepResult(ok=True, duration_s=dt, artifact=ckpt_dir)


# ── Step 3: anvil-eval ───────────────────────────────────────────────────────

def run_step_eval(sc: Scenario, force: bool, steps_override: int) -> StepResult:
    ckpt_dir = sc.ckpt_dir(steps_override)
    if not (ckpt_dir / "pretrained_model" / "config.json").exists():
        return _missing(ckpt_dir / "pretrained_model" / "config.json")

    expected = sc.eval_out / "metrics_summary.json"
    if force and sc.eval_out.exists():
        shutil.rmtree(sc.eval_out)
    if expected.exists() and not force:
        return StepResult(ok=True, duration_s=0.0, artifact=expected, notes="cached")

    t0 = time.monotonic()
    rc = _run([
        "uv", "run", "anvil-eval",
        "--checkpoint", str(ckpt_dir),
        "--dataset", str(sc.dataset_dir),
        "--num-eps", "1",
        "--output-dir", str(sc.eval_out),
    ])
    dt = time.monotonic() - t0

    if rc != 0:
        return StepResult(ok=False, duration_s=dt, artifact=expected, notes=f"exit {rc}")
    if not expected.exists():
        return StepResult(ok=False, duration_s=dt, artifact=expected,
                          notes="missing metrics_summary.json")
    return StepResult(ok=True, duration_s=dt, artifact=expected)


# ── Step 4: anvil-eval-ros ───────────────────────────────────────────────────

def run_step_eval_ros(sc: Scenario, force: bool, steps_override: int,
                      with_docker: bool) -> StepResult:
    ckpt_dir = sc.ckpt_dir(steps_override)
    if not (ckpt_dir / "pretrained_model" / "config.json").exists():
        return _missing(ckpt_dir / "pretrained_model" / "config.json")

    expected = (
        (sc.eval_ros_out / "metrics_summary.json") if with_docker
        else (sc.eval_ros_out / "eval_plan.json")
    )
    if force and sc.eval_ros_out.exists():
        _rmtree(sc.eval_ros_out)
    if expected.exists() and not force:
        return StepResult(ok=True, duration_s=0.0, artifact=expected, notes="cached")

    monitor_dir = sc.eval_ros_out / "monitor"
    base_cfg = (
        sc.inference_config
        if sc.inference_config is not None
        else FIXTURES / "configs" / "inference-eval-smoke-test.yaml"
    )
    cmd = [
        "uv", "run", "anvil-eval-ros",
        "--checkpoint", str(ckpt_dir),
        "--mcap-root", str(sc.mcap_root),
        "--dataset-dir", str(sc.dataset_dir),
        "--base-inference-config", str(base_cfg),
        "--num-eps", "1",
        "--output-dir", str(sc.eval_ros_out),
    ]
    if not with_docker:
        cmd.append("--no-docker")
    else:
        subprocess.run(
            ["docker", "rm", "-f",
             "lerobot-eval-inference", "lerobot-eval-player", "lerobot-eval-recorder",
             "lerobot-eval-monitor"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        # Pre-create monitor dir as current user so Docker writes files into a
        # user-owned directory (otherwise Docker creates it as root and we can't
        # write the inference_report.png from the host afterwards).
        monitor_dir.mkdir(parents=True, exist_ok=True)
        cmd.append("--monitor")

    t0 = time.monotonic()
    rc = _run(cmd)
    dt = time.monotonic() - t0

    # Capture Docker container logs for post-mortem inspection
    if with_docker:
        _save_docker_logs(sc.eval_ros_out)

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

        # ── Monitor CSV plot ─────────────────────────────────────────────
        monitor_csv = monitor_dir / "inference_data.csv"
        monitor_png = monitor_dir / "inference_report.png"
        if monitor_csv.exists():
            print(f"  [monitor] Plotting {monitor_csv.relative_to(REPO)} ...", flush=True)
            plot_rc = _run([
                "uv", "run", "python", str(REPO / "scripts" / "plot_monitor_csv.py"),
                str(monitor_csv),
                "-o", str(monitor_png),
            ])
            if plot_rc == 0 and monitor_png.exists():
                notes_bits.append(f"monitor→{monitor_png.relative_to(REPO)}")
            else:
                notes_bits.append("monitor plot FAILED")
        else:
            notes_bits.append("monitor CSV missing")
    else:
        plan = json.loads(expected.read_text())
        notes_bits.append(f"{len(plan.get('episodes', []))} eps")
    return StepResult(ok=True, duration_s=dt, artifact=expected, notes=", ".join(notes_bits))


# ── Driver ───────────────────────────────────────────────────────────────────

STEP_NAMES: dict[int, str] = {
    1: "mcap-convert",
    2: "anvil-trainer",
    3: "anvil-eval",
    4: "anvil-eval-ros",
}


def run_step(step_no: int, sc: Scenario, force: bool, steps_override: int,
             with_docker: bool, exclude_observs: str = "",
             do_resume: bool = False) -> StepResult:
    if step_no == 1:
        return run_step_convert(sc, force)
    elif step_no == 2:
        return run_step_train(sc, force, steps_override, exclude_observs, do_resume)
    elif step_no == 3:
        return run_step_eval(sc, force, steps_override)
    elif step_no == 4:
        return run_step_eval_ros(sc, force, steps_override, with_docker)
    raise ValueError(f"invalid step: {step_no}")


def parse_select(raw: str) -> list[int]:
    valid = set(STEP_NAMES)
    if raw == "all":
        return sorted(valid)
    out = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        n = int(chunk)
        if n not in valid:
            raise SystemExit(f"invalid step: {n}; valid: {sorted(valid)}")
        out.append(n)
    return out


def format_row(scenario_key: str, step_no: int, step_total: int, name: str,
               res: StepResult) -> str:
    status = "PASS" if res.ok else "FAIL"
    rel_art = (res.artifact.relative_to(REPO)
               if res.artifact.is_relative_to(REPO) else res.artifact)
    tail = f"  [{res.notes}]" if res.notes else ""
    return (f"  [{scenario_key.upper()}] [{step_no}/{step_total}] "
            f"{name:<15} ... {status:<4} ({res.duration_s:5.1f}s)  → {rel_art}{tail}")


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenario", default="all",
                   help=(
                       "comma-separated scenario keys or 'all' (default: all). "
                       f"Valid: {', '.join(SCENARIOS)}"
                   ))
    p.add_argument("--select", default="all",
                   help="comma-separated step numbers or 'all' (default: all)")
    p.add_argument("--force", action="store_true",
                   help="delete existing artifacts before each selected step")
    p.add_argument("--keep-going", action="store_true",
                   help="don't stop on first failure")
    p.add_argument("--steps-override", type=int, default=10,
                   help="training --steps value (default: 10)")
    p.add_argument("--no-docker", action="store_true",
                   help="step 4 skips the Docker stack and only generates eval_plan.json")
    p.add_argument("--exclude-observs", default="",
                   metavar="SUFFIXES",
                   help=(
                       "comma-separated observation suffixes to DROP during training "
                       "(e.g. images.chest  or  images.wrist_r,velocity). "
                       "Only affects step 2.  An isolated output dir is used to avoid "
                       "cache collision with the baseline run.  "
                       "Validates docs/training.md Data Filter section."
                   ))
    p.add_argument("--resume", action="store_true",
                   help=(
                       "test the anvil-trainer resume path in step 2: "
                       "phase 1 trains to steps//2, phase 2 resumes to steps. "
                       "Uses --resume=PATH (equals form) to exercise the fixed resume detection."
                   ))
    args = p.parse_args()

    selected_steps = parse_select(args.select)
    if args.scenario == "all":
        scenarios = list(SCENARIOS.values())
    else:
        keys = [k.strip() for k in args.scenario.split(",") if k.strip()]
        unknown = [k for k in keys if k not in SCENARIOS]
        if unknown:
            raise SystemExit(f"unknown scenario(s): {unknown}; valid: {list(SCENARIOS)}")
        scenarios = [SCENARIOS[k] for k in keys]

    exclude_observs = (args.exclude_observs or "").strip()

    # When --exclude-observs is set, use isolated output dirs so artifacts from
    # "drop images.chest" and "keep everything" never collide in the cache.
    if exclude_observs:
        excl_tag = "excl_" + exclude_observs.replace(",", "_").replace(".", "_")
        new_scenarios = []
        for sc in scenarios:
            new_scenarios.append(Scenario(
                key=sc.key,
                label=f"{sc.label} [{excl_tag}]",
                mcap_root=sc.mcap_root,
                dataset_dir=sc.dataset_dir,       # dataset is unchanged (step 1 not modified)
                train_out=sc.train_out.parent / (sc.train_out.name + "_" + excl_tag),
                eval_out=sc.eval_out.parent / (sc.eval_out.name + "_" + excl_tag),
                eval_ros_out=sc.eval_ros_out.parent / (sc.eval_ros_out.name + "_" + excl_tag),
                convert_config=sc.convert_config,
                action_type=sc.action_type,
            ))
        scenarios = new_scenarios

    all_results: list[tuple[str, int, str, StepResult]] = []
    overall_t0 = time.monotonic()
    abort = False

    for sc in scenarios:
        print(f"\n{'═'*70}", flush=True)
        print(f"  SCENARIO: {sc.label}", flush=True)
        print(f"{'═'*70}", flush=True)

        for pos, step_no in enumerate(selected_steps, start=1):
            name = STEP_NAMES[step_no]
            print(f"\n  ─── Step {step_no}: {name} ───", flush=True)
            res = run_step(step_no, sc, args.force, args.steps_override,
                           with_docker=not args.no_docker,
                           exclude_observs=exclude_observs,
                           do_resume=args.resume)
            row = format_row(sc.key, pos, len(selected_steps), name, res)
            print(row, flush=True)
            all_results.append((sc.key, step_no, name, res))

            if not res.ok and not args.keep_going:
                abort = True
                break

        if abort:
            break

    passed = sum(1 for _, _, _, r in all_results if r.ok)
    failed = len(all_results) - passed
    dt = time.monotonic() - overall_t0
    print()
    print(f"{'─'*70}")

    # Print per-scenario summary
    for sc in scenarios:
        sc_results = [(sn, nm, r) for (sk, sn, nm, r) in all_results if sk == sc.key]
        if not sc_results:
            continue
        sc_pass = sum(1 for _, _, r in sc_results if r.ok)
        sc_fail = len(sc_results) - sc_pass
        status = "OK" if sc_fail == 0 else "FAIL"
        print(f"  [{sc.key.upper()}] {sc.label}: {sc_pass} passed, {sc_fail} failed  [{status}]")

    print(f"{'─'*70}")
    print(f"Total: {passed} passed, {failed} failed in {dt:.1f}s")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
