#!/usr/bin/env python3
"""End-to-end CLI smoke test for the anvil training / eval stack.

Five scenarios are tested against the fixture at data/raw/test-session (5 stub MCAPs,
single right arm):

  AFO  — action_from_observation=true,  action_type=absolute
  CMD  — action_from_observation=false, action_type=absolute
  AFO_DELTA_OBS_T      — AFO dataset + --action-type=delta_obs_t
  AFO_DELTA_SEQUENTIAL — AFO dataset + --action-type=delta_sequential
  CMD_DELTA_OBS_T      — CMD dataset + --action-type=delta_obs_t

delta scenarios reuse the step-1 dataset produced by afo/cmd and only
differ in training (steps 2–4).  They share convert_config with their
base scenario so step 1 is always a no-op (cached) for delta variants.

Each scenario runs all 4 steps: mcap-convert → anvil-trainer → anvil-eval → anvil-eval-ros

Usage:
  uv run python tests/smoke/scripts/pipeline_smoke_test.py                             # all scenarios, all 4 steps
  uv run python tests/smoke/scripts/pipeline_smoke_test.py --scenario afo              # AFO only
  uv run python tests/smoke/scripts/pipeline_smoke_test.py --scenario afo,cmd          # subset
  uv run python tests/smoke/scripts/pipeline_smoke_test.py --select 1,2               # steps 1+2 for all scenarios
  uv run python tests/smoke/scripts/pipeline_smoke_test.py --force                    # wipe + rerun
  uv run python tests/smoke/scripts/pipeline_smoke_test.py --no-docker                # step 4 skips Docker
  uv run python tests/smoke/scripts/pipeline_smoke_test.py --keep-going               # don't stop on failure

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

MCAP_ROOT = FIXTURES / "test-session"


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
    action_type: str = "absolute"  # absolute | delta_obs_t | delta_sequential

    @property
    def checkpoint(self) -> Path:
        return self.train_out / "checkpoints"

    def ckpt_dir(self, steps: int) -> Path:
        return self.train_out / "checkpoints" / f"{steps:06d}"


# mcap-convert appends the input directory name to the output path, so the dataset
# ends up at <output_parent>/<mcap_root_name>/.  Use scenario-specific parent dirs
# under outputs/ to keep artifacts separate.
# Delta scenarios share the SAME dataset as their base (afo/cmd) — step 1 is a no-op.
_MCAP_NAME = MCAP_ROOT.name  # "test-session"

SCENARIOS: dict[str, Scenario] = {
    "afo": Scenario(
        key="afo",
        label="AFO absolute",
        mcap_root=MCAP_ROOT,
        dataset_dir=OUTPUTS / "datasets" / "afo" / _MCAP_NAME,
        train_out=OUTPUTS / "model_zoo" / "afo" / "smoke",
        eval_out=OUTPUTS / "eval_results" / "afo" / "raw",
        eval_ros_out=OUTPUTS / "eval_results" / "afo" / "ros",
        convert_config=FIXTURES / "configs" / "mcap-converter-smoke-test-afo.yaml",
    ),
    "cmd": Scenario(
        key="cmd",
        label="CMD absolute",
        mcap_root=MCAP_ROOT,
        dataset_dir=OUTPUTS / "datasets" / "cmd" / _MCAP_NAME,
        train_out=OUTPUTS / "model_zoo" / "cmd" / "smoke",
        eval_out=OUTPUTS / "eval_results" / "cmd" / "raw",
        eval_ros_out=OUTPUTS / "eval_results" / "cmd" / "ros",
        convert_config=FIXTURES / "configs" / "mcap-converter-smoke-test-cmd.yaml",
    ),
    "afo_delta_obs_t": Scenario(
        key="afo_delta_obs_t",
        label="AFO delta_obs_t",
        mcap_root=MCAP_ROOT,
        dataset_dir=OUTPUTS / "datasets" / "afo" / _MCAP_NAME,  # shared with afo
        train_out=OUTPUTS / "model_zoo" / "afo_delta_obs_t" / "smoke",
        eval_out=OUTPUTS / "eval_results" / "afo_delta_obs_t" / "raw",
        eval_ros_out=OUTPUTS / "eval_results" / "afo_delta_obs_t" / "ros",
        convert_config=FIXTURES / "configs" / "mcap-converter-smoke-test-afo.yaml",
        action_type="delta_obs_t",
    ),
    "afo_delta_sequential": Scenario(
        key="afo_delta_sequential",
        label="AFO delta_sequential",
        mcap_root=MCAP_ROOT,
        dataset_dir=OUTPUTS / "datasets" / "afo" / _MCAP_NAME,  # shared with afo
        train_out=OUTPUTS / "model_zoo" / "afo_delta_sequential" / "smoke",
        eval_out=OUTPUTS / "eval_results" / "afo_delta_sequential" / "raw",
        eval_ros_out=OUTPUTS / "eval_results" / "afo_delta_sequential" / "ros",
        convert_config=FIXTURES / "configs" / "mcap-converter-smoke-test-afo.yaml",
        action_type="delta_sequential",
    ),
    "cmd_delta_obs_t": Scenario(
        key="cmd_delta_obs_t",
        label="CMD delta_obs_t",
        mcap_root=MCAP_ROOT,
        dataset_dir=OUTPUTS / "datasets" / "cmd" / _MCAP_NAME,  # shared with cmd
        train_out=OUTPUTS / "model_zoo" / "cmd_delta_obs_t" / "smoke",
        eval_out=OUTPUTS / "eval_results" / "cmd_delta_obs_t" / "raw",
        eval_ros_out=OUTPUTS / "eval_results" / "cmd_delta_obs_t" / "ros",
        convert_config=FIXTURES / "configs" / "mcap-converter-smoke-test-cmd.yaml",
        action_type="delta_obs_t",
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


def _rmtree(path: Path) -> None:
    try:
        shutil.rmtree(path)
    except PermissionError:
        subprocess.run(["sudo", "rm", "-rf", str(path)], check=True)


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

    t0 = time.monotonic()
    rc = _run([
        "uv", "run", "mcap-convert",
        "-i", str(sc.mcap_root),
        "-o", str(sc.dataset_dir.parent),
        "--config", str(sc.convert_config),
        "--robot-type", "anvil_openarm",
    ])
    dt = time.monotonic() - t0

    if rc != 0:
        return StepResult(ok=False, duration_s=dt, artifact=sc.dataset_dir, notes=f"exit {rc}")
    if not expected.exists():
        return StepResult(ok=False, duration_s=dt, artifact=sc.dataset_dir,
                          notes=f"missing {expected.relative_to(REPO)}")
    return StepResult(ok=True, duration_s=dt, artifact=sc.dataset_dir)


# ── Step 2: anvil-trainer ────────────────────────────────────────────────────

def run_step_train(sc: Scenario, force: bool, steps_override: int) -> StepResult:
    if not (sc.dataset_dir / "meta" / "info.json").exists():
        return _missing(sc.dataset_dir / "meta" / "info.json")

    ckpt_dir = sc.ckpt_dir(steps_override)
    expected = ckpt_dir / "pretrained_model" / "model.safetensors"

    if force and sc.train_out.exists():
        shutil.rmtree(sc.train_out)
    if expected.exists() and not force:
        return StepResult(ok=True, duration_s=0.0, artifact=ckpt_dir, notes="cached")

    t0 = time.monotonic()
    train_cmd = [
        "uv", "run", "anvil-trainer",
        f"--dataset.root={sc.dataset_dir}",
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
        f"--output_dir={sc.train_out}",
        "--job_name=smoke",
    ]
    if sc.action_type != "absolute":
        train_cmd.append(f"--action-type={sc.action_type}")
    rc = _run(train_cmd, env_extra={"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
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
    cmd = [
        "uv", "run", "anvil-eval-ros",
        "--checkpoint", str(ckpt_dir),
        "--mcap-root", str(sc.mcap_root),
        "--dataset-dir", str(sc.dataset_dir),
        "--base-inference-config",
        str(FIXTURES / "configs" / "inference-eval-smoke-test.yaml"),
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
             with_docker: bool) -> StepResult:
    if step_no == 1:
        return run_step_convert(sc, force)
    elif step_no == 2:
        return run_step_train(sc, force, steps_override)
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
                           with_docker=not args.no_docker)
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
