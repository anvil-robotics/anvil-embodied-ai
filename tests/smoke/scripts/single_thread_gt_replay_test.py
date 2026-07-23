#!/usr/bin/env python3
"""Correctness test for the experimental single-thread GT replayer.

Mirrors ``gt_replay_correctness_test.py``'s staged bring-up (mock-robot ->
verifier -> replayer, with a DDS-discovery delay), but targets the
``replay-verify-single-thread`` profile (``single_thread_gt_replayer_node`` +
``single_thread_gt_replay_verifier_node``) instead of the production
``dataset_gt_replayer_node``/``gt_replay_verifier_node`` pair — see both
nodes' module docstrings for why this variant exists.

Usage:
    uv run python tests/smoke/scripts/single_thread_gt_replay_test.py
    uv run python tests/smoke/scripts/single_thread_gt_replay_test.py --repeat 10
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

sys.path.insert(0, str(REPO / "ros2" / "src" / "lerobot_control"))
from lerobot_control import dataset_reader  # noqa: E402

COMPOSE_FILE = REPO / "docker-compose.fake-hardware.yml"
REPORTS_DIR = REPO / "tests" / "smoke" / ".gt_replay_reports_single_thread"
PROFILE = "replay-verify-single-thread"

FIXTURES = [
    ("ee-abs", REPO / "data" / "debug" / "ee-abs" / "ee-space-testing"),
    ("ee-delta", REPO / "data" / "debug" / "ee-delta" / "ee-space-testing"),
]

EPISODE = 0
DDS_DISCOVERY_SLEEP_SEC = 3.0
REPORT_POLL_INTERVAL_SEC = 1.0
REPORT_POLL_MARGIN_SEC = 10.0
MOCK_CONTAINER = "lerobot-fake-robot"
CONTAINERS = [
    MOCK_CONTAINER, "lerobot-fake-replay-single-thread", "lerobot-gt-replay-verify-single-thread",
]


def _compose(*args: str, env_extra: dict | None = None) -> int:
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    cmd = ["docker", "compose", "-f", str(COMPOSE_FILE), "--profile", PROFILE, *args]
    print(f"  $ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, cwd=REPO, env=env).returncode


def _wait_healthy(container: str, timeout_sec: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        result = subprocess.run(
            ["docker", "inspect", "--format", "{{.State.Health.Status}}", container],
            capture_output=True, text=True,
        )
        if result.returncode == 0 and result.stdout.strip() == "healthy":
            return True
        time.sleep(1.0)
    return False


def _save_docker_logs(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    for name in CONTAINERS:
        result = subprocess.run(
            ["docker", "logs", "--timestamps", name], capture_output=True, text=True,
        )
        if result.returncode == 0 or result.stdout or result.stderr:
            (log_dir / f"{name}.log").write_text(result.stdout + result.stderr)


def _compute_seed(dataset_root: Path) -> str:
    obs = dataset_reader.load_episode_observations_quat(dataset_root, EPISODE)
    return ",".join(f"{v:.10g}" for v in obs[0])


def run_scenario(
    name: str, dataset_root: Path, timeout_sec: float, run_idx: int, wait_until_arrived: bool,
) -> bool:
    mode = "wait-until-arrived" if wait_until_arrived else "30hz-no-gate"
    print(f"\n=== {name} [{mode}] (run {run_idx}): {dataset_root} ===")

    scenario_reports_dir = REPORTS_DIR / mode / name
    scenario_reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = scenario_reports_dir / "gt_replay_report.json"
    if report_path.exists():
        report_path.unlink()

    seed = _compute_seed(dataset_root)
    env = {
        "EE_MODE": "true",
        "DATASET_PATH": str(dataset_root),
        "EPISODE": str(EPISODE),
        "EE_SEED_POSE": seed,
        "REPORTS_DIR": str(scenario_reports_dir),
        "VERIFY_TIMEOUT_SEC": str(timeout_sec),
        "WAIT_UNTIL_ARRIVED": "true" if wait_until_arrived else "false",
    }

    try:
        if _compose("up", "-d", "mock-robot", env_extra=env) != 0:
            print("  mock-robot failed to start")
            return False
        if not _wait_healthy(MOCK_CONTAINER):
            print("  mock-robot never became healthy")
            return False

        if _compose("up", "-d", "gt-replay-verify-single-thread", env_extra=env) != 0:
            print("  verifier failed to start")
            return False
        time.sleep(DDS_DISCOVERY_SLEEP_SEC)

        if _compose("up", "-d", "replay-single-thread", env_extra=env) != 0:
            print("  replayer failed to start")
            return False

        deadline = time.monotonic() + timeout_sec + REPORT_POLL_MARGIN_SEC
        while time.monotonic() < deadline and not report_path.exists():
            time.sleep(REPORT_POLL_INTERVAL_SEC)

        if not report_path.exists():
            print(f"  TIMEOUT waiting for report at {report_path}")
            _save_docker_logs(scenario_reports_dir / "docker_logs")
            return False

        report = json.loads(report_path.read_text())
        all_passed = bool(report.get("all_passed"))
        print(f"  {'PASS' if all_passed else 'FAIL'}")
        print(json.dumps(report, indent=2))
        if not all_passed:
            _save_docker_logs(scenario_reports_dir / "docker_logs")
        return all_passed

    finally:
        _compose("down", env_extra=env)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout-sec", type=float, default=60.0)
    parser.add_argument("--no-build", action="store_true")
    parser.add_argument("--repeat", type=int, default=1, help="Repeat each fixture N times (flakiness check)")
    parser.add_argument(
        "--modes", choices=["both", "wait-until-arrived", "30hz-no-gate"], default="both",
        help="Which replayer mode(s) to A/B (default: both)",
    )
    args = parser.parse_args()

    if not args.no_build:
        print("Building docker image...")
        if _compose("build") != 0:
            print("Docker build failed")
            return 1

    modes = (
        [("wait-until-arrived", True), ("30hz-no-gate", False)] if args.modes == "both"
        else [(args.modes, args.modes == "wait-until-arrived")]
    )

    # results[mode][fixture] = [bool, ...] one per run
    results: dict[str, dict[str, list[bool]]] = {m: {name: [] for name, _ in FIXTURES} for m, _ in modes}

    for mode_name, wait_until_arrived in modes:
        for run_idx in range(1, args.repeat + 1):
            for name, dataset_root in FIXTURES:
                if not dataset_root.exists():
                    print(f"SKIP {name}: dataset not found at {dataset_root}")
                    continue
                ok = run_scenario(
                    name, dataset_root, timeout_sec=args.timeout_sec, run_idx=run_idx,
                    wait_until_arrived=wait_until_arrived,
                )
                results[mode_name][name].append(ok)
                if not ok:
                    print(f"  !! {name} [{mode_name}] run {run_idx} FAILED")

    print("\n=== Comparison summary ===")
    all_ok = True
    for mode_name, _ in modes:
        for name, _ in FIXTURES:
            runs = results[mode_name][name]
            if not runs:
                continue
            n_pass = sum(runs)
            ok = n_pass == len(runs)
            all_ok = all_ok and ok
            print(f"  {mode_name:20s} {name:10s}: {n_pass}/{len(runs)} passed  {runs}")

    print(f"\n=== Overall: {'PASS' if all_ok else 'FAIL'} ===")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
