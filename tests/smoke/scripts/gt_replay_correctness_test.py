#!/usr/bin/env python3
"""GT-Replayer correctness test — validates dataset_gt_replayer_node against fake hardware.

For each fixture dataset (ee_abs, ee_delta), seeds the fake-hardware mock's initial
EE pose from the dataset's own first recorded observation.state row, replays the
dataset through the real inference pipeline, and has gt_replay_verifier_node compare
the live published commands against the dataset's own recorded trajectory
(``published_cmd[t] == dataset.observation.state[t+1]``, converted to quat) — see
claude_docs/gt-replayer-correctness-test-plan.md for the full design.

Orchestrates docker-compose.fake-hardware.yml's ``replay-verify`` profile in
explicit stages (mock-robot -> gt-replay-verify -> replay, with a DDS-discovery
delay between the last two) rather than bringing all three up at once, so the
verifier is guaranteed to be subscribed before the replay run starts publishing
(compose's ``depends_on`` only gates start order, not "has finished subscribing").

Usage:
    uv run python tests/smoke/scripts/gt_replay_correctness_test.py
    uv run python tests/smoke/scripts/gt_replay_correctness_test.py --timeout-sec 120
    uv run python tests/smoke/scripts/gt_replay_correctness_test.py --no-build
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]   # tests/smoke/scripts/ -> repo root

# dataset_reader.py has no rclpy dependency, so it's importable directly here —
# reuse it instead of re-implementing the observation-row read + quat conversion.
sys.path.insert(0, str(REPO / "ros2" / "src" / "lerobot_control"))
from lerobot_control import dataset_reader  # noqa: E402

COMPOSE_FILE = REPO / "docker-compose.fake-hardware.yml"
REPORTS_DIR = REPO / "tests" / "smoke" / ".gt_replay_reports"
GT_REPLAY_TEST_CONFIG = REPO / "configs" / "lerobot_control" / "inference_ee_gt_replay_test.yaml"

FIXTURES = [
    ("ee-abs", REPO / "data" / "debug" / "ee-abs" / "ee-space-testing"),
    ("ee-delta", REPO / "data" / "debug" / "ee-delta" / "ee-space-testing"),
]

EPISODE = 0
DDS_DISCOVERY_SLEEP_SEC = 3.0
REPORT_POLL_INTERVAL_SEC = 1.0
REPORT_POLL_MARGIN_SEC = 10.0
MOCK_CONTAINER = "lerobot-fake-robot"
CONTAINERS = [MOCK_CONTAINER, "lerobot-fake-replay", "lerobot-gt-replay-verify"]


def _compose(*args: str, env_extra: dict | None = None) -> int:
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    cmd = ["docker", "compose", "-f", str(COMPOSE_FILE), "--profile", "replay-verify", *args]
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
    """The dataset's own first observation.state row, quat layout, comma-formatted."""
    obs = dataset_reader.load_episode_observations_quat(dataset_root, EPISODE)
    return ",".join(f"{v:.10g}" for v in obs[0])


def run_scenario(name: str, dataset_root: Path, timeout_sec: float) -> bool:
    print(f"\n=== {name}: {dataset_root} ===")

    # gt-replay-verify writes <REPORTS_DIR>/gt_replay_report.json (fixed name inside
    # the container); give each scenario its own report dir so they don't clobber.
    scenario_reports_dir = REPORTS_DIR / name
    scenario_reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = scenario_reports_dir / "gt_replay_report.json"
    if report_path.exists():
        report_path.unlink()

    seed = _compute_seed(dataset_root)
    env = {
        "EE_MODE": "true",
        "DATASET_PATH": str(dataset_root),
        "CONFIG_FILE": str(GT_REPLAY_TEST_CONFIG),
        "EPISODE": str(EPISODE),
        "EE_SEED_POSE": seed,
        "REPORTS_DIR": str(scenario_reports_dir),
        "VERIFY_TIMEOUT_SEC": str(timeout_sec),
    }

    try:
        if _compose("up", "-d", "mock-robot", env_extra=env) != 0:
            print("  mock-robot failed to start")
            return False
        if not _wait_healthy(MOCK_CONTAINER):
            print("  mock-robot never became healthy")
            return False

        if _compose("up", "-d", "gt-replay-verify", env_extra=env) != 0:
            print("  gt-replay-verify failed to start")
            return False
        time.sleep(DDS_DISCOVERY_SLEEP_SEC)

        if _compose("up", "-d", "replay", env_extra=env) != 0:
            print("  replay failed to start")
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
    parser.add_argument(
        "--timeout-sec", type=float, default=60.0,
        help="Verifier safety timeout per scenario (default: 60.0)",
    )
    parser.add_argument(
        "--no-build", action="store_true",
        help="Skip `docker compose build` (use the already-built image as-is)",
    )
    args = parser.parse_args()

    if not args.no_build:
        print("Building docker image...")
        if _compose("build") != 0:
            print("Docker build failed")
            return 1

    results: dict[str, bool | None] = {}
    for name, dataset_root in FIXTURES:
        if not dataset_root.exists():
            print(f"SKIP {name}: dataset not found at {dataset_root}")
            results[name] = None
            continue
        results[name] = run_scenario(name, dataset_root, timeout_sec=args.timeout_sec)

    print("\n=== Summary ===")
    all_ok = True
    for name, ok in results.items():
        label = "SKIPPED" if ok is None else ("PASS" if ok else "FAIL")
        print(f"  {name}: {label}")
        if ok is False:
            all_ok = False
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
