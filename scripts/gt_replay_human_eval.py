#!/usr/bin/env python3
"""GT-replay evaluation via human judgment — real hardware (or fake, as a dry run).

Unlike ``gt_replay_verifier_node``/``gt_replay_correctness_test.py`` (fake-hardware-only,
numeric tolerance against the dataset's own recorded trajectory — see
``claude_docs/dataset-gt-replayer-and-fake-hardware-architecture.md``), this tool replays
one or more episodes and asks a HUMAN OPERATOR to judge task success, since a real robot's
physically-measured pose legitimately deviates from the recording (actuation dynamics,
tracking error, latency) in ways no numeric tolerance should flag as a bug.

For each selected episode, in order:
  1. Bring up one episode's replayer container (``--target real``:
     ``docker-compose.yml``'s ``gt-replay-real`` service, against a real robot's own
     already-running controller stack; ``--target fake``: ``docker-compose.fake-hardware.yml``'s
     ``mock-robot`` + ``replay`` services, a free rehearsal of this tool's own control flow).
  2. Poll for the completion-signal sentinel the replayer writes (see
     ``dataset_gt_replayer_node.py``'s ``_write_signal``) up to a per-episode timeout that
     scales with the episode's own nominal duration. Classify ``homing_status``
     (confirmed/failed/skipped) and ``replay_status`` (completed/timed_out/crashed/
     not_attempted) from whatever the sentinel (or its absence) says.
  3. If replay completed: prompt the operator for a pass/fail judgment + optional comment.
     If it didn't (homing failure, timeout, crash): skip the prompt — there's nothing to
     judge — and record ``operator_verdict: null``. These are never conflated.
  4. Tear the episode's container(s) down before the next episode.

After all episodes: write a JSON report (episode-by-episode + summary counts, including a
``pass_rate`` computed only over episodes that actually completed replay) and print a
compact summary to stdout.

See claude_docs/real-hardware-gt-replay-eval-plan.md for the full design.

Usage:
    scripts/gt_replay_human_eval.py --target real --dataset /path/to/dataset --episodes 0:5 \\
        --config-file configs/lerobot_control/inference_ee.yaml

    scripts/gt_replay_human_eval.py --target fake --dataset data/debug/ee-delta/ee-space-testing \\
        --episodes 0:2 --config-file configs/lerobot_control/inference_ee.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(REPO / "ros2" / "src" / "lerobot_control"))
from lerobot_control import dataset_reader  # noqa: E402

FAKE_COMPOSE_FILE = REPO / "docker-compose.fake-hardware.yml"
REAL_COMPOSE_FILE = REPO / "docker-compose.yml"

MOCK_CONTAINER = "lerobot-fake-robot"
FAKE_REPLAY_CONTAINER = "lerobot-fake-replay"
REAL_REPLAY_CONTAINER = "lerobot-gt-replay-real"

DEFAULT_REPORT_PATH = REPO / "gt_replay_human_eval_report.json"
DEFAULT_SIGNAL_DIR = REPO / "tests" / "smoke" / ".gt_replay_reports" / "human_eval_signals"

# (multiplier, fixed_margin_sec) applied to an episode's nominal duration
# (n_frames / control_frequency) to compute the default completion-poll timeout.
#
# The fixed margin exists mainly to cover node-startup latency, NOT episode-to-
# episode variance: `docker compose up -d` returns as soon as the container is
# scheduled, well before dataset_gt_replayer_node has finished spawning image
# workers and completing DDS discovery — that startup latency (observed ~12s,
# more on a cold cache) elapses entirely *inside* the completion-poll loop,
# silently eating into this budget before any GT row is even replayed. This
# node-startup cost is identical for `replay` (fake) and `gt-replay-real`
# (real) — same LeRobotInferenceNode-derived startup machinery either way —
# so the fake margin must be comfortably above it too, not just a small
# constant: an under-provisioned margin here reads as a false "timed_out"
# on the very first (coldest) episode of a run, not a real failure.
#
# Real hardware additionally gets a larger allowance on top of that shared
# startup cost: tracking error/operator intervention can genuinely extend
# wall-clock time beyond the nominal duration in a way the mock's instantaneous
# echo never does.
TIMEOUT_PROFILE = {
    "fake": (1.5, 25.0),
    "real": (3.0, 30.0),
}

POLL_INTERVAL_SEC = 1.0


# --------------------------------------------------------------------------- #
# Pure logic (unit-testable without docker) — episode selection, timeout/report
# math, and the operator prompt.
# --------------------------------------------------------------------------- #


def resolve_timeout_sec(
    explicit: float | None,
    target: str,
    n_frames: int,
    control_frequency: float,
    homing_enabled: bool,
    homing_timeout_sec: float,
) -> float:
    """Per-episode completion-poll timeout — wrapper-only concern (see plan doc).

    ``dataset_gt_replayer_node`` has no notion of "replay timed out" itself; it
    just replays until done or killed. Scales with the episode's own nominal
    duration rather than a flat guess, since episode lengths vary across
    datasets. ``explicit`` (--completion-timeout-sec), if given, overrides the
    computation entirely. Homing's own timeout is additive on top, since the
    outer poll must outlast both phases combined.
    """
    if explicit is not None:
        base = explicit
    else:
        multiplier, margin = TIMEOUT_PROFILE[target]
        nominal_duration = n_frames / control_frequency
        base = nominal_duration * multiplier + margin
    if homing_enabled:
        base += homing_timeout_sec
    return base


def classify_signal(signal: dict | None, container_exited: bool, timed_out: bool) -> tuple[str | None, str]:
    """Map a raw completion sentinel (or its absence) to (homing_status, replay_status).

    Three orthogonal outcomes, never conflated (see plan doc's report schema):
      - homing_status: "confirmed" | "failed" | "skipped" | None (unknown — no
        signal ever arrived, e.g. a crash before the node could write one).
      - replay_status: "completed" | "timed_out" | "crashed" | "not_attempted"
        ("not_attempted" is set exactly when homing failed — GT playback never
        starts in that case).
    """
    if signal is not None:
        status = signal.get("status")
        homing_status = signal.get("homing_status")
        if status == "complete":
            return homing_status, "completed"
        if status == "homing_failed":
            return "failed", "not_attempted"
        if status == "interrupted":
            return homing_status, "crashed"
        # Unrecognized status value — treat conservatively as a crash rather
        # than silently guessing "completed".
        return homing_status, "crashed"
    if container_exited:
        return None, "crashed"
    if timed_out:
        return None, "timed_out"
    # Should not be reachable (caller only calls this once one of the above is
    # true), but never fabricate a status if it somehow is.
    return None, "crashed"


def prompt_operator_verdict(episode: int, rows_replayed: int, elapsed_sec: float) -> tuple[str, str]:
    """Foreground, blocking pass/fail + optional-comment prompt.

    Mirrors mcap_converter's migrate_config.py _confirm() precedent (raw
    input(), no framework) with one deliberate deviation: a pass/fail judgment
    has no safe default, so this re-asks on unrecognized input instead of
    treating it as a definitive answer either way.
    """
    print(
        f"\nEpisode {episode} replay complete "
        f"({rows_replayed} rows replayed over {elapsed_sec:.1f}s)."
    )
    while True:
        answer = input("Did the robot complete the task successfully? [y/n]: ").strip().lower()
        if answer in ("y", "yes"):
            verdict = "pass"
            break
        if answer in ("n", "no"):
            verdict = "fail"
            break
        print("Please enter 'y' or 'n'.")
    comment = input("Comment (optional, Enter to skip): ").strip()
    return verdict, comment


def build_episode_record(
    episode: int,
    homing_status: str | None,
    replay_status: str,
    operator_verdict: str | None,
    comment: str | None,
    timestamp: str,
) -> dict:
    return {
        "episode": episode,
        "homing_status": homing_status,
        "replay_status": replay_status,
        "operator_verdict": operator_verdict,
        "comment": comment,
        "timestamp": timestamp,
    }


def build_report(
    dataset: str,
    target: str,
    episodes_requested: str,
    episode_records: list[dict],
    started_at: str,
    finished_at: str,
) -> dict:
    n_total = len(episode_records)
    n_homing_confirmed = sum(1 for r in episode_records if r["homing_status"] == "confirmed")
    n_homing_failed = sum(1 for r in episode_records if r["homing_status"] == "failed")
    n_homing_skipped = sum(1 for r in episode_records if r["homing_status"] == "skipped")
    n_completed_replay = sum(1 for r in episode_records if r["replay_status"] == "completed")
    n_failed_to_replay = n_total - n_completed_replay
    n_operator_pass = sum(1 for r in episode_records if r["operator_verdict"] == "pass")
    n_operator_fail = sum(1 for r in episode_records if r["operator_verdict"] == "fail")
    pass_rate = (n_operator_pass / n_completed_replay) if n_completed_replay else None

    return {
        "dataset": dataset,
        "target": target,
        "episodes_requested": episodes_requested,
        "episodes_run": [r["episode"] for r in episode_records],
        "started_at": started_at,
        "finished_at": finished_at,
        "summary": {
            "n_total": n_total,
            "n_homing_confirmed": n_homing_confirmed,
            "n_homing_failed": n_homing_failed,
            "n_homing_skipped": n_homing_skipped,
            "n_completed_replay": n_completed_replay,
            "n_failed_to_replay": n_failed_to_replay,
            "n_operator_pass": n_operator_pass,
            "n_operator_fail": n_operator_fail,
            "pass_rate": pass_rate,
        },
        "episodes": episode_records,
    }


def print_summary(report: dict) -> None:
    s = report["summary"]
    print("\n=== GT-replay human-eval summary ===")
    print(f"  dataset: {report['dataset']} (target={report['target']})")
    print(f"  episodes requested: {report['episodes_requested']} -> ran {s['n_total']}")
    print(
        f"  homing: {s['n_homing_confirmed']} confirmed, {s['n_homing_failed']} failed, "
        f"{s['n_homing_skipped']} skipped"
    )
    print(f"  replay completed: {s['n_completed_replay']}/{s['n_total']}")
    if s["pass_rate"] is not None:
        print(
            f"  operator verdict: {s['n_operator_pass']} pass, {s['n_operator_fail']} fail "
            f"(pass_rate={s['pass_rate']:.2f})"
        )
    else:
        print("  operator verdict: n/a (no episode completed replay)")
    for r in report["episodes"]:
        print(
            f"    episode {r['episode']}: homing={r['homing_status']} "
            f"replay={r['replay_status']} verdict={r['operator_verdict']}"
            + (f" — {r['comment']}" if r["comment"] else "")
        )


# --------------------------------------------------------------------------- #
# Docker orchestration
# --------------------------------------------------------------------------- #


def _compose(compose_file: Path, profile: str, *args: str, env_extra: dict | None = None) -> int:
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    cmd = ["docker", "compose", "-f", str(compose_file), "--profile", profile, *args]
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


def _container_exited(container: str) -> bool:
    result = subprocess.run(
        ["docker", "inspect", "--format", "{{.State.Status}}", container],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return True  # container not found at all — treat as exited/gone
    return result.stdout.strip() not in ("running", "created")


def _read_signal(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None  # mid-write race — caller just polls again next tick


def _compute_ee_seed(dataset_root: Path, episode: int) -> str:
    """The episode's own first observation.state row, quat layout, comma-formatted.

    Mirrors gt_replay_correctness_test.py's _compute_seed. Always used for
    --target fake (never left empty): passing an empty string through to the
    mock's `-p ee_seed_pose:=` ROS CLI override crashes rclpy's argument
    parser outright (a pre-existing gap in that plumbing, unrelated to homing),
    and seeding from the episode's own recorded start pose is the sensible
    default here regardless.
    """
    obs = dataset_reader.load_episode_observations_quat(dataset_root, episode)
    return ",".join(f"{v:.10g}" for v in obs[0])


def run_episode(
    episode: int,
    dataset_root: Path,
    args: argparse.Namespace,
    signal_dir: Path,
) -> dict:
    """Bring up one episode's replay, wait for it, prompt if applicable, tear down."""
    print(f"\n=== Episode {episode} ({args.target}) ===")
    signal_path = signal_dir / f"episode_{episode}.json"
    if signal_path.exists():
        signal_path.unlink()

    _actions = dataset_reader.load_episode_actions(dataset_root, episode)
    n_frames = len(_actions) if _actions is not None else 0
    timeout_sec = resolve_timeout_sec(
        args.completion_timeout_sec, args.target, max(n_frames, 1), args.control_frequency,
        homing_enabled=args.home_before_replay, homing_timeout_sec=args.homing_timeout_sec,
    )

    env = {
        "DATASET_PATH": str(dataset_root),
        "CONFIG_FILE": str(args.config_file),
        "EPISODE": str(episode),
        "CONTROL_FREQ": str(args.control_frequency),
        "DEBUG": str(args.debug).lower(),
        "COMPLETION_SIGNAL_DIR": str(signal_dir),
        "HOME_BEFORE_REPLAY": str(args.home_before_replay).lower(),
        "HOME_ATOL_POS_M": str(args.home_atol_pos_m),
        "HOME_ATOL_ROT_DEG": str(args.home_atol_rot_deg),
        "HOMING_TIMEOUT_SEC": str(args.homing_timeout_sec),
        "HOME_MAX_POS_DELTA_M": str(args.home_max_pos_delta_m),
        "HOME_MAX_ROT_DELTA_DEG": str(args.home_max_rot_delta_deg),
    }

    replay_container = REAL_REPLAY_CONTAINER if args.target == "real" else FAKE_REPLAY_CONTAINER
    start = time.monotonic()

    try:
        if args.target == "fake":
            env["EE_MODE"] = "true"
            env["EE_ARMS"] = args.arms
            env["EE_SEED_POSE"] = _compute_ee_seed(dataset_root, episode)
            if _compose(FAKE_COMPOSE_FILE, "replay", "up", "-d", "mock-robot", env_extra=env) != 0:
                print("  mock-robot failed to start")
                return build_episode_record(episode, None, "crashed", None, None, _now())
            if not _wait_healthy(MOCK_CONTAINER):
                print("  mock-robot never became healthy")
                return build_episode_record(episode, None, "crashed", None, None, _now())
            if _compose(FAKE_COMPOSE_FILE, "replay", "up", "-d", "replay", env_extra=env) != 0:
                print("  replay failed to start")
                return build_episode_record(episode, None, "crashed", None, None, _now())
        else:
            if _compose(REAL_COMPOSE_FILE, "gt-replay-real", "up", "-d", "gt-replay-real", env_extra=env) != 0:
                print("  gt-replay-real failed to start")
                return build_episode_record(episode, None, "crashed", None, None, _now())

        deadline = time.monotonic() + timeout_sec
        signal = None
        exited = False
        while time.monotonic() < deadline:
            signal = _read_signal(signal_path)
            if signal is not None:
                break
            if _container_exited(replay_container):
                exited = True
                break
            time.sleep(POLL_INTERVAL_SEC)
        timed_out = signal is None and not exited

        homing_status, replay_status = classify_signal(signal, exited, timed_out)

        if replay_status == "completed":
            elapsed = time.monotonic() - start
            rows_replayed = signal.get("rows_replayed", n_frames) if signal else n_frames
            verdict, comment = prompt_operator_verdict(episode, rows_replayed, elapsed)
        else:
            verdict, comment = None, None
            print(f"  homing_status={homing_status} replay_status={replay_status} — skipping operator prompt")

        return build_episode_record(episode, homing_status, replay_status, verdict, comment, _now())

    finally:
        if args.target == "fake":
            _compose(FAKE_COMPOSE_FILE, "replay", "down", env_extra=env)
        else:
            _compose(REAL_COMPOSE_FILE, "gt-replay-real", "down", env_extra=env)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="Path to the converted dataset")
    parser.add_argument(
        "--episodes", required=True,
        help="Episode selection: comma list and/or start:end ranges, 0-based, end-exclusive "
        "(e.g. '0,1,2', '0:10', '0,1:3,5'). No negative indices or step.",
    )
    parser.add_argument("--target", choices=["real", "fake"], required=True)
    parser.add_argument("--config-file", required=True, help="Path to the inference config YAML")
    parser.add_argument("--report-path", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--signal-dir", default=str(DEFAULT_SIGNAL_DIR))
    parser.add_argument(
        "--completion-timeout-sec", type=float, default=None,
        help="Override the per-episode completion-poll timeout (default: scales with the "
        "episode's own nominal duration — see resolve_timeout_sec).",
    )
    parser.add_argument("--control-frequency", type=float, default=30.0)
    parser.add_argument("--arms", default="left,right", help="Comma-separated arm ids (fake target only)")
    parser.add_argument(
        "--home-before-replay", dest="home_before_replay", action="store_true", default=None,
        help="Force homing on, even for --target fake (rehearses the homing feature itself "
        "against the mock; off by default for fake since ee_seed_pose already seeds it exactly).",
    )
    parser.add_argument(
        "--no-home-before-replay", dest="home_before_replay", action="store_false",
        help="Force homing off, even for --target real.",
    )
    parser.add_argument("--home-atol-pos-m", type=float, default=0.01)
    parser.add_argument("--home-atol-rot-deg", type=float, default=5.0)
    parser.add_argument("--homing-timeout-sec", type=float, default=30.0)
    parser.add_argument(
        "--home-max-pos-delta-m", type=float, default=0.01,
        help="Max homing approach speed, position: metres per publish tick. "
        "inference_node's joint-space action_limiter isn't applied in EE mode, so this "
        "ramps the one-shot homing command instead of jumping straight to the target.",
    )
    parser.add_argument(
        "--home-max-rot-delta-deg", type=float, default=2.0,
        help="Max homing approach speed, orientation: degrees per publish tick.",
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(argv)

    if args.home_before_replay is None:
        args.home_before_replay = args.target == "real"

    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    # Docker Compose treats a bare relative path in a volume mount as a named
    # volume reference, not a bind path — always resolve to absolute.
    dataset_root = Path(args.dataset).resolve()
    args.config_file = str(Path(args.config_file).resolve())
    signal_dir = Path(args.signal_dir)
    signal_dir.mkdir(parents=True, exist_ok=True)

    total_episodes = dataset_reader.load_info(dataset_root)["total_episodes"]
    try:
        episodes = dataset_reader.parse_episode_spec(args.episodes, total_episodes)
    except ValueError as e:
        print(f"Invalid --episodes spec: {e}", file=sys.stderr)
        return 1
    if not episodes:
        print("No episodes resolved from --episodes spec — nothing to do.", file=sys.stderr)
        return 1

    print(f"Resolved episodes: {episodes} (of {total_episodes} total)")

    started_at = _now()
    records = [run_episode(ep, dataset_root, args, signal_dir) for ep in episodes]
    finished_at = _now()

    report = build_report(
        dataset=str(dataset_root), target=args.target, episodes_requested=args.episodes,
        episode_records=records, started_at=started_at, finished_at=finished_at,
    )

    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\nReport written to {report_path}")
    print_summary(report)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
