#!/usr/bin/env python3
"""Generate timing and latency reports from Anvil/OpenArm MCAP recordings."""

from __future__ import annotations

import argparse
import glob
import json
import math
import statistics
import subprocess
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

JOINT_STATE_TOPIC = "/joint_states"

COMMAND_TOPICS = {
    "/follower_l_forward_position_controller/commands": ("left", "follower_l"),
    "/follower_r_forward_position_controller/commands": ("right", "follower_r"),
}

JOINT_ORDER = [
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "joint7",
    "finger_joint1",
]

EXPECTED_HZ = {
    JOINT_STATE_TOPIC: 500.0,
    "/cam_waist/image_raw/compressed": 30.0,
    "/cam_wrist_r/image_raw/compressed": 30.0,
    "/cam_chest/image_raw/compressed": 30.0,
    "/cam_wrist_l/image_raw/compressed": 30.0,
    "/follower_l_forward_position_controller/commands": 30.0,
    "/follower_r_forward_position_controller/commands": 30.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize MCAP timing, gaps, header skew, and command-to-state lag",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  uv run --with 'mcap~=1.3' --with 'mcap-ros2-support~=0.5.7' python scripts/mcap_timing_report.py data/raw/openarm-calibration-20260706
  uv run --with 'mcap~=1.3' --with 'mcap-ros2-support~=0.5.7' python scripts/mcap_timing_report.py data/raw/session/0001/*.mcap --output-dir eval_results/calibration/session
""",
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="MCAP files, directories containing MCAP files, or glob patterns",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eval_results/calibration_timing"),
        help="directory for timing_report.json and timing_report.md",
    )
    parser.add_argument(
        "--gap-multiple",
        type=float,
        default=2.5,
        help="flag gaps above expected period * multiple (default: 2.5)",
    )
    parser.add_argument(
        "--max-lag-ms",
        type=float,
        default=500.0,
        help="maximum command-to-state lag to test (default: 500 ms)",
    )
    parser.add_argument(
        "--lag-sample-hz",
        type=float,
        default=60.0,
        help="resampling frequency for lag estimation (default: 60 Hz)",
    )
    parser.add_argument(
        "--min-correlation",
        type=float,
        default=0.15,
        help="minimum derivative correlation to report a lag estimate (default: 0.15)",
    )
    return parser.parse_args()


def load_mcap_deps():
    try:
        from mcap_ros2.reader import read_ros2_messages
    except ImportError as exc:
        print("[ERROR] Missing MCAP dependencies.", file=sys.stderr)
        print(
            "Run with: uv run --with 'mcap~=1.3' --with 'mcap-ros2-support~=0.5.7' "
            "python scripts/mcap_timing_report.py ...",
            file=sys.stderr,
        )
        print(f"Details: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    return read_ros2_messages


def expand_inputs(inputs: list[str]) -> list[Path]:
    paths: list[Path] = []
    for raw in inputs:
        matches = [Path(p) for p in glob.glob(raw)] or [Path(raw)]
        for path in matches:
            if path.is_dir():
                paths.extend(sorted(path.rglob("*.mcap")))
            elif path.is_file():
                paths.append(path)
            else:
                print(f"[WARN] Skipping missing input: {path}", file=sys.stderr)

    unique = sorted({p.resolve() for p in paths})
    if not unique:
        raise SystemExit("[ERROR] No MCAP files found")
    return unique


def normalize_timestamp(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.timestamp()
    if isinstance(value, (int, float)):
        return float(value) / 1e9 if abs(value) > 1e6 else float(value)
    return None


def header_stamp_seconds(msg: Any) -> float | None:
    header = getattr(msg, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is None:
        return None
    sec = getattr(stamp, "sec", None)
    nanosec = getattr(stamp, "nanosec", None)
    if sec is None or nanosec is None:
        return None
    return float(sec) + float(nanosec) / 1e9


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    low = math.floor(index)
    high = math.ceil(index)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - index) + ordered[high] * (index - low)


def stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p95": None,
            "p99": None,
            "min": None,
            "max": None,
        }
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "min": min(values),
        "max": max(values),
    }


def rounded(value: Any, digits: int = 3) -> Any:
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, dict):
        return {k: rounded(v, digits) for k, v in value.items()}
    if isinstance(value, list):
        return [rounded(v, digits) for v in value]
    return value


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def topic_expected_hz(topic: str) -> float | None:
    if topic in EXPECTED_HZ:
        return EXPECTED_HZ[topic]
    if topic.endswith("/commands"):
        return 30.0
    if "/image_raw" in topic:
        return 30.0
    return None


def append_joint_observation(
    joint_observations: dict[str, dict[str, list[tuple[float, float]]]],
    timestamp: float,
    msg: Any,
) -> None:
    names = list(getattr(msg, "name", []) or [])
    positions = list(getattr(msg, "position", []) or [])
    if not names or not positions:
        return

    by_name = {name: positions[i] for i, name in enumerate(names[: len(positions)])}
    for arm_name, prefix in (("left", "follower_l"), ("right", "follower_r")):
        for joint in JOINT_ORDER:
            value = by_name.get(f"{prefix}_{joint}")
            if value is not None:
                joint_observations[arm_name][joint].append((timestamp, float(value)))


def append_command(
    command_series: dict[str, dict[str, list[tuple[float, float]]]],
    timestamp: float,
    topic: str,
    msg: Any,
) -> None:
    mapping = COMMAND_TOPICS.get(topic)
    if mapping is None:
        return
    arm_name, _prefix = mapping
    data = list(getattr(msg, "data", []) or [])
    for joint, value in zip(JOINT_ORDER, data):
        command_series[arm_name][joint].append((timestamp, float(value)))


def interpolate_series(series: list[tuple[float, float]], grid: list[float]) -> list[float]:
    if not series:
        return []
    series = sorted(series)
    out: list[float] = []
    idx = 0
    for timestamp in grid:
        while idx + 1 < len(series) and series[idx + 1][0] <= timestamp:
            idx += 1
        if idx + 1 >= len(series):
            out.append(series[-1][1])
            continue
        t0, v0 = series[idx]
        t1, v1 = series[idx + 1]
        if t1 <= t0:
            out.append(v0)
            continue
        alpha = max(0.0, min(1.0, (timestamp - t0) / (t1 - t0)))
        out.append(v0 + alpha * (v1 - v0))
    return out


def derivative(values: list[float]) -> list[float]:
    return [values[i + 1] - values[i] for i in range(len(values) - 1)]


def correlation(a_values: list[float], b_values: list[float]) -> float | None:
    if len(a_values) != len(b_values) or len(a_values) < 10:
        return None
    mean_a = statistics.fmean(a_values)
    mean_b = statistics.fmean(b_values)
    da = [value - mean_a for value in a_values]
    db = [value - mean_b for value in b_values]
    denom_a = math.sqrt(sum(value * value for value in da))
    denom_b = math.sqrt(sum(value * value for value in db))
    if denom_a <= 1e-9 or denom_b <= 1e-9:
        return None
    return sum(a * b for a, b in zip(da, db)) / (denom_a * denom_b)


def estimate_lags(
    command_series: dict[str, dict[str, list[tuple[float, float]]]],
    joint_observations: dict[str, dict[str, list[tuple[float, float]]]],
    max_lag_ms: float,
    sample_hz: float,
    min_correlation: float,
) -> dict[str, Any]:
    step_s = 1.0 / sample_hz
    max_lag_steps = max(1, int((max_lag_ms / 1000.0) / step_s))
    results: dict[str, Any] = {}

    for arm_name in sorted(command_series):
        joint_results = []
        for joint in JOINT_ORDER:
            cmd = sorted(command_series[arm_name].get(joint, []))
            obs = sorted(joint_observations[arm_name].get(joint, []))
            if len(cmd) < 5 or len(obs) < 20:
                continue

            start = max(cmd[0][0], obs[0][0])
            end = min(cmd[-1][0], obs[-1][0])
            if end - start < 1.0:
                continue

            grid_count = int((end - start) / step_s)
            if grid_count < max_lag_steps + 20:
                continue

            grid = [start + i * step_s for i in range(grid_count)]
            cmd_d = derivative(interpolate_series(cmd, grid))
            obs_d = derivative(interpolate_series(obs, grid))

            if max(cmd_d, default=0.0) - min(cmd_d, default=0.0) < 1e-4:
                continue
            if max(obs_d, default=0.0) - min(obs_d, default=0.0) < 1e-4:
                continue

            best_corr = None
            best_lag_steps = None
            usable_max_lag = min(max_lag_steps, len(cmd_d) - 10)
            for lag_steps in range(usable_max_lag + 1):
                cmd_window = cmd_d[: len(cmd_d) - lag_steps] if lag_steps else cmd_d
                obs_window = obs_d[lag_steps:] if lag_steps else obs_d
                corr = correlation(cmd_window, obs_window)
                if corr is not None and (best_corr is None or corr > best_corr):
                    best_corr = corr
                    best_lag_steps = lag_steps

            if (
                best_corr is not None
                and best_lag_steps is not None
                and best_corr >= min_correlation
            ):
                joint_results.append(
                    {
                        "joint": joint,
                        "lag_ms": best_lag_steps * step_s * 1000.0,
                        "correlation": best_corr,
                    }
                )

        lags = [item["lag_ms"] for item in joint_results]
        results[arm_name] = {
            "joints_used": len(joint_results),
            "lag_ms": stats(lags),
            "per_joint": joint_results,
        }

    return results


def read_recordings(paths: list[Path]) -> dict[str, Any]:
    read_ros2_messages = load_mcap_deps()
    topic_times: dict[str, list[float]] = defaultdict(list)
    topic_file_times: dict[str, list[list[float]]] = defaultdict(list)
    header_lags_ms: dict[str, list[float]] = defaultdict(list)
    lag_inputs: list[dict[str, Any]] = []
    read_errors: list[str] = []

    for path_index, path in enumerate(paths):
        file_topic_times: dict[str, list[float]] = defaultdict(list)
        file_command_series: dict[str, dict[str, list[tuple[float, float]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        file_joint_observations: dict[
            str, dict[str, list[tuple[float, float]]]
        ] = defaultdict(lambda: defaultdict(list))
        try:
            for message in read_ros2_messages(str(path)):
                timestamp = normalize_timestamp(message.log_time)
                if timestamp is None:
                    continue
                topic = message.channel.topic
                msg = message.ros_msg

                topic_times[topic].append(timestamp)
                file_topic_times[topic].append(timestamp)

                header_seconds = header_stamp_seconds(msg)
                if header_seconds is not None:
                    header_lags_ms[topic].append((timestamp - header_seconds) * 1000.0)

                if topic == JOINT_STATE_TOPIC:
                    append_joint_observation(file_joint_observations, timestamp, msg)
                elif topic in COMMAND_TOPICS:
                    append_command(file_command_series, timestamp, topic, msg)
        except Exception as exc:
            read_errors.append(f"{path}: {exc}")
        finally:
            for topic, times in file_topic_times.items():
                if times:
                    topic_file_times[topic].append(times)
            if file_command_series or file_joint_observations:
                lag_inputs.append(
                    {
                        "path_index": path_index,
                        "path": str(path),
                        "command_series": file_command_series,
                        "joint_observations": file_joint_observations,
                    }
                )

    return {
        "topic_times": topic_times,
        "topic_file_times": topic_file_times,
        "header_lags_ms": header_lags_ms,
        "lag_inputs": lag_inputs,
        "read_errors": read_errors,
    }


def aggregate_lag_reports(lag_reports: list[tuple[dict[str, Any], dict[str, Any]]]) -> dict[str, Any]:
    aggregated: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for lag_input, lag_report in lag_reports:
        for arm_name, arm_report in lag_report.items():
            for item in arm_report["per_joint"]:
                item_with_source = dict(item)
                item_with_source["source_file"] = lag_input["path"]
                item_with_source["source_index"] = lag_input["path_index"]
                aggregated[arm_name].append(item_with_source)

    return {
        arm_name: {
            "joints_used": len(items),
            "lag_ms": stats([item["lag_ms"] for item in items]),
            "per_joint": items,
        }
        for arm_name, items in sorted(aggregated.items())
    }


def build_topic_report(
    topic_file_times: dict[str, list[list[float]]], gap_multiple: float
) -> dict[str, dict[str, Any]]:
    report = {}
    for topic, per_file_times in sorted(topic_file_times.items()):
        ordered_files = [sorted(times) for times in per_file_times if times]
        all_times = [timestamp for times in ordered_files for timestamp in times]
        intervals_ms = []
        duration_s = 0.0
        nonempty_files = 0
        for ordered in ordered_files:
            nonempty_files += 1
            if len(ordered) > 1:
                duration_s += ordered[-1] - ordered[0]
                intervals_ms.extend(
                    (ordered[i + 1] - ordered[i]) * 1000.0
                    for i in range(len(ordered) - 1)
                )
        first = min(all_times) if all_times else None
        last = max(all_times) if all_times else None
        expected_hz = topic_expected_hz(topic)
        gap_threshold_ms = None
        if expected_hz:
            gap_threshold_ms = (1000.0 / expected_hz) * gap_multiple
        elif intervals_ms:
            period_p95 = percentile(intervals_ms, 0.95)
            gap_threshold_ms = period_p95 * gap_multiple if period_p95 else None
        gaps = [gap for gap in intervals_ms if gap_threshold_ms and gap > gap_threshold_ms]

        report[topic] = {
            "count": len(all_times),
            "file_count": nonempty_files,
            "first_log_time_s": first,
            "last_log_time_s": last,
            "summed_file_duration_s": duration_s,
            "observed_hz": ((len(all_times) - nonempty_files) / duration_s)
            if duration_s > 0
            else None,
            "expected_hz": expected_hz,
            "period_ms": stats(intervals_ms),
            "gap_threshold_ms": gap_threshold_ms,
            "gap_count": len(gaps),
            "max_gap_ms": max(gaps) if gaps else None,
        }
    return report


def build_warnings(topic_report: dict[str, dict[str, Any]]) -> list[str]:
    warnings = []
    for topic, info in topic_report.items():
        observed_hz = info.get("observed_hz")
        expected_hz = info.get("expected_hz")
        if expected_hz and observed_hz and observed_hz < expected_hz * 0.8:
            warnings.append(
                f"{topic}: observed rate {observed_hz:.1f} Hz is below 80% of expected {expected_hz:.1f} Hz"
            )
        if info.get("gap_count", 0) > 0:
            warnings.append(
                f"{topic}: {info['gap_count']} timing gaps above {info['gap_threshold_ms']:.1f} ms"
            )
    return warnings


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join("" if value is None else str(value) for value in row) + " |")
    return "\n".join(lines)


def fmt_ms(value: Any) -> str:
    return "" if value is None else f"{float(value):.1f}"


def fmt_hz(value: Any) -> str:
    return "" if value is None else f"{float(value):.1f}"


def render_markdown(report: dict[str, Any]) -> str:
    topic_rows = []
    for topic, info in report["topics"].items():
        topic_rows.append(
            [
                topic,
                info["count"],
                fmt_hz(info["observed_hz"]),
                fmt_hz(info["expected_hz"]),
                fmt_ms(info["period_ms"]["median"]),
                fmt_ms(info["period_ms"]["p95"]),
                info["gap_count"],
                fmt_ms(info["max_gap_ms"]),
            ]
        )

    skew_rows = []
    for topic, info in report["header_lag_ms"].items():
        skew_rows.append(
            [
                topic,
                info["count"],
                fmt_ms(info["median"]),
                fmt_ms(info["p95"]),
                fmt_ms(info["max"]),
            ]
        )

    lag_rows = []
    for arm, info in report["command_to_state_lag"].items():
        lag_rows.append(
            [
                arm,
                info["joints_used"],
                fmt_ms(info["lag_ms"]["median"]),
                fmt_ms(info["lag_ms"]["p95"]),
                fmt_ms(info["lag_ms"]["max"]),
            ]
        )

    warning_text = "\n".join(f"- {item}" for item in report["warnings"]) or "- None"
    skew_section = (
        markdown_table(["Topic", "Samples", "Median ms", "P95 ms", "Max ms"], skew_rows)
        if skew_rows
        else "No header stamps found."
    )
    lag_section = (
        markdown_table(["Arm", "Joints used", "Median ms", "P95 ms", "Max ms"], lag_rows)
        if lag_rows
        else "No command-to-state lag estimate available. Record a deliberate timing sweep with command topics."
    )

    return "\n\n".join(
        [
            "# MCAP Timing Report",
            f"Generated: {report['generated_at']}",
            f"Git commit: {report.get('git_commit') or 'unknown'}",
            f"MCAP files: {len(report['mcap_files'])}",
            "## Topic Timing",
            markdown_table(
                [
                    "Topic",
                    "Count",
                    "Hz",
                    "Expected Hz",
                    "Median period ms",
                    "P95 period ms",
                    "Gaps",
                    "Max gap ms",
                ],
                topic_rows,
            ),
            "## Header Stamp Lag",
            skew_section,
            "## Estimated Command-To-State Lag",
            lag_section,
            "## Warnings",
            warning_text,
        ]
    ) + "\n"


def main() -> int:
    args = parse_args()
    paths = expand_inputs(args.inputs)
    recordings = read_recordings(paths)

    topic_report = build_topic_report(recordings["topic_file_times"], args.gap_multiple)
    header_report = {
        topic: stats(values) for topic, values in sorted(recordings["header_lags_ms"].items())
    }
    lag_report = aggregate_lag_reports(
        [
            (
                lag_input,
                estimate_lags(
                    lag_input["command_series"],
                    lag_input["joint_observations"],
                    max_lag_ms=args.max_lag_ms,
                    sample_hz=args.lag_sample_hz,
                    min_correlation=args.min_correlation,
                ),
            )
            for lag_input in recordings["lag_inputs"]
        ]
    )

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "git_commit": git_commit(),
        "mcap_files": [str(path) for path in paths],
        "topics": topic_report,
        "header_lag_ms": header_report,
        "command_to_state_lag": lag_report,
        "read_errors": recordings["read_errors"],
        "warnings": build_warnings(topic_report) + recordings["read_errors"],
    }
    report = rounded(report)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "timing_report.json"
    md_path = args.output_dir / "timing_report.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    md_path.write_text(render_markdown(report))

    print(f"[OK] wrote {json_path}")
    print(f"[OK] wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
