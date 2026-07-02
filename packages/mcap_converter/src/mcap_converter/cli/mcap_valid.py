"""mcap-valid: scan raw MCAP sessions for topic coverage/gap issues before conversion."""

import argparse
import json
from pathlib import Path
from typing import List

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from mcap_converter.config.loader import ConfigLoader
from mcap_converter.core.quality import (
    SEVERITY_CRITICAL,
    SEVERITY_OK,
    SEVERITY_WARNING,
    QualityThresholds,
    apply_batch_fps_check,
    scan_episode,
)

console = Console()

_SEVERITY_COLOR = {SEVERITY_OK: "green", SEVERITY_WARNING: "yellow", SEVERITY_CRITICAL: "red"}


def _render_table(reports, *, verbose: bool) -> None:
    table = Table(title="mcap-valid report")
    table.add_column("Episode")
    table.add_column("Duration", justify="right")
    table.add_column("Status")

    for r in reports:
        if r.read_error:
            table.add_row(Path(r.path).name, "-", "[red]error[/red]")
            continue
        color = _SEVERITY_COLOR[r.severity]
        table.add_row(Path(r.path).name, f"{r.duration_s:.1f}s", f"[{color}]{r.severity}[/{color}]")

    console.print(table)

    for r in reports:
        if r.read_error:
            # File-level failure — topics is always empty here, so the
            # topic-based detail panel below would silently show nothing.
            # Print the read error explicitly regardless of --verbose.
            console.print(Panel(r.read_error, title=Path(r.path).name, border_style="red"))
            continue
        if r.severity == SEVERITY_OK and not verbose:
            continue
        flagged = [t for t in r.topics if t.severity != SEVERITY_OK] if not verbose else r.topics
        if not flagged:
            continue
        lines = [f"[{_SEVERITY_COLOR[t.severity]}]{t.label}[/{_SEVERITY_COLOR[t.severity]}]: {t.reason}" for t in flagged]
        console.print(Panel("\n".join(lines), title=Path(r.path).name, border_style=_SEVERITY_COLOR[r.severity]))

    n_error = sum(1 for r in reports if r.read_error)
    n_ok = sum(1 for r in reports if not r.read_error and r.severity == SEVERITY_OK)
    n_warn = sum(1 for r in reports if not r.read_error and r.severity == SEVERITY_WARNING)
    n_crit = sum(1 for r in reports if not r.read_error and r.severity == SEVERITY_CRITICAL)
    console.print(
        f"\n{len(reports)} episodes: {n_ok} ok, {n_warn} warning, {n_crit} critical"
        + (f", {n_error} unreadable" if n_error else "")
    )


def main(args: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Scan raw MCAP sessions for coverage/gap quality issues before conversion.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  mcap-valid -i data/raw/my-session --config configs/mcap_converter/openarm_bimanual_quest.yaml
  mcap-valid -i recording.mcap --format json --output report.json
  mcap-valid -i data/raw/my-session --fail-on-critical   # CI gate, exit 1 on any critical episode
""",
    )
    parser.add_argument("-i", "--input", required=True, help="MCAP file or directory (recursive **/*.mcap)")
    parser.add_argument("--config", default=None, help="conversion config YAML (same one used by mcap-convert)")
    parser.add_argument("--format", choices=["table", "json"], default="table")
    parser.add_argument("--output", default=None, help="write the report to this file instead of / in addition to stdout")
    parser.add_argument("--stream-gap-factor", type=float, default=5.0)
    parser.add_argument("--stream-min-gap", type=float, default=0.5)
    parser.add_argument("--action-warn-gap", type=float, default=1.0)
    parser.add_argument("--fps-tolerance", type=float, default=0.15)
    parser.add_argument("--fail-on-critical", action="store_true", help="exit 1 if any episode has a critical issue")
    parser.add_argument("--verbose", action="store_true", help="show per-topic detail even for healthy episodes")
    parsed = parser.parse_args(args)

    config = ConfigLoader.from_yaml(parsed.config) if parsed.config else ConfigLoader.get_default()
    thresholds = QualityThresholds(
        stream_gap_factor=parsed.stream_gap_factor,
        stream_min_gap_s=parsed.stream_min_gap,
        action_warn_gap_s=parsed.action_warn_gap,
        fps_degradation_tolerance=parsed.fps_tolerance,
    )

    input_path = Path(parsed.input)
    mcap_files = [input_path] if input_path.is_file() else sorted(input_path.glob("**/*.mcap"))

    reports = [scan_episode(str(p), config, thresholds) for p in mcap_files]
    if len(reports) > 1:
        reports = apply_batch_fps_check(reports, thresholds)

    if parsed.format == "json":
        payload = json.dumps({"episodes": [r.to_dict() for r in reports]}, indent=2)
        if parsed.output:
            Path(parsed.output).write_text(payload)
        else:
            print(payload)
    else:
        _render_table(reports, verbose=parsed.verbose)
        if parsed.output:
            Path(parsed.output).write_text(json.dumps({"episodes": [r.to_dict() for r in reports]}, indent=2))

    return 1 if (parsed.fail_on_critical and any(not r.passed for r in reports)) else 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
