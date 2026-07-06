"""mcap-valid: scan raw MCAP sessions for topic coverage/gap issues before conversion.

Config-free: which topics get analyzed, and what role each plays, is inferred entirely
from each topic's ROS2 message type (see core/quality.classify_topic) — no conversion
config is needed or accepted.

By default (no flags needed), a JSON report and a comprehensive Markdown
report covering every episode and topic are always written to
./mcap_valid_reports/<name>.{json,md}, in addition to whatever --format /
--output produce.

`--topic`/`--max-samples` fold in the old standalone `mcap-inspect` tool's deep
per-message field-structure dump for a single topic (opt-in, off by default).
"""

import argparse
import json
from pathlib import Path
from typing import List

from mcap.exceptions import McapError
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from mcap_converter.core.quality import (
    SEVERITY_CRITICAL,
    SEVERITY_OK,
    SEVERITY_WARNING,
    QualityThresholds,
    apply_batch_fps_check,
    apply_batch_topic_presence_check,
    scan_episode,
)
from mcap_converter.core.schema_inspect import inspect_message_structure, render_structure_text

console = Console()
# Status/progress notices (e.g. "report written to ...") go to stderr so they
# never pollute `--format json` stdout, which downstream tooling parses as
# pure JSON.
_status_console = Console(stderr=True)

_SEVERITY_COLOR = {SEVERITY_OK: "green", SEVERITY_WARNING: "yellow", SEVERITY_CRITICAL: "red"}


def _summary_line(reports) -> str:
    """Build the one-line episode-count summary shared by the table and Markdown report."""
    n_error = sum(1 for r in reports if r.read_error)
    n_ok = sum(1 for r in reports if not r.read_error and r.severity == SEVERITY_OK)
    n_warn = sum(1 for r in reports if not r.read_error and r.severity == SEVERITY_WARNING)
    n_crit = sum(1 for r in reports if not r.read_error and r.severity == SEVERITY_CRITICAL)
    return f"{len(reports)} episodes: {n_ok} ok, {n_warn} warning, {n_crit} critical" + (
        f", {n_error} unreadable" if n_error else ""
    )


def _render_topics_table(reports) -> None:
    """Print a baseline "what's in this file" table for one representative episode.

    Folds in the old `mcap-inspect` tool's topic-summary table: every topic present in
    the file (including role="unclassified" ones), regardless of severity. Uses the
    first readable report in the batch as the representative episode — real recordings
    in the same session all share the same topic layout, so any one of them is
    representative for this purpose.
    """
    representative = next((r for r in reports if r.read_error is None), None)
    if representative is None:
        return

    topics_table = Table(title=f"Topics in {Path(representative.path).name}")
    # no_wrap=True: topic/type strings are long and have no spaces to word-wrap
    # on, so they'd otherwise fold mid-word across lines on a narrow terminal.
    # Using the shared `console` (real terminal-aware width) instead of a
    # separate fixed-width Console keeps this table consistent with every
    # other table/panel in this command's output; no_wrap keeps names intact
    # by letting Rich shrink other columns instead.
    topics_table.add_column("Topic", no_wrap=True)
    topics_table.add_column("Type", no_wrap=True)
    topics_table.add_column("Messages", justify="right")
    topics_table.add_column("Role")
    for t in representative.topics:
        topics_table.add_row(t.topic, t.message_type or "-", str(t.message_count), t.role)
    console.print(topics_table)


def _render_table(reports, *, verbose: bool) -> None:
    _render_topics_table(reports)

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
        # escape(): labels like "action[left]" would otherwise be parsed as Rich
        # markup tags, silently dropping the bracketed arm suffix from the output.
        lines = [
            f"[{_SEVERITY_COLOR[t.severity]}]{escape(t.label)}[/{_SEVERITY_COLOR[t.severity]}]: {escape(t.reason)}"
            for t in flagged
        ]
        console.print(
            Panel(
                "\n".join(lines), title=Path(r.path).name, border_style=_SEVERITY_COLOR[r.severity]
            )
        )

    console.print(f"\n{_summary_line(reports)}")


def default_report_paths(input_path: Path) -> tuple[Path, Path]:
    """
    Compute the default (JSON, Markdown) report paths for an input path.

    Reports are always written to ./mcap_valid_reports/<name>.{json,md}
    (relative to the current working directory), named after the input:
    a directory's own name, or a single file's stem (extension stripped).
    """
    resolved = input_path.resolve()
    name = resolved.stem if resolved.is_file() else resolved.name
    report_dir = Path.cwd() / "mcap_valid_reports"
    return report_dir / f"{name}.json", report_dir / f"{name}.md"


def render_markdown_report(reports, *, input_path: str) -> str:
    """Render a comprehensive Markdown report listing every episode and topic.

    Unlike the terminal table (which hides healthy detail by default), this
    always lists every episode and every topic (including unclassified ones),
    regardless of severity.
    """
    summary = _summary_line(reports)

    lines = [
        "# mcap-valid Report",
        "",
        f"- Input: `{input_path}`",
        "",
        "## Summary",
        "",
        summary,
        "",
        "## Episodes",
        "",
    ]

    for r in reports:
        lines.append(f"### `{Path(r.path).name}` — {r.severity} (passed: {r.passed})")
        lines.append("")
        if r.read_error:
            lines.append(f"**Read error:** `{r.read_error}`")
            lines.append("")
            continue

        lines.append(f"- Path: `{r.path}`")
        lines.append(f"- Duration: `{r.duration_s:.1f}s`")
        lines.append("")
        lines.append(
            "| Topic | Label | Type | Role | Messages | Avg FPS | Coverage | Total Gap (s) | Longest Gap (s) | Severity | Reason |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for t in r.topics:
            avg_fps = f"{t.avg_fps:.2f}" if t.avg_fps is not None else "-"
            reason = t.reason.replace("|", "\\|")
            lines.append(
                f"| {t.topic} | {t.label} | {t.message_type or '-'} | {t.role} | {t.message_count} | {avg_fps} | "
                f"{t.coverage_ratio:.2f} | {t.total_gap_s:.2f} | {t.longest_gap_s:.2f} | "
                f"{t.severity} | {reason} |"
            )
        lines.append("")

    return "\n".join(lines)


def main(args: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Scan raw MCAP sessions for coverage/gap quality issues before conversion.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  mcap-valid -i data/raw/my-session   # no config needed - topics are auto-detected by message type
  mcap-valid -i recording.mcap --format json --output report.json
  mcap-valid -i data/raw/my-session --fail-on-critical   # CI gate, exit 1 on any critical episode
  mcap-valid -i recording.mcap --topic /joint_states     # deep field-structure dump for one topic

  (by default, a JSON + Markdown report is always written to ./mcap_valid_reports/<name>.{json,md})
""",
    )
    parser.add_argument(
        "-i", "--input", required=True, help="MCAP file or directory (recursive **/*.mcap)"
    )
    parser.add_argument("--format", choices=["table", "json"], default="table")
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "also write the report to this file, IN ADDITION to the default "
            "./mcap_valid_reports/<name>.{json,md} report files that are always written"
        ),
    )
    parser.add_argument("--stream-gap-factor", type=float, default=5.0)
    parser.add_argument("--stream-min-gap", type=float, default=0.5)
    parser.add_argument("--action-warn-gap", type=float, default=1.0)
    parser.add_argument("--fps-tolerance", type=float, default=0.15)
    parser.add_argument(
        "--fail-on-critical", action="store_true", help="exit 1 if any episode has a critical issue"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="show per-topic detail even for healthy episodes"
    )
    parser.add_argument(
        "--topic",
        default=None,
        help="deep field-structure dump for one topic (folds in the old mcap-inspect tool)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=5,
        help="max message samples for --topic field dump (default: 5)",
    )
    parsed = parser.parse_args(args)

    thresholds = QualityThresholds(
        stream_gap_factor=parsed.stream_gap_factor,
        stream_min_gap_s=parsed.stream_min_gap,
        action_warn_gap_s=parsed.action_warn_gap,
        fps_degradation_tolerance=parsed.fps_tolerance,
    )

    input_path = Path(parsed.input)
    if not input_path.exists():
        _status_console.print(f"[red]✗ input path does not exist: {input_path}[/red]")
        return 1

    mcap_files = [input_path] if input_path.is_file() else sorted(input_path.glob("**/*.mcap"))

    reports = [scan_episode(str(p), thresholds) for p in mcap_files]
    if len(reports) > 1:
        reports = apply_batch_fps_check(reports, thresholds)
        reports = apply_batch_topic_presence_check(reports)

    structure = None
    if parsed.topic and mcap_files:
        representative_file = mcap_files[0]
        try:
            structure = inspect_message_structure(
                str(representative_file), topic=parsed.topic, max_samples=parsed.max_samples
            )
        except (OSError, McapError) as exc:
            # inspect_message_structure() deliberately lets I/O/parse errors propagate
            # (see its docstring) rather than swallowing them like the old standalone
            # mcap-inspect tool did — so this CLI layer must catch them itself, the
            # same way scan_episode()'s callers rely on it to turn read errors into a
            # clean report instead of a crash.
            _status_console.print(
                f"[red]✗ failed to read {representative_file} for --topic: {exc}[/red]"
            )
            return 1

    default_json_path, default_md_path = default_report_paths(input_path)
    default_json_path.parent.mkdir(parents=True, exist_ok=True)
    # default_payload: always written to the default on-disk report path below.
    # It's always episodes-only — --topic's structure dump is a --format json/
    # --output-only convenience and has no place in the always-on disk report.
    default_payload = {"episodes": [r.to_dict() for r in reports]}
    payload_json = json.dumps(default_payload, indent=2)
    default_json_path.write_text(payload_json)
    default_md_path.write_text(render_markdown_report(reports, input_path=str(input_path)))
    _status_console.print(f"[dim]報告已寫入: {default_json_path}, {default_md_path}[/dim]")

    if parsed.format == "json":
        # json_payload: the ad-hoc `--format json` stdout/`--output` payload, derived
        # from default_payload but with `topic_structure` folded in when --topic was
        # given — kept separate from default_payload so the always-on disk report
        # above never accidentally gains an extra key based on this run's flags.
        json_payload = dict(default_payload)
        if structure is not None:
            json_payload["topic_structure"] = structure
        payload = json.dumps(json_payload, indent=2)
        if parsed.output:
            Path(parsed.output).write_text(payload)
        else:
            print(payload)
    else:
        _render_table(reports, verbose=parsed.verbose)
        if structure is not None:
            # escape(): field types like "List[float]" would otherwise be parsed as
            # Rich markup tags, silently dropping the "[float]" part from the output —
            # the same class of bug the "action[left]"/"action[right]" escaping above
            # guards against.
            console.print(escape(render_structure_text(structure)))
        if parsed.output:
            Path(parsed.output).write_text(payload_json)

    return 1 if (parsed.fail_on_critical and any(not r.passed for r in reports)) else 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
