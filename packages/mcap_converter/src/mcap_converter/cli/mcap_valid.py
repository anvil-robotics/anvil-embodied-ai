"""mcap-valid: scan raw MCAP sessions for topic coverage/gap issues before conversion.

Config-free: which topics get analyzed, and what role each plays, is inferred entirely
from each topic's ROS2 message type (see core/quality.classify_topic) — no conversion
config is needed or accepted.

By default (no flags needed), a JSON report and a comprehensive Markdown
report covering every episode and topic are always written inside the input
itself, at <input>/mcap_valid_reports/report.{json,md} (or
<input's-parent>/mcap_valid_reports/<stem>/report.{json,md} for a single
file), in addition to whatever --format / --output produce.

`--topic`/`--max-samples` fold in the old standalone `mcap-inspect` tool's deep
per-message field-structure dump for a single topic (opt-in, off by default).
"""

import argparse
import json
import statistics
from pathlib import Path
from typing import Dict, List, Tuple

from mcap.exceptions import McapError
from rich.console import Console
from rich.markup import escape
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TimeElapsedColumn
from rich.table import Table

from mcap_converter.core.quality import (
    ROLE_ACTION,
    ROLE_STREAM,
    ROLE_UNCLASSIFIED,
    SEVERITY_CRITICAL,
    SEVERITY_PASS,
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

_SEVERITY_COLOR = {SEVERITY_PASS: "green", SEVERITY_WARNING: "yellow", SEVERITY_CRITICAL: "red"}
_SEVERITY_ICON = {SEVERITY_PASS: "🟢", SEVERITY_WARNING: "🟡", SEVERITY_CRITICAL: "🔴"}
# Row-grouping order for the Topic Health Overview table: stream topics first,
# then action, then unclassified — reads more naturally than the flat
# alphabetical-by-topic-path order the per-episode tables use.
_ROW_ROLE_ORDER = {ROLE_STREAM: 0, ROLE_ACTION: 1, ROLE_UNCLASSIFIED: 2}
# Recurring-issue grouping order for Flagged Topics: critical groups before warning.
_FLAG_SEVERITY_ORDER = {SEVERITY_CRITICAL: 0, SEVERITY_WARNING: 1}
# Cap on how many episode filenames get listed inline in a report bullet —
# beyond this, list the rest by count instead so a batch with hundreds of
# flagged/unreadable episodes doesn't produce one unreadable line (or,
# pre-this-cap, hundreds of individual bullets).
_MAX_NAMES_SHOWN = 20


def _truncate_names(names: List[str], limit: int = _MAX_NAMES_SHOWN) -> str:
    """Comma-join up to `limit` names; beyond that, summarize the rest by count
    instead of listing every one, so a report with hundreds of flagged
    episodes stays readable instead of producing one enormous line."""
    if len(names) <= limit:
        return ", ".join(names)
    shown = ", ".join(names[:limit])
    return f"{shown}, ... and {len(names) - limit} more (see the JSON report for the full list)"


def _flagged_topics(r) -> List:
    """This episode's non-pass topics — the shared "what's actually flagged" predicate
    used to build the Reason cell/column in both the terminal table and the
    Markdown per-episode overview table, so the two never drift apart."""
    return [t for t in r.topics if t.severity != SEVERITY_PASS]


def _summary_line(reports) -> str:
    """Build the one-line episode-count summary shared by the table and Markdown report."""
    n_error = sum(1 for r in reports if r.read_error)
    n_pass = sum(1 for r in reports if not r.read_error and r.severity == SEVERITY_PASS)
    n_warn = sum(1 for r in reports if not r.read_error and r.severity == SEVERITY_WARNING)
    n_crit = sum(1 for r in reports if not r.read_error and r.severity == SEVERITY_CRITICAL)
    return f"{len(reports)} episodes: {n_pass} pass, {n_warn} warning, {n_crit} critical" + (
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
    table.add_column("Reason")

    for r in reports:
        if r.read_error:
            # File-level failure — topics is always empty here, so the
            # topic-selection logic below would silently show nothing. Show
            # the read error itself as the Reason, regardless of --verbose.
            table.add_row(Path(r.path).name, "-", "[red]error[/red]", escape(r.read_error))
            continue
        color = _SEVERITY_COLOR[r.severity]
        flagged = r.topics if verbose else _flagged_topics(r)
        if flagged:
            # escape(): labels like "action[left]" would otherwise be parsed as Rich
            # markup tags, silently dropping the bracketed arm suffix from the output.
            # A literal "\n" is fine here (unlike the Markdown table's cells) since
            # Rich table cells natively support multi-line wrapped content.
            reason = "\n".join(
                f"[{_SEVERITY_COLOR[t.severity]}]{escape(t.label)}[/{_SEVERITY_COLOR[t.severity]}]: "
                f"{escape(t.reason)}"
                for t in flagged
            )
        else:
            reason = "-"
        table.add_row(
            Path(r.path).name, f"{r.duration_s:.1f}s", f"[{color}]{r.severity}[/{color}]", reason
        )

    console.print(table)

    console.print(f"\n{_summary_line(reports)}")


def default_report_paths(input_path: Path) -> tuple[Path, Path]:
    """
    Compute the default (JSON, Markdown) report paths for an input path.

    Reports always live inside the input's own location, never relative to
    the current working directory — this is what lets a session directory
    carry its own report and lets `mcap-convert`'s auto-discovery find it
    regardless of where either command was invoked from.

    - Directory input (the normal case, a session directory): reports go to
      <resolved_input_dir>/mcap_valid_reports/report.{json,md}. No extra
      per-name subfolder is needed here — the directory itself is already
      the namespace.
    - Single-file input: reports go to
      <file's_parent_dir>/mcap_valid_reports/<file_stem>/report.{json,md} —
      the per-file subfolder is kept because multiple files can share the
      same parent directory and would otherwise collide.
    """
    resolved = input_path.resolve()
    if resolved.is_file():
        report_dir = resolved.parent / "mcap_valid_reports" / resolved.stem
    else:
        report_dir = resolved / "mcap_valid_reports"
    return report_dir / "report.json", report_dir / "report.md"


def _readable_reports(reports) -> List:
    """Episodes that were actually readable (read_error is None).

    read_error episodes have topics=[] and contribute nothing to per-topic
    comparisons or recurring-issue grouping — they still get their own row
    in the per-episode overview table, with their read_error text as Reason.
    """
    return [r for r in reports if r.read_error is None]


def _ordered_labels(readable) -> List[str]:
    """Distinct topic labels across readable episodes (union, not intersection),
    grouped by role (stream, then action, then unclassified) and alphabetical
    within each group — see _ROW_ROLE_ORDER."""
    role_by_label: Dict[str, str] = {}
    for r in readable:
        for t in r.topics:
            role_by_label.setdefault(t.label, t.role)
    return sorted(
        role_by_label, key=lambda label: (_ROW_ROLE_ORDER.get(role_by_label[label], 99), label)
    )


def _episode_overview_table_lines(reports) -> List[str]:
    """One row per episode (including unreadable ones): Episode/Duration/Status/Reason.

    This is what "merges the warning message into the report table" for the
    Markdown report: instead of a separate per-episode-column matrix (which
    broke down at high episode counts, since columns don't scroll) or a
    scattered Batch-Overview/Conclusion split, every episode gets exactly one
    row here, with its non-pass topics folded into a single Reason cell. Rows
    scale fine no matter how many episodes there are, so this is never
    truncated the way _truncate_names's comma-lists are.
    """
    lines = [
        "| Episode | Duration | Status | Reason |",
        "|---|---|---|---|",
    ]
    for r in reports:
        name = Path(r.path).name
        if r.read_error:
            reason = r.read_error.replace("|", "\\|")
            lines.append(f"| {name} | - | error | {reason} |")
            continue
        flagged = _flagged_topics(r)
        if flagged:
            reason = "; ".join(f"{t.label}: {t.reason}" for t in flagged).replace("|", "\\|")
        else:
            reason = "-"
        lines.append(f"| {name} | {r.duration_s:.1f}s | {r.severity} | {reason} |")
    lines.append("")
    return lines


def _topic_health_table_lines(readable) -> List[str]:
    """Single fixed-8-column table, one row per distinct topic label.

    Size depends only on topic count (small and fixed for a given
    robot/config), never on episode count — this is what actually fixes the
    500-episode-columns scalability bug. Every label gets a row, including
    ones with no numeric fps anywhere (rendered as `-`, not omitted).
    """
    labels = _ordered_labels(readable)
    type_by_label: Dict[str, str] = {}
    severity_counts_by_label: Dict[str, Dict[str, int]] = {}
    fps_by_label: Dict[str, List[float]] = {}
    for r in readable:
        for t in r.topics:
            # First-seen message_type per label wins — a label's type should
            # be consistent across episodes in practice, same "first
            # representative" convention _flagged_topic_groups uses for reasons.
            type_by_label.setdefault(t.label, t.message_type)
            counts = severity_counts_by_label.setdefault(
                t.label, {SEVERITY_PASS: 0, SEVERITY_WARNING: 0, SEVERITY_CRITICAL: 0}
            )
            # Direct indexing (not .get(..., 0)): counts is pre-seeded with
            # all 3 valid severities, so an unexpected severity value should
            # raise here rather than silently vanish from the rendered row.
            counts[t.severity] += 1
            if t.avg_fps is not None:
                fps_by_label.setdefault(t.label, []).append(t.avg_fps)

    lines = [
        "### Topic Health Overview",
        "",
        "| Topic | Type | Pass | Warning | Critical | Min FPS | Median FPS | Max FPS |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for label in labels:
        counts = severity_counts_by_label.get(label, {})
        fps_values = fps_by_label.get(label)
        if fps_values:
            min_fps = f"{min(fps_values):.2f}"
            max_fps = f"{max(fps_values):.2f}"
            median_fps = f"{statistics.median(fps_values):.2f}"
        else:
            min_fps = median_fps = max_fps = "-"
        lines.append(
            f"| {label} | {type_by_label.get(label) or '-'} | {counts.get(SEVERITY_PASS, 0)} | "
            f"{counts.get(SEVERITY_WARNING, 0)} | {counts.get(SEVERITY_CRITICAL, 0)} | "
            f"{min_fps} | {median_fps} | {max_fps} |"
        )
    lines.append("")
    return lines


def _flagged_topic_groups(
    readable,
) -> List[Tuple[Tuple[str, str], Tuple[List[str], str]]]:
    """Group non-pass (label, severity) pairs across readable episodes' topics.

    Rule-based only (no LLM / free-text generation): each group carries the
    count + list of episodes it occurs in, plus ONE verbatim representative
    reason string (from the first episode encountered) — never a paraphrase
    or merge of multiple episodes' reasons. Sorted critical-before-warning,
    then alphabetically by label within the same severity.
    """
    episodes_by_key: Dict[Tuple[str, str], List[str]] = {}
    reason_by_key: Dict[Tuple[str, str], str] = {}
    for r in readable:
        name = Path(r.path).name
        seen_in_episode = set()
        for t in r.topics:
            if t.severity == SEVERITY_PASS:
                continue
            key = (t.label, t.severity)
            if key in seen_in_episode:
                continue
            seen_in_episode.add(key)
            episodes_by_key.setdefault(key, []).append(name)
            reason_by_key.setdefault(key, t.reason)

    ordered_keys = sorted(episodes_by_key, key=lambda k: (_FLAG_SEVERITY_ORDER.get(k[1], 99), k[0]))
    return [(key, (episodes_by_key[key], reason_by_key[key])) for key in ordered_keys]


def _readiness_verdict_lines(reports, readable) -> List[str]:
    """Rule-based readiness verdict: ❌ critical / ⚠️ warning / ✅ all-clean, or the
    "no episodes found" edge case for an empty batch.

    Unreadable episodes already carry severity=SEVERITY_CRITICAL upstream (see
    scan_episode's read_error branch) but are excluded from `readable`, so
    they're added back in explicitly here for the critical count.
    """
    unreadable = [r for r in reports if r.read_error is not None]
    total = len(reports)
    n_critical = sum(1 for r in readable if r.severity == SEVERITY_CRITICAL) + len(unreadable)
    n_warning = sum(1 for r in readable if r.severity == SEVERITY_WARNING)

    if total == 0:
        verdict = "**No episodes found** — nothing was scanned, nothing to convert."
    elif n_critical:
        verdict = (
            f"**❌ Not ready to convert** — {n_critical}/{total} episode(s) flagged critical. "
            "Fix the underlying recordings, or note that `mcap-convert` skips them "
            "automatically by default (pass `--include-flagged critical` to convert them anyway)."
        )
    elif n_warning:
        verdict = (
            f"**⚠️ Ready to convert with warnings** — {n_warning}/{total} episode(s) have "
            "non-blocking issues (see below). Warnings never block `mcap-convert` unless you "
            "explicitly pass `--include-flagged pass`."
        )
    else:
        verdict = "**✅ All episodes clean** — ready to convert."
    return [verdict, ""]


def _flagged_topics_lines(readable) -> List[str]:
    """### Flagged Topics: recurring (label, severity) issues across readable episodes,
    sorted critical-before-warning. Empty (no lines at all) when nothing is flagged.
    """
    groups = _flagged_topic_groups(readable)
    if not groups:
        return []

    lines = ["### Flagged Topics", ""]
    total_readable = len(readable)
    for (label, severity), (episodes, reason) in groups:
        icon = _SEVERITY_ICON[severity]
        ep_list = _truncate_names(episodes)
        lines.append(
            f"- {icon} **{label}**: {severity} in {len(episodes)}/{total_readable} "
            f'episode(s) ({ep_list}) — e.g. "{reason}"'
        )
    lines.append("")
    return lines


def render_markdown_report(reports, *, input_path: str) -> str:
    """Render a comprehensive Markdown report listing every episode and topic.

    Everything comprehensive lives up front under `## Summary`: the stats
    line, a per-episode overview table (Episode/Duration/Status/Reason — one
    row per episode, including unreadable ones, with non-pass topics folded
    into the Reason cell), the rule-based readiness verdict, `### Flagged
    Topics` (recurring cross-episode issues), and `### Topic Health Overview`
    (a single fixed-column table, batch-only, omitted with fewer than 2
    readable episodes since a single episode has nothing to compare against).
    Only the exhaustive per-topic detail per episode (`## Episodes`) stays at
    the bottom as the drill-down/raw-data section — unlike the terminal table
    (which hides healthy detail by default), it always lists every episode
    and every topic (including unclassified ones), regardless of severity.
    """
    readable = _readable_reports(reports)
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
    ]
    lines.extend(_episode_overview_table_lines(reports))
    lines.extend(_readiness_verdict_lines(reports, readable))
    lines.extend(_flagged_topics_lines(readable))
    if len(readable) >= 2:
        lines.extend(_topic_health_table_lines(readable))
    lines.append("## Episodes")
    lines.append("")

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

  (by default, a JSON + Markdown report is always written inside the input, at
   <input>/mcap_valid_reports/report.{json,md})
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
            "<input>/mcap_valid_reports/report.{json,md} report files that are always written"
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

    if len(mcap_files) > 1:
        # Rendered to _status_console (stderr), never `console` (stdout) — scan_episode()
        # does a real full-message decode for stream/action/unclassified topics, so a
        # large batch can take a while, but --format json's stdout must stay pure,
        # parseable JSON for downstream tooling.
        with Progress(
            SpinnerColumn(),
            "[progress.description]{task.description}",
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=_status_console,
        ) as progress:
            scan_task = progress.add_task("[bold blue]Scanning episodes", total=len(mcap_files))
            reports = []
            for p in mcap_files:
                reports.append(scan_episode(str(p), thresholds))
                progress.advance(scan_task)
    else:
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
