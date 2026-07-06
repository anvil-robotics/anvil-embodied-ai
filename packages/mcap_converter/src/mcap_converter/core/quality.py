"""Pre-conversion quality scanning for raw MCAP sessions.

Splits into two layers, mirroring the pattern used for action forward-fill
in extractor.py:
  - Pure analysis functions (analyze_topic_coverage, detect_fps_degradation,
    apply_batch_fps_check, apply_batch_topic_presence_check, classify_topic,
    classify_topics, worst_severity) that take already-extracted data and
    are fully unit-testable without any MCAP file I/O.
  - A thin I/O adapter (scan_episode, and its helpers) that reads a real
    MCAP file and feeds the pure functions.

Topic monitoring is config-free: which topics get analyzed, and what role
each plays (camera/joint stream vs. action command), is inferred entirely
from the ROS2 message type recorded in the MCAP's own schema metadata (see
classify_topic). No DataConfig is consulted anywhere in this module.
"""

import re
import statistics
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from mcap.exceptions import McapError
from mcap.reader import make_reader as make_mcap_reader

from .extractor import message_timestamp
from .reader import McapReader

SEVERITY_PASS = "pass"
SEVERITY_WARNING = "warning"
SEVERITY_CRITICAL = "critical"
_SEVERITY_RANK = {SEVERITY_PASS: 0, SEVERITY_WARNING: 1, SEVERITY_CRITICAL: 2}

ROLE_STREAM = "stream"
ROLE_ACTION = "action"
ROLE_UNCLASSIFIED = "unclassified"

_SCHEMA_JOINT_STATE = "sensor_msgs/JointState"
_SCHEMA_COMPRESSED_IMAGE = "sensor_msgs/CompressedImage"
_SCHEMA_IMAGE = "sensor_msgs/Image"
_SCHEMA_FLOAT64_MULTIARR = "std_msgs/Float64MultiArray"

_CAMERA_LABEL_RE = re.compile(r"^/cam_(?P<name>[^/]+)/image_raw")
# Anchored (fullmatch-style) rather than a bare .search() substring match, so a topic that
# merely happens to CONTAIN this pattern somewhere in the middle of a longer/nested path
# (e.g. a future multi-robot namespace like "/robot2/follower_l_x_controller/commands/debug")
# can't be misclassified as an action topic. No real topic in this repo is nested/suffixed
# like that today, but anchoring costs nothing and removes the risk entirely.
_ARM_SIDE_RE = re.compile(r"^/follower_(?P<side>[lr])_.*controller/commands$")
_ARM_SIDE_TO_LABEL = {"l": "left", "r": "right"}


def worst_severity(severities: Iterable[str]) -> str:
    """Return the most severe value in severities, defaulting to PASS if empty."""
    worst = SEVERITY_PASS
    for sev in severities:
        if _SEVERITY_RANK[sev] > _SEVERITY_RANK[worst]:
            worst = sev
    return worst


@dataclass
class GapInterval:
    """A single interval where a topic went quiet longer than expected."""

    start_s: float
    end_s: float
    duration_s: float
    kind: str  # "dropframe" | "idle" | "leading" | "trailing"


@dataclass
class TopicQualityReport:
    """Coverage/gap analysis result for one topic in one episode."""

    topic: str
    label: str
    role: str  # "stream" | "action" | "unclassified"
    message_count: int
    avg_fps: Optional[float]  # only meaningful for role="stream"; None for "action"
    coverage_ratio: float
    total_gap_s: float
    longest_gap_s: float
    gaps: List[GapInterval] = field(default_factory=list)
    severity: str = SEVERITY_PASS
    reason: str = ""
    message_type: Optional[str] = None  # normalized ROS2 schema name, e.g. "sensor_msgs/JointState"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EpisodeQualityReport:
    """Aggregated quality report for one MCAP episode across all monitored topics."""

    path: str  # str(Path(mcap_path).resolve())
    duration_s: float
    severity: str
    passed: bool
    topics: List[TopicQualityReport] = field(default_factory=list)
    read_error: Optional[str] = None  # set when the file itself couldn't be read/parsed

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class QualityThresholds:
    """Tunable thresholds for the quality analysis. All CLI-overridable."""

    stream_gap_factor: float = 5.0
    stream_min_gap_s: float = 0.5
    action_warn_gap_s: float = 1.0
    fps_degradation_tolerance: float = 0.15


@dataclass
class MonitoredTopic:
    """A single topic to check, classified purely from its ROS2 message type."""

    topic: str
    label: str
    role: str  # "stream" | "action" | "unclassified"
    message_type: Optional[str] = None


def _normalize_schema_name(name: Optional[str]) -> Optional[str]:
    """Normalize a ROS2 schema/type name across recorder variants.

    Some recorders/readers report e.g. "sensor_msgs/msg/JointState" (with a /msg/ infix)
    or a "ros2msg://" URI prefix instead of the bare "sensor_msgs/JointState" form. Strip
    both so classification is robust to either convention.
    """
    if name is None:
        return None
    return name.replace("ros2msg://", "").replace("/msg/", "/")


def _fallback_label(topic: str) -> str:
    """Deterministic, human-readable label when no topic-name pattern matches."""
    return topic.lstrip("/").replace("/", "_")


def classify_topic(topic: str, schema_name: Optional[str]) -> MonitoredTopic:
    """Map one (topic, message-type) pair to a MonitoredTopic (role + label).

    Driven purely by the message type recorded in the MCAP's own schema metadata — this
    reproduces the same classification every real conversion config in this repo would have
    produced via explicit config, for every real recording. A topic whose message type isn't
    one of the 3 known robot-pipeline types is returned with role=ROLE_UNCLASSIFIED
    (informational only — never dropped, never crashes, never affects episode severity).
    """
    norm = _normalize_schema_name(schema_name)
    if norm == _SCHEMA_JOINT_STATE:
        return MonitoredTopic(topic, label="joint_states", role=ROLE_STREAM, message_type=norm)
    if norm in (_SCHEMA_COMPRESSED_IMAGE, _SCHEMA_IMAGE):
        m = _CAMERA_LABEL_RE.match(topic)
        label = m.group("name") if m else _fallback_label(topic)
        return MonitoredTopic(topic, label=label, role=ROLE_STREAM, message_type=norm)
    if norm == _SCHEMA_FLOAT64_MULTIARR:
        m = _ARM_SIDE_RE.match(topic)
        arm = _ARM_SIDE_TO_LABEL.get(m.group("side")) if m else None
        label = f"action[{arm}]" if arm else f"action[{_fallback_label(topic)}]"
        return MonitoredTopic(topic, label=label, role=ROLE_ACTION, message_type=norm)
    return MonitoredTopic(
        topic, label=_fallback_label(topic), role=ROLE_UNCLASSIFIED, message_type=norm
    )


def classify_topics(topic_schemas: Dict[str, Optional[str]]) -> List[MonitoredTopic]:
    """Classify every present topic (sorted by topic name for a deterministic report order)."""
    return [classify_topic(t, s) for t, s in sorted(topic_schemas.items())]


def analyze_topic_coverage(
    timestamps: List[float],
    session_start: float,
    session_end: float,
    *,
    topic: str,
    label: str,
    role: str,
    thresholds: QualityThresholds,
    action_from_observation: bool = False,
) -> TopicQualityReport:
    """
    Analyze one topic's coverage within one episode.

    Fallback/severity rules (see design doc for rationale):
    - role="stream" (camera / joint_states): zero messages, a single message,
      or any gap (mid-stream / leading / trailing) beyond threshold -> CRITICAL.
      These topics should be dense and continuous; any interruption is a real
      recording problem.
    - role="action": zero messages -> WARNING unless action_from_observation
      is True (then OK) — a silent arm could legitimately be an unused arm in
      a single-arm task, not a recording bug. Idle gaps mid-episode -> WARNING,
      never CRITICAL (TASK-001 confirmed idle arms are normal teleop behavior).
      Leading/trailing gaps are not flagged for action topics at all — an arm
      simply not yet engaged at the start, or already released at the end, is
      exactly the same normal idle behavior as a mid-episode gap.

    Note: `avg_fps` on the returned report is only computed for role="stream"
    (a fixed publish rate makes an average meaningful); it is always None for
    role="action", whose event-driven timing has no single "rate" to average.
    """
    span = max(session_end - session_start, 1e-9)

    if len(timestamps) == 0:
        if role == "action" and action_from_observation:
            severity, reason = (
                SEVERITY_PASS,
                "action topic 零訊息（action_from_observation=true，可接受）",
            )
        elif role == "action":
            severity, reason = (
                SEVERITY_WARNING,
                "action topic 全程零訊息，可能為單手任務或設定不符，請人工確認",
            )
        else:
            severity, reason = SEVERITY_CRITICAL, "stream topic 零訊息，完全沒錄到"
        return TopicQualityReport(
            topic=topic,
            label=label,
            role=role,
            message_count=0,
            avg_fps=None,
            coverage_ratio=0.0,
            total_gap_s=span,
            longest_gap_s=span,
            gaps=[],
            severity=severity,
            reason=reason,
        )

    ts = sorted(timestamps)
    avg_fps: Optional[float] = None
    gaps: List[GapInterval] = []

    if role == "stream":
        if len(ts) == 1:
            return TopicQualityReport(
                topic=topic,
                label=label,
                role=role,
                message_count=1,
                avg_fps=None,
                coverage_ratio=0.0,
                total_gap_s=span,
                longest_gap_s=span,
                gaps=[],
                severity=SEVERITY_CRITICAL,
                reason="stream 僅 1 則訊息，幾乎沒錄到",
            )
        avg_fps = (len(ts) - 1) / (ts[-1] - ts[0]) if ts[-1] > ts[0] else None
        intervals = [b - a for a, b in zip(ts, ts[1:])]
        median = statistics.median(intervals)
        drop_threshold = max(thresholds.stream_min_gap_s, thresholds.stream_gap_factor * median)

        for a, b, iv in zip(ts, ts[1:], intervals):
            if iv > drop_threshold:
                gaps.append(GapInterval(a - session_start, b - session_start, iv, "dropframe"))

        leading = ts[0] - session_start
        if leading > drop_threshold:
            gaps.append(GapInterval(0.0, leading, leading, "leading"))

        trailing = session_end - ts[-1]
        if trailing > drop_threshold:
            gaps.append(GapInterval(ts[-1] - session_start, span, trailing, "trailing"))

        if gaps:
            severity = SEVERITY_CRITICAL
            reason = f"{len(gaps)} 個異常斷點，最長 {max(g.duration_s for g in gaps):.2f}s"
        else:
            severity, reason = SEVERITY_PASS, "PASS"

    else:  # role == "action"
        intervals = [b - a for a, b in zip(ts, ts[1:])]
        for a, b, iv in zip(ts, ts[1:], intervals):
            if iv > thresholds.action_warn_gap_s:
                gaps.append(GapInterval(a - session_start, b - session_start, iv, "idle"))

        if gaps:
            severity = SEVERITY_WARNING
            longest = max(gaps, key=lambda g: g.duration_s)
            reason = (
                f"{len(gaps)} 個 idle gap，範圍 {longest.start_s:.2f}s~{longest.end_s:.2f}s"
                f"（持續 {longest.duration_s:.2f}s，正常，手臂未操作）"
            )
        else:
            severity, reason = SEVERITY_PASS, "PASS"

    total_gap = sum(g.duration_s for g in gaps)
    longest_gap = max((g.duration_s for g in gaps), default=0.0)
    coverage = max(0.0, 1.0 - total_gap / span)

    return TopicQualityReport(
        topic=topic,
        label=label,
        role=role,
        message_count=len(ts),
        avg_fps=avg_fps,
        coverage_ratio=coverage,
        total_gap_s=total_gap,
        longest_gap_s=longest_gap,
        gaps=gaps,
        severity=severity,
        reason=reason,
    )


def detect_fps_degradation(
    episode_fps: Dict[str, float],
    thresholds: QualityThresholds,
) -> Dict[str, Tuple[bool, str]]:
    """
    Compare each episode's fps for one topic against the median of the
    OTHER episodes in the batch (leave-one-out).

    Uses the median (not max) as the reference so a single noisy high outlier
    doesn't set an unreachable bar for the rest of the batch. Leave-one-out
    (excluding the episode under test from its own reference) matters most
    for small batches: an inclusive median gets pulled toward the very
    episode it's supposed to judge, which can mask real degradation. A topic
    present in only one episode of the batch (no other episode has a
    measurement to compare against) is treated as not degraded rather than
    raising an error.
    """
    if not episode_fps:
        return {}
    result = {}
    for path, fps in episode_fps.items():
        others = [v for p, v in episode_fps.items() if p != path]
        if not others:
            result[path] = (False, "")
            continue
        reference = statistics.median(others)
        if fps < reference * (1 - thresholds.fps_degradation_tolerance):
            reason = f"fps 退化：本集 {fps:.1f}fps vs 同批中位數 {reference:.1f}fps"
            result[path] = (True, reason)
        else:
            result[path] = (False, "")
    return result


def apply_batch_fps_check(
    reports: List[EpisodeQualityReport],
    thresholds: QualityThresholds,
) -> List[EpisodeQualityReport]:
    """
    Cross-episode pass: detect stream topics whose fps has degraded relative
    to the rest of the batch, and upgrade PASS -> WARNING for those topics.
    Never downgrades an existing CRITICAL/WARNING severity.
    """
    # Group avg_fps by (topic, label) across all episodes that have it.
    by_key: Dict[Tuple[str, str], Dict[str, float]] = {}
    for ep in reports:
        for t in ep.topics:
            if t.role == "stream" and t.avg_fps is not None:
                by_key.setdefault((t.topic, t.label), {})[ep.path] = t.avg_fps

    degraded_by_path_and_key: Dict[Tuple[str, str, str], str] = {}
    for key, episode_fps in by_key.items():
        for path, (is_degraded, reason) in detect_fps_degradation(episode_fps, thresholds).items():
            if is_degraded:
                degraded_by_path_and_key[(path, *key)] = reason

    updated_reports = []
    for ep in reports:
        new_topics = []
        for t in ep.topics:
            reason_key = (ep.path, t.topic, t.label)
            if reason_key in degraded_by_path_and_key and t.severity == SEVERITY_PASS:
                new_topics.append(
                    replace(
                        t,
                        severity=SEVERITY_WARNING,
                        reason=f"{t.reason}; {degraded_by_path_and_key[reason_key]}".strip("; "),
                    )
                )
            else:
                new_topics.append(t)
        new_severity = worst_severity(t.severity for t in new_topics)
        updated_reports.append(
            replace(
                ep,
                severity=new_severity,
                passed=(new_severity != SEVERITY_CRITICAL),
                topics=new_topics,
            )
        )
    return updated_reports


def apply_batch_topic_presence_check(
    reports: List[EpisodeQualityReport],
) -> List[EpisodeQualityReport]:
    """
    Config-free auto-detection can only classify topics that actually exist as declared
    channels in a file — if an entire stream/camera never published anything (no channel
    declared at all, e.g. a camera driver never started), it's simply absent from that
    episode's topic list rather than flagged, since there's no config declaring "this
    topic should exist."

    This function closes that gap using cross-episode evidence within the same batch: any
    (topic, role) pair present in a MAJORITY of the batch's episodes is treated as
    "expected." Any episode missing an expected pair gets a synthesized CRITICAL
    "topic completely absent" entry appended to its topic list, and its overall severity
    is recomputed. Pure function — no I/O. No-ops for a single-episode batch (nothing to
    compare against).

    Design rationale: majority quorum (not "any episode has it") avoids one episode's
    stray/noise topic making every OTHER episode look like it's "missing" something that
    was never actually expected; and avoids false positives for topics that legitimately
    only appear in a minority of episodes (e.g. some rare event-log topic). This does NOT
    replace what a config could express (e.g. "this task always has exactly 4 cameras") —
    it only catches within-batch inconsistency, but that's sufficient for the most common
    real failure mode (a driver completely failing for one episode). Single-episode scans
    (`-i` pointing at one file) have no batch to compare against, so this mechanism
    naturally doesn't apply there.
    """
    if len(reports) <= 1:
        return reports

    valid_reports = [r for r in reports if r.read_error is None]
    if len(valid_reports) <= 1:
        return reports

    presence_count: Dict[Tuple[str, str], int] = {}
    topic_role_label: Dict[Tuple[str, str], str] = {}
    for r in valid_reports:
        for t in r.topics:
            if t.role in (ROLE_STREAM, ROLE_ACTION):
                key = (t.topic, t.role)
                presence_count[key] = presence_count.get(key, 0) + 1
                topic_role_label.setdefault(key, t.label)

    quorum = len(valid_reports) // 2 + 1  # strict majority
    expected = {key for key, cnt in presence_count.items() if cnt >= quorum}

    new_reports = []
    for r in reports:
        if r.read_error is not None:
            new_reports.append(r)
            continue
        present_keys = {(t.topic, t.role) for t in r.topics}
        missing = expected - present_keys
        if not missing:
            new_reports.append(r)
            continue
        extra_topics = list(r.topics)
        for topic, role in sorted(missing):
            extra_topics.append(
                TopicQualityReport(
                    topic=topic,
                    label=topic_role_label[(topic, role)],
                    role=role,
                    message_count=0,
                    avg_fps=None,
                    coverage_ratio=0.0,
                    total_gap_s=0.0,
                    longest_gap_s=0.0,
                    gaps=[],
                    severity=SEVERITY_CRITICAL,
                    reason=(
                        f"topic completely absent (present in {presence_count[(topic, role)]}/"
                        f"{len(valid_reports)} sibling episodes in this batch)"
                    ),
                    message_type=None,
                )
            )
        new_severity = worst_severity(t.severity for t in extra_topics)
        new_reports.append(
            replace(
                r,
                topics=extra_topics,
                severity=new_severity,
                passed=(new_severity != SEVERITY_CRITICAL),
            )
        )
    return new_reports


@dataclass
class _TopicSummary:
    """Per-topic (message_count, schema_name) as read from the MCAP footer summary."""

    count: int
    schema_name: Optional[str]  # normalized, e.g. "sensor_msgs/JointState"; None if no schema


def _summary_topic_info(mcap_path: str) -> Dict[str, _TopicSummary]:
    """Per-topic (message_count, schema_name) from the MCAP footer summary (O(1), no full scan).

    Iterates ALL declared channels (summary.channels), not just those with nonzero message
    counts, so a declared-but-silent channel still surfaces (count 0) and gets correctly
    classified/analyzed for its own zero-message severity — an improvement over the prior
    implementation, which iterated summary.statistics.channel_message_counts and so silently
    dropped any channel that recorded zero messages.

    Raises OSError / McapError on unreadable/unparseable files — callers must catch.
    """
    with open(mcap_path, "rb") as f:
        summary = make_mcap_reader(f).get_summary()
    if summary is None:
        return {}
    counts_by_cid = summary.statistics.channel_message_counts if summary.statistics else {}
    out: Dict[str, _TopicSummary] = {}
    for cid, ch in summary.channels.items():
        schema = summary.schemas.get(ch.schema_id) if ch.schema_id else None
        schema_name = _normalize_schema_name(schema.name) if schema else None
        cnt = counts_by_cid.get(cid, 0)
        if ch.topic in out:
            # >1 channel publishing to the same topic name (rare): sum counts, keep first schema
            prev = out[ch.topic]
            out[ch.topic] = _TopicSummary(prev.count + cnt, prev.schema_name or schema_name)
        else:
            out[ch.topic] = _TopicSummary(cnt, schema_name)
    return out


def _collect_timestamps(mcap_path: str, topics: List[str]) -> Dict[str, List[float]]:
    """Single-pass scan collecting message_timestamp() for each requested topic."""
    reader = McapReader(mcap_path)
    out: Dict[str, List[float]] = {t: [] for t in topics}
    for msg in reader.read_messages(topics=topics):
        out[msg.channel.topic].append(message_timestamp(msg))
    return out


def scan_episode(
    mcap_path: str,
    thresholds: QualityThresholds,
) -> EpisodeQualityReport:
    """
    Scan one MCAP episode file and produce a full quality report.

    Which topics get analyzed, and what role each plays, is inferred purely
    from each topic's ROS2 message type (see classify_topic) — no config is
    consulted. Topics whose message type isn't one of the known robot-
    pipeline types are still surfaced in the report (role="unclassified",
    severity always PASS) rather than silently dropped. Their avg_fps IS
    computed from their own real message timestamps (via a best-effort,
    separate scan — see below), but everything else about them stays purely
    informational: coverage/gaps/severity are never derived from it.

    Session start/end are computed from the actual collected message
    timestamps across all analyzable (stream/action) topics — never from
    MCAP summary file-level fields (those reflect the whole file, not any
    single topic, and would misclassify legitimate action-topic idle gaps
    as dropframes).

    If the file itself cannot be opened or parsed, this returns a report
    with severity=CRITICAL, passed=False, no topics, and read_error set to
    a human-readable message — distinct from a genuinely-recorded-but-empty
    file, so a caller (e.g. a CLI) can tell "bad file" from "bad recording."
    """
    try:
        topic_info = _summary_topic_info(mcap_path)
    except (OSError, McapError) as exc:
        return EpisodeQualityReport(
            path=str(Path(mcap_path).resolve()),
            duration_s=0.0,
            severity=SEVERITY_CRITICAL,
            passed=False,
            topics=[],
            read_error=f"{type(exc).__name__}: {exc}",
        )

    monitored = classify_topics({t: ti.schema_name for t, ti in topic_info.items()})
    analyzable = [m for m in monitored if m.role in (ROLE_STREAM, ROLE_ACTION)]
    unclassified = [m for m in monitored if m.role == ROLE_UNCLASSIFIED]

    scan_topics = [m.topic for m in analyzable if topic_info[m.topic].count > 0]
    # Unlike the unclassified-topics decode below, a decode failure here means the
    # actual monitored (stream/action) message stream is truncated/corrupted — e.g. a
    # readable footer but a recording process killed mid-write. That's just as much a
    # "this file is broken" situation as the footer-read failure above, so it must fail
    # the whole episode the same way, not degrade gracefully like the informational-only
    # unclassified topics do.
    try:
        ts_map = _collect_timestamps(mcap_path, scan_topics) if scan_topics else {}
    except (OSError, McapError) as exc:
        return EpisodeQualityReport(
            path=str(Path(mcap_path).resolve()),
            duration_s=0.0,
            severity=SEVERITY_CRITICAL,
            passed=False,
            topics=[],
            read_error=f"{type(exc).__name__}: {exc}",
        )

    all_ts = [t for lst in ts_map.values() for t in lst]
    session_start = min(all_ts) if all_ts else 0.0
    session_end = max(all_ts) if all_ts else 0.0

    # Unclassified topics are informational-only and must never affect session
    # bounds or stream/action severity — but a real fps is still useful to show
    # (this is the whole point of this block). Collected into a SEPARATE map so
    # it can never be mixed into all_ts/session_start/session_end above. Unlike
    # the 3 well-tested monitored message types, an unclassified topic can in
    # principle be any ROS2 message type (or even a non-ROS2-encoded channel),
    # so a decode failure here must degrade to "no fps" rather than fail the
    # whole scan — these topics are informational-only by design and must
    # never be able to break anything.
    unclassified_scan_topics = [m.topic for m in unclassified if topic_info[m.topic].count > 0]
    try:
        unclassified_ts_map = (
            _collect_timestamps(mcap_path, unclassified_scan_topics)
            if unclassified_scan_topics
            else {}
        )
    except (OSError, McapError):
        unclassified_ts_map = {}

    topic_reports = []
    for m in monitored:
        if m.role == ROLE_UNCLASSIFIED:
            ti = topic_info[m.topic]
            ts = sorted(unclassified_ts_map.get(m.topic, []))
            avg_fps = (len(ts) - 1) / (ts[-1] - ts[0]) if len(ts) >= 2 and ts[-1] > ts[0] else None
            topic_reports.append(
                TopicQualityReport(
                    topic=m.topic,
                    label=m.label,
                    role=ROLE_UNCLASSIFIED,
                    message_count=ti.count,
                    avg_fps=avg_fps,
                    coverage_ratio=1.0,
                    total_gap_s=0.0,
                    longest_gap_s=0.0,
                    gaps=[],
                    severity=SEVERITY_PASS,
                    reason=f"unmonitored message type ({m.message_type or 'unknown'}) — informational only",
                    message_type=m.message_type,
                )
            )
        else:
            result = analyze_topic_coverage(
                ts_map.get(m.topic, []),
                session_start,
                session_end,
                topic=m.topic,
                label=m.label,
                role=m.role,
                thresholds=thresholds,
                action_from_observation=False,
            )
            topic_reports.append(replace(result, message_type=m.message_type))

    severity = worst_severity(r.severity for r in topic_reports)
    return EpisodeQualityReport(
        path=str(Path(mcap_path).resolve()),
        duration_s=session_end - session_start,
        severity=severity,
        passed=(severity != SEVERITY_CRITICAL),
        topics=topic_reports,
    )
