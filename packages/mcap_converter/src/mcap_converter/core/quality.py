"""Pre-conversion quality scanning for raw MCAP sessions.

Splits into two layers, mirroring the pattern used for action forward-fill
in extractor.py:
  - Pure analysis functions (analyze_topic_coverage, detect_fps_degradation,
    apply_batch_fps_check, resolve_monitored_topics, worst_severity) that
    take already-extracted data and are fully unit-testable without any
    MCAP file I/O.
  - A thin I/O adapter (scan_episode, and its helpers) that reads a real
    MCAP file and feeds the pure functions.
"""

import statistics
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from mcap.exceptions import McapError
from mcap.reader import make_reader as make_mcap_reader

from ..config.schema import DataConfig
from .extractor import message_timestamp
from .reader import McapReader

SEVERITY_OK = "ok"
SEVERITY_WARNING = "warning"
SEVERITY_CRITICAL = "critical"
_SEVERITY_RANK = {SEVERITY_OK: 0, SEVERITY_WARNING: 1, SEVERITY_CRITICAL: 2}


def worst_severity(severities: Iterable[str]) -> str:
    """Return the most severe value in severities, defaulting to OK if empty."""
    worst = SEVERITY_OK
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
    role: str  # "stream" | "action"
    message_count: int
    avg_fps: Optional[float]  # only meaningful for role="stream"; None for "action"
    coverage_ratio: float
    total_gap_s: float
    longest_gap_s: float
    gaps: List[GapInterval] = field(default_factory=list)
    severity: str = SEVERITY_OK
    reason: str = ""

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
    """A single topic to check, resolved from DataConfig against what's actually in the file."""

    topic: str
    label: str
    role: str  # "stream" | "action"


def resolve_monitored_topics(config: DataConfig, available_topics: Set[str]) -> List[MonitoredTopic]:
    """
    Decide which topics to check based on DataConfig, matching camera topic
    variants against what's actually present in the file.

    Leader-follower mode (config.action_topics empty) produces no separate
    "action" entries — action is embedded in robot_state_topic in that mode,
    which this MVP does not independently verify (see design doc §"MVP 範圍").
    """
    monitored = [MonitoredTopic(topic=config.robot_state_topic, label="joint_states", role="stream")]

    for cam_topic in config.camera_topics:
        candidates = [cam_topic, cam_topic + "/compressed"]
        resolved = next((c for c in candidates if c in available_topics), cam_topic)
        label = config.camera_topic_mapping.get(cam_topic, cam_topic)
        monitored.append(MonitoredTopic(topic=resolved, label=label, role="stream"))

    for topic, topic_cfg in config.action_topics.items():
        monitored.append(MonitoredTopic(topic=topic, label=f"action[{topic_cfg.arm}]", role="action"))

    return monitored


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
            severity, reason = SEVERITY_OK, "action topic 零訊息（action_from_observation=true，可接受）"
        elif role == "action":
            severity, reason = SEVERITY_WARNING, "action topic 全程零訊息，可能為單手任務或設定不符，請人工確認"
        else:
            severity, reason = SEVERITY_CRITICAL, "stream topic 零訊息，完全沒錄到"
        return TopicQualityReport(
            topic=topic, label=label, role=role, message_count=0, avg_fps=None,
            coverage_ratio=0.0, total_gap_s=span, longest_gap_s=span,
            gaps=[], severity=severity, reason=reason,
        )

    ts = sorted(timestamps)
    avg_fps: Optional[float] = None
    gaps: List[GapInterval] = []

    if role == "stream":
        if len(ts) == 1:
            return TopicQualityReport(
                topic=topic, label=label, role=role, message_count=1, avg_fps=None,
                coverage_ratio=0.0, total_gap_s=span, longest_gap_s=span, gaps=[],
                severity=SEVERITY_CRITICAL, reason="stream 僅 1 則訊息，幾乎沒錄到",
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
            severity, reason = SEVERITY_OK, "OK"

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
            severity, reason = SEVERITY_OK, "OK"

    total_gap = sum(g.duration_s for g in gaps)
    longest_gap = max((g.duration_s for g in gaps), default=0.0)
    coverage = max(0.0, 1.0 - total_gap / span)

    return TopicQualityReport(
        topic=topic, label=label, role=role, message_count=len(ts), avg_fps=avg_fps,
        coverage_ratio=coverage, total_gap_s=total_gap, longest_gap_s=longest_gap,
        gaps=gaps, severity=severity, reason=reason,
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
    to the rest of the batch, and upgrade OK -> WARNING for those topics.
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
            if reason_key in degraded_by_path_and_key and t.severity == SEVERITY_OK:
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
            replace(ep, severity=new_severity, passed=(new_severity != SEVERITY_CRITICAL), topics=new_topics)
        )
    return updated_reports


def _summary_message_counts(mcap_path: str) -> Dict[str, int]:
    """
    O(1) per-topic message counts from the MCAP footer summary (no full scan).

    Raises OSError (including FileNotFoundError) or McapError if the file
    cannot be opened or parsed — callers (scan_episode) must catch these to
    produce a clear "file unreadable" result rather than letting a bad file
    look like a real recording with zero messages.
    """
    with open(mcap_path, "rb") as f:
        reader = make_mcap_reader(f)
        summary = reader.get_summary()

    if summary is None or summary.statistics is None:
        return {}

    id_to_topic = {cid: ch.topic for cid, ch in summary.channels.items()}
    counts: Dict[str, int] = {}
    for cid, count in summary.statistics.channel_message_counts.items():
        topic = id_to_topic.get(cid)
        if topic is not None:
            counts[topic] = counts.get(topic, 0) + count
    return counts


def _collect_timestamps(mcap_path: str, topics: List[str]) -> Dict[str, List[float]]:
    """Single-pass scan collecting message_timestamp() for each requested topic."""
    reader = McapReader(mcap_path)
    out: Dict[str, List[float]] = {t: [] for t in topics}
    for msg in reader.read_messages(topics=topics):
        out[msg.channel.topic].append(message_timestamp(msg))
    return out


def scan_episode(
    mcap_path: str,
    config: DataConfig,
    thresholds: QualityThresholds,
) -> EpisodeQualityReport:
    """
    Scan one MCAP episode file and produce a full quality report.

    Session start/end are computed from the actual collected message
    timestamps across all monitored topics — never from MCAP summary
    file-level fields (those reflect the whole file, not any single topic,
    and would misclassify legitimate action-topic idle gaps as dropframes).

    If the file itself cannot be opened or parsed, this returns a report
    with severity=CRITICAL, passed=False, no topics, and read_error set to
    a human-readable message — distinct from a genuinely-recorded-but-empty
    file, so a caller (e.g. a CLI) can tell "bad file" from "bad recording."
    """
    try:
        counts = _summary_message_counts(mcap_path)
    except (OSError, McapError) as exc:
        return EpisodeQualityReport(
            path=str(Path(mcap_path).resolve()),
            duration_s=0.0,
            severity=SEVERITY_CRITICAL,
            passed=False,
            topics=[],
            read_error=f"{type(exc).__name__}: {exc}",
        )

    available = set(counts)
    monitored = resolve_monitored_topics(config, available)

    scan_topics = [m.topic for m in monitored if counts.get(m.topic, 0) > 0]
    ts_map = _collect_timestamps(mcap_path, scan_topics) if scan_topics else {}

    all_ts = [t for lst in ts_map.values() for t in lst]
    session_start = min(all_ts) if all_ts else 0.0
    session_end = max(all_ts) if all_ts else 0.0

    topic_reports = [
        analyze_topic_coverage(
            ts_map.get(m.topic, []), session_start, session_end,
            topic=m.topic, label=m.label, role=m.role,
            thresholds=thresholds, action_from_observation=config.action_from_observation,
        )
        for m in monitored
    ]

    severity = worst_severity(r.severity for r in topic_reports)
    return EpisodeQualityReport(
        path=str(Path(mcap_path).resolve()),
        duration_s=session_end - session_start,
        severity=severity,
        passed=(severity != SEVERITY_CRITICAL),
        topics=topic_reports,
    )
