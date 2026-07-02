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
from dataclasses import asdict, dataclass, field
from typing import Iterable, List, Optional, Set

from ..config.schema import DataConfig

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
            reason = f"{len(gaps)} 個 idle gap，最長 {max(g.duration_s for g in gaps):.2f}s（正常，手臂未操作）"
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
