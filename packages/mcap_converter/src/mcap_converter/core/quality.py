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
from dataclasses import dataclass, field
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
        import dataclasses

        return dataclasses.asdict(self)


@dataclass
class EpisodeQualityReport:
    """Aggregated quality report for one MCAP episode across all monitored topics."""

    path: str  # str(Path(mcap_path).resolve())
    duration_s: float
    severity: str
    passed: bool
    topics: List[TopicQualityReport] = field(default_factory=list)

    def to_dict(self) -> dict:
        import dataclasses

        return dataclasses.asdict(self)


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
