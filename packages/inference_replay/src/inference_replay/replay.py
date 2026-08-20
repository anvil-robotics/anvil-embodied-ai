"""Log an inference trace to Rerun as two robots on one timeline.

Layout:

    obs/mesh/<link>            observed robot: per-frame pose (link pose x visual origin) + mesh
    obs/mesh/<link>/stalled    a red dot, present only while a joint driving that link stalls
    cmd/mesh/<link>            commanded robot, same shape, tinted
    cmd/marker/<link>          a dot at each commanded link origin -- the non-occluding shadow
    plots/<joint>/{observed,commanded,gap,stall,clamp}
    events                     one entry per detected stall

The three groups (`obs/mesh`, `cmd/mesh`, `cmd/marker`) are deliberately siblings under
plain prefixes rather than interleaved as `<link>/geom` children. That is what lets each
viewing mode be a single prefix filter -- and a single click in the viewer -- instead of
toggling 23 individual meshes.

Zero-order hold is not implemented here because Rerun already behaves that way: a logged value
stays current until the next one on that timeline, which is exactly what the arm experienced
between 30 Hz command updates.

Meshes and dots are logged once as static data; only the transforms are time-indexed. Logging
geometry per frame instead would multiply the recording by ~5000.
"""

from __future__ import annotations

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from .analysis import Analysis
from .kinematics import RobotKinematics
from .trace import ARM_CHANNELS, NAMES, Trace

TIMELINE = "trace_time"

# Matches the palette build_trace_replay.py already established, so the two artifacts read
# the same way: teal for what happened, amber for what was asked for.
_COMMANDED_RGB = (0xF2, 0xB3, 0x3D)
# Alpha applies if the viewer honours it on meshes. If it does not, the commanded robot renders
# solid -- which is still correct in the Commanded tab, where the observed robot is not drawn.
_COMMANDED_MESH_RGBA = (*_COMMANDED_RGB, 0x60)
_STALL_RGB = (0xE2, 0x56, 0x4D)

_MARKER_RADIUS_M = 0.008


def _world_poses(kin: RobotKinematics, samples: np.ndarray) -> tuple[list[str], np.ndarray]:
    """FK over every sample. Returns (links, (n, n_links, 4, 4) homogeneous transforms)."""
    links = list(dict.fromkeys(kin.link_names))
    index = {link: j for j, link in enumerate(links)}
    poses = np.zeros((len(samples), len(links), 4, 4), dtype=np.float64)
    poses[..., 3, 3] = 1.0
    for i, sample in enumerate(samples):
        for link, matrix in kin.link_poses(sample).items():
            poses[i, index[link]] = matrix
    return links, poses


def _send_transforms(stream: rr.RecordingStream, path: str, matrices: np.ndarray, rel: np.ndarray) -> None:
    """Send an (n, 4, 4) transform series as one column per component."""
    rr.send_columns(
        path,
        indexes=[rr.TimeColumn(TIMELINE, duration=rel)],
        columns=rr.Transform3D.columns(
            translation=matrices[:, :3, 3].astype(np.float32),
            mat3x3=matrices[:, :3, :3].astype(np.float32).reshape(len(rel), 9),
        ),
        recording=stream,
    )


def _log_robot(
    stream: rr.RecordingStream,
    root: str,
    kin: RobotKinematics,
    samples: np.ndarray,
    rel: np.ndarray,
    *,
    tint: tuple[int, int, int, int] | None,
    with_markers: bool,
) -> None:
    links, poses = _world_poses(kin, samples)
    index = {link: j for j, link in enumerate(links)}

    for visual in kin.visuals:
        # The mesh entity carries link_pose x visual_origin directly, so the mesh needs no
        # child entity of its own and the whole robot is one prefix.
        path = f"{root}/mesh/{visual.link}"
        _send_transforms(stream, path, poses[:, index[visual.link]] @ visual.origin, rel)
        stream.log(path, rr.Asset3D(path=visual.mesh_path, albedo_factor=tint), static=True)

    if with_markers:
        for link in links:
            path = f"{root}/marker/{link}"
            _send_transforms(stream, path, poses[:, index[link]], rel)
            # One static dot per link rides its animated transform, so the commanded pose stays
            # readable without a second mesh that would occlude the observed robot.
            stream.log(
                path,
                rr.Points3D([[0.0, 0.0, 0.0]], colors=[_COMMANDED_RGB], radii=[_MARKER_RADIUS_M]),
                static=True,
            )


def _log_plots(stream: rr.RecordingStream, trace: Trace, analysis: Analysis) -> None:
    time_column = rr.TimeColumn(TIMELINE, duration=trace.rel)
    for channel, name in enumerate(NAMES):
        series = {
            "observed": trace.obs[:, channel],
            "commanded": trace.cmd[:, channel],
            "gap": analysis.gap[:, channel],
            "clamp": analysis.clamp_mask(channel, trace.n).astype(np.float32),
        }
        # Grippers are exempt from stall detection, so a flat zero line here would imply
        # "checked and clean" rather than "not assessed".
        if channel in ARM_CHANNELS:
            series["stall"] = analysis.stall_mask(channel, trace.n).astype(np.float32)
        for label, values in series.items():
            rr.send_columns(
                f"plots/{name}/{label}",
                indexes=[time_column],
                columns=rr.Scalars.columns(scalars=values),
                recording=stream,
            )


def _log_events(stream: rr.RecordingStream, trace: Trace, analysis: Analysis, kin: RobotKinematics) -> None:
    for event in analysis.events:
        stream.set_time(TIMELINE, duration=float(trace.rel[event.start]))
        stream.log(
            "events",
            rr.TextLog(
                f"{event.name} stalled {event.duration_sec:.2f}s "
                f"({event.steps} steps), max gap {event.max_gap:.3f} rad, "
                f"moved {event.obs_range:.4f} rad",
                level=rr.TextLogLevel.WARN,
            ),
        )

    # Parented to the observed mesh of the link each joint drives, so the dot rides the arm.
    # Logged as a child of an already-posed entity, it needs no transform of its own.
    for channel in ARM_CHANNELS:
        link = kin.child_link_by_channel.get(channel)
        if link is None:
            continue
        path = f"obs/mesh/{link}/stalled"
        for start, end in analysis.stalls.get(channel, []):
            stream.set_time(TIMELINE, duration=float(trace.rel[start]))
            stream.log(
                path,
                rr.Points3D(
                    [[0.0, 0.0, 0.0]], colors=[_STALL_RGB], radii=[_MARKER_RADIUS_M * 2.5]
                ),
            )
            # Clearing at the end of the span is what makes it a transient marker rather than
            # one that persists for the rest of the recording.
            stream.set_time(TIMELINE, duration=float(trace.rel[min(end, trace.n - 1)]))
            stream.log(path, rr.Points3D([]))


def default_blueprint() -> rrb.Blueprint:
    """The three viewing modes as tabs, so switching is one click.

    Visibility overrides were the obvious alternative but would leave 23 individual meshes to
    toggle by hand; excluding paths from a single view's contents hides them from its tree
    entirely, with no way back except editing the filter. Separate views sidestep both.
    """
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Tabs(
                rrb.Spatial3DView(
                    name="Observed + commanded shadow",
                    origin="/",
                    contents=["+ /obs/mesh/**", "+ /cmd/marker/**"],
                ),
                rrb.Spatial3DView(name="Observed only", origin="/", contents=["+ /obs/mesh/**"]),
                rrb.Spatial3DView(name="Commanded only", origin="/", contents=["+ /cmd/mesh/**"]),
                active_tab=0,
            ),
            rrb.Vertical(
                rrb.TimeSeriesView(name="Commanded vs observed", origin="/plots"),
                rrb.TextLogView(name="Stall events", origin="/events"),
                row_shares=[3, 1],
            ),
            column_shares=[3, 2],
        ),
        collapse_panels=False,
    )


def log_trace(stream: rr.RecordingStream, trace: Trace, analysis: Analysis, kin: RobotKinematics) -> None:
    """Log a full trace: both robots, the per-joint plots, and the stall events."""
    stream.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    _log_robot(stream, "obs", kin, trace.obs, trace.rel, tint=None, with_markers=False)
    _log_robot(
        stream, "cmd", kin, trace.cmd, trace.rel, tint=_COMMANDED_MESH_RGBA, with_markers=True
    )
    _log_plots(stream, trace, analysis)
    _log_events(stream, trace, analysis, kin)
