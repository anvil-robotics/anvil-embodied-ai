"""G2 mechanism analysis — why chunk-anchor (n-0) targets collapse on Diffusion.

The closed-loop matrix shows a puzzle: within the *goal family* (same 10-dim
rot6d observation, same encoding), an **absolute** target trains fine on
Diffusion (``goal-abs`` 98) while a **chunk-anchor relative** target of the same
trajectory collapses (``goal-world-n0`` 16); yet a **per-frame** relative target
(``native_n0`` 98) is robust. So neither "relative" nor "rot6d" is the cause —
the *anchor construction* is.

This module isolates that mechanism from data alone. It takes ONE source of
truth — the absolute goal trajectory stored in the ``goalabs`` dataset — and
re-derives the three target constructions off the *same* trajectory, so the
only thing that varies is the anchor:

    absolute      target[k] = goal_pose[t0+k]                 (world coords)
    chunk-anchor  target[k] = goal_pose[t0+k] (-) state[t0]   (anchor = chunk start)
    per-frame     target[k] = goal_pose[t0+k] (-) state[t0+k] (anchor = own frame)

over sliding chunks of the diffusion training ``horizon`` (16), and reports the
within-chunk dynamic range (how the target magnitude/variance grows across a
chunk).

NEGATIVE RESULT (honest): at the diffusion horizon (16) the *position* target
magnitude does NOT ramp — chunk-anchor, per-frame and absolute all have a
within-chunk dynamic range near 1.0. So the naive "chunk-anchor makes the target
grow large" intuition is refuted; raw target magnitude is not the differentiator.

The load-bearing evidence is instead the DIRECT closed-loop comparison
(``compare_closed_loop`` / ``--collapse-trace``): the trained chunk-anchor
diffusion policy mode-collapses — its delivered commands shrink to ~1/3 the
magnitude and ~1/2 the per-axis spread of both the robust per-frame policy and
the ground-truth demos, so the arm under-actuates and times out. ACT survives
the same target (82) because it regresses the conditional mean; a distributional
generator degrades to the low-magnitude marginal. The mechanism is a property of
how a distributional model fits the chunk-anchor target family, not of the
target's magnitude.

Run:
    uv run --package anvil-sim python -m \
        anvil_sim.studies.libero_ee.analysis.mechanism_analysis \
        --dataset data/datasets/ee-space/libero-task10-goalabs \
        --out outputs/bench/analysis
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from anvil_shared.ee_transform import (
    quats_to_matrices,
    rot6ds_to_matrices,
)
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# goalabs layout: action = [xyz(3), rot6d(6), gripper(1)] (10); state = [xyz(3), quat(4), gripper(1)] (8).
_POS = slice(0, 3)
_ACT_R6D = slice(3, 9)
_STATE_QUAT = slice(3, 7)
DEFAULT_HORIZON = 16  # diffusion policy horizon (config.json: horizon=16, n_action_steps=8)


def load_abs_goal_and_state(dataset_root: Path) -> list[dict[str, np.ndarray]]:
    """Load per-episode absolute goal actions and observation states.

    Returns a list (episode order) of ``{"goal": (T,10), "state": (T,8)}`` from
    the ``goalabs`` dataset (whose action column IS the absolute goal pose).
    """
    ds = LeRobotDataset(repo_id="local", root=str(dataset_root))
    hf = ds.hf_dataset.select_columns(
        ["episode_index", "action", "observation.state"]
    ).with_format(None)
    episodes: dict[int, dict[str, list]] = {}
    for ep_idx, action, state in zip(
        hf["episode_index"], hf["action"], hf["observation.state"], strict=True
    ):
        ep = int(ep_idx)
        bucket = episodes.setdefault(ep, {"goal": [], "state": []})
        bucket["goal"].append(np.asarray(action, dtype=np.float64))
        bucket["state"].append(np.asarray(state, dtype=np.float64))
    return [
        {"goal": np.stack(episodes[k]["goal"]), "state": np.stack(episodes[k]["state"])}
        for k in sorted(episodes)
    ]


def _rot_angle(r_a_6d: np.ndarray, r_from_mat: np.ndarray) -> np.ndarray:
    """Geodesic angle (rad) of R(action_rot6d) @ R(anchor).T, batched over (...,)."""
    r_act = rot6ds_to_matrices(r_a_6d)
    r_rel = r_act @ np.swapaxes(r_from_mat, -2, -1)
    trace = np.trace(r_rel, axis1=-2, axis2=-1)
    cos = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    return np.arccos(cos)


def chunk_profiles(
    episodes: list[dict[str, np.ndarray]], horizon: int
) -> dict[str, np.ndarray]:
    """Aggregate per-within-chunk-index (k=0..horizon-1) target profiles.

    Slides a length-``horizon`` window over every episode (one window per start
    offset, mirroring how training samples chunks at every frame) and, for each
    of the three anchor constructions, records the per-element position L2 norm
    and rotation geodesic angle. Returns arrays shaped (horizon,) of the mean
    and std across all windows.
    """
    pos = {"absolute": [], "chunk_anchor": [], "per_frame": []}
    rot = {"absolute": [], "chunk_anchor": [], "per_frame": []}
    # A shared world origin for "absolute" magnitude (workspace centroid) so its
    # numbers are comparable — absolute target scale is otherwise an arbitrary
    # offset; what matters is that it does NOT ramp within a chunk.
    all_goal_xyz = np.concatenate([e["goal"][:, _POS] for e in episodes], axis=0)
    origin = all_goal_xyz.mean(axis=0)

    for e in episodes:
        goal, state = e["goal"], e["state"]
        goal_xyz, goal_r6d = goal[:, _POS], goal[:, _ACT_R6D]
        state_xyz = state[:, _POS]
        state_R = quats_to_matrices(state[:, _STATE_QUAT])
        t_max = goal.shape[0] - horizon
        if t_max <= 0:
            continue
        for t0 in range(t_max):
            idx = slice(t0, t0 + horizon)
            gx, gr = goal_xyz[idx], goal_r6d[idx]
            # absolute (relative to a fixed world origin, not an anchor)
            pos["absolute"].append(np.linalg.norm(gx - origin, axis=-1))
            rot["absolute"].append(_rot_angle(gr, np.broadcast_to(np.eye(3), (horizon, 3, 3))))
            # chunk-anchor: single anchor = chunk-start frame
            pos["chunk_anchor"].append(np.linalg.norm(gx - state_xyz[t0], axis=-1))
            anchor_R = np.broadcast_to(state_R[t0], (horizon, 3, 3))
            rot["chunk_anchor"].append(_rot_angle(gr, anchor_R))
            # per-frame: anchor = each element's own frame
            pos["per_frame"].append(np.linalg.norm(gx - state_xyz[idx], axis=-1))
            rot["per_frame"].append(_rot_angle(gr, state_R[idx]))

    out: dict[str, np.ndarray] = {}
    for kind, store in (("pos", pos), ("rot", rot)):
        for name, rows in store.items():
            arr = np.stack(rows)  # (n_windows, horizon)
            out[f"{kind}.{name}.mean"] = arr.mean(axis=0)
            out[f"{kind}.{name}.std"] = arr.std(axis=0)
    return out


def summarize(prof: dict[str, np.ndarray], horizon: int) -> dict:
    """Reduce profiles to the head-line dynamic-range metrics per construction."""
    h0, hL = 0, horizon - 1
    summary: dict = {"horizon": horizon, "constructions": {}}
    for name in ("absolute", "chunk_anchor", "per_frame"):
        pm = prof[f"pos.{name}.mean"]
        ps = prof[f"pos.{name}.std"]
        rm = prof[f"rot.{name}.mean"]
        summary["constructions"][name] = {
            "pos_norm_head": float(pm[h0]),
            "pos_norm_tail": float(pm[hL]),
            "pos_dynamic_range": float(pm[hL] / (pm[h0] + 1e-9)),
            "pos_tail_cv": float(ps[hL] / (pm[hL] + 1e-9)),  # coeff. of variation at horizon tail
            "rot_angle_head_rad": float(rm[h0]),
            "rot_angle_tail_rad": float(rm[hL]),
            "rot_dynamic_range": float(rm[hL] / (rm[h0] + 1e-9)),
        }
    return summary


def _plot(prof: dict[str, np.ndarray], horizon: int, out_png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    k = np.arange(horizon)
    fig, (ax_p, ax_r) = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = {"absolute": "tab:green", "chunk_anchor": "tab:red", "per_frame": "tab:blue"}
    labels = {
        "absolute": "absolute (goal-abs, robust 98)",
        "chunk_anchor": "chunk-anchor n-0 (goal-world-n0, COLLAPSE 16)",
        "per_frame": "per-frame (native_n0, robust 98)",
    }
    for name in ("absolute", "chunk_anchor", "per_frame"):
        pm, ps = prof[f"pos.{name}.mean"], prof[f"pos.{name}.std"]
        ax_p.plot(k, pm, color=colors[name], label=labels[name])
        ax_p.fill_between(k, pm - ps, pm + ps, color=colors[name], alpha=0.15)
        ax_r.plot(k, np.degrees(prof[f"rot.{name}.mean"]), color=colors[name], label=labels[name])
    ax_p.set(title="Position target magnitude within one predicted chunk",
             xlabel="within-chunk index k (0..horizon-1)", ylabel="mean ||Δpos|| (raw)")
    ax_r.set(title="Rotation target magnitude within one predicted chunk",
             xlabel="within-chunk index k", ylabel="mean geodesic angle (deg)")
    for ax in (ax_p, ax_r):
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle(
        f"Why chunk-anchor collapses on Diffusion — within-chunk dynamic range (horizon={horizon})",
        fontsize=11,
    )
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def compare_closed_loop(
    collapse_trace: Path, robust_trace: Path, gt_native_trace: Path | None = None
) -> dict:
    """Compare the *delivered* command (``native_cmd``, the common 7-dim world
    command both policies emit into the env) of a collapsed vs a robust closed-
    loop diffusion rollout, to document the collapse empirically.

    A mode-collapsed diffusion policy emits low-diversity, mean-reverting
    commands (it samples the marginal, not the obs-conditioned mode), so its
    per-axis command std and step-to-step variation shrink relative to a robust
    policy and to the ground-truth demonstration command distribution.
    """

    def _cmds(p: Path) -> np.ndarray:
        rows = [json.loads(line) for line in p.open()]
        return np.asarray([r["native_cmd"] for r in rows if r.get("native_cmd")], dtype=np.float64)

    def _stats(cmd: np.ndarray) -> dict:
        pos, rot = cmd[:, 0:3], cmd[:, 3:6]
        step_dpos = np.linalg.norm(np.diff(pos, axis=0), axis=-1)
        return {
            "n_steps": int(cmd.shape[0]),
            "pos_axis_std": [float(x) for x in pos.std(axis=0)],
            "pos_norm_mean": float(np.linalg.norm(pos, axis=-1).mean()),
            "pos_norm_std": float(np.linalg.norm(pos, axis=-1).std()),
            "rot_axis_std": [float(x) for x in rot.std(axis=0)],
            "step_dpos_mean": float(step_dpos.mean()),
            "step_dpos_std": float(step_dpos.std()),
        }

    out = {
        "collapse (goal-world-n0 diffusion, 16%)": _stats(_cmds(collapse_trace)),
        "robust (native_n0 diffusion, 98%)": _stats(_cmds(robust_trace)),
    }
    if gt_native_trace and gt_native_trace.exists():
        # GT trace uses the 'provided' column as the reference native command.
        rows = [json.loads(line) for line in gt_native_trace.open()]
        gt = np.asarray([r["provided"] for r in rows], dtype=np.float64)
        out["gt (native demos)"] = _stats(gt)
    return out


def _plot_collapse(cmp: dict, out_png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = list(cmp)  # collapse, robust, gt (insertion order)
    colors = ["tab:red", "tab:blue", "0.4"]
    axes_lbl = ["x", "y", "z"]
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.2))
    # per-axis command std (diversity)
    x = np.arange(3)
    w = 0.26
    for i, name in enumerate(order):
        ax0.bar(x + (i - 1) * w, cmp[name]["pos_axis_std"], w, label=name.split(" ")[0], color=colors[i])
    ax0.set(title="Delivered command diversity (per-axis std)", xticks=x, xlabel="position axis",
            ylabel="std of native_cmd (raw)")
    ax0.set_xticklabels(axes_lbl)
    ax0.legend(fontsize=8)
    # command magnitude
    mags = [cmp[name]["pos_norm_mean"] for name in order]
    ax1.bar(range(len(order)), mags, color=colors[: len(order)])
    ax1.set(title="Delivered command magnitude (mean ||pos||)", xticks=range(len(order)),
            ylabel="mean ||native_cmd pos|| (raw)")
    ax1.set_xticklabels([n.split(" ")[0] for n in order])
    for ax in (ax0, ax1):
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Diffusion mode collapse: chunk-anchor policy under-actuates vs robust / GT", fontsize=11)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path,
                    default=Path("data/datasets/ee-space/libero-task10-goalabs"))
    ap.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    ap.add_argument("--out", type=Path, default=Path("outputs/bench/analysis"))
    ap.add_argument("--collapse-trace", type=Path,
                    default=Path("outputs/bench/analysis/traces/collapse_worldn0_diffusion/trace.jsonl"))
    ap.add_argument("--robust-trace", type=Path,
                    default=Path("outputs/bench/analysis/traces/robust_native_n0_diffusion/trace.jsonl"))
    ap.add_argument("--gt-trace", type=Path,
                    default=Path("outputs/bench/replay/native-baseline/trace.jsonl"))
    args = ap.parse_args()

    episodes = load_abs_goal_and_state(args.dataset)
    prof = chunk_profiles(episodes, args.horizon)
    summary = summarize(prof, args.horizon)
    summary["dataset"] = str(args.dataset)
    summary["n_episodes"] = len(episodes)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "mechanism.json").write_text(json.dumps(summary, indent=2))
    _plot(prof, args.horizon, args.out / "mechanism.png")
    print(f"== within-chunk target profiles (horizon={args.horizon}) ==")
    print(json.dumps(summary["constructions"], indent=2))

    # Direct closed-loop evidence of the collapse (the load-bearing artifact).
    if args.collapse_trace.exists() and args.robust_trace.exists():
        cmp = compare_closed_loop(args.collapse_trace, args.robust_trace, args.gt_trace)
        (args.out / "closed_loop_collapse.json").write_text(json.dumps(cmp, indent=2))
        _plot_collapse(cmp, args.out / "collapse.png")
        print("\n== closed-loop delivered-command comparison ==")
        print(json.dumps(cmp, indent=2))
        print(f"\nwrote {args.out/'closed_loop_collapse.json'} and {args.out/'collapse.png'}")
    print(f"\nwrote {args.out/'mechanism.json'} and {args.out/'mechanism.png'}")


if __name__ == "__main__":
    main()
