#!/usr/bin/env python3
"""Vendor a slim, viz-only OpenArm asset set from an anvil-workcell checkout.

Why this exists: `inference-replay` needs a URDF-accurate robot to draw, but the only
OpenArm description lives in anvil-workcell (`ros2/src/openarm_v2_description`) and is
117 MB of CAD-resolution meshes behind a tree of xacro macros. Neither repo wants a
submodule (anvil-workcell deliberately migrated off them), so this script generates a
committed, viewer-sized snapshot instead: one flat URDF plus decimated visual meshes.

Re-run it whenever the upstream description changes. SOURCE_COMMIT records where the
current snapshot came from so drift is detectable.

Usage:
    uv run --package inference-replay --extra sync python scripts/sync_openarm_assets.py --workcell ~/anvil-workcell
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import types
import xml.etree.ElementTree as ET
from pathlib import Path

# Relative to the workcell repo root.
_DESCRIPTION_PKG = "openarm_v2_description"
_DESCRIPTION_PATH = Path("ros2/src") / _DESCRIPTION_PKG

# `urdf/robot/teleop.xacro` is the entrypoint the real robot uses -- see
# anvil-workcell `control/workcell_config.py` ARM_SPECS and `robot_nodes.py::_process_xacro`.
# `v10.urdf.xacro` looks like the more obvious choice but is broken in this private fork:
# it never passes `ee_inertials`, so the hand macro falls back to a
# `config/hand/openarm_hand/inertials.yaml` that does not exist here.
_XACRO_ENTRYPOINT = Path("urdf/robot/teleop.xacro")

# Arms in teleop.xacro are gated on a non-empty CAN interface, so naming only the two
# followers yields exactly the follower pair -- no leader arms parked 1 m away, and no
# need to post-process the URDF to remove them. The values are arbitrary labels here:
# ros2_control is off, so nothing ever touches a CAN bus.
_XACRO_MAPPINGS = {
    "ros2_control": "false",
    "hand": "true",
    "follower_l_can_interface": "follower_l",
    "follower_r_can_interface": "follower_r",
}

# Fingers read gripper divergence, which is the whole point of the ghost view, so they
# keep more detail than the big structural links.
_DEFAULT_FACE_RATIO = 0.05
_FINGER_FACE_RATIO = 0.20
_MIN_FACES = 200


def _install_ament_shim(share_dir: Path) -> None:
    """Let xacro resolve `$(find openarm_v2_description)` without a ROS install.

    xacro imports `ament_index_python.packages.get_package_share_directory` lazily, from
    a single call site in `substitution_args.py`, and that package is not on PyPI. A stub
    in `sys.modules` is enough, and beats the alternatives: copying the 117 MB tree to
    rewrite `$(find ...)` by hand, or requiring a sourced ROS environment to run this.
    """
    packages = types.ModuleType("ament_index_python.packages")

    def get_package_share_directory(name: str) -> str:
        if name != _DESCRIPTION_PKG:
            raise KeyError(f"only {_DESCRIPTION_PKG} is available, not {name!r}")
        return str(share_dir)

    packages.get_package_share_directory = get_package_share_directory  # type: ignore[attr-defined]
    root = types.ModuleType("ament_index_python")
    root.packages = packages  # type: ignore[attr-defined]
    sys.modules.setdefault("ament_index_python", root)
    sys.modules.setdefault("ament_index_python.packages", packages)


def _source_commit(workcell: Path) -> str:
    """Best-effort provenance stamp for the generated snapshot."""
    try:
        out = subprocess.run(
            ["git", "-C", str(workcell), "log", "-1", "--format=%H %cs", "--", str(_DESCRIPTION_PATH)],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip() or "unknown"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def generate_urdf(description_dir: Path) -> ET.Element:
    _install_ament_shim(description_dir)
    # Imported after the shim so substitution_args can resolve $(find ...). Only present
    # with the `sync` extra installed, hence the ignore.
    import xacro  # type: ignore[import-not-found]

    # process_file() fills the mapping dict with every default arg it resolves, so hand it
    # a copy -- otherwise _XACRO_MAPPINGS grows and the provenance stamp becomes unreadable.
    doc = xacro.process_file(str(description_dir / _XACRO_ENTRYPOINT), mappings=dict(_XACRO_MAPPINGS))
    return ET.fromstring(doc.toxml())


def _face_ratio(mesh_rel: Path) -> float:
    return _FINGER_FACE_RATIO if "finger" in mesh_rel.name or "hand" in mesh_rel.name else _DEFAULT_FACE_RATIO


def _scale_suffix(scale: tuple[float, float, float]) -> str:
    """Distinguish mirrored variants of the same source mesh in the output filename."""
    negative = "".join(axis for axis, value in zip("xyz", scale) if value < 0)
    return f"_mirror{negative}" if negative else ""


def _convert_mesh(src: Path, dst: Path, ratio: float, scale: tuple[float, float, float]) -> tuple[int, int]:
    """Load a CAD mesh, decimate it, bake in the URDF scale, and write a GLB.

    Scale is baked rather than left on the URDF's `<mesh scale=...>` because these arms use a
    negative Y scale to mirror one side, and a reflection in the render transform inverts face
    winding. trimesh corrects the winding when it applies the reflection, so doing it here
    keeps the viewer free of reflected transforms entirely.

    Returns (faces_before, faces_after).
    """
    import numpy as np
    import trimesh

    # force="mesh" collapses a scene to a single Trimesh; the annotation is what the type
    # stubs cannot infer from the flag.
    mesh: trimesh.Trimesh = trimesh.load(src, force="mesh")  # type: ignore[assignment]
    before = len(mesh.faces)
    # CAD exports arrive with duplicated vertices per triangle, which starves the quadric
    # decimator of the connectivity it needs -- merge before simplifying.
    mesh.merge_vertices()
    target = max(_MIN_FACES, int(before * ratio))
    if target < before:
        mesh = mesh.simplify_quadric_decimation(face_count=target)
    mesh.apply_transform(np.diag([*scale, 1.0]))
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(mesh.export(file_type="glb"))  # type: ignore[arg-type]
    return before, len(mesh.faces)


def sync(workcell: Path, out_dir: Path) -> None:
    description_dir = (workcell / _DESCRIPTION_PATH).resolve()
    if not (description_dir / _XACRO_ENTRYPOINT).is_file():
        raise SystemExit(
            f"No {_DESCRIPTION_PKG} at {description_dir}.\n"
            f"Pass --workcell pointing at an anvil-workcell checkout."
        )

    print(f"Reading  {description_dir}")
    root = generate_urdf(description_dir)

    # Collision geometry is dead weight for a viewer and is over half the payload.
    removed = 0
    for link in root.iter("link"):
        for collision in list(link.findall("collision")):
            link.remove(collision)
            removed += 1
    print(f"Dropped  {removed} collision geometries")

    if out_dir.exists():
        shutil.rmtree(out_dir)

    # Keyed by (source uri, scale): the same source mesh appears twice, mirrored, for the two
    # arms, and each mirroring needs its own baked GLB.
    converted: dict[tuple[str, tuple[float, float, float]], str] = {}
    for geometry in root.iter("geometry"):
        for mesh_el in geometry.findall("mesh"):
            uri = mesh_el.get("filename", "")
            prefix = f"package://{_DESCRIPTION_PKG}/"
            if not uri.startswith(prefix):
                raise SystemExit(f"Unexpected mesh URI, cannot vendor it: {uri!r}")
            # `rel` already starts with "meshes/", so it doubles as the output layout.
            rel = Path(uri[len(prefix) :])
            raw_scale = mesh_el.get("scale", "1 1 1").split()
            scale = (float(raw_scale[0]), float(raw_scale[1]), float(raw_scale[2]))

            key = (uri, scale)
            if key not in converted:
                dst_rel = rel.with_name(rel.stem + _scale_suffix(scale)).with_suffix(".glb")
                before, after = _convert_mesh(
                    description_dir / rel, out_dir / dst_rel, _face_ratio(rel), scale
                )
                converted[key] = str(dst_rel)
                print(f"  {dst_rel.name:<26} {before:>8} -> {after:>6} faces")
            # Paths are relative to the URDF, which is what yourdfpy's filename handler wants.
            mesh_el.set("filename", converted[key])
            # Scale is baked into the GLB now; leaving the attribute would apply it twice.
            mesh_el.attrib.pop("scale", None)

    urdf_path = out_dir / "openarm_v2_followers.urdf"
    ET.ElementTree(root).write(urdf_path, encoding="unicode", xml_declaration=True)
    (out_dir / "SOURCE_COMMIT").write_text(
        f"{_DESCRIPTION_PKG} @ {_source_commit(workcell)}\n"
        f"xacro: {_XACRO_ENTRYPOINT} {' '.join(f'{k}:={v}' for k, v in _XACRO_MAPPINGS.items())}\n"
        f"regenerate: uv run --package inference-replay --extra sync python scripts/sync_openarm_assets.py\n"
    )

    total_kb = sum(p.stat().st_size for p in out_dir.rglob("*") if p.is_file()) / 1024
    print(f"\nWrote    {urdf_path}")
    print(f"         {len(converted)} meshes, {total_kb / 1024:.1f} MB total")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--workcell",
        type=Path,
        default=Path.home() / "anvil-workcell",
        help="Path to an anvil-workcell checkout (default: ~/anvil-workcell)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "packages/inference_replay/src/inference_replay/assets/openarm_v2",
        help="Destination for the generated URDF + meshes",
    )
    args = parser.parse_args()
    sync(args.workcell.expanduser(), args.out)


if __name__ == "__main__":
    main()
