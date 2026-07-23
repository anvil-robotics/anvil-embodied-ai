#!/usr/bin/env python3
"""List, show, and select the checkpoint used by inference."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_ZOO = REPO_ROOT / "model_zoo"
CURRENT_LINK = MODEL_ZOO / "inference" / "current"
DEFAULT_ENV_FILE = REPO_ROOT / ".env"

POLICY_EXTRAS = {
    "act": "",
    "diffusion": "diffusion",
    "smolvla": "smolvla",
    "pi0": "pi",
    "pi05": "pi",
    "molmoact2": "molmoact2",
    "groot": "groot",
    "multi_task_dit": "multi_task_dit",
    "evo1": "evo1",
    "fastwam": "fastwam",
    "vla_jepa": "vla_jepa",
}

MODEL_PROFILES = {
    "lego-in-cup/act": {
        "config_file": "./configs/lerobot_control/inference_lego_in_cup_act.yaml",
        "inference_arm": "left",
        "extras": "vla_jepa",
    },
    "lego-in-cup/vla-jepa": {
        "config_file": "./configs/lerobot_control/inference_lego_in_cup_vla_jepa.yaml",
        "inference_arm": "left",
    },
}


@dataclass(frozen=True)
class Checkpoint:
    path: Path
    selector: str
    policy_type: str
    action_type: str
    task_description: str | None


def _absolute_no_resolve(path: Path) -> Path:
    return Path(os.path.abspath(path))


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON: {path}: {exc}") from exc


def _config_dir(checkpoint_path: Path) -> Path | None:
    pretrained = checkpoint_path / "pretrained_model"
    if (pretrained / "config.json").is_file():
        return pretrained
    if (checkpoint_path / "config.json").is_file():
        return checkpoint_path
    return None


def _format_env_path(path: Path) -> str:
    abs_path = _absolute_no_resolve(path)
    try:
        rel = abs_path.relative_to(REPO_ROOT)
    except ValueError:
        return str(abs_path)
    return f"./{rel.as_posix()}"


def _selector_for(checkpoint_path: Path) -> str:
    abs_path = _absolute_no_resolve(checkpoint_path)
    try:
        rel = abs_path.relative_to(MODEL_ZOO)
    except ValueError:
        return _format_env_path(checkpoint_path)

    parts = rel.parts
    if len(parts) >= 3 and parts[-2] == "checkpoints":
        job = Path(*parts[:-2]).as_posix()
        return f"{job}:{parts[-1]}"
    return rel.as_posix()


def _checkpoint_from_path(path: Path) -> Checkpoint | None:
    config_dir = _config_dir(path)
    if config_dir is None:
        return None

    config = _read_json(config_dir / "config.json")
    anvil_config = _read_json(config_dir / "anvil_config.json")
    policy_type = str(config.get("type") or "unknown")
    action_type = str(anvil_config.get("action_type") or "absolute")
    if action_type == "absolute" and anvil_config.get("use_delta_actions"):
        action_type = "delta_obs_t"
    task_description = anvil_config.get("task_description") or config.get("task_description")

    return Checkpoint(
        path=path,
        selector=_selector_for(path),
        policy_type=policy_type,
        action_type=action_type,
        task_description=task_description,
    )


def _is_under_cache(path: Path) -> bool:
    return ".cache" in path.parts


def discover_checkpoints() -> list[Checkpoint]:
    checkpoints: list[Checkpoint] = []
    if not MODEL_ZOO.exists():
        return checkpoints

    for checkpoints_dir in MODEL_ZOO.rglob("checkpoints"):
        if _is_under_cache(checkpoints_dir):
            continue
        if not checkpoints_dir.is_dir():
            continue
        for child in sorted(checkpoints_dir.iterdir(), key=lambda p: p.name):
            if _is_under_cache(child) or not child.is_dir():
                continue
            checkpoint = _checkpoint_from_path(child)
            if checkpoint is not None:
                checkpoints.append(checkpoint)

    return sorted(checkpoints, key=lambda c: c.selector)


def _expand_model_path(value: str) -> Path:
    raw = value.strip().strip("'\"")
    path = Path(os.path.expanduser(raw))
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _read_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values

    env_re = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$")
    for line in path.read_text().splitlines():
        match = env_re.match(line)
        if match is None:
            continue
        value = match.group(2).strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[match.group(1)] = value
    return values


def _write_env(path: Path, updates: dict[str, str]) -> None:
    if not path.exists():
        example = REPO_ROOT / ".env.example"
        if example.exists():
            shutil.copyfile(example, path)
        else:
            path.write_text("")

    env_re = re.compile(r"^(\s*)(export\s+)?([A-Za-z_][A-Za-z0-9_]*)=.*$")
    lines = path.read_text().splitlines()
    seen: set[str] = set()
    output: list[str] = []

    for line in lines:
        match = env_re.match(line)
        if match is None or match.group(3) not in updates:
            output.append(line)
            continue

        key = match.group(3)
        prefix = match.group(1)
        export = match.group(2) or ""
        output.append(f"{prefix}{export}{key}={updates[key]}")
        seen.add(key)

    if output and output[-1] != "":
        output.append("")
    for key, value in updates.items():
        if key not in seen:
            output.append(f"{key}={value}")

    path.write_text("\n".join(output).rstrip() + "\n")


def _resolve_selector(value: str) -> Path:
    explicit_candidates: list[Path] = []
    raw = Path(os.path.expanduser(value))
    if raw.is_absolute():
        explicit_candidates.append(raw)
    else:
        explicit_candidates.append(REPO_ROOT / raw)
        explicit_candidates.append(MODEL_ZOO / raw)

    if ":" in value and not raw.exists():
        model_name, checkpoint_name = value.rsplit(":", 1)
        explicit_candidates.append(MODEL_ZOO / model_name / "checkpoints" / checkpoint_name)

    for candidate in explicit_candidates:
        if _checkpoint_from_path(candidate) is not None:
            return candidate

    matches: list[Checkpoint] = []
    for checkpoint in discover_checkpoints():
        job = checkpoint.selector.rsplit(":", 1)[0]
        names = {
            checkpoint.selector,
            checkpoint.path.name,
            job,
            Path(job).name,
            _format_env_path(checkpoint.path),
        }
        if checkpoint.path.name != "last":
            names.discard(Path(job).name)
            names.discard(job)
        if value in names:
            matches.append(checkpoint)

    unique = {match.path: match for match in matches}
    if len(unique) == 1:
        return next(iter(unique.values())).path
    if len(unique) > 1:
        choices = "\n  ".join(sorted(match.selector for match in unique.values()))
        raise SystemExit(f"Ambiguous checkpoint selector '{value}'. Matches:\n  {choices}")

    raise SystemExit(f"Checkpoint not found or missing config.json: {value}")


def _print_table(rows: list[tuple[str, str, str, str]]) -> None:
    if not rows:
        print("No local inference checkpoints found under model_zoo/.")
        return

    headers = ("selector", "policy", "action", "path")
    widths = [len(header) for header in headers]
    for row in rows:
        widths = [max(width, len(value)) for width, value in zip(widths, row, strict=True)]

    fmt = "  ".join(f"{{:<{width}}}" for width in widths)
    print(fmt.format(*headers))
    print(fmt.format(*(('-' * width) for width in widths)))
    for row in rows:
        print(fmt.format(*row))


def command_list(_args: argparse.Namespace) -> None:
    rows: list[tuple[str, str, str, str]] = []
    for checkpoint in discover_checkpoints():
        path = _format_env_path(checkpoint.path)
        if checkpoint.path.is_symlink():
            path = f"{path} -> {os.readlink(checkpoint.path)}"
        rows.append((checkpoint.selector, checkpoint.policy_type, checkpoint.action_type, path))
    _print_table(rows)


def _source_metadata(checkpoint_path: Path) -> dict:
    if checkpoint_path.parent.name == "checkpoints":
        return _read_json(checkpoint_path.parent.parent / "source.json")
    return {}


def _show_checkpoint(checkpoint_path: Path, env: dict[str, str] | None = None) -> None:
    checkpoint = _checkpoint_from_path(checkpoint_path)
    if checkpoint is None:
        raise SystemExit(f"Checkpoint not found or missing config.json: {checkpoint_path}")

    expected_extras = POLICY_EXTRAS.get(checkpoint.policy_type)
    print(f"Selector: {checkpoint.selector}")
    print(f"Path: {_format_env_path(checkpoint.path)}")
    print(f"Policy: {checkpoint.policy_type}")
    print(f"Action type: {checkpoint.action_type}")
    if checkpoint.task_description:
        print(f"Task: {checkpoint.task_description}")
    if expected_extras is not None:
        print(f"Expected LEROBOT_EXTRAS: {expected_extras or '<empty>'}")
    if env is not None:
        print(f"Configured LEROBOT_EXTRAS: {env.get('LEROBOT_EXTRAS', '<unset>') or '<empty>'}")
        if "HF_CACHE" in env:
            print(f"HF_CACHE: {env['HF_CACHE']}")

    source = _source_metadata(checkpoint.path)
    if source:
        print(f"Source repo: {source.get('repo_id', '<unknown>')}")
        print(f"Source revision: {source.get('revision', '<unknown>')}")


def command_show(args: argparse.Namespace) -> None:
    env_path = _absolute_no_resolve(Path(args.env_file))
    env = _read_env(env_path)

    if args.target:
        checkpoint_path = _resolve_selector(args.target)
    elif "MODEL_PATH" in env:
        checkpoint_path = _expand_model_path(env["MODEL_PATH"])
    elif CURRENT_LINK.exists():
        checkpoint_path = CURRENT_LINK
    else:
        raise SystemExit("No MODEL_PATH in .env and no model_zoo/inference/current link.")

    print(f"Env file: {_format_env_path(env_path)}")
    if "MODEL_PATH" in env:
        print(f"MODEL_PATH: {env['MODEL_PATH']}")
    _show_checkpoint(checkpoint_path, env)

    if CURRENT_LINK.exists() or CURRENT_LINK.is_symlink():
        print(f"Current link: {_format_env_path(CURRENT_LINK)} -> {os.readlink(CURRENT_LINK)}")


def command_set(args: argparse.Namespace) -> None:
    env_path = _absolute_no_resolve(Path(args.env_file))
    old_env = _read_env(env_path)
    checkpoint_path = _resolve_selector(args.target)
    checkpoint = _checkpoint_from_path(checkpoint_path)
    if checkpoint is None:
        raise SystemExit(f"Checkpoint not found or missing config.json: {checkpoint_path}")

    extras = POLICY_EXTRAS.get(checkpoint.policy_type)
    if extras is None:
        known = ", ".join(sorted(POLICY_EXTRAS))
        raise SystemExit(f"No LEROBOT_EXTRAS mapping for policy '{checkpoint.policy_type}'. Known: {known}")

    selector_job = checkpoint.selector.rsplit(":", 1)[0]
    profile = MODEL_PROFILES.get(selector_job, {})
    configured_extras = profile.get("extras", extras or old_env.get("LEROBOT_EXTRAS", ""))
    CURRENT_LINK.parent.mkdir(parents=True, exist_ok=True)
    if CURRENT_LINK.exists() or CURRENT_LINK.is_symlink():
        CURRENT_LINK.unlink()
    target_abs = _absolute_no_resolve(checkpoint.path)
    rel_target = os.path.relpath(target_abs, CURRENT_LINK.parent)
    CURRENT_LINK.symlink_to(rel_target)

    updates = {
        "MODEL_PATH": _format_env_path(checkpoint.path),
        "LEROBOT_EXTRAS": configured_extras,
        "ACTION_TYPE": checkpoint.action_type,
    }
    if profile:
        updates["CONFIG_FILE"] = profile["config_file"]
        updates["INFERENCE_ARM"] = profile["inference_arm"]
    _write_env(env_path, updates)

    print("Set inference model:")
    for key, value in updates.items():
        print(f"  {key}={value or '<empty>'}")
    print(f"  current -> {rel_target}")
    print("")
    _show_checkpoint(checkpoint.path, _read_env(env_path))

    if old_env.get("LEROBOT_EXTRAS") != configured_extras:
        print("")
        print("LEROBOT_EXTRAS changed; rebuild the inference image before running:")
        print("  docker compose build")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env-file",
        default=str(DEFAULT_ENV_FILE),
        help="Env file to read or update (default: ./.env)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list", help="List local checkpoints under model_zoo/")

    show_parser = subparsers.add_parser("show", help="Show the selected inference checkpoint")
    show_parser.add_argument("target", nargs="?", help="Optional checkpoint path or selector")

    set_parser = subparsers.add_parser("set", aliases=["use"], help="Set .env to a checkpoint")
    set_parser.add_argument("target", help="Checkpoint path or selector from the list command")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "list":
        command_list(args)
    elif args.command == "show":
        command_show(args)
    elif args.command in {"set", "use"}:
        command_set(args)
    else:
        parser.error(f"Unknown command: {args.command}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
