"""dataset-config-migrate: upgrade a dataset's conversion_config.yaml to the current schema.

Permanent, reusable CLI companion to ``mcap_converter.config.versioning`` — reads an
already-converted dataset's ``conversion_config.yaml``, applies the registered migration
chain if it's behind the current schema version, and writes the upgraded config back in
place, backing up the original under a version-tagged filename first. See
``claude_docs/mcap-converter-encoding-refactor-plan.md`` Part 0b for the full design.

Any tool that auto-discovers ``conversion_config.yaml`` by its canonical name (e.g.
``anvil_eval_ros``'s config-discovery logic) transparently picks up the upgraded file with
no change to its own lookup logic, once this has been run.

Behavior, precisely (see the design doc's "Gap" numbering for why each of these exists):

- Already at the current schema version: prints an informational message and performs
  ZERO file operations — no re-serialize, no rename, not even with identical content
  (Gap 4).
- An existing version-tagged backup file (``conversion_config_v<old>.yaml``) already
  present at the target path: refuses by default, explaining the directory appears to
  already have been migrated; ``--force`` overwrites it (Gap 3).
- Before any file is touched, prints the exact plan (what will be renamed, what will be
  written) and requires an explicit ``y``/``yes`` confirmation. ``--force`` does NOT skip
  this prompt — its only effect is the backup-collision check above (Gap 2).
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from mcap_converter.config.loader import ConfigLoader
from mcap_converter.config.versioning import CURRENT_SCHEMA_VERSION, detected_version

log = logging.getLogger(__name__)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )


def parse_args(args=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Upgrade a dataset's conversion_config.yaml to the current schema version "
            f"({CURRENT_SCHEMA_VERSION})."
        ),
    )
    parser.add_argument(
        "--dataset", type=str, required=True,
        help="path to the dataset directory containing conversion_config.yaml",
    )
    parser.add_argument(
        "--force", action="store_true",
        help=(
            "overwrite an existing version-tagged backup file if one is already present. "
            "Does NOT skip the yes/no confirmation prompt — see module docstring."
        ),
    )
    return parser.parse_args(args)


def _confirm(prompt: str) -> bool:
    answer = input(prompt).strip().lower()
    return answer in ("y", "yes")


def main(args=None) -> int:
    setup_logging()
    parsed = parse_args(args)

    cfg_path = Path(parsed.dataset) / "conversion_config.yaml"
    if not cfg_path.exists():
        log.error("[config-migrate] %s not found.", cfg_path)
        return 1

    raw = ConfigLoader.load_yaml(str(cfg_path))
    detected = detected_version(raw)

    # Gap 4: already-current is a true no-op — zero file operations, checked before
    # anything else touches the filesystem. Not even a re-serialize with identical
    # content, which would still touch mtime and could trip unrelated downstream
    # staleness logic.
    if detected == CURRENT_SCHEMA_VERSION:
        log.info(
            "[config-migrate] %s is already at current schema version (v%s) — nothing to "
            "do.", cfg_path, detected,
        )
        return 0

    backup_path = cfg_path.with_name(f"conversion_config_v{detected}.yaml")

    # Gap 3: reject-by-default on backup collision; --force overwrites it. Checked before
    # the confirmation prompt so we never ask the user to confirm an operation we're about
    # to refuse anyway.
    if backup_path.exists() and not parsed.force:
        log.error(
            "[config-migrate] %s already exists — this directory appears to have already "
            "been migrated. Use --force to overwrite the existing backup.",
            backup_path,
        )
        return 1

    # Gap 2: confirmation is independent of --force — --force's only effect is the
    # backup-collision check above, never this prompt.
    print(
        "This will:\n"
        f"  1. Rename {cfg_path} -> {backup_path}\n"
        f"  2. Write the upgraded config (schema v{CURRENT_SCHEMA_VERSION}) to {cfg_path}\n"
    )
    if not _confirm("Proceed? [y/N]: "):
        log.info("[config-migrate] Aborted — no file operations performed.")
        return 1

    cfg_path.rename(backup_path)
    # Lenient: this tool's job is upgrading the version, not policing unrelated stray keys
    # left over in the old file — any truly-unknown key is simply dropped from the
    # rewritten output, the desired "clean upgrade" behavior.
    upgraded = ConfigLoader.from_dict(raw, strict=False)
    ConfigLoader.to_yaml(upgraded, str(cfg_path))

    log.info(
        "[config-migrate] Migrated %s: v%s -> v%s. Original backed up at %s.",
        cfg_path, detected, CURRENT_SCHEMA_VERSION, backup_path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
