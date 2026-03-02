# Beautify `mcap-convert` CLI with Rich Progress & Report

**Date**: 2025-02-25
**Task**: Replace plain `print()` output in `mcap-convert` CLI with Rich panels, progress bars, and tables.

## Context

The `mcap-convert` CLI used plain `print(f"[{timestamp}] ...")` statements for all output — startup banners, progress updates, and final reports were text walls with `=` separators. The goal was to make the CLI professional and visually polished using the `rich` library while keeping core modules (`extractor.py`, `writer.py`) free of presentation concerns.

## Implementation

### Files Modified (4)

1. **`packages/mcap_converter/pyproject.toml`** — Added `"rich>=13.0"` to dependencies.

2. **`packages/mcap_converter/src/mcap_converter/core/extractor.py`** — Added `quiet: bool` and `progress_callback: Optional[Callable[[int], None]]` parameters to `BufferedStreamExtractor.__init__()`. All 5 print statements in `extract_frames()` are guarded with `if not self.quiet:`. The callback is invoked after each yielded frame (both in main loop and flush loop), enabling the caller to drive Rich progress updates.

3. **`packages/mcap_converter/src/mcap_converter/core/writer.py`** — Added `quiet: bool` parameter to `LeRobotWriter.__init__()`. All prints in `create_dataset()`, `add_episode()`, and `finalize()` are guarded.

4. **`packages/mcap_converter/src/mcap_converter/cli/convert.py`** — Major rewrite:
   - **Startup banner**: Blue `rich.Panel` with a `Table` grid showing all config parameters.
   - **Detection logs**: `console.log()` with auto-timestamps (removed `get_timestamp()` helper).
   - **Episode progress**: `rich.Progress` with overall + per-episode bars. Per-episode bar shows real-time frame count and speed (f/s), turns green on completion.
   - **Final report**: Green `rich.Panel` containing summary table, per-episode breakdown table, and timing table.
   - **Error handling**: `console.print_exception()` replaces `traceback.print_exc()`.
   - **Hub upload**: `console.status()` spinner.
   - Both `LeRobotWriter` and `BufferedStreamExtractor` are created with `quiet=True`.

### Design Decision: `quiet`/`callback` vs. Rich in Core

Rich imports are confined to `convert.py` only. Core modules (`extractor.py`, `writer.py`) received generic `quiet`/`callback` parameters so other CLI tools (`mcap-inspect`, `dataset-validate`, etc.) remain unaffected and don't gain a Rich dependency at the module level.

## Validation

- `uv sync --all-packages` — Installed `rich==14.3.3` + deps (`markdown-it-py`, `mdurl`).
- `python3 -c "from mcap_converter.cli.convert import main; print('OK')"` — Import OK.
- `mcap-convert --help` — CLI fully functional with correct argument parsing.
- Rich rendering test — Startup banner and final report Panel/Table render correctly in terminal.
- All other CLI tools (`mcap-inspect --help`, `dataset-validate --help`) — Unaffected, working.

## Architecture Note

Changes are strictly in the CLI presentation layer. The core data pipeline (`McapReader -> BufferedStreamExtractor -> LeRobotWriter`) is architecturally unchanged. New parameters (`quiet`, `progress_callback`) are backward-compatible with default values that preserve existing behavior. Consistent with the modular architecture defined in `docs/architecture.md`.
