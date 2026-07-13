#!/usr/bin/env python3
"""Smoke test for `mcap-valid` and its `mcap-convert` quality-flag integration.

Complements the pytest unit tests for mcap_converter/core/quality.py and the
--include-flagged / --skip-episode-idx plumbing in mcap_converter/cli/convert.py
with real CLI-level coverage — every command below is a real `uv run
mcap-valid` / `uv run mcap-convert` subprocess invocation against the
committed fixture at tests/smoke/fixtures/test-session (5 stub MCAPs, single
right arm), not a direct Python import.

Sections
--------
A  `mcap-valid` basic CLI behavior (fast, no mutation)
   A1. table format runs cleanly, summary line reports "5 episodes"
   A2. --format json produces valid JSON with the expected shape
   A3. --output PATH writes the JSON report to a file (table format too)
   A4. --fail-on-critical exits 0 on the healthy fixture
   A5. --fail-on-critical exits 1 when the input file itself is unreadable
   A6. default behavior (no flags) always writes a JSON + Markdown report to
       <input>/mcap_valid_reports/report.{json,md}, inside the input itself

Note: mcap-valid unconditionally writes <input>/mcap_valid_reports/report.{json,md}
inside the `-i` input's own resolved location — not relative to cwd. So every
`mcap-valid` subprocess below that needs the default (non---output) report
path runs against a temp COPY of the committed test-session fixture (see
`_copy_session_into`) rather than the real committed `MCAP_ROOT` directly;
pointing `-i` at MCAP_ROOT would otherwise leave mcap_valid_reports/ behind
inside the tracked fixture tree as a side effect of this smoke test. `cwd` is
still pointed at an isolated temp directory too, both to host that fixture
copy and to keep any other cwd-relative side effects out of the repo.

B  `mcap-convert` quality-flag integration (subprocess, real conversion)
   B1. generate a real mcap-valid JSON report, then build a synthetic variant
       with one episode forced to "critical" and another to "warning"
   B2. --include-flagged warning (the default) skips the critical episode only
   B3. --include-flagged pass skips both critical and warning episodes
   B4. --skip-episode-idx "2:4" uses Python-slice exclusive-end semantics
       (skips episodes 2 and 3, NOT 4) at the CLI level
   B5. --skip-episode-idx with an out-of-range index fails cleanly (no
       traceback) and does not mutate a pre-existing, non-empty output dir

Usage
-----
  # All sections
  uv run python tests/smoke/scripts/mcap_valid_smoke_test.py

  # Section A only (fast, no mcap-convert subprocesses)
  uv run python tests/smoke/scripts/mcap_valid_smoke_test.py --skip-convert
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]   # tests/smoke/scripts/ → repo root
SMOKE_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = SMOKE_ROOT / "fixtures"
MCAP_ROOT = FIXTURES / "test-session"
CONFIG = FIXTURES / "configs" / "mcap-converter-smoke-test-cmd.yaml"

_EXPECTED_EPISODES = 5

# ── result tracking ───────────────────────────────────────────────────────────

_results: list[tuple[str, bool, str]] = []   # (name, ok, detail)


def _assert(name: str, condition: bool, detail: str = "") -> bool:
    status = "PASS" if condition else "FAIL"
    line = f"  {status:<4}  {name}"
    if detail:
        line += f"  [{detail}]"
    print(line, flush=True)
    _results.append((name, condition, detail))
    return condition


def _skip(name: str, reason: str) -> None:
    print(f"  SKIP  {name}  [{reason}]", flush=True)
    _results.append((name, True, f"skipped: {reason}"))


# ── helpers ───────────────────────────────────────────────────────────────────


def _run(cmd: list[str], *, cwd: Path) -> subprocess.CompletedProcess:
    """Run a `uv run <cli>` subprocess with an explicit, isolated `cwd`.

    `mcap-valid` unconditionally writes <input>/mcap_valid_reports/report.{json,md}
    inside its `-i` input's own resolved location (not relative to cwd), so
    callers that exercise the default report path must pass `-i` pointed at a
    temp COPY of the fixture (see `_copy_session_into`), never the real
    committed `MCAP_ROOT` directly — otherwise the write would land inside
    the tracked fixture tree. `cwd` is still isolated here too, both to host
    that fixture copy and to keep any other cwd-relative output (e.g.
    --output PATH targets) out of the repo. `--project REPO` is injected so
    `uv run` still finds the project regardless of cwd.
    """
    if cmd[:2] == ["uv", "run"]:
        cmd = cmd[:2] + ["--project", str(REPO)] + cmd[2:]
    print(f"  $ (cwd={cwd}) {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def _copy_session_into(dest_root: Path) -> Path:
    """Copy the real committed test-session fixture into an isolated temp
    directory and return the copy's path.

    mcap-valid's default report now writes inside `-i`'s own resolved
    location, not cwd — pointing `-i` at MCAP_ROOT directly would write
    mcap_valid_reports/ into the tracked fixture tree on every smoke test run.
    """
    dest = dest_root / "test-session"
    shutil.copytree(MCAP_ROOT, dest)
    return dest


def _assert_episode_shape(name: str, payload: dict) -> bool:
    """Assert payload has an 'episodes' list of the expected length, each with
    the required keys. Returns True iff every check passes."""
    if not _assert(f"{name}: has 'episodes' key", "episodes" in payload):
        return False
    episodes = payload["episodes"]
    ok = _assert(
        f"{name}: {_EXPECTED_EPISODES} episodes reported",
        len(episodes) == _EXPECTED_EPISODES,
        f"got {len(episodes)}",
    )
    required_keys = {"path", "severity", "passed"}
    missing = [i for i, ep in enumerate(episodes) if not required_keys.issubset(ep)]
    ok = _assert(
        f"{name}: each episode has path/severity/passed keys",
        not missing,
        f"missing in episodes at index {missing}" if missing else "",
    ) and ok
    return ok


# ── Section A: mcap-valid basic CLI behavior ──────────────────────────────────


def run_section_a() -> None:
    print(f"\n{'═'*70}")
    print("  SECTION A — mcap-valid CLI behavior (no mutation)")
    print(f"{'═'*70}")

    with tempfile.TemporaryDirectory() as base_tmpdir:
        base_cwd = Path(base_tmpdir)
        session_copy = _copy_session_into(base_cwd)
        base_cmd = [
            "uv", "run", "mcap-valid",
            "-i", str(session_copy),
        ]

        # A1 — table format runs cleanly
        print("\n  A1. table format")
        proc = _run(base_cmd, cwd=base_cwd)
        _assert("A1 exit code 0", proc.returncode == 0, f"exit {proc.returncode}")
        _assert(
            "A1 summary line reports 5 episodes",
            "5 episodes:" in proc.stdout,
            proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else "(empty stdout)",
        )

        # A2 — JSON format has the expected shape
        print("\n  A2. --format json")
        proc = _run(base_cmd + ["--format", "json"], cwd=base_cwd)
        _assert("A2 exit code 0", proc.returncode == 0, f"exit {proc.returncode}")
        try:
            payload = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            _assert("A2 stdout parses as JSON", False, str(exc))
        else:
            _assert("A2 stdout parses as JSON", True)
            _assert_episode_shape("A2", payload)

        # A4 — --fail-on-critical exits 0 on the healthy fixture
        print("\n  A4. --fail-on-critical on healthy fixture")
        proc = _run(base_cmd + ["--fail-on-critical"], cwd=base_cwd)
        _assert(
            "A4 --fail-on-critical exits 0 (no critical episodes)",
            proc.returncode == 0,
            f"exit {proc.returncode}",
        )

    # A3 — --output PATH writes the report to a file (table format too)
    print("\n  A3. --output PATH (table format)")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        a3_session = _copy_session_into(tmp)
        a3_cmd = ["uv", "run", "mcap-valid", "-i", str(a3_session)]
        out_path = tmp / "report.json"
        proc = _run(a3_cmd + ["--output", str(out_path)], cwd=tmp)
        _assert("A3 exit code 0", proc.returncode == 0, f"exit {proc.returncode}")
        _assert("A3 output file created", out_path.exists(), str(out_path))
        if out_path.exists():
            try:
                payload = json.loads(out_path.read_text())
            except json.JSONDecodeError as exc:
                _assert("A3 output file parses as JSON", False, str(exc))
            else:
                _assert("A3 output file parses as JSON", True)
                _assert_episode_shape("A3", payload)

    # A5 — --fail-on-critical exits 1 when the file itself is unreadable
    print("\n  A5. --fail-on-critical with an unreadable .mcap file")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        bad_mcap = tmp / "garbage.mcap"
        bad_mcap.write_bytes(b"not a real mcap file")
        proc = _run([
            "uv", "run", "mcap-valid",
            "-i", str(tmp),
            "--fail-on-critical",
        ], cwd=tmp)
        _assert(
            "A5 --fail-on-critical exits 1 on read_error (critical, not passed)",
            proc.returncode == 1,
            f"exit {proc.returncode}",
        )

    # A6 — default behavior (no flags): JSON + Markdown report always written
    print("\n  A6. default report files (no flags needed)")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        a6_session = _copy_session_into(tmp)
        a6_cmd = ["uv", "run", "mcap-valid", "-i", str(a6_session)]
        proc = _run(a6_cmd, cwd=tmp)
        _assert("A6 exit code 0", proc.returncode == 0, f"exit {proc.returncode}")

        # Report now lives inside the input's own location (session dir),
        # not under cwd: <session_copy>/mcap_valid_reports/report.{json,md}.
        default_json = a6_session / "mcap_valid_reports" / "report.json"
        default_md = a6_session / "mcap_valid_reports" / "report.md"
        _assert(
            "A6 default JSON report written and non-empty",
            default_json.exists() and default_json.stat().st_size > 0,
            str(default_json),
        )
        _assert(
            "A6 default Markdown report written and non-empty",
            default_md.exists() and default_md.stat().st_size > 0,
            str(default_md),
        )
        if default_json.exists():
            try:
                payload = json.loads(default_json.read_text())
            except json.JSONDecodeError as exc:
                _assert("A6 default JSON report parses as JSON", False, str(exc))
            else:
                _assert("A6 default JSON report parses as JSON", True)
                _assert_episode_shape("A6", payload)
        if default_md.exists():
            md_text = default_md.read_text()
            # Episode headers are uniquely "### `filename`" (backtick right after
            # the space) — scoped this way so it doesn't also match the "###
            # Topic Health Overview" / "### Flagged Topics" subsection headings
            # that live under the same report's Summary section.
            _assert(
                "A6 default Markdown report has a header per episode",
                md_text.count("### `") == _EXPECTED_EPISODES,
                f"got {md_text.count('### `')} headers",
            )


# ── Section B: mcap-convert quality-flag integration ──────────────────────────


def run_section_b() -> None:
    print(f"\n{'═'*70}")
    print("  SECTION B — mcap-convert quality-flag integration")
    print(f"{'═'*70}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        # A temp copy of the fixture, not MCAP_ROOT directly: mcap-valid's
        # unconditional default report write would otherwise land inside the
        # tracked fixture tree. mcap-convert below is also pointed at this
        # same copy (not MCAP_ROOT) so its episode paths match the ones
        # baked into the quality report generated from it.
        session_copy = _copy_session_into(tmp)

        # B1 — generate a real report, then build a synthetic critical+warning variant
        print("\n  B1. generate real quality report, synthesize critical+warning variant")
        report_path = tmp / "report.json"
        proc = _run([
            "uv", "run", "mcap-valid",
            "-i", str(session_copy),
            "--format", "json",
            "--output", str(report_path),
        ], cwd=tmp)
        if not _assert("B1 mcap-valid report generated (exit 0)", proc.returncode == 0,
                       f"exit {proc.returncode}"):
            return
        if not _assert("B1 report file exists", report_path.exists()):
            return

        report = json.loads(report_path.read_text())
        episodes = report["episodes"]
        if not _assert("B1 report has 5 episodes", len(episodes) == _EXPECTED_EPISODES,
                       f"got {len(episodes)}"):
            return

        critical_path = episodes[0]["path"]
        warning_path = episodes[1]["path"]
        for ep in episodes:
            if ep["path"] == critical_path:
                ep["severity"] = "critical"
                ep["passed"] = False
            elif ep["path"] == warning_path:
                ep["severity"] = "warning"
        synthetic_path = tmp / "synthetic_report.json"
        synthetic_path.write_text(json.dumps(report, indent=2))
        _assert(
            "B1 synthetic report written with 1 critical + 1 warning episode",
            True,
            f"critical={Path(critical_path).name}, warning={Path(warning_path).name}",
        )

        base_convert_cmd = [
            "uv", "run", "mcap-convert",
            "-i", str(session_copy),
            "--config", str(CONFIG),
            "--robot-type", "anvil_openarm",
        ]

        # B2 — --include-flagged warning (the default): critical only skipped
        print("\n  B2. --include-flagged warning (the default, skips critical only)")
        out1 = tmp / "out1"
        proc = _run(base_convert_cmd + [
            "-o", str(out1),
            "--quality-report", str(synthetic_path),
            "--include-flagged", "warning",
        ], cwd=tmp)
        _assert("B2 exit code 0", proc.returncode == 0, f"exit {proc.returncode}")
        _assert(
            "B2 stdout reports critical episode skipped",
            "skipped (quality: critical)" in proc.stdout,
        )
        _assert(
            "B2 stdout does NOT report warning episode skipped (converts normally)",
            "skipped (quality: warning)" not in proc.stdout,
        )
        info1 = out1 / "test-session" / "meta" / "info.json"
        if _assert("B2 dataset info.json exists", info1.exists(), str(info1)):
            total1 = json.loads(info1.read_text()).get("total_episodes")
            _assert(
                "B2 dataset has 4 episodes (5 - 1 critical skip)",
                total1 == 4,
                f"total_episodes={total1}",
            )

        # B3 — --include-flagged pass: critical + warning both skipped
        print("\n  B3. --include-flagged pass (skips critical AND warning)")
        out2 = tmp / "out2"
        proc = _run(base_convert_cmd + [
            "-o", str(out2),
            "--quality-report", str(synthetic_path),
            "--include-flagged", "pass",
        ], cwd=tmp)
        _assert("B3 exit code 0", proc.returncode == 0, f"exit {proc.returncode}")
        _assert(
            "B3 stdout reports critical episode skipped",
            "skipped (quality: critical)" in proc.stdout,
        )
        _assert(
            "B3 stdout reports warning episode skipped",
            "skipped (quality: warning)" in proc.stdout,
        )
        info2 = out2 / "test-session" / "meta" / "info.json"
        if _assert("B3 dataset info.json exists", info2.exists(), str(info2)):
            total2 = json.loads(info2.read_text()).get("total_episodes")
            _assert(
                "B3 dataset has 3 episodes (5 - 2 skips)",
                total2 == 3,
                f"total_episodes={total2}",
            )

        # B4 — --skip-episode-idx exclusive-end range
        print("\n  B4. --skip-episode-idx \"2:4\" (exclusive end: skips 2,3 not 4)")
        out3 = tmp / "out3"
        proc = _run(base_convert_cmd + [
            "-o", str(out3),
            "--quality-report", str(report_path),
            "--skip-episode-idx", "2:4",
        ], cwd=tmp)
        _assert("B4 exit code 0", proc.returncode == 0, f"exit {proc.returncode}")
        manual_skip_count = proc.stdout.count("skipped (manual index)")
        _assert(
            "B4 exactly 2 episodes skipped by manual index (episodes 2 and 3)",
            manual_skip_count == 2,
            f"count={manual_skip_count}",
        )
        info3 = out3 / "test-session" / "meta" / "info.json"
        if _assert("B4 dataset info.json exists", info3.exists(), str(info3)):
            total3 = json.loads(info3.read_text()).get("total_episodes")
            _assert(
                "B4 dataset has 3 episodes (1, 4, 5)",
                total3 == 3,
                f"total_episodes={total3}",
            )

        # B5 — out-of-range --skip-episode-idx fails cleanly, no mutation
        print("\n  B5. --skip-episode-idx 99 (out of range) fails without mutating output dir")
        out4_base = tmp / "out4"
        out4_dataset = out4_base / "test-session"
        out4_dataset.mkdir(parents=True)
        marker = out4_dataset / "marker.txt"
        marker.write_text("pre-existing content")
        proc = _run(base_convert_cmd + [
            "-o", str(out4_base),
            "--quality-report", str(report_path),
            "--skip-episode-idx", "99",
        ], cwd=tmp)
        _assert(
            "B5 exit code != 0 (out-of-range index rejected)",
            proc.returncode != 0,
            f"exit {proc.returncode}",
        )
        _assert(
            "B5 no raw traceback in output",
            "Traceback" not in proc.stdout and "Traceback" not in proc.stderr,
        )
        _assert(
            "B5 pre-existing output dir NOT mutated (marker file survives)",
            marker.exists(),
            str(marker),
        )


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--skip-convert", action="store_true",
                   help="run section A only (skip mcap-convert subprocess section B)")
    args = p.parse_args()

    print(f"mcap_valid_smoke_test  repo={REPO}")

    run_section_a()

    if not args.skip_convert:
        run_section_b()
    else:
        print("\n[skipping section B — --skip-convert]")

    # ── summary ───────────────────────────────────────────────────────────────
    passed = sum(1 for _, ok, _ in _results if ok)
    failed = sum(1 for _, ok, d in _results if not ok and not d.startswith("skipped"))
    skipped = sum(1 for _, ok, d in _results if ok and d.startswith("skipped"))

    print(f"\n{'─'*70}")
    if failed:
        failures = [(n, d) for n, ok, d in _results if not ok]
        print(f"FAILURES ({len(failures)}):")
        for name, detail in failures:
            print(f"  ✗ {name}")
            if detail:
                print(f"    {detail}")
    print(f"Total: {passed - skipped} passed, {failed} failed, {skipped} skipped")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
