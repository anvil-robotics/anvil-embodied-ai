#!/usr/bin/env python3
"""Smoke test for `mcap-valid` and its `mcap-convert` quality-flag integration.

Complements the pytest unit tests for mcap_converter/core/quality.py and the
--skip-flagged / --skip-episode-idx plumbing in mcap_converter/cli/convert.py
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

B  `mcap-convert` quality-flag integration (subprocess, real conversion)
   B1. generate a real mcap-valid JSON report, then build a synthetic variant
       with one episode forced to "critical" and another to "warning"
   B2. --skip-flagged (bare = critical only) skips the critical episode only
   B3. --skip-flagged warning skips both critical and warning episodes
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


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)


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

    base_cmd = [
        "uv", "run", "mcap-valid",
        "-i", str(MCAP_ROOT),
        "--config", str(CONFIG),
    ]

    # A1 — table format runs cleanly
    print("\n  A1. table format")
    proc = _run(base_cmd)
    _assert("A1 exit code 0", proc.returncode == 0, f"exit {proc.returncode}")
    _assert(
        "A1 summary line reports 5 episodes",
        "5 episodes:" in proc.stdout,
        proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else "(empty stdout)",
    )

    # A2 — JSON format has the expected shape
    print("\n  A2. --format json")
    proc = _run(base_cmd + ["--format", "json"])
    _assert("A2 exit code 0", proc.returncode == 0, f"exit {proc.returncode}")
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        _assert("A2 stdout parses as JSON", False, str(exc))
    else:
        _assert("A2 stdout parses as JSON", True)
        _assert_episode_shape("A2", payload)

    # A3 — --output PATH writes the report to a file (table format too)
    print("\n  A3. --output PATH (table format)")
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "report.json"
        proc = _run(base_cmd + ["--output", str(out_path)])
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

    # A4 — --fail-on-critical exits 0 on the healthy fixture
    print("\n  A4. --fail-on-critical on healthy fixture")
    proc = _run(base_cmd + ["--fail-on-critical"])
    _assert(
        "A4 --fail-on-critical exits 0 (no critical episodes)",
        proc.returncode == 0,
        f"exit {proc.returncode}",
    )

    # A5 — --fail-on-critical exits 1 when the file itself is unreadable
    print("\n  A5. --fail-on-critical with an unreadable .mcap file")
    with tempfile.TemporaryDirectory() as tmpdir:
        bad_mcap = Path(tmpdir) / "garbage.mcap"
        bad_mcap.write_bytes(b"not a real mcap file")
        proc = _run([
            "uv", "run", "mcap-valid",
            "-i", str(tmpdir),
            "--fail-on-critical",
        ])
        _assert(
            "A5 --fail-on-critical exits 1 on read_error (critical, not passed)",
            proc.returncode == 1,
            f"exit {proc.returncode}",
        )


# ── Section B: mcap-convert quality-flag integration ──────────────────────────


def run_section_b() -> None:
    print(f"\n{'═'*70}")
    print("  SECTION B — mcap-convert quality-flag integration")
    print(f"{'═'*70}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # B1 — generate a real report, then build a synthetic critical+warning variant
        print("\n  B1. generate real quality report, synthesize critical+warning variant")
        report_path = tmp / "report.json"
        proc = _run([
            "uv", "run", "mcap-valid",
            "-i", str(MCAP_ROOT),
            "--config", str(CONFIG),
            "--format", "json",
            "--output", str(report_path),
        ])
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
            "-i", str(MCAP_ROOT),
            "--config", str(CONFIG),
            "--robot-type", "anvil_openarm",
        ]

        # B2 — bare --skip-flagged: critical only
        print("\n  B2. --skip-flagged (bare, default = critical only)")
        out1 = tmp / "out1"
        proc = _run(base_convert_cmd + [
            "-o", str(out1),
            "--quality-report", str(synthetic_path),
            "--skip-flagged",
        ])
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

        # B3 — --skip-flagged warning: critical + warning
        print("\n  B3. --skip-flagged warning (skips critical AND warning)")
        out2 = tmp / "out2"
        proc = _run(base_convert_cmd + [
            "-o", str(out2),
            "--quality-report", str(synthetic_path),
            "--skip-flagged", "warning",
        ])
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
            "--skip-episode-idx", "2:4",
        ])
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
            "--skip-episode-idx", "99",
        ])
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
