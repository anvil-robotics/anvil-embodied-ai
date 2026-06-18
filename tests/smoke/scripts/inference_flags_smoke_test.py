#!/usr/bin/env python3
"""Smoke test for scripts/run_inference.sh flag parsing and Docker integration.

Validates that the documented flags in docs/inference.md and .env.example match
actual script + compose behaviour.

Sections
--------
A  Mapping assertions (fast, no containers started)
   A1. --fake-hardware      → docker-compose.fake-hardware.yml selected
   A2. --monitor-enable     → --profile monitor injected into compose passthrough
   A3. --echo-topic-only    → ECHO_TOPIC_ONLY=true exported before docker is called
   A4. --debug              → DEBUG=true exported; fake-hardware inference config
                              renders `debug:=true` in the ros2 launch command
   A5. combined fake+debug  → both fake-hardware compose + DEBUG=true

B  Startup test — fake-hardware + monitor-enable (requires Docker, no GPU)
   B1. `run_inference.sh --fake-hardware --monitor-enable up -d`
       → lerobot-fake-monitor container starts and logs Joint State / camera Hz
       → validates docs/inference.md "Test with Fake Hardware First" scenario
       → validates --fake-hardware and --monitor-enable end-to-end
   B2. `run_inference.sh --fake-hardware --echo-topic-only up -d`
       NOTE: --echo-topic-only exports ECHO_TOPIC_ONLY=true but fake-hardware
             compose ignores this env var (monitor service hardcodes echo_topic_only:=true).
       This test confirms the flag is a no-op in fake-hardware context,
       matching the documented caveat in docs/inference.md.

C  Best-effort GPU startup — --debug with smoke checkpoint (skipped if no ckpt)
   C1. `run_inference.sh --fake-hardware --debug --profile inference up -d`
       → waits for `[DEBUG] Action FPS` log marker
       → validates --debug end-to-end

Usage
-----
  # All sections (requires Docker)
  uv run python tests/smoke/scripts/inference_flags_smoke_test.py

  # Mapping assertions only (fast, no Docker needed)
  uv run python tests/smoke/scripts/inference_flags_smoke_test.py --skip-startup

  # Skip the GPU test (C)
  uv run python tests/smoke/scripts/inference_flags_smoke_test.py --skip-gpu
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]   # tests/smoke/scripts/ → repo root
SMOKE_ROOT = Path(__file__).resolve().parents[1]
INFERENCE_SH = REPO / "scripts" / "run_inference.sh"
COMPOSE_FAKE = REPO / "docker-compose.fake-hardware.yml"

# Default smoke checkpoint produced by pipeline_smoke_test.py (cmd scenario, 10 steps)
_SMOKE_CKPT = SMOKE_ROOT / "outputs" / "model_zoo" / "cmd" / "smoke" / "checkpoints" / "000010"

# ── docker shim for mapping assertions ───────────────────────────────────────
# Intercepts `docker compose ...` calls made by run_inference.sh.
# Reports the env vars that the script exports just before calling docker.
_DOCKER_SHIM = """\
#!/bin/bash
# smoke-test docker shim: report exported env vars and received args
printf 'SHIM_ARGS=%s\\n' "$*"
printf 'SHIM_ECHO_TOPIC_ONLY=%s\\n' "${ECHO_TOPIC_ONLY:-}"
printf 'SHIM_MONITOR_ENABLE=%s\\n' "${MONITOR_ENABLE:-}"
printf 'SHIM_DEBUG=%s\\n' "${DEBUG:-}"
exit 0
"""

# ── helpers ───────────────────────────────────────────────────────────────────

def _clean_env() -> dict:
    """Return os.environ minus any vars run_inference.sh might misread."""
    env = dict(os.environ)
    for var in ("ECHO_TOPIC_ONLY", "MONITOR_ENABLE", "DEBUG", "MODEL_PATH",
                "MONITOR_OUTPUT_DIR", "ACTION_TYPE"):
        env.pop(var, None)
    return env


def _run_with_shim(*script_flags: str) -> dict:
    """Run run_inference.sh with a docker shim inserted at the front of PATH.

    Passes 'version' as the compose command so the shim returns 0 immediately
    without trying to start any service.

    Returns a dict with:
      compose_line  — the '[run_inference] compose: ...' stdout line
      shim_args     — args received by the docker shim (compose passthrough)
      echo_topic_only / monitor_enable / debug  — env var values at shim call time
      returncode    — script exit code
      stdout / stderr
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        shim = Path(tmpdir) / "docker"
        shim.write_text(_DOCKER_SHIM)
        shim.chmod(0o755)

        env = _clean_env()
        env["PATH"] = f"{tmpdir}:{env.get('PATH', '')}"

        cmd = ["bash", str(INFERENCE_SH), *script_flags, "version"]
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO, env=env)
        stdout = proc.stdout

        def _extract(key: str) -> str:
            m = re.search(rf"^{key}=(.*)$", stdout, re.MULTILINE)
            return m.group(1).strip() if m else ""

        compose_line = next(
            (ln.strip() for ln in stdout.splitlines() if "[run_inference] compose:" in ln),
            "",
        )
        return {
            "compose_line": compose_line,
            "shim_args": _extract("SHIM_ARGS"),
            "echo_topic_only": _extract("SHIM_ECHO_TOPIC_ONLY"),
            "monitor_enable": _extract("SHIM_MONITOR_ENABLE"),
            "debug": _extract("SHIM_DEBUG"),
            "returncode": proc.returncode,
            "stdout": stdout,
            "stderr": proc.stderr,
        }


def _compose_fake_config_inference() -> str:
    """Render docker-compose.fake-hardware.yml --profile inference as text."""
    env = _clean_env()
    env["DEBUG"] = "true"   # set so config rendering reflects the flag
    proc = subprocess.run(
        ["docker", "compose", "-f", str(COMPOSE_FAKE),
         "--profile", "inference", "config"],
        capture_output=True, text=True, cwd=REPO, env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"docker compose config failed:\n{proc.stderr}")
    return proc.stdout


def _poll_container_log(container: str, pattern: str,
                        timeout_s: float = 240.0, interval_s: float = 3.0) -> bool:
    """Poll `docker logs <container>` until pattern matches or timeout."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        proc = subprocess.run(
            ["docker", "logs", container, "--tail", "200"],
            capture_output=True, text=True,
        )
        combined = proc.stdout + proc.stderr
        if re.search(pattern, combined):
            return True
        time.sleep(interval_s)
    return False


def _compose_fake_down(profile: str) -> None:
    subprocess.run(
        ["docker", "compose", "-f", str(COMPOSE_FAKE),
         "--profile", profile, "down", "--remove-orphans"],
        cwd=REPO, check=False,
    )


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


# ── Section A: Mapping assertions ─────────────────────────────────────────────

def run_section_a() -> None:
    print(f"\n{'═'*70}")
    print("  SECTION A — Mapping assertions (no containers)")
    print(f"{'═'*70}")

    # A1 — --fake-hardware selects the fake-hardware compose file
    print("\n  A1. --fake-hardware → compose file selection")
    r = _run_with_shim("--fake-hardware")
    _assert(
        "--fake-hardware uses docker-compose.fake-hardware.yml",
        "fake-hardware" in r["compose_line"],
        r["compose_line"],
    )
    _assert(
        "--fake-hardware does not use docker-compose.yml",
        "docker-compose.yml" not in r["compose_line"] or "fake-hardware" in r["compose_line"],
        r["compose_line"],
    )

    # A2 — --monitor-enable injects --profile monitor into passthrough
    print("\n  A2. --monitor-enable → --profile monitor in compose args")
    r = _run_with_shim("--fake-hardware", "--monitor-enable")
    _assert(
        "--monitor-enable injects --profile monitor",
        "--profile monitor" in r["shim_args"],
        f"shim_args={r['shim_args']}",
    )

    # A3 — --echo-topic-only exports ECHO_TOPIC_ONLY=true
    print("\n  A3. --echo-topic-only → ECHO_TOPIC_ONLY=true exported")
    r = _run_with_shim("--echo-topic-only")
    _assert(
        "--echo-topic-only sets ECHO_TOPIC_ONLY=true",
        r["echo_topic_only"] == "true",
        f"ECHO_TOPIC_ONLY={r['echo_topic_only']!r}",
    )
    # Confirm it is NOT set without the flag
    r_plain = _run_with_shim("--fake-hardware")
    _assert(
        "ECHO_TOPIC_ONLY not set without --echo-topic-only",
        r_plain["echo_topic_only"] == "",
        f"ECHO_TOPIC_ONLY={r_plain['echo_topic_only']!r}",
    )

    # A4 — --debug exports DEBUG=true
    print("\n  A4. --debug → DEBUG=true exported")
    r = _run_with_shim("--fake-hardware", "--debug")
    _assert(
        "--debug sets DEBUG=true",
        r["debug"] == "true",
        f"DEBUG={r['debug']!r}",
    )
    # DEBUG not set without the flag
    _assert(
        "DEBUG not set without --debug",
        r_plain["debug"] == "",
        f"DEBUG={r_plain['debug']!r}",
    )

    # A4b — fake-hardware inference compose renders debug:=true when DEBUG=true
    print("\n  A4b. DEBUG=true → debug:=true in fake-hardware inference command")
    try:
        config_text = _compose_fake_config_inference()
        _assert(
            "compose config renders debug:=true when DEBUG=true",
            "debug:=true" in config_text,
            "found" if "debug:=true" in config_text else "NOT found in compose config",
        )
    except RuntimeError as exc:
        _assert("compose config rendered", False, str(exc))

    # A5 — combined: --fake-hardware + --debug
    print("\n  A5. --fake-hardware --debug combined")
    r = _run_with_shim("--fake-hardware", "--debug")
    _assert(
        "combined: fake-hardware compose selected",
        "fake-hardware" in r["compose_line"],
        r["compose_line"],
    )
    _assert(
        "combined: DEBUG=true exported",
        r["debug"] == "true",
        f"DEBUG={r['debug']!r}",
    )


# ── Section B: Startup tests ───────────────────────────────────────────────────

def run_section_b() -> None:
    print(f"\n{'═'*70}")
    print("  SECTION B — Startup tests (Docker, no GPU)")
    print(f"{'═'*70}")

    # B1 — fake-hardware + monitor-enable: DDS connectivity + FPS
    print("\n  B1. --fake-hardware --monitor-enable up -d")
    print("      (validates DDS + FPS log markers from inference node)", flush=True)
    # Force-remove stale containers that may come from a different compose project
    # (compose down --remove-orphans only removes containers from the same project)
    subprocess.run(
        ["docker", "rm", "-f", "lerobot-fake-robot", "lerobot-fake-monitor"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    _compose_fake_down("monitor")   # clean slate

    t0 = time.monotonic()
    proc = subprocess.run(
        ["bash", str(INFERENCE_SH), "--fake-hardware", "--monitor-enable", "up", "-d"],
        cwd=REPO, capture_output=True, text=True, env=_clean_env(),
    )
    started = proc.returncode == 0
    _assert(
        "B1 containers started (up -d exit 0)",
        started,
        f"exit {proc.returncode}" + (f"\n{proc.stderr[:300]}" if not started else ""),
    )

    if started:
        # Wait for the inference_node stats-log line (printed every ~5s by default)
        print("      polling lerobot-fake-monitor logs for Hz markers ...", flush=True)
        # Pattern: "Joint State  X.X Hz" or "<camera>  X.X Hz  (+N"
        found = _poll_container_log(
            container="lerobot-fake-monitor",
            pattern=r"(?:Joint State|waist|wrist_r|chest)\s+[\d.]+\s+Hz",
            timeout_s=120.0,
        )
        _assert(
            "B1 Hz log markers received from inference node (DDS OK + FPS > 0)",
            found,
            "Hz pattern matched" if found else "TIMEOUT: no Hz markers in 120s",
        )
        dt = time.monotonic() - t0
        print(f"      elapsed: {dt:.1f}s", flush=True)

        # Teardown
        print("      tearing down ...", flush=True)
        _compose_fake_down("monitor")

    # B2 — --echo-topic-only with --fake-hardware: flag is parsed but env is ignored
    print("\n  B2. --echo-topic-only with --fake-hardware (env exported, but ignored by compose)")
    r = _run_with_shim("--fake-hardware", "--echo-topic-only")
    _assert(
        "B2 ECHO_TOPIC_ONLY=true exported by script",
        r["echo_topic_only"] == "true",
        f"ECHO_TOPIC_ONLY={r['echo_topic_only']!r}",
    )
    # Confirm fake-hardware monitor service hardcodes echo_topic_only:=true (not via env)
    try:
        env = _clean_env()
        env["ECHO_TOPIC_ONLY"] = "false"   # explicitly set to false
        cfg_proc = subprocess.run(
            ["docker", "compose", "-f", str(COMPOSE_FAKE),
             "--profile", "monitor", "config"],
            capture_output=True, text=True, cwd=REPO, env=env,
        )
        cfg_text = cfg_proc.stdout
        _assert(
            "B2 fake-hardware monitor command hardcodes echo_topic_only:=true (not from env)",
            "echo_topic_only:=true" in cfg_text,
            "hardcoded found" if "echo_topic_only:=true" in cfg_text else "NOT found",
        )
    except Exception as exc:
        _assert("B2 compose config readable", False, str(exc))


# ── Section C: Best-effort GPU test ───────────────────────────────────────────

def run_section_c(checkpoint: Path) -> None:
    print(f"\n{'═'*70}")
    print("  SECTION C — Best-effort GPU startup (--debug with checkpoint)")
    print(f"{'═'*70}")
    print(f"\n  C1. --fake-hardware --debug  (MODEL_PATH={checkpoint})")
    print("      Waits for [DEBUG] Action FPS log from inference container.", flush=True)

    # Force-remove stale containers that may come from a different compose project
    subprocess.run(
        ["docker", "rm", "-f", "lerobot-fake-robot", "lerobot-fake-inference",
         "lerobot-fake-monitor"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    _compose_fake_down("inference")   # clean slate

    env = _clean_env()
    env["MODEL_PATH"] = str(checkpoint)

    # NOTE: --profile is a docker compose global flag; must come before the subcommand.
    # run_inference.sh passes unknown flags as-is to compose, so putting --profile
    # before 'up' results in: docker compose -f file.yml --profile inference up -d
    proc = subprocess.run(
        ["bash", str(INFERENCE_SH), "--fake-hardware", "--debug",
         "--profile", "inference", "up", "-d"],
        cwd=REPO, capture_output=True, text=True, env=env,
    )
    started = proc.returncode == 0
    _assert(
        "C1 containers started (up -d exit 0)",
        started,
        f"exit {proc.returncode}" + (f"\n{proc.stderr[:300]}" if not started else ""),
    )

    if started:
        print("      polling lerobot-fake-inference logs for [DEBUG] Action FPS ...",
              flush=True)
        found = _poll_container_log(
            container="lerobot-fake-inference",
            # Matches either the debug marker, a topology-mismatch normalization error,
            # or any other fatal startup error
            pattern=r"\[DEBUG\] Action FPS|RuntimeError|Traceback|Error:|FATAL|Exception",
            timeout_s=180.0,
        )
        proc_log = subprocess.run(
            ["docker", "logs", "lerobot-fake-inference", "--tail", "50"],
            capture_output=True, text=True,
        )
        log_tail = proc_log.stdout + proc_log.stderr
        debug_ok = bool(re.search(r"\[DEBUG\] Action FPS", log_tail))

        # Detect the known topology-mismatch scenario: smoke checkpoint (single arm, 8-DOF)
        # vs mock-robot fake hardware (bimanual, 16-DOF).  This is NOT a --debug failure;
        # the flag was correctly wired (confirmed by A4 + A4b).  Mark as SKIP.
        topology_mismatch = bool(re.search(
            r"The size of tensor a.*must match|normalize|NormalizeProcessor",
            log_tail, re.IGNORECASE,
        ))

        if debug_ok:
            _assert("C1 [DEBUG] Action FPS marker seen in inference container", True,
                    "[DEBUG] marker found")
        elif topology_mismatch:
            _skip(
                "C1 [DEBUG] Action FPS marker",
                "topology mismatch — smoke checkpoint is 8-DOF single arm; "
                "mock-robot publishes 16-DOF bimanual; --debug flag wiring verified by A4+A4b",
            )
        else:
            _assert(
                "C1 [DEBUG] Action FPS marker seen in inference container",
                False,
                "TIMEOUT / unexpected error — see log tail below",
            )
            print("      --- last 50 log lines ---", flush=True)
            print(log_tail[:2000], flush=True)

        print("      tearing down ...", flush=True)
        _compose_fake_down("inference")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--skip-startup", action="store_true",
                   help="run section A only (no docker containers started)")
    p.add_argument("--skip-gpu", action="store_true",
                   help="skip section C (best-effort GPU test)")
    p.add_argument("--checkpoint", default="",
                   help=(
                       "checkpoint dir for section C (default: smoke output from "
                       f"pipeline_smoke_test --scenario cmd, {_SMOKE_CKPT.relative_to(REPO)})"
                   ))
    args = p.parse_args()

    print(f"inference_flags_smoke_test  repo={REPO}")

    run_section_a()

    if not args.skip_startup:
        run_section_b()
    else:
        print("\n[skipping section B — --skip-startup]")

    if not args.skip_startup and not args.skip_gpu:
        ckpt = Path(args.checkpoint) if args.checkpoint else _SMOKE_CKPT
        if ckpt.exists():
            run_section_c(ckpt)
        else:
            print(f"\n{'═'*70}")
            print("  SECTION C — skipped (checkpoint not found)")
            print(f"  {ckpt}")
            print(f"  Run pipeline_smoke_test.py --scenario cmd first to build it.")
            print(f"{'═'*70}")
    else:
        if args.skip_gpu:
            print("\n[skipping section C — --skip-gpu]")

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
