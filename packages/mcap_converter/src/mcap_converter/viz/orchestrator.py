"""Docker/git orchestration for dataset-viz: acquiring the pinned upstream
lerobot-dataset-visualizer source, building docker-compose invocations, and
polling for stack readiness.

Command-building functions here are pure (return an argv list; they don't
execute anything) so they're unit-testable without Docker installed.
Everything that actually shells out or hits the network takes an injectable
`runner`/`opener` parameter defaulting to the real implementation, so tests
can substitute a fake and assert on what WOULD have been executed/fetched.
"""

import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from mcap_converter.viz.config import VISUALIZER_PINNED_SHA, VISUALIZER_REPO_URL

COMPOSE_PROJECT_NAME = "mcap-viz"
VISUALIZER_SOURCE_DIRNAME = "lerobot-dataset-visualizer"


def compose_file_path() -> Path:
    """Path to the checked-in docker-compose.yml, resolved relative to this package."""
    return Path(__file__).parent / "compose" / "docker-compose.yml"


# ── Pure command-argv builders (no execution) ────────────────────────────


def build_compose_up_argv(
    compose_path: Path, env_file_path: Path, *, rebuild: bool = False
) -> List[str]:
    argv = [
        "docker",
        "compose",
        "-p",
        COMPOSE_PROJECT_NAME,
        "-f",
        str(compose_path),
        "--env-file",
        str(env_file_path),
        "up",
        "-d",
    ]
    if rebuild:
        argv.append("--build")
    return argv


def build_compose_down_argv(compose_path: Path, env_file_path: Path) -> List[str]:
    return [
        "docker",
        "compose",
        "-p",
        COMPOSE_PROJECT_NAME,
        "-f",
        str(compose_path),
        "--env-file",
        str(env_file_path),
        "down",
    ]


def build_compose_logs_argv(compose_path: Path, env_file_path: Path) -> List[str]:
    return [
        "docker",
        "compose",
        "-p",
        COMPOSE_PROJECT_NAME,
        "-f",
        str(compose_path),
        "--env-file",
        str(env_file_path),
        "logs",
        "--no-color",
    ]


# ── Preflight checks ──────────────────────────────────────────────────────


def is_port_available(port: int, host: str = "127.0.0.1") -> bool:
    """True if a TCP socket can be bound to (host, port) right now."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
        except OSError:
            return False
        return True


def _run_or_error(
    runner: Callable, argv: List[str], failure_prefix: str
) -> Tuple[Optional[subprocess.CompletedProcess], Optional[str]]:
    """
    Run `runner(argv, capture_output=True, text=True)`, converting an OSError
    raised by the runner itself (e.g. PermissionError because the caller
    can't access the Docker socket, or FileNotFoundError if the executable
    disappeared) into a clean error message instead of letting it propagate.

    Returns (completed_process, None) on success, or (None, error_message)
    if the runner raised.
    """
    try:
        return runner(argv, capture_output=True, text=True), None
    except OSError as exc:
        return None, f"{failure_prefix}: {exc}"


def check_docker_available(runner: Callable = subprocess.run) -> Tuple[bool, str]:
    """
    Returns (ok, message). message is empty on success, or a clear
    human-readable reason on failure (Docker missing vs. daemon down vs.
    compose plugin missing). Never raises, even if `runner` itself raises
    an OSError (e.g. PermissionError on the Docker socket).
    """
    if shutil.which("docker") is None:
        return False, "Docker is not installed or not on PATH."

    result, error = _run_or_error(runner, ["docker", "version"], "failed to run 'docker version'")
    if error is not None:
        return False, error
    if result.returncode != 0:
        return False, "Docker is installed but the daemon is not running (or not accessible)."

    compose_result, error = _run_or_error(
        runner, ["docker", "compose", "version"], "failed to run 'docker compose version'"
    )
    if error is not None:
        return False, error
    if compose_result.returncode != 0:
        return False, "Docker Compose plugin is not available (`docker compose version` failed)."
    return True, ""


# ── Pinned-source acquisition ─────────────────────────────────────────────


def _checkout_pinned_sha(runner: Callable, source_dir: Path) -> str:
    """
    Checkout VISUALIZER_PINNED_SHA in `source_dir`. Returns an empty string
    on success, or a human-readable error message on failure (whether from a
    nonzero exit code or an OSError raised by `runner` itself).
    """
    failure_prefix = f"git checkout {VISUALIZER_PINNED_SHA} failed"
    result, error = _run_or_error(
        runner, ["git", "-C", str(source_dir), "checkout", VISUALIZER_PINNED_SHA], failure_prefix
    )
    if error is not None:
        return error
    if result.returncode != 0:
        return f"{failure_prefix}: {result.stderr.strip()}"
    return ""


def ensure_visualizer_source(
    cache_dir: Path,
    *,
    refresh: bool = False,
    runner: Callable = subprocess.run,
) -> Tuple[bool, Path, str]:
    """
    Ensure the pinned-commit lerobot-dataset-visualizer source is present at
    <cache_dir>/lerobot-dataset-visualizer, cloning/checking out as needed.

    - If `refresh` is True, delete any existing checkout first.
    - If the directory doesn't exist: blobless clone + checkout the pinned SHA.
    - If it exists: check `git -C <dir> rev-parse HEAD`. If it matches
      VISUALIZER_PINNED_SHA, reuse as-is (no network). If it doesn't
      (pin was bumped, or the checkout was tampered with), `git fetch` then
      checkout the pinned SHA.

    Returns (ok, source_dir, error_message). error_message is empty on
    success. On failure, source_dir is still returned (best-effort) but
    should not be used. Never raises, even if `runner` itself raises an
    OSError (e.g. `git` not being executable).
    """
    source_dir = cache_dir / VISUALIZER_SOURCE_DIRNAME

    if refresh and source_dir.exists():
        shutil.rmtree(source_dir)

    if not source_dir.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        clone_result, error = _run_or_error(
            runner,
            ["git", "clone", "--filter=blob:none", VISUALIZER_REPO_URL, str(source_dir)],
            "git clone failed",
        )
        if error is not None:
            return False, source_dir, error
        if clone_result.returncode != 0:
            return False, source_dir, f"git clone failed: {clone_result.stderr.strip()}"

        checkout_error = _checkout_pinned_sha(runner, source_dir)
        if checkout_error:
            return False, source_dir, checkout_error
        return True, source_dir, ""

    rev_parse_result, error = _run_or_error(
        runner,
        ["git", "-C", str(source_dir), "rev-parse", "HEAD"],
        "git rev-parse HEAD failed",
    )
    if error is not None:
        return False, source_dir, error
    current_sha = rev_parse_result.stdout.strip() if rev_parse_result.returncode == 0 else None
    if current_sha == VISUALIZER_PINNED_SHA:
        return True, source_dir, ""

    fetch_result, error = _run_or_error(
        runner, ["git", "-C", str(source_dir), "fetch"], "git fetch failed"
    )
    if error is not None:
        return False, source_dir, error
    if fetch_result.returncode != 0:
        return False, source_dir, f"git fetch failed: {fetch_result.stderr.strip()}"

    checkout_error = _checkout_pinned_sha(runner, source_dir)
    if checkout_error:
        return False, source_dir, checkout_error
    return True, source_dir, ""


# ── Readiness polling ──────────────────────────────────────────────────────


def wait_for_ready(
    urls: List[str],
    *,
    timeout_s: float = 120.0,
    interval_s: float = 2.0,
    opener: Callable[[str], object] = urllib.request.urlopen,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> bool:
    """
    Poll every URL in `urls` until ALL return HTTP 200 (via `opener`), or
    `timeout_s` elapses. Returns True if all became ready in time, False on
    timeout. `sleep`/`clock` are injectable for fast, deterministic tests.
    """
    deadline = clock() + timeout_s
    while clock() < deadline:
        all_ready = True
        for url in urls:
            try:
                with opener(url) as response:
                    status = getattr(response, "status", None) or response.getcode()
                    if status != 200:
                        all_ready = False
                        break
            except (urllib.error.URLError, OSError):
                all_ready = False
                break
        if all_ready:
            return True
        sleep(interval_s)
    return False
