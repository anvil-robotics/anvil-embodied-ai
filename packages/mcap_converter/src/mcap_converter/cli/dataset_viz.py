"""dataset-viz: stand up a local Docker stack (nginx + the official
HuggingFace lerobot-dataset-visualizer web app) so you can interactively
browse a converted LeRobot dataset in a browser.

This module is orchestration glue: it wires together the pure helpers in
mcap_converter.viz.{dataset_check,nginx_config,config,orchestrator} into a
CLI. See those modules for the actual validation/rendering/Docker logic.
"""

import argparse
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from typing import List, Optional

from rich.console import Console

from mcap_converter.viz.config import (
    DEFAULT_FRONTEND_PORT,
    DEFAULT_STATIC_PORT,
    browse_url,
    default_repo_id,
    frontend_probe_url,
    render_run_env,
    resolve_cache_dir,
    static_probe_url,
)
from mcap_converter.viz.dataset_check import validate_dataset_root
from mcap_converter.viz.nginx_config import render_nginx_conf
from mcap_converter.viz.orchestrator import (
    build_compose_down_argv,
    build_compose_logs_argv,
    build_compose_up_argv,
    check_docker_available,
    compose_file_path,
    ensure_visualizer_source,
    is_port_available,
    wait_for_ready,
)

console = Console()

# Building the frontend Docker image from scratch (--rebuild, or the very
# first run) takes noticeably longer than starting an already-built image,
# so allow more time before declaring the stack unready.
_READY_TIMEOUT_DEFAULT_S = 120.0
_READY_TIMEOUT_REBUILD_S = 300.0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Stand up a local Docker stack (nginx + the official HuggingFace "
            "lerobot-dataset-visualizer web app) to interactively browse a "
            "converted LeRobot dataset."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  dataset-viz data/datasets/afo/my-session
  dataset-viz data/datasets/afo/my-session --repo-id anvil/my-session --episode 3
  dataset-viz data/datasets/afo/my-session --frontend-port 3000 --static-port 8090
  dataset-viz --stop                         # tear down a running stack
  dataset-viz data/datasets/afo/my-session --detach --no-open
""",
    )
    parser.add_argument(
        "root",
        nargs="?",
        metavar="ROOT",
        default=None,
        help=(
            "Path to a converted LeRobot v2.0/v2.1/v3.0 dataset root directory "
            "(the directory containing meta/info.json). Required unless --stop is given."
        ),
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        metavar="ORG/NAME",
        help=(
            "Cosmetic org/dataset name shown in the browse URL and the visualizer's own UI. "
            'Default: "local/<basename of ROOT>". Any value works -- the local nginx server '
            "ignores the org/dataset URL segments entirely."
        ),
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=0,
        metavar="N",
        help="Episode index to open first in the printed browse URL. Default: 0.",
    )
    parser.add_argument(
        "--frontend-port",
        type=int,
        default=DEFAULT_FRONTEND_PORT,
        metavar="PORT",
        help=f"Host port for the web app. Default: {DEFAULT_FRONTEND_PORT}.",
    )
    parser.add_argument(
        "--static-port",
        type=int,
        default=DEFAULT_STATIC_PORT,
        metavar="PORT",
        help=f"Host port for the nginx dataset server. Default: {DEFAULT_STATIC_PORT}.",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Do not automatically open a browser; just print the URL.",
    )
    parser.add_argument(
        "--detach",
        action="store_true",
        help=(
            "Start the stack and return immediately, leaving it running in the background. "
            "Default: stay attached and block until Ctrl-C, then tear the stack down."
        ),
    )
    parser.add_argument(
        "--stop",
        action="store_true",
        help="Tear down a previously started stack and exit. ROOT is not required when this flag is given.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force-rebuild the frontend Docker image.",
    )
    parser.add_argument(
        "--refresh-source",
        action="store_true",
        help="Delete and re-clone the pinned lerobot-dataset-visualizer source before building.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Override the cache directory used for the cloned source, generated nginx.conf, "
            "and generated run.env. Default: $XDG_CACHE_HOME/anvil-mcap-viz or "
            "~/.cache/anvil-mcap-viz."
        ),
    )
    return parser


def _stop_stack(cache_dir_override: Optional[Path]) -> int:
    """Tear down a previously started stack. Returns the process exit code."""
    cache_dir = resolve_cache_dir(cache_dir_override)
    env_file_path = cache_dir / "run.env"
    if not env_file_path.exists():
        console.print("[dim]No running stack found (nothing to stop).[/dim]")
        return 0

    result = subprocess.run(build_compose_down_argv(compose_file_path(), env_file_path))
    if result.returncode != 0:
        console.print(f"[red]✗ failed to stop the stack (exit {result.returncode}).[/red]")
        return result.returncode
    console.print("[green]Stack stopped.[/green]")
    return 0


def main(args: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    parsed = parser.parse_args(args)

    if parsed.stop:
        return _stop_stack(parsed.cache_dir)

    if parsed.root is None:
        console.print("[red]✗ ROOT is required unless --stop is given.[/red]")
        return 1

    docker_ok, docker_message = check_docker_available()
    if not docker_ok:
        console.print(f"[red]✗ {docker_message}[/red]")
        return 1

    dataset_root = Path(parsed.root)
    check = validate_dataset_root(dataset_root)
    if not check.ok:
        for error in check.errors:
            console.print(f"[red]✗ {error}[/red]")
        return 1
    for warning in check.warnings:
        console.print(f"[yellow]⚠ {warning}[/yellow]")

    if not is_port_available(parsed.frontend_port):
        console.print(
            f"[red]✗ port {parsed.frontend_port} is already in use. "
            "Pick a different port with --frontend-port.[/red]"
        )
        return 1
    if not is_port_available(parsed.static_port):
        console.print(
            f"[red]✗ port {parsed.static_port} is already in use. "
            "Pick a different port with --static-port.[/red]"
        )
        return 1

    cache_dir = resolve_cache_dir(parsed.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    console.print("[dim]Ensuring lerobot-dataset-visualizer source is available...[/dim]")
    source_ok, source_dir, source_error = ensure_visualizer_source(
        cache_dir, refresh=parsed.refresh_source
    )
    if not source_ok:
        console.print(f"[red]✗ {source_error}[/red]")
        return 1

    nginx_conf_path = cache_dir / "nginx.conf"
    nginx_conf_path.write_text(render_nginx_conf(parsed.static_port))

    env_file_path = cache_dir / "run.env"
    env_file_path.write_text(
        render_run_env(
            dataset_root=dataset_root,
            nginx_conf_path=nginx_conf_path,
            visualizer_dir=source_dir,
            static_port=parsed.static_port,
            frontend_port=parsed.frontend_port,
        )
    )

    compose_path = compose_file_path()
    compose_up_argv = build_compose_up_argv(compose_path, env_file_path, rebuild=parsed.rebuild)
    console.print(
        "[dim]Starting the Docker stack"
        + (
            " (building the frontend image, this may take a few minutes)..."
            if parsed.rebuild
            else "..."
        )
        + "[/dim]"
    )
    up_result = subprocess.run(compose_up_argv)
    if up_result.returncode != 0:
        subprocess.run(build_compose_logs_argv(compose_path, env_file_path))
        console.print(
            f"[red]✗ failed to start the Docker stack (exit {up_result.returncode}).[/red]"
        )
        return 1

    repo_id = parsed.repo_id or default_repo_id(dataset_root)
    probe_urls = [
        static_probe_url(static_port=parsed.static_port, repo_id=repo_id),
        frontend_probe_url(frontend_port=parsed.frontend_port),
    ]
    timeout_s = _READY_TIMEOUT_REBUILD_S if parsed.rebuild else _READY_TIMEOUT_DEFAULT_S
    console.print("[dim]Waiting for the stack to become ready...[/dim]")
    ready = wait_for_ready(probe_urls, timeout_s=timeout_s)
    if not ready:
        console.print(
            f"[red]✗ timed out waiting for the stack to become ready after {timeout_s:.0f}s.[/red]"
        )
        console.print("Check these URLs manually:")
        for probe_url in probe_urls:
            console.print(f"  {probe_url}")
        console.print(
            "[yellow]The stack has NOT been torn down automatically -- inspect it, then run "
            "`dataset-viz --stop` when done.[/yellow]"
        )
        return 1

    url = browse_url(frontend_port=parsed.frontend_port, repo_id=repo_id, episode=parsed.episode)
    console.print(f"[green]Ready![/green] Browse at: [bold]{url}[/bold]")
    if not parsed.no_open:
        try:
            webbrowser.open(url)
        except Exception as exc:  # noqa: BLE001 - never let browser-launch failure be fatal
            console.print(f"[dim]Could not open a browser automatically: {exc}[/dim]")

    if parsed.detach:
        stop_cmd = "dataset-viz --stop"
        if parsed.cache_dir is not None:
            stop_cmd += f" --cache-dir {parsed.cache_dir}"
        console.print(f"[dim]Running in the background. Stop it later with: {stop_cmd}[/dim]")
        return 0

    console.print("[dim]Press Ctrl-C to stop.[/dim]")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        console.print("\n[dim]Stopping the stack...[/dim]")
        subprocess.run(build_compose_down_argv(compose_path, env_file_path))
        console.print("[green]Stack stopped.[/green]")
        return 130


if __name__ == "__main__":
    sys.exit(main())
