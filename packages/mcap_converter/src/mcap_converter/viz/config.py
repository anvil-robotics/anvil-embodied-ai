"""Constants and small pure helpers for dataset-viz: the pinned upstream
lerobot-dataset-visualizer source, cache-directory resolution, and the
various string-building helpers (repo-id, browse URL, run.env content) that
tie the CLI's inputs to what gets rendered into the Docker stack.
"""

import os
from pathlib import Path
from typing import Optional

# The official HuggingFace web app we orchestrate. See viz/nginx_config.py's
# module docstring for why the exact pinned commit matters: its
# src/utils/versionUtils.ts defines the DATASET_URL + /{repoId}/resolve/main/
# URL contract our nginx config depends on. Pin a full commit SHA (not a
# branch/tag) so that contract can't silently change under us.
VISUALIZER_REPO_URL = "https://github.com/huggingface/lerobot-dataset-visualizer.git"
VISUALIZER_PINNED_SHA = "92ddb488bd1628089119971684289cc9b6f88715"

DEFAULT_FRONTEND_PORT = 7860
DEFAULT_STATIC_PORT = 8080

CACHE_DIR_ENV_VAR = "MCAP_VIZ_CACHE_DIR"


def resolve_cache_dir(override: Optional[Path] = None) -> Path:
    """
    Resolve the cache directory used to store the cloned visualizer source,
    the generated nginx.conf, and the generated run.env.

    Precedence: explicit `override` argument > $XDG_CACHE_HOME/anvil-mcap-viz
    > ~/.cache/anvil-mcap-viz. Does NOT create the directory (callers are
    responsible for mkdir(parents=True, exist_ok=True) when they actually
    need to write into it).
    """
    if override is not None:
        return Path(override)
    xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg_cache_home) if xdg_cache_home else Path.home() / ".cache"
    return base / "anvil-mcap-viz"


def default_repo_id(dataset_root: Path) -> str:
    """
    Derive a cosmetic default "org/dataset" identifier from a dataset root
    path, used when the user doesn't pass --repo-id. This value is purely
    decorative (shown in the browse URL and the visualizer's own UI) — the
    nginx config discards the org/dataset URL segments entirely, so this
    never has to correspond to anything on disk. Example:
    default_repo_id(Path("/data/datasets/my-session")) -> "local/my-session"
    """
    return f"local/{Path(dataset_root).resolve().name}"


def browse_url(*, frontend_port: int, repo_id: str, episode: int) -> str:
    """
    Build the URL the user should open to start browsing, e.g.:
    browse_url(frontend_port=7860, repo_id="local/my-session", episode=0)
    -> "http://localhost:7860/local/my-session/0"
    """
    return f"http://localhost:{frontend_port}/{repo_id}/{episode}"


def static_probe_url(*, static_port: int, repo_id: str) -> str:
    """
    Build the URL used by the CLI's own readiness probe: a GET to this URL
    (through nginx) should return 200 with the dataset's meta/info.json
    content, proving nginx + the bind mount + the location regex are all
    correctly wired up. Example:
    static_probe_url(static_port=8080, repo_id="local/my-session")
    -> "http://localhost:8080/local/my-session/resolve/main/meta/info.json"
    """
    return f"http://localhost:{static_port}/{repo_id}/resolve/main/meta/info.json"


def frontend_probe_url(*, frontend_port: int) -> str:
    """
    Build the URL used by the CLI's own readiness probe to confirm the
    frontend app itself has started (independent of dataset content).
    """
    return f"http://localhost:{frontend_port}/"


def render_run_env(
    *,
    dataset_root: Path,
    nginx_conf_path: Path,
    visualizer_dir: Path,
    static_port: int,
    frontend_port: int,
) -> str:
    """
    Render the contents of the --env-file passed to `docker compose`,
    supplying the per-invocation values the checked-in docker-compose.yml
    template references via ${VAR} substitution. All paths are rendered as
    their resolved absolute-path string form.
    """
    return (
        f"MCAP_VIZ_DATASET_ROOT={Path(dataset_root).resolve()}\n"
        f"MCAP_VIZ_NGINX_CONF={Path(nginx_conf_path).resolve()}\n"
        f"MCAP_VIZ_VISUALIZER_DIR={Path(visualizer_dir).resolve()}\n"
        f"STATIC_PORT={static_port}\n"
        f"FRONTEND_PORT={frontend_port}\n"
    )
