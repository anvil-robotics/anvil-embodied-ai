"""Generates the nginx config that serves a local LeRobot dataset directory
at the exact URL shape the official lerobot-dataset-visualizer frontend
expects: /{org}/{dataset}/resolve/main/{path}.

IMPORTANT — fragile external dependency: this URL shape is NOT a documented
API of lerobot-dataset-visualizer. It was derived by reading the upstream
source at src/utils/versionUtils.ts (buildVersionedUrl()), which builds
`${DATASET_URL}/${repoId}/resolve/main/${path}`. If the pinned upstream
commit (see viz/config.py — built in a later task) is ever bumped, re-verify
this contract still holds by re-reading that file at the new pinned SHA.

The org/dataset URL segments are matched by a wildcard regex and discarded
(never validated against anything) — the actual dataset directory is
bind-mounted at a fixed container path, /srv/dataset, regardless of what
org/dataset name the frontend URL uses.
"""


def render_nginx_conf(static_port: int) -> str:
    """
    Render a complete nginx.conf that:
    - listens on `static_port`
    - serves files bind-mounted at /srv/dataset under the URL path
      /{org}/{dataset}/resolve/main/{path} (org/dataset segments ignored)
    - supports HTTP Range requests (206 Partial Content) for <video> seeking
      — this is nginx's default behavior for static files, no special config
      needed beyond serving via `alias`, so this function's only Range-related
      work is advertising `Accept-Ranges: bytes` explicitly for clarity
    - sets permissive CORS headers, since the frontend (a different port) and
      this static server are different origins from the browser's perspective
    """
    return f"""\
worker_processes auto;
events {{ worker_connections 1024; }}
http {{
    include       /etc/nginx/mime.types;
    default_type  application/octet-stream;
    sendfile      on;
    server {{
        listen {static_port};

        # Match /{{org}}/{{dataset}}/resolve/main/{{path...}} ; org+dataset are ignored.
        location ~ "^/[^/]+/[^/]+/resolve/main/(?<subpath>.+)$" {{
            alias /srv/dataset/$subpath;

            # <video> scrubbing: nginx serves Range (206) for static files
            # automatically; advertise it explicitly for clarity.
            add_header Accept-Ranges bytes always;

            # Frontend origin (localhost:FRONTEND_PORT) differs from this
            # origin (localhost:{static_port}) -> cross-origin fetch of
            # parquet/video/json.
            add_header Access-Control-Allow-Origin  "*" always;
            add_header Access-Control-Allow-Methods "GET, HEAD, OPTIONS" always;
            add_header Access-Control-Allow-Headers "Range" always;
            if ($request_method = OPTIONS) {{ return 204; }}
        }}
    }}
}}
"""
