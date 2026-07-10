"""dataset-viz: browse a converted LeRobot dataset with a Rerun-based viewer.

This module is a thin wrapper: no subprocess, no Docker, no vendored
frontend. It validates the dataset root with
`mcap_converter.viz.dataset_check.validate_dataset_root`, then either prints
the episode count (`--list-episodes`) or loads the requested episodes
(`--episodes`, defaulting to the first 10) into one Rerun viewer session,
each as an independent, switchable `rerun.RecordingStream` recording.
"""

import argparse
import json
import sys
import time
import uuid
from pathlib import Path
from typing import List, Optional
from urllib.parse import quote

from rich.console import Console

from mcap_converter.viz.config import default_repo_id
from mcap_converter.viz.dataset_check import validate_dataset_root

console = Console()

# Matches lerobot.scripts.lerobot_dataset_viz's own --web-port default.
_DEFAULT_WEB_PORT = 9090

# Matches lerobot.scripts.lerobot_dataset_viz's own --grpc-port default; also
# the default local port rr.spawn() listens on.
_DEFAULT_GRPC_PORT = 9876

# rr.RecordingStream.serve_grpc()'s own default is "25%" -- far too low for a
# deliberate, bounded multi-episode load (video frames across several
# episodes routinely exceed that, silently evicting the earliest-loaded
# episodes once the limit is hit). Raise it well above what the default
# --episodes load is expected to need; --server-memory-limit can override
# this for very large --episodes specs on memory-constrained machines.
_SERVER_MEMORY_LIMIT = "75%"

# rr.Image.compress()'s own default (95) is very high quality and, in
# practice, still not enough size reduction to reliably keep a full
# --episodes load under the server memory limit above. 20 is a much more
# aggressive but still visually workable tradeoff for browsing purposes (not
# print-quality) -- confirmed by real-world testing to still show
# recognizable video content while meaningfully shrinking load time;
# --jpeg-quality can override this per invocation.
_DEFAULT_JPEG_QUALITY = 20

# Above this many episodes, --episodes prints a slowness warning (not a hard
# block -- each episode is a full LeRobotDataset load + Rerun log pass).
_MANY_EPISODES_WARNING_THRESHOLD = 20

# When --episodes is omitted, load this many episodes (or all of them, if the
# dataset has fewer) by default, instead of requiring an explicit spec.
_DEFAULT_EPISODE_COUNT = 10


def parse_episodes_spec(spec: str, total_episodes: int) -> List[int]:
    """
    Parse a 0-based episode index spec (for --episodes) into a sorted list of
    concrete indices.

    This mirrors the comma-list + colon-range SHAPE of
    `mcap_converter.cli.convert.parse_episode_index_spec`, but is 0-based to
    match how `LeRobotDataset`/lerobot's own tooling numbers episodes,
    rather than mcap-convert's 1-based episode numbering. Colon ranges follow
    Python slice convention: the end is EXCLUSIVE, e.g. "1:4" selects episodes
    1, 2, 3 (not 4) -- same as Python's range(1, 4). An omitted start defaults
    to 0; an omitted end reaches the actual last episode inclusively (there's
    nothing to exclude when no end is given).
    """
    result: set = set()
    for raw_token in spec.split(","):
        token = raw_token.strip()
        if not token:
            continue
        if ":" in token:
            start_str, end_str = token.split(":", 1)
            start_str, end_str = start_str.strip(), end_str.strip()
            try:
                start = int(start_str) if start_str else 0
                end = int(end_str) if end_str else total_episodes
            except ValueError:
                raise ValueError(f"invalid episode range token: '{token}'")
            if start >= end:
                raise ValueError(f"invalid range '{token}': start must be less than end (end is exclusive)")
            if start < 0 or end > total_episodes:
                raise ValueError(
                    f"range '{token}' out of bounds -- episodes are numbered 0 to {total_episodes - 1}"
                )
            result.update(range(start, end))
        else:
            try:
                idx = int(token)
            except ValueError:
                raise ValueError(f"invalid episode index token: '{token}'")
            if not (0 <= idx <= total_episodes - 1):
                raise ValueError(f"episode index {idx} out of range (0-{total_episodes - 1})")
            result.add(idx)
    return sorted(result)


def _detect_lan_ip() -> str:
    """
    Best-effort detection of this machine's LAN-reachable IP address.

    Rerun's own `serve_grpc()`/`serve_web_viewer()` hardcode `127.0.0.1` in
    the connection URI they build -- correct for a browser on the SAME
    machine, but useless for a REMOTE browser (whose own loopback interface
    that address refers to has nothing listening on it). We substitute this
    machine's actual LAN IP into that URI before handing it to the web
    viewer, so a remote device can actually connect.

    Opens a UDP socket "connected" to a public address without sending any
    data -- this doesn't require internet access to succeed, it just asks
    the OS routing table which local interface/IP it would use to reach
    that address, which is normally the LAN-facing one. Falls back to
    "127.0.0.1" if this fails for any reason (e.g. no network interfaces).
    """
    import socket

    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            probe.connect(("8.8.8.8", 80))
            return probe.getsockname()[0]
        finally:
            probe.close()
    except OSError:
        return "127.0.0.1"


def _log_episode_to_stream(
    dataset,
    episode_index: int,
    stream,
    *,
    compress_images: bool = True,
    jpeg_quality: int = _DEFAULT_JPEG_QUALITY,
) -> None:
    """
    Log one episode's frames (video, action, state, done/reward/success) to a
    single Rerun `RecordingStream` instance. This is the same per-frame logic
    as `lerobot.scripts.lerobot_dataset_viz.visualize_dataset`'s inner loop,
    replicated with `stream.log`/`stream.set_time` instance calls instead of
    the global `rr.log`/`rr.set_time` functions, so that N episodes can be
    logged to N independent, simultaneously-alive recordings.

    `compress_images` mirrors `visualize_dataset`'s own `display_compressed_images`
    flag (JPEG-encodes each frame via `rr.Image(...).compress()` instead of
    logging it raw) -- but we default it to True, the opposite of upstream's
    default, since raw uncompressed frames across several full episodes are
    the main reason the --web gRPC server's memory buffer gets exceeded and
    silently drops the earliest-loaded episodes (confirmed by real-world
    testing). Dataset browsing doesn't need pixel-perfect precision, so this
    tradeoff is taken by default; --full-quality-images opts back out.

    `jpeg_quality` is passed straight through to `Image.compress()` (Rerun's
    own default is 95 -- "saves a lot of space, but is still visually very
    similar" per its own docstring -- which still wasn't enough headroom in
    practice for a full --episodes load; see _DEFAULT_JPEG_QUALITY).
    """
    import torch.utils.data
    import rerun as rr
    from lerobot.scripts.lerobot_dataset_viz import to_hwc_uint8_numpy
    from lerobot.utils.constants import ACTION, DONE, OBS_STATE, REWARD

    dataloader = torch.utils.data.DataLoader(dataset, num_workers=0, batch_size=32)

    first_index = None
    for batch in dataloader:
        if first_index is None:
            first_index = batch["index"][0].item()
        for i in range(len(batch["index"])):
            stream.set_time("frame_index", sequence=batch["index"][i].item() - first_index)
            stream.set_time("timestamp", timestamp=batch["timestamp"][i].item())
            for key in dataset.meta.camera_keys:
                img = to_hwc_uint8_numpy(batch[key][i])
                img_entity = (
                    rr.Image(img).compress(jpeg_quality=jpeg_quality)
                    if compress_images
                    else rr.Image(img)
                )
                stream.log(key, img_entity)
            if ACTION in batch:
                for dim_idx, val in enumerate(batch[ACTION][i]):
                    stream.log(f"{ACTION}/{dim_idx}", rr.Scalars(val.item()))
            if OBS_STATE in batch:
                for dim_idx, val in enumerate(batch[OBS_STATE][i]):
                    stream.log(f"state/{dim_idx}", rr.Scalars(val.item()))
            if DONE in batch:
                stream.log(DONE, rr.Scalars(batch[DONE][i].item()))
            if REWARD in batch:
                stream.log(REWARD, rr.Scalars(batch[REWARD][i].item()))
            if "next.success" in batch:
                stream.log("next.success", rr.Scalars(batch["next.success"][i].item()))
        # Flush after every batch instead of once at the very end. A user
        # testing this observed the first-processed episode's opening video
        # frames (curves/scalars were unaffected) go missing even after
        # seeking back to the start -- consistent with a fresh gRPC/HTTP2
        # connection's small initial flow-control window being overwhelmed
        # by an unbroken burst of image data before it's had a chance to
        # grow via normal send/acknowledge traffic. This wasn't independently
        # confirmed (no way to inspect the underlying transport here), but
        # flushing incrementally is a reasonable mitigation regardless, and
        # also reduces how much of one episode sits unflushed in memory at
        # once.
        stream.flush()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Browse a converted LeRobot dataset with lerobot's own Rerun-based "
            "viewer -- video, actions, and state synced on one timeline."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""\
examples:
  dataset-viz data/datasets/afo/my-session
  dataset-viz data/datasets/afo/my-session --episodes "0,2:5"
  dataset-viz data/datasets/afo/my-session --episodes all
  dataset-viz data/datasets/afo/my-session --list-episodes
  dataset-viz data/datasets/afo/my-session --web --web-port 9090

If --episodes is omitted, the first {_DEFAULT_EPISODE_COUNT} episodes (or
all of them, if there are fewer) are loaded by default.
""",
    )
    parser.add_argument(
        "root",
        metavar="ROOT",
        help="Path to a converted LeRobot v2.0/v2.1/v3.0 dataset root directory "
        "(the directory containing meta/info.json).",
    )
    parser.add_argument(
        "--episodes",
        default=None,
        metavar="SPEC",
        help="Load one or more episodes into one Rerun viewer session, each as an "
        "independent, switchable recording. 0-based comma-list and colon-range "
        'syntax, end exclusive, e.g. "0,2:5" -> episodes 0, 2, 3, 4, or "all" for '
        f"every episode. Default: the first {_DEFAULT_EPISODE_COUNT} episodes (or "
        "all of them, if there are fewer).",
    )
    parser.add_argument(
        "--list-episodes",
        action="store_true",
        help="Print the dataset's total episode count and exit, instead of opening the viewer.",
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        metavar="ORG/NAME",
        help='Cosmetic org/dataset name passed through to lerobot. Default: "local/<basename of ROOT>".',
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="Serve the viewer over the network (Rerun's `mode=distant`) instead of spawning a "
        "local window -- use this to view from another machine on the LAN.",
    )
    parser.add_argument(
        "--web-port",
        type=int,
        default=_DEFAULT_WEB_PORT,
        metavar="PORT",
        help=f"Web port for the Rerun web viewer when --web is set. Default: {_DEFAULT_WEB_PORT}.",
    )
    parser.add_argument(
        "--host",
        default=None,
        metavar="IP",
        help="Override the auto-detected IP used in the printed connect/browser URLs when "
        "--web is set. Auto-detection picks whichever local IP the OS would use to reach "
        "the public internet, which is wrong if you're actually reachable via a different "
        "path (e.g. a Tailscale/VPN IP, or a machine with multiple network interfaces) -- "
        "the web page itself may load fine over that path while the embedded gRPC address "
        "still points at the wrong interface, showing Rerun's built-in example instead of "
        "your data. Pass the IP the remote browser will actually use to reach this machine.",
    )
    parser.add_argument(
        "--server-memory-limit",
        default=_SERVER_MEMORY_LIMIT,
        metavar="LIMIT",
        help="Max memory the --web gRPC server buffers before dropping the earliest-loaded "
        f'episode data (e.g. "50%%" or "4GB"). Default: {_SERVER_MEMORY_LIMIT.replace("%", "%%")}. '
        "Raise this if earlier episodes disappear from the recording list when loading "
        "many/large episodes.",
    )
    parser.add_argument(
        "--full-quality-images",
        action="store_true",
        help="Log frames uncompressed instead of JPEG-compressing them. Uses substantially "
        "more memory per episode (the main cause of earlier episodes getting dropped from "
        "the --web recording list) -- only use this if you need pixel-perfect precision.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=_DEFAULT_JPEG_QUALITY,
        metavar="N",
        help="JPEG quality (1-100, higher = larger/better) used when compressing frames. "
        f"Default: {_DEFAULT_JPEG_QUALITY} -- lower this further if episodes are still "
        "getting dropped from the --web recording list. Ignored with --full-quality-images.",
    )
    return parser


def main(args: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    parsed = parser.parse_args(args)

    dataset_root = Path(parsed.root)
    check = validate_dataset_root(dataset_root)
    if not check.ok:
        for error in check.errors:
            console.print(f"[red]✗ {error}[/red]")
        return 1
    for warning in check.warnings:
        console.print(f"[yellow]⚠ {warning}[/yellow]")

    if parsed.list_episodes:
        info = json.loads((dataset_root / "meta" / "info.json").read_text())
        total_episodes = info.get("total_episodes")
        console.print(f"total_episodes: {total_episodes}")
        return 0

    repo_id = parsed.repo_id or default_repo_id(dataset_root)

    info = json.loads((dataset_root / "meta" / "info.json").read_text())
    total_episodes = info.get("total_episodes")

    if parsed.episodes is None:
        episode_indices = list(range(min(_DEFAULT_EPISODE_COUNT, total_episodes)))
    elif parsed.episodes.strip().lower() == "all":
        episode_indices = list(range(total_episodes))
    else:
        try:
            episode_indices = parse_episodes_spec(parsed.episodes, total_episodes)
        except ValueError as exc:
            console.print(f"[red]✗ --episodes error: {exc}[/red]")
            return 1

    if not episode_indices:
        console.print("[red]✗ --episodes error: spec selected zero episodes[/red]")
        return 1
    if len(episode_indices) > _MANY_EPISODES_WARNING_THRESHOLD:
        console.print(
            f"[yellow]⚠ Loading {len(episode_indices)} episodes into one session may be "
            "slow (each episode is a full dataset load + Rerun log pass).[/yellow]"
        )
    return _run_episodes(dataset_root, repo_id, episode_indices, parsed)


def _run_episodes(
    dataset_root: Path,
    repo_id: str,
    episode_indices: List[int],
    parsed: argparse.Namespace,
) -> int:
    """
    Load each episode in `episode_indices` into its own `rerun.RecordingStream`,
    all connected to the same viewer/session (same `application_id`, distinct
    `recording_id`s) so they show up as independently switchable recordings in
    Rerun's own recording-list UI.
    """
    # Imported lazily so `--help` and `--list-episodes` stay fast and don't
    # require torch/rerun to be importable.
    import rerun as rr
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    # rr.serve_grpc()/rr.spawn() need an application_id to attach to. We never
    # call rr.init() (it would flush/destroy other recordings as a side
    # effect, which is exactly what we're avoiding here -- N independent,
    # simultaneously-alive recordings), so there's no "current active global
    # recording" for the bare module-level rr.serve_grpc() to fall back on --
    # it raises "No application id found" if called with no recording= and no
    # prior rr.init(). We sidestep this with a dedicated, throwaway
    # "bootstrap" RecordingStream used ONLY to call .serve_grpc() (which
    # passes recording=self internally) and get back the server's connect
    # URL -- it never logs any episode data itself. Every real episode
    # stream (including what would otherwise be the "first" one) then
    # uniformly .connect_grpc()s to that URL, exactly like every
    # non-first episode already did. Previously the first episode REUSED
    # the bootstrap stream directly instead of a fresh .connect_grpc(), and
    # started logging immediately after serve_grpc() returned -- serve_grpc()
    # only guarantees the server is *listening*, not that it's fully ready
    # to reliably receive a connected stream's data yet, and that race
    # silently dropped the first chunk of whichever episode went first
    # (confirmed by real-world testing: only the first-loaded episode was
    # missing its opening portion, every other episode was intact). rr.spawn
    # (connect=False) has no such requirement (recording is only used "if
    # connect=True", per its own docstring), so the local (non-`--web`)
    # branch never had this problem.
    remote_connect_url = None
    web_url = None
    if parsed.web:
        bootstrap_stream = rr.RecordingStream(application_id=repo_id, recording_id=str(uuid.uuid4()))
        # serve_grpc()'s server_memory_limit defaults to "25%" (of system
        # RAM) -- data for "late connecting" clients (which the web viewer
        # always is, since it connects only after the page loads) is
        # buffered up to that limit, and the EARLIEST logged data is dropped
        # once it's exceeded. We're deliberately loading a bounded, known
        # set of episodes that should ALL stay available for the whole
        # session, not act as a rolling live-tail buffer -- video frames
        # across several episodes routinely blow past 25%, which silently
        # evicted the earliest-loaded episodes (confirmed by testing: only
        # the last-loaded episode remained visible). Raise the limit well
        # above what the default multi-episode load is expected to need.
        connect_url = bootstrap_stream.serve_grpc(
            grpc_port=_DEFAULT_GRPC_PORT, server_memory_limit=parsed.server_memory_limit
        )
        # connect_url is "rerun+http://127.0.0.1:{port}/proxy" (serve_grpc()
        # hardcodes 127.0.0.1). That's correct for a client running on THIS
        # machine (used below for the remaining streams' .connect_grpc()),
        # but useless as the web viewer's connect_to target: a REMOTE
        # browser loading the served page would try to reach ITS OWN
        # loopback interface, not this server, and silently show nothing
        # (Rerun's web viewer falls back to its built-in example). Substitute
        # this machine's actual reachable IP for the browser-facing URLs
        # only. Auto-detection picks the OS's default-route interface, which
        # is wrong if the remote browser is actually reaching this machine
        # via a different path (confirmed by real-world testing: a Tailscale
        # IP loaded the page fine but the embedded gRPC address still pointed
        # at the plain LAN IP, which wasn't reachable from wherever the
        # browser was) -- --host lets the user override it explicitly.
        lan_ip = parsed.host or _detect_lan_ip()
        remote_connect_url = connect_url.replace("127.0.0.1", lan_ip)
        # rr.serve_web_viewer()'s connect_to only takes effect when
        # open_browser=True (it's used to build the URL that gets
        # webbrowser.open()'d locally -- per its own docstring: "If
        # open_browser is true, then this is the URL the web viewer will
        # connect to"). We pass open_browser=False (there's no local browser
        # to open when serving remotely), so connect_to is otherwise
        # discarded and a bare http://host:port/ shows Rerun's built-in
        # example instead of our data. The actual mechanism is a `?url=`
        # query parameter on the served page itself -- build that URL
        # ourselves and print it instead of the bare host:port.
        web_url = f"http://{lan_ip}:{parsed.web_port}/?url={quote(remote_connect_url, safe='')}"
        rr.serve_web_viewer(open_browser=False, web_port=parsed.web_port, connect_to=remote_connect_url)
    else:
        rr.spawn(port=_DEFAULT_GRPC_PORT, connect=False)
        connect_url = f"rerun+http://127.0.0.1:{_DEFAULT_GRPC_PORT}/proxy"

    # The connect/browser URLs are deliberately NOT printed here, before
    # loading starts. A user testing this over --web found the actual
    # trigger for episodes losing data: connecting a browser WHILE episodes
    # are still being logged races against the server replaying its
    # (evicting, shared, cross-recording) buffer to that newly-connecting
    # client -- waiting until loading is fully done before opening the
    # viewer avoided it entirely. Printing the URL only after the loop
    # below finishes makes that the natural default instead of relying on
    # the user to already know not to click early.
    from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task("Loading episodes", total=len(episode_indices))
        for episode_index in episode_indices:
            progress.update(task_id, description=f"Loading episode {episode_index}")
            dataset = LeRobotDataset(repo_id=repo_id, root=dataset_root, episodes=[episode_index])
            stream = rr.RecordingStream(application_id=repo_id, recording_id=str(uuid.uuid4()))
            stream.connect_grpc(connect_url)
            stream.send_recording_name(f"episode_{episode_index}")
            _log_episode_to_stream(
                dataset,
                episode_index,
                stream,
                compress_images=not parsed.full_quality_images,
                jpeg_quality=parsed.jpeg_quality,
            )
            stream.flush()
            progress.update(task_id, advance=1)

    console.print("All requested episodes loaded -- select any of them from the viewer's recording list.")
    if parsed.web:
        console.print(f"Connect a native viewer with: rerun {remote_connect_url}")
        console.print(f"Open in a browser: {web_url}")

    if parsed.web:
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            console.print("Ctrl-C received. Exiting.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
