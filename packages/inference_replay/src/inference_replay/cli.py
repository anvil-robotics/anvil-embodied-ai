"""`inference-replay` -- watch an inference_data.csv as a URDF-accurate 3D replay.

    inference-replay monitor_output/inference_data.csv                  # write a .rrd
    inference-replay monitor_output/inference_data.csv --spawn          # native viewer
    inference-replay monitor_output/inference_data.csv --web            # browser viewer
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import uuid
from pathlib import Path
from urllib.parse import quote

from rich.console import Console

console = Console()

# Same defaults as mcap_converter's dataset-viz, so the two CLIs behave alike.
_DEFAULT_WEB_PORT = 9090
_DEFAULT_GRPC_PORT = 9876


def _detect_lan_ip() -> str:
    """This machine's address on its default route, for a browser on another host."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))  # no traffic is sent; this just picks an interface
            return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="inference-replay",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("csv", type=Path, help="Path to an inference_data.csv written by inference_monitor_node")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Where to write the .rrd recording (default: alongside the CSV). Ignored with --spawn/--web.",
    )
    parser.add_argument("--spawn", action="store_true", help="Open the native Rerun viewer instead of writing a file")
    parser.add_argument("--web", action="store_true", help="Serve the Rerun web viewer instead of writing a file")
    parser.add_argument("--web-port", type=int, default=_DEFAULT_WEB_PORT, help="Port for --web (default: %(default)s)")
    parser.add_argument(
        "--host",
        default=None,
        help="Address to advertise to the browser with --web (default: auto-detect this machine's LAN IP)",
    )
    parser.add_argument(
        "--urdf",
        type=Path,
        default=None,
        help="Override the vendored URDF (default: the package's assets/openarm_v2)",
    )
    parser.add_argument(
        "--control-frequency",
        type=float,
        default=None,
        help="The run's inference control_frequency, to warn when the CSV undersampled it",
    )
    return parser


def main() -> int:
    parsed = _build_parser().parse_args()

    # Imported here so `--help` and argument errors stay fast and do not need the heavy deps.
    import rerun as rr

    from .analysis import analyse
    from .kinematics import RobotKinematics
    from .replay import default_blueprint, log_trace
    from .trace import TraceAlignmentError, load_trace, undersampling_warning

    if not parsed.csv.is_file():
        console.print(f"[red]No such file:[/red] {parsed.csv}")
        return 2

    # Checked before the (slow) load so an SSH session fails immediately rather than after
    # half a minute of work it cannot show. The native viewer needs a display on THIS machine.
    if parsed.spawn and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        console.print(
            "[red]--spawn needs a local display[/red] and neither DISPLAY nor WAYLAND_DISPLAY "
            "is set — this looks like a headless or SSH session, where rerun's viewer would "
            "start with nowhere to draw.\n"
            "Use [bold]--web[/bold] and open the printed URL from your own machine (see the "
            "README for the SSH-tunnel recipe), or [bold]--out replay.rrd[/bold] to write a "
            "file you can copy and open in a local viewer."
        )
        return 2

    try:
        trace = load_trace(parsed.csv)
    except TraceAlignmentError as e:
        console.print(f"[red]Cannot replay this trace[/red]\n{e}")
        return 1

    analysis = analyse(trace)
    try:
        kin = RobotKinematics(parsed.urdf)
    except (FileNotFoundError, ValueError) as e:
        console.print(f"[red]Robot description unusable[/red]\n{e}")
        return 1

    console.print(
        f"[bold]{trace.path.name}[/bold]  {trace.n} steps, {trace.duration_sec:.1f}s at "
        f"{trace.hz:.1f} Hz, action_type={trace.action_type}"
    )
    console.print(f"clamp cap {trace.cap:.4f} rad (alignment residual {trace.residual:.2e})")
    console.print(
        f"[yellow]{len(analysis.events)}[/yellow] stall events across "
        f"[yellow]{len(analysis.windows)}[/yellow] windows"
    )
    for window in analysis.windows:
        console.print(
            f"    t={window.t_sec:>6.1f}s  {window.duration_sec:>5.2f}s  {len(window.joints)} joints"
        )

    warning = undersampling_warning(trace, parsed.control_frequency)
    if warning:
        console.print(f"[yellow]warning:[/yellow] {warning}")

    application_id = f"inference-replay/{trace.path.stem}"
    stream = rr.RecordingStream(application_id=application_id, recording_id=str(uuid.uuid4()))

    # Pick the sink before logging: a stream started without one buffers everything in memory.
    # The blueprint is sent after the sink for the same reason -- sent first it has nowhere to
    # go, and the viewer falls back to an auto-generated layout with no tabs.
    web_url = None
    out_path = None
    if parsed.web:
        connect_url = stream.serve_grpc(grpc_port=_DEFAULT_GRPC_PORT)
        # serve_grpc() hardcodes 127.0.0.1, which is useless to a browser on another machine --
        # it would resolve to that machine's own loopback. Advertise a reachable address instead.
        lan_ip = parsed.host or _detect_lan_ip()
        remote_url = connect_url.replace("127.0.0.1", lan_ip)
        # serve_web_viewer()'s connect_to only takes effect with open_browser=True; the served
        # page otherwise needs an explicit ?url= or it shows Rerun's built-in example.
        web_url = f"http://{lan_ip}:{parsed.web_port}/?url={quote(remote_url, safe='')}"
        rr.serve_web_viewer(open_browser=False, web_port=parsed.web_port, connect_to=remote_url)
    elif parsed.spawn:
        stream.spawn()
    else:
        out_path = parsed.out or parsed.csv.with_suffix(".rrd")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        stream.save(str(out_path))

    stream.send_blueprint(default_blueprint())

    console.print("logging…")
    log_trace(stream, trace, analysis, kin)
    stream.flush()

    if out_path is not None:
        console.print(f"wrote [bold]{out_path}[/bold] ({out_path.stat().st_size / 1e6:.1f} MB)")
        console.print(f"open it with: rerun {out_path}")
    if web_url is not None:
        console.print(f"web viewer: [bold]{web_url}[/bold]")
        console.print("Ctrl-C to stop serving.")
        try:
            # serve_web_viewer() runs in the background; hold the process open for the browser.
            import time

            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            console.print("stopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
