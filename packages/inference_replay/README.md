# inference-replay

URDF-accurate virtual replay of an `inference_data.csv` monitor trace, rendered with Rerun.

The observed trajectory (`obs_state`) draws as a solid robot; the commanded trajectory
(`control_cmd`) rides alongside as a ghost, so commanded-vs-observed divergence is visible
in space instead of inferred from a plot. Per-joint stall and clamp analysis rides the same
timeline.

Nothing here touches hardware — it is a viewer, not a replay onto the arms.

## Install

Everything needed to render a trace is committed, including the robot description — a clone
and a sync is the whole setup, with no other repo required:

```bash
git clone https://github.com/anvil-robotics/anvil-embodied-ai.git
cd anvil-embodied-ai
uv sync --package inference-replay    # viewer only: no torch, no lerobot
```

Deps are deliberately light (`rerun-sdk`, `yourdfpy`, `numpy`, `rich`) so looking at a CSV does
not pull the torch/lerobot stack the training packages need.

Note that `--package` *prunes* the shared workspace venv to just this package's deps. On a
machine that also trains or converts, use the repo-wide form from the root README instead —
`uv sync --all-packages` — which includes this package, then prefix commands with the package
so the right entry point resolves:

```bash
uv run --package inference-replay inference-replay monitor_output/inference_data.csv
```

## Use

```bash
inference-replay monitor_output/inference_data.csv           # write a .rrd
inference-replay monitor_output/inference_data.csv --spawn   # native viewer
inference-replay monitor_output/inference_data.csv --web     # browser viewer
```

Pass `--control-frequency 50` to be warned when the monitor's fixed ~30 Hz logging
undersampled a faster command stream (as in `monitor_output/control_freq_50/`).

## Viewing it from another machine

The traces live on the inference box, which usually has no display, so `--spawn` is only for a
workstation you are sitting at — it refuses outright when `DISPLAY`/`WAYLAND_DISPLAY` are unset
rather than starting a viewer with nowhere to draw.

**Over Tailscale / any routable address** — the usual case, and no tunnel needed:

```bash
inference-replay monitor_output/inference_data.csv --web --host <the-address-you-connect-over>
# e.g. --host my-inference-box.my-tailnet.ts.net
```

Then open the printed URL on your own machine. Both ports have to be reachable: 9090 serves
the page, 9876 is the gRPC data proxy. Rerun UI but an empty recording means 9090 got through
and 9876 did not.

`--host` matters, and defaults to the box's *default-route* address. If you reach the box some
other way -- Tailscale, a VPN, a second NIC -- that default is wrong: the page loads, but the
gRPC address embedded in it points somewhere your browser cannot reach and the viewer sits
empty. Pass the address you actually connect over.

**SSH tunnel** — only if the box runs a real `sshd`. Note that **Tailscale SSH does not
support port forwarding**, so `-L` silently gets you nowhere on a tailnet-only box (and if
nothing listens on :22, `ssh -L` just refuses the connection). Check with
`ss -ltn | grep :22` before reaching for this.

```bash
# from your own machine, not from the box
ssh -L 9090:localhost:9090 -L 9876:localhost:9876 <you>@<the-box>
# then, on the box
inference-replay monitor_output/inference_data.csv --web --host 127.0.0.1
```

**Or skip the network** and copy a file:

```bash
inference-replay monitor_output/inference_data.csv --out replay.rrd
# from your own machine
scp <you>@<the-box>:~/anvil-embodied-ai/replay.rrd . && rerun replay.rrd
```

## Three ways to look at it

Both robots are always logged; the modes are visibility toggles in the viewer, not separate
recordings:

| Mode | show |
|---|---|
| observed only | `obs/mesh/**` |
| observed + commanded shadow (default) | `obs/mesh/**` and `cmd/marker/**` |
| commanded only | `cmd/mesh/**` |

## Robot description

The URDF and meshes under `src/inference_replay/assets/openarm_v2` are generated from an
anvil-workcell checkout and committed here, because that repo deliberately does not use
submodules and its CAD meshes are 117 MB. Regenerate after an upstream description change:

```bash
uv run --package inference-replay --extra sync python scripts/sync_openarm_assets.py --workcell ~/anvil-workcell
```

`assets/openarm_v2/SOURCE_COMMIT` records which upstream commit the current snapshot came
from, and the exact xacro invocation used.
