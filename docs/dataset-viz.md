[← Back to README](../README.md)

# Dataset Visualization

Browse a converted LeRobot dataset's episodes, videos, and action/state curves with a [Rerun](https://rerun.io/)-based viewer. Fully local and offline -- no Docker, no network access, no third-party web app. Not part of the core conversion pipeline -- used as needed, after `mcap-convert`.

---

## Quick Start

```bash
uv run dataset-viz data/datasets/my-session
```

By default this loads the first 10 episodes (or all of them, if there are fewer) into one Rerun viewer window, each as an independent, switchable recording -- video, actions, and state synced on one timeline per episode.

```bash
uv run dataset-viz data/datasets/my-session --list-episodes       # how many episodes are there?
uv run dataset-viz data/datasets/my-session --episodes "0,2:5"    # only these episodes
uv run dataset-viz data/datasets/my-session --episodes all        # every episode
```

---

## Flags

| Flag | Default | Description |
|------|---------|-------------|
| `ROOT` | _(required)_ | Path to a LeRobot v2.0/v2.1/v3.0 dataset root |
| `--episodes SPEC` | first 10 (or all, if fewer) | Which episodes to load, each as its own switchable Rerun recording (see below). `"all"` loads every episode. |
| `--list-episodes` | — | Print the dataset's total episode count and exit -- use this to find valid values for `--episodes` |
| `--repo-id ORG/NAME` | `local/<basename of ROOT>` | Cosmetic name passed through to lerobot -- not used for any lookup |
| `--web` | — | Serve the viewer over the network (Rerun's `mode=distant`) instead of spawning a local window |
| `--web-port PORT` | `9090` | Web port for the Rerun web viewer when `--web` is set |
| `--host IP` | auto-detected | Override the IP used in the printed connect/browser URLs. Auto-detection guesses wrong if you're reachable via a different network path than the default route (e.g. Tailscale/VPN) -- see Remote / LAN Viewing below |
| `--server-memory-limit LIMIT` | `75%` | Max memory the `--web` gRPC server buffers before dropping the earliest-loaded episode data (e.g. `"50%"` or `"4GB"`). Raise this if earlier episodes disappear from the recording list when loading many/large episodes |
| `--full-quality-images` | — | Log frames uncompressed instead of JPEG-compressing them. Uses substantially more memory per episode (the main cause of episodes getting dropped, see below) -- only use this if you need pixel-perfect precision |
| `--jpeg-quality N` | `20` | JPEG quality (1-100, higher = larger/better) used when compressing frames. Lower this further if episodes are still getting dropped. Ignored with `--full-quality-images` |

---

## How It Works

1. Validate the dataset root: `meta/info.json` exists and parses, `codebase_version` is one of `v2.0`/`v2.1`/`v3.0`, `data/` directory exists (missing `videos/` is a warning only, not fatal).
2. If `--list-episodes`: read `total_episodes` from `meta/info.json`, print it, and exit -- no dataset loading, no viewer.
3. Otherwise, resolve which episodes to load: an explicit `--episodes` spec, `"all"`, or (if `--episodes` is omitted) the first 10 -- or all of them, if the dataset has fewer than 10.
4. Each requested episode gets its own `LeRobotDataset(..., episodes=[i])` and its own `rerun.RecordingStream` (same `application_id` -- the dataset's repo-id, so the viewer groups them into one app; distinct `recording_id` per episode). All streams connect to the same underlying viewer/gRPC server, so a single `dataset-viz` invocation opens one Rerun session containing every requested episode as an independently switchable recording. A progress bar tracks this loop (`Loading episode N ━━━ i/total`).
5. Without `--web`, Rerun spawns a local viewer window directly and the command returns once logging finishes; the window persists independently afterward.
6. With `--web`, Rerun instead serves the recordings over gRPC + a web viewer on `--web-port`. **The connect/browser URLs are only printed after every requested episode has finished loading** -- opening the browser earlier, while episodes are still being logged, was confirmed (real-world testing) to race against the server's buffer replay and cause the first-loaded episode to lose its opening portion permanently (see the note below). Once printed, the command blocks (Ctrl-C to stop).

To switch between episodes, use **Rerun's own recording list** in the viewer UI.

**Episodes losing data (whole episodes disappearing, or an episode missing its opening seconds) is a real bug, not a UI quirk, and high-resolution cameras make it worse.** `--web`'s gRPC server buffers logged data in ONE memory pool shared across every recording connected to it (not a separate pool per episode) so that late-connecting viewers (which the web viewer always is) get everything -- but Rerun's own default buffer limit is only 25% of system RAM, and video frames across several episodes routinely exceed that, silently dropping the *earliest*-buffered data once the limit is hit, regardless of which episode it belongs to. Two mitigations, both on by default:
- Frames are JPEG-compressed before logging (`--jpeg-quality`, default `20` -- much more aggressive than Rerun's own default of 95, confirmed by testing to still show recognizable video while shrinking both memory use and load time). `--full-quality-images` opts out if you need pixel-perfect precision, at the cost of reintroducing this problem.
- `--server-memory-limit` (default `75%`) raises Rerun's own 25% ceiling. If episodes are still losing data, raise it further -- an absolute value like `"16GB"` may be more predictable than a percentage on a shared/multi-tenant machine.

**Confirmed trigger: connecting the browser WHILE episodes are still loading.** Because the buffer above is shared and FIFO-evicted, replaying it to a client that connects mid-load races against both new data arriving (from episodes still being logged) and old data being evicted to make room -- real-world testing traced a missing opening segment on the *first*-loaded episode specifically to this. `dataset-viz` now only prints the connect/browser URL after every requested episode has finished loading (a progress bar tracks the load instead) precisely to make "wait, then connect" the default -- opening a previously-printed or bookmarked URL early can still hit this.

If more than 20 episodes are requested (including via `--episodes all` on a large dataset), a warning is printed (not a hard block) that loading that many into one session may be slow, since each episode is still a full dataset load + Rerun log pass -- and the memory pressure described above scales with it too.

**Refreshing the browser page re-triggers a full replay of the buffered history.** This is inherent to the same buffering mechanism above (a page refresh looks like a brand-new "late-connecting" viewer to the gRPC server, which resends everything it has buffered so far) -- expect the same sequential-loading behavior to play out again on every refresh, not a sign that logging re-ran. Since loading has already finished by the time you have a URL to refresh, this replay is safe (no new data is arriving to race against).

---

## Remote / LAN Viewing

```bash
uv run dataset-viz data/datasets/my-session --web --web-port 9090
uv run dataset-viz data/datasets/my-session --episodes "0,2:5" --web --web-port 9090
```

The command prints a ready-to-use browser URL, e.g. `http://192.168.1.42:9090/?url=rerun%2Bhttp%3A%2F%2F192.168.1.42%3A9876%2Fproxy` -- open it from another device on the same network. The `?url=` part matters: Rerun's own `connect_to` parameter only auto-connects a locally-opened browser, so a bare `http://<ip>:<port>/` shows Rerun's built-in example instead of your data. The IP is auto-detected (via the local route used to reach the public internet, no actual internet access required) and substituted for the `127.0.0.1` that Rerun's own APIs hardcode -- otherwise a remote browser would try to connect to its own loopback interface instead of this machine. This replaces the old Docker/nginx-based browsing approach with the mechanism Rerun itself supports for remote viewing.

**If you're reachable via a different network path than the one auto-detection guesses (e.g. Tailscale/VPN, or a machine with multiple NICs), use `--host`.** Confirmed by real-world testing: opening the printed URL via a Tailscale IP loaded the web page fine (Tailscale routed that request correctly), but the embedded `?url=` still pointed at the auto-detected *plain LAN IP* (whatever the OS's default route to the public internet uses), which the remote browser couldn't reach at all -- back to Rerun's built-in example page, same symptom as the original `127.0.0.1` bug, different cause. Auto-detection has no way to know which of a machine's several possible addresses you'll actually be reached through; tell it explicitly:

```bash
uv run dataset-viz data/datasets/my-session --web --host 100.101.20.109   # e.g. a Tailscale IP
```

Honesty note: whether Rerun's gRPC/web server actually binds to all network interfaces (reachable from another machine) or only to `localhost` is determined by Rerun's own compiled viewer, not something configurable from this CLI -- it was not possible to verify the literal bind address from Python source in this environment (no display/network testing available). Rerun's `--serve`/web-viewer mode is specifically designed and documented for cross-machine viewing, so this is expected to work, but if the printed URL isn't reachable from another device even with the right `--host`, that's a Rerun-level networking question (e.g. a firewall blocking the port), not a `dataset-viz` bug.

---

## Troubleshooting

- **Invalid dataset root** — the path must contain a `meta/info.json` that parses as JSON, with `codebase_version` in `v2.0`/`v2.1`/`v3.0`, and a `data/` directory. Run `mcap-convert` first if you haven't converted the session yet.
- **Only seeing the first 10 episodes** — that's the default when `--episodes` is omitted, not a bug. Use `--episodes "10:20"` (or similar), or `--episodes all` for the whole dataset. Run `--list-episodes` to see the total count first.
- **Episodes disappear or lose their opening seconds over `--web`** — first, make sure you waited for `dataset-viz` to print the connect/browser URL (i.e. the progress bar finished) before opening it; connecting early is a confirmed trigger. If it still happens, the gRPC server's shared memory buffer was exceeded and dropped the oldest data — images are compressed by default already (`--jpeg-quality`, default `20`); if it still happens (e.g. very high-resolution cameras or many episodes), raise `--server-memory-limit` further (see How It Works above).
- **`--web` port already in use** — pick a different `--web-port`.
- **The web page loads (over Tailscale/VPN/a non-default network path) but shows Rerun's example instead of your data** — the auto-detected IP embedded in the URL doesn't match the path you're actually connecting through. Pass `--host` with the IP the remote browser uses to reach this machine (see Remote / LAN Viewing above).

---

[← Back to README](../README.md)
