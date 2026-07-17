"""CLI-level tests for dataset-viz: verify main()'s control flow -- validate
-> (--list-episodes early-exit | resolve the --episodes spec, defaulting to
the first 10 episodes | load each into its own rerun.RecordingStream) -- and
that flags map to the right call arguments. `rerun.RecordingStream`/`rr.spawn`
and `LeRobotDataset` are mocked so tests never actually spawn a Rerun viewer
or load real dataset frames.

Pure-function tests for the underlying viz.{dataset_check,config} helpers
live in test_dataset_viz.py; this file is specifically about
mcap_converter.cli.dataset_viz.main().
"""

import json
from pathlib import Path

import pytest

from mcap_converter.cli.dataset_viz import _detect_lan_ip, main, parse_episodes_spec
from mcap_converter.viz.config import default_repo_id


def _make_dataset(tmp_path: Path, *, total_episodes: int = 5) -> Path:
    """Build a minimal valid synthetic LeRobot v3.0 dataset root under tmp_path."""
    root = tmp_path / "my-dataset"
    meta_dir = root / "meta"
    meta_dir.mkdir(parents=True)
    (meta_dir / "info.json").write_text(
        json.dumps({"codebase_version": "v3.0", "total_episodes": total_episodes})
    )
    (root / "data").mkdir()
    return root


class _FakeDataset:
    """Stand-in for LeRobotDataset that just records constructor args."""

    def __init__(self, repo_id, root, episodes):
        self.repo_id = repo_id
        self.root = root
        self.episodes = episodes


class TestHelp:
    def test_help_exits_zero_and_mentions_key_flags(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0

        out = capsys.readouterr().out
        for flag in (
            "--episodes",
            "--list-episodes",
            "--repo-id",
            "--web",
            "--web-port",
        ):
            assert flag in out


class TestRootRequiredness:
    def test_root_missing_is_a_usage_error(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main([])
        assert exc_info.value.code == 2
        err = capsys.readouterr().err
        assert "ROOT" in err or "root" in err


class TestDatasetValidation:
    def test_invalid_dataset_root_returns_1_and_short_circuits(self, tmp_path, capsys):
        bad_root = tmp_path / "not-a-dataset"
        bad_root.mkdir()
        rc = main([str(bad_root)])

        assert rc == 1
        out = capsys.readouterr().out
        assert "info.json" in out or "does not exist" in out

    def test_missing_root_directory_returns_1(self, tmp_path, capsys):
        rc = main([str(tmp_path / "does-not-exist")])

        assert rc == 1
        out = capsys.readouterr().out
        assert "does not exist" in out or "not a directory" in out


class TestListEpisodes:
    def test_prints_total_episodes_and_exits_zero(self, tmp_path, capsys):
        dataset_root = _make_dataset(tmp_path, total_episodes=7)

        rc = main([str(dataset_root), "--list-episodes"])

        assert rc == 0
        out = capsys.readouterr().out
        assert "7" in out

    def test_invalid_root_short_circuits_before_list_episodes(self, tmp_path, capsys):
        bad_root = tmp_path / "not-a-dataset"
        bad_root.mkdir()

        rc = main([str(bad_root), "--list-episodes"])

        assert rc == 1
        out = capsys.readouterr().out
        assert "info.json" in out or "does not exist" in out


class TestRepoIdDefault:
    def test_default_repo_id_used_when_not_passed(self, monkeypatch, tmp_path):
        dataset_root = _make_dataset(tmp_path)
        captured = {}

        monkeypatch.setattr(
            "lerobot.datasets.lerobot_dataset.LeRobotDataset", _FakeDataset
        )
        monkeypatch.setattr("rerun.RecordingStream", _FakeStream)
        monkeypatch.setattr("rerun.spawn", lambda **kwargs: None)
        monkeypatch.setattr(
            "mcap_converter.cli.dataset_viz._log_episode_to_stream",
            lambda dataset, episode_index, stream, **kw: captured.setdefault("repo_id", dataset.repo_id),
        )

        rc = main([str(dataset_root)])

        assert rc == 0
        assert captured["repo_id"] == default_repo_id(dataset_root)

    def test_explicit_repo_id_overrides_default(self, monkeypatch, tmp_path):
        dataset_root = _make_dataset(tmp_path)
        captured = {}

        monkeypatch.setattr(
            "lerobot.datasets.lerobot_dataset.LeRobotDataset", _FakeDataset
        )
        monkeypatch.setattr("rerun.RecordingStream", _FakeStream)
        monkeypatch.setattr("rerun.spawn", lambda **kwargs: None)
        monkeypatch.setattr(
            "mcap_converter.cli.dataset_viz._log_episode_to_stream",
            lambda dataset, episode_index, stream, **kw: captured.setdefault("repo_id", dataset.repo_id),
        )

        rc = main([str(dataset_root), "--repo-id", "anvil/custom-name"])

        assert rc == 0
        assert captured["repo_id"] == "anvil/custom-name"


class TestParseEpisodesSpec:
    """Unit tests for the 0-based --episodes spec parser."""

    def test_single_index(self):
        assert parse_episodes_spec("2", total_episodes=5) == [2]

    def test_comma_list(self):
        assert parse_episodes_spec("0,2,4", total_episodes=5) == [0, 2, 4]

    def test_colon_range_is_end_exclusive(self):
        # "1:4" -> 1, 2, 3 (not 4), matching Python's range(1, 4).
        assert parse_episodes_spec("1:4", total_episodes=5) == [1, 2, 3]

    def test_mixed_comma_list_and_range(self):
        assert parse_episodes_spec("0,2:4", total_episodes=5) == [0, 2, 3]

    def test_open_ended_start_defaults_to_zero(self):
        assert parse_episodes_spec(":3", total_episodes=5) == [0, 1, 2]

    def test_open_ended_end_reaches_last_episode_inclusively(self):
        assert parse_episodes_spec("3:", total_episodes=5) == [3, 4]

    def test_duplicate_indices_are_deduplicated(self):
        assert parse_episodes_spec("1,1,1:3", total_episodes=5) == [1, 2]

    def test_out_of_range_single_index_raises(self):
        with pytest.raises(ValueError, match="out of range"):
            parse_episodes_spec("5", total_episodes=5)

    def test_negative_single_index_raises(self):
        with pytest.raises(ValueError, match="out of range"):
            parse_episodes_spec("-1", total_episodes=5)

    def test_out_of_bounds_range_raises(self):
        with pytest.raises(ValueError, match="out of bounds"):
            parse_episodes_spec("3:6", total_episodes=5)

    def test_invalid_token_raises(self):
        with pytest.raises(ValueError, match="invalid episode index token"):
            parse_episodes_spec("abc", total_episodes=5)

    def test_invalid_range_token_raises(self):
        with pytest.raises(ValueError, match="invalid episode range token"):
            parse_episodes_spec("a:b", total_episodes=5)

    def test_range_start_not_less_than_end_raises(self):
        with pytest.raises(ValueError, match="start must be less than end"):
            parse_episodes_spec("3:3", total_episodes=5)

    def test_empty_spec_returns_empty_list(self):
        assert parse_episodes_spec("", total_episodes=5) == []


class TestDetectLanIp:
    """
    _detect_lan_ip() is a best-effort helper: it substitutes Rerun's
    hardcoded 127.0.0.1 with this machine's actual LAN IP for the
    browser-facing --web URL, since a remote browser's own loopback
    interface has nothing listening on it. These tests fake the socket
    module so they don't depend on this test machine's real network state.
    """

    def test_returns_getsockname_result(self, monkeypatch):
        class _FakeSocket:
            def connect(self, addr):
                pass

            def getsockname(self):
                return ("10.1.2.3", 54321)

            def close(self):
                pass

        monkeypatch.setattr("socket.socket", lambda *a, **kw: _FakeSocket())

        assert _detect_lan_ip() == "10.1.2.3"

    def test_falls_back_to_loopback_on_failure(self, monkeypatch):
        class _FailingSocket:
            def connect(self, addr):
                raise OSError("Network is unreachable")

            def close(self):
                pass

        monkeypatch.setattr("socket.socket", lambda *a, **kw: _FailingSocket())

        assert _detect_lan_ip() == "127.0.0.1"


class _FakeStream:
    """Stand-in for rerun.RecordingStream that records calls instead of touching real Rerun."""

    instances: list = []

    def __init__(self, application_id, recording_id):
        self.application_id = application_id
        self.recording_id = recording_id
        self.connected_to = None
        self.recording_name = None
        self.flushed = False
        self.served_grpc_kwargs = None
        _FakeStream.instances.append(self)

    def connect_grpc(self, url):
        self.connected_to = url

    def serve_grpc(self, **kwargs):
        self.served_grpc_kwargs = kwargs
        self.connected_to = "rerun+http://127.0.0.1:9876/proxy"
        return self.connected_to

    def send_recording_name(self, name):
        self.recording_name = name

    def flush(self):
        self.flushed = True


class TestEpisodesFlagMapping:
    """
    Verifies --episodes wires up N independent RecordingStreams (one per
    requested episode) without ever touching a real Rerun process: rr.spawn,
    RecordingStream, and the per-frame logging helper are all mocked.
    """

    def _patch_common(self, monkeypatch, tmp_path, logged_calls):
        dataset_root = _make_dataset(tmp_path, total_episodes=5)
        _FakeStream.instances = []

        monkeypatch.setattr(
            "lerobot.datasets.lerobot_dataset.LeRobotDataset", _FakeDataset
        )
        monkeypatch.setattr("rerun.RecordingStream", _FakeStream)
        monkeypatch.setattr("rerun.spawn", lambda **kwargs: logged_calls.setdefault("spawn_kwargs", kwargs))
        monkeypatch.setattr(
            "mcap_converter.cli.dataset_viz._log_episode_to_stream",
            lambda dataset, episode_index, stream, **kw: logged_calls.setdefault("logged", []).append(
                (episode_index, stream)
            ),
        )
        return dataset_root

    def test_episodes_spec_creates_one_stream_per_episode(self, monkeypatch, tmp_path):
        logged_calls = {}
        dataset_root = self._patch_common(monkeypatch, tmp_path, logged_calls)

        rc = main([str(dataset_root), "--episodes", "0,2:4"])

        assert rc == 0
        assert len(_FakeStream.instances) == 3
        logged_episode_indices = [ep for ep, _stream in logged_calls["logged"]]
        assert logged_episode_indices == [0, 2, 3]
        for stream in _FakeStream.instances:
            assert stream.flushed is True
            assert stream.connected_to == "rerun+http://127.0.0.1:9876/proxy"
        # Same application_id (repo_id) across all streams so the viewer groups
        # them into one app; distinct recording_ids so they're separate recordings.
        assert len({s.application_id for s in _FakeStream.instances}) == 1
        assert len({s.recording_id for s in _FakeStream.instances}) == 3
        assert "spawn_kwargs" in logged_calls

    def test_omitting_episodes_loads_first_default_count(self, monkeypatch, tmp_path):
        # total_episodes (5) is below the default count (10) -- every episode
        # should be loaded, not just the first 5 of some hardcoded larger set.
        logged_calls = {}
        dataset_root = self._patch_common(monkeypatch, tmp_path, logged_calls)

        rc = main([str(dataset_root)])

        assert rc == 0
        logged_episode_indices = [ep for ep, _stream in logged_calls["logged"]]
        assert logged_episode_indices == [0, 1, 2, 3, 4]

    def test_omitting_episodes_caps_at_default_count_for_larger_datasets(self, monkeypatch, tmp_path):
        logged_calls = {}
        dataset_root = _make_dataset(tmp_path, total_episodes=100)
        _FakeStream.instances = []
        monkeypatch.setattr(
            "lerobot.datasets.lerobot_dataset.LeRobotDataset", _FakeDataset
        )
        monkeypatch.setattr("rerun.RecordingStream", _FakeStream)
        monkeypatch.setattr("rerun.spawn", lambda **kwargs: logged_calls.setdefault("spawn_kwargs", kwargs))
        monkeypatch.setattr(
            "mcap_converter.cli.dataset_viz._log_episode_to_stream",
            lambda dataset, episode_index, stream, **kw: logged_calls.setdefault("logged", []).append(
                (episode_index, stream)
            ),
        )

        rc = main([str(dataset_root)])

        assert rc == 0
        logged_episode_indices = [ep for ep, _stream in logged_calls["logged"]]
        assert logged_episode_indices == list(range(10))

    def test_episodes_all_loads_every_episode(self, monkeypatch, tmp_path):
        logged_calls = {}
        dataset_root = self._patch_common(monkeypatch, tmp_path, logged_calls)  # total_episodes=5

        rc = main([str(dataset_root), "--episodes", "all"])

        assert rc == 0
        logged_episode_indices = [ep for ep, _stream in logged_calls["logged"]]
        assert logged_episode_indices == [0, 1, 2, 3, 4]

    def test_out_of_range_episodes_spec_returns_1(self, tmp_path, capsys):
        dataset_root = _make_dataset(tmp_path, total_episodes=5)

        rc = main([str(dataset_root), "--episodes", "10"])

        assert rc == 1
        out = capsys.readouterr().out
        assert "out of range" in out

    def test_many_episodes_prints_slowness_warning(self, monkeypatch, tmp_path):
        logged_calls = {}
        dataset_root = _make_dataset(tmp_path, total_episodes=25)
        _FakeStream.instances = []
        monkeypatch.setattr(
            "lerobot.datasets.lerobot_dataset.LeRobotDataset", _FakeDataset
        )
        monkeypatch.setattr("rerun.RecordingStream", _FakeStream)
        monkeypatch.setattr("rerun.spawn", lambda **kwargs: None)
        monkeypatch.setattr(
            "mcap_converter.cli.dataset_viz._log_episode_to_stream",
            lambda dataset, episode_index, stream, **kw: None,
        )

        import io
        from contextlib import redirect_stdout

        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = main([str(dataset_root), "--episodes", "0:25"])

        assert rc == 0
        assert len(_FakeStream.instances) == 25
        assert "may be" in buf.getvalue() or "slow" in buf.getvalue()

    def test_web_flag_uses_serve_grpc_and_web_viewer(self, monkeypatch, tmp_path, capsys):
        # rr.serve_grpc() (the bare module-level function) requires an
        # application_id from an already-active recording, which we
        # deliberately never establish via rr.init(). The real code sidesteps
        # this with a dedicated, throwaway "bootstrap" RecordingStream used
        # ONLY to call .serve_grpc() (an INSTANCE method, which passes
        # recording=self internally) — it never logs any episode data
        # itself. This test locks that in: it deliberately does NOT patch a
        # module-level "rerun.serve_grpc" at all, so if the code regressed to
        # calling that instead, this test would fail with an
        # AttributeError/real-Rerun-call rather than silently passing.
        #
        # Every REAL episode stream (including the first requested episode)
        # uniformly .connect_grpc()s to the bootstrap stream's URL — none of
        # them reuse the bootstrap stream directly. This was a real, fixed
        # bug: the first episode used to log through the bootstrap stream
        # right after serve_grpc() returned, and serve_grpc() only
        # guarantees the server is listening, not that it's immediately
        # ready to reliably receive a stream's data — that race silently
        # dropped the first episode's opening frames in real-world testing.
        logged_calls = {}
        dataset_root = _make_dataset(tmp_path, total_episodes=5)
        _FakeStream.instances = []

        monkeypatch.setattr(
            "lerobot.datasets.lerobot_dataset.LeRobotDataset", _FakeDataset
        )
        monkeypatch.setattr("rerun.RecordingStream", _FakeStream)
        monkeypatch.setattr(
            "rerun.serve_web_viewer",
            lambda **kwargs: logged_calls.setdefault("serve_web_viewer_kwargs", kwargs),
        )
        monkeypatch.setattr(
            "mcap_converter.cli.dataset_viz._log_episode_to_stream",
            lambda dataset, episode_index, stream, **kw: logged_calls.setdefault(
                "logged_streams", []
            ).append(stream),
        )
        # Avoid actually blocking on the "serve forever" loop for --web.
        monkeypatch.setattr(
            "mcap_converter.cli.dataset_viz.time.sleep",
            lambda _seconds: (_ for _ in ()).throw(KeyboardInterrupt()),
        )
        # serve_grpc() hardcodes 127.0.0.1, which is only reachable from this
        # same machine -- the real code substitutes the actual LAN IP before
        # handing the URL to serve_web_viewer(), since that's what a REMOTE
        # browser needs to connect. Pin the detected IP so this assertion is
        # deterministic across machines.
        monkeypatch.setattr(
            "mcap_converter.cli.dataset_viz._detect_lan_ip", lambda: "10.0.0.5"
        )

        rc = main([str(dataset_root), "--episodes", "0,1", "--web", "--web-port", "1234"])

        assert rc == 0
        # 1 throwaway bootstrap stream + 2 real episode streams.
        assert len(_FakeStream.instances) == 3
        bootstrap_stream, episode_stream_0, episode_stream_1 = _FakeStream.instances
        assert bootstrap_stream.served_grpc_kwargs == {
            "grpc_port": 9876,
            "server_memory_limit": "75%",
        }
        assert bootstrap_stream.connected_to == "rerun+http://127.0.0.1:9876/proxy"
        # Every real episode stream uniformly .connect_grpc()s -- none of
        # them reuse the bootstrap stream's .serve_grpc() connection.
        for episode_stream in (episode_stream_0, episode_stream_1):
            assert episode_stream.served_grpc_kwargs is None
            assert episode_stream.connected_to == "rerun+http://127.0.0.1:9876/proxy"
        # The bootstrap stream never receives any logging calls; only the
        # two real episode streams do.
        assert logged_calls["logged_streams"] == [episode_stream_0, episode_stream_1]
        assert logged_calls["serve_web_viewer_kwargs"]["web_port"] == 1234
        # But the browser-facing connect_to IS substituted to the LAN IP.
        assert logged_calls["serve_web_viewer_kwargs"]["connect_to"] == "rerun+http://10.0.0.5:9876/proxy"
        # serve_web_viewer()'s connect_to is only applied when open_browser=
        # True (per its own docstring) -- we pass open_browser=False, so it's
        # otherwise a no-op. The actual mechanism for a bare served page to
        # auto-connect is a `?url=` query parameter; assert the printed
        # browser URL actually includes one, pointing at the LAN IP.
        out = capsys.readouterr().out
        assert (
            "http://10.0.0.5:1234/?url=rerun%2Bhttp%3A%2F%2F10.0.0.5%3A9876%2Fproxy" in out
        )

    def test_host_flag_overrides_auto_detected_ip(self, monkeypatch, tmp_path, capsys):
        # Auto-detection picks the OS's default-route interface, which is
        # wrong if the remote browser actually reaches this machine via a
        # different path (confirmed by real-world testing: a Tailscale IP
        # loaded the page but the embedded gRPC address still pointed at the
        # auto-detected plain LAN IP, unreachable from the browser's side).
        # --host must override _detect_lan_ip()'s result everywhere it's used.
        logged_calls = {}
        dataset_root = _make_dataset(tmp_path, total_episodes=5)
        _FakeStream.instances = []

        monkeypatch.setattr(
            "lerobot.datasets.lerobot_dataset.LeRobotDataset", _FakeDataset
        )
        monkeypatch.setattr("rerun.RecordingStream", _FakeStream)
        monkeypatch.setattr(
            "rerun.serve_web_viewer",
            lambda **kwargs: logged_calls.setdefault("serve_web_viewer_kwargs", kwargs),
        )
        monkeypatch.setattr(
            "mcap_converter.cli.dataset_viz._log_episode_to_stream",
            lambda dataset, episode_index, stream, **kw: None,
        )
        monkeypatch.setattr(
            "mcap_converter.cli.dataset_viz.time.sleep",
            lambda _seconds: (_ for _ in ()).throw(KeyboardInterrupt()),
        )
        # _detect_lan_ip must NOT be consulted at all when --host is given --
        # raise if it is, so this test would fail loudly instead of silently
        # passing if --host were ever ignored.
        monkeypatch.setattr(
            "mcap_converter.cli.dataset_viz._detect_lan_ip",
            lambda: (_ for _ in ()).throw(AssertionError("_detect_lan_ip should not be called when --host is set")),
        )

        rc = main(
            [
                str(dataset_root),
                "--episodes",
                "0",
                "--web",
                "--web-port",
                "1234",
                "--host",
                "100.101.20.109",
            ]
        )

        assert rc == 0
        assert logged_calls["serve_web_viewer_kwargs"]["connect_to"] == "rerun+http://100.101.20.109:9876/proxy"
        out = capsys.readouterr().out
        assert "http://100.101.20.109:1234/?url=" in out
        assert "192.168" not in out

    def test_server_memory_limit_flag_overrides_default(self, monkeypatch, tmp_path):
        # serve_grpc()'s own default (25%) is too low for a deliberate,
        # bounded multi-episode load -- video data across several episodes
        # routinely exceeds it, silently dropping the earliest-loaded
        # episodes (confirmed by real-world testing). --server-memory-limit
        # lets a user raise it further for very large --episodes specs.
        logged_calls = {}
        dataset_root = _make_dataset(tmp_path, total_episodes=5)
        _FakeStream.instances = []

        monkeypatch.setattr(
            "lerobot.datasets.lerobot_dataset.LeRobotDataset", _FakeDataset
        )
        monkeypatch.setattr("rerun.RecordingStream", _FakeStream)
        monkeypatch.setattr(
            "rerun.serve_web_viewer",
            lambda **kwargs: logged_calls.setdefault("serve_web_viewer_kwargs", kwargs),
        )
        monkeypatch.setattr(
            "mcap_converter.cli.dataset_viz._log_episode_to_stream",
            lambda dataset, episode_index, stream, **kw: None,
        )
        monkeypatch.setattr(
            "mcap_converter.cli.dataset_viz.time.sleep",
            lambda _seconds: (_ for _ in ()).throw(KeyboardInterrupt()),
        )
        monkeypatch.setattr(
            "mcap_converter.cli.dataset_viz._detect_lan_ip", lambda: "10.0.0.5"
        )

        rc = main(
            [str(dataset_root), "--episodes", "0", "--web", "--server-memory-limit", "8GB"]
        )

        assert rc == 0
        assert _FakeStream.instances[0].served_grpc_kwargs == {
            "grpc_port": 9876,
            "server_memory_limit": "8GB",
        }

    def test_images_compressed_by_default_uncompressed_with_full_quality_flag(
        self, monkeypatch, tmp_path
    ):
        # Uncompressed frames across several episodes are the main reason the
        # --web gRPC server's memory buffer gets exceeded (confirmed by
        # real-world testing) -- compression is on by default; --full-quality-images
        # opts back out.
        logged_calls = {}
        dataset_root = _make_dataset(tmp_path, total_episodes=5)
        _FakeStream.instances = []

        monkeypatch.setattr(
            "lerobot.datasets.lerobot_dataset.LeRobotDataset", _FakeDataset
        )
        monkeypatch.setattr("rerun.RecordingStream", _FakeStream)
        monkeypatch.setattr("rerun.spawn", lambda **kwargs: None)
        monkeypatch.setattr(
            "mcap_converter.cli.dataset_viz._log_episode_to_stream",
            lambda dataset, episode_index, stream, **kw: logged_calls.setdefault(
                "compress_images_by_episode", []
            ).append(kw.get("compress_images")),
        )

        rc = main([str(dataset_root), "--episodes", "0,1"])
        assert rc == 0
        assert logged_calls["compress_images_by_episode"] == [True, True]

        logged_calls.clear()
        _FakeStream.instances = []
        rc = main([str(dataset_root), "--episodes", "0,1", "--full-quality-images"])
        assert rc == 0
        assert logged_calls["compress_images_by_episode"] == [False, False]

    def test_jpeg_quality_defaults_and_overrides(self, monkeypatch, tmp_path):
        # rr.Image.compress()'s own default (95) wasn't aggressive enough in
        # practice to keep a full --episodes load under the memory limit;
        # --jpeg-quality (default 20) lets it go lower still.
        logged_calls = {}
        dataset_root = _make_dataset(tmp_path, total_episodes=5)
        _FakeStream.instances = []

        monkeypatch.setattr(
            "lerobot.datasets.lerobot_dataset.LeRobotDataset", _FakeDataset
        )
        monkeypatch.setattr("rerun.RecordingStream", _FakeStream)
        monkeypatch.setattr("rerun.spawn", lambda **kwargs: None)
        monkeypatch.setattr(
            "mcap_converter.cli.dataset_viz._log_episode_to_stream",
            lambda dataset, episode_index, stream, **kw: logged_calls.setdefault(
                "jpeg_quality_by_episode", []
            ).append(kw.get("jpeg_quality")),
        )

        rc = main([str(dataset_root), "--episodes", "0"])
        assert rc == 0
        assert logged_calls["jpeg_quality_by_episode"] == [20]

        logged_calls.clear()
        _FakeStream.instances = []
        rc = main([str(dataset_root), "--episodes", "0", "--jpeg-quality", "20"])
        assert rc == 0
        assert logged_calls["jpeg_quality_by_episode"] == [20]
