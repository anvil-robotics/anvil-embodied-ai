"""Tests for the dataset-viz CLI's pure helper functions (Task 1: dataset_check + nginx_config;
Task 2: config; Task 3: orchestrator)."""

import json
import socket
import subprocess
import urllib.error
from pathlib import Path

from mcap_converter.viz.config import (
    CACHE_DIR_ENV_VAR,
    DEFAULT_FRONTEND_PORT,
    DEFAULT_STATIC_PORT,
    VISUALIZER_PINNED_SHA,
    VISUALIZER_REPO_URL,
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


def _make_dataset(
    tmp_path: Path,
    *,
    codebase_version: str = "v3.0",
    include_data_dir: bool = True,
    include_videos: bool = True,
) -> Path:
    """Build a minimal synthetic dataset directory under tmp_path for testing."""
    root = tmp_path / "my-dataset"
    meta_dir = root / "meta"
    meta_dir.mkdir(parents=True)
    (meta_dir / "info.json").write_text(json.dumps({"codebase_version": codebase_version}))
    if include_data_dir:
        (root / "data").mkdir()
    if include_videos:
        (root / "videos" / "observation.images.waist").mkdir(parents=True)
    return root


class TestValidateDatasetRoot:
    def test_valid_v3_dataset_passes(self, tmp_path):
        root = _make_dataset(tmp_path)
        result = validate_dataset_root(root)
        assert result.ok is True
        assert result.codebase_version == "v3.0"
        assert result.errors == []
        assert result.warnings == []

    def test_valid_v2_and_v21_datasets_pass(self, tmp_path):
        for version in ("v2.0", "v2.1"):
            root = _make_dataset(tmp_path / version, codebase_version=version)
            result = validate_dataset_root(root)
            assert result.ok is True, f"version {version} should pass"
            assert result.codebase_version == version

    def test_nonexistent_root_fails(self, tmp_path):
        result = validate_dataset_root(tmp_path / "does-not-exist")
        assert result.ok is False
        assert any("does not exist" in e or "not a directory" in e for e in result.errors)

    def test_file_instead_of_directory_fails(self, tmp_path):
        f = tmp_path / "not-a-dir"
        f.write_text("x")
        result = validate_dataset_root(f)
        assert result.ok is False

    def test_missing_info_json_fails(self, tmp_path):
        root = tmp_path / "no-meta"
        root.mkdir()
        result = validate_dataset_root(root)
        assert result.ok is False
        assert any("info.json" in e for e in result.errors)

    def test_malformed_info_json_fails(self, tmp_path):
        root = tmp_path / "bad-json"
        (root / "meta").mkdir(parents=True)
        (root / "meta" / "info.json").write_text("{not valid json")
        result = validate_dataset_root(root)
        assert result.ok is False

    def test_info_json_not_a_dict_fails_gracefully(self, tmp_path):
        root = tmp_path / "weird-info"
        (root / "meta").mkdir(parents=True)
        (root / "meta" / "info.json").write_text(json.dumps(["not", "a", "dict"]))
        result = validate_dataset_root(root)  # must not raise
        assert result.ok is False
        assert any("info.json" in e or "JSON object" in e for e in result.errors)

    def test_unsupported_codebase_version_fails(self, tmp_path):
        root = _make_dataset(tmp_path, codebase_version="v1.0")
        result = validate_dataset_root(root)
        assert result.ok is False
        assert result.codebase_version == "v1.0"
        assert any("v1.0" in e for e in result.errors)

    def test_missing_data_dir_fails(self, tmp_path):
        root = _make_dataset(tmp_path, include_data_dir=False)
        result = validate_dataset_root(root)
        assert result.ok is False
        assert any("data" in e for e in result.errors)

    def test_missing_videos_dir_warns_but_still_ok(self, tmp_path):
        root = _make_dataset(tmp_path, include_videos=False)
        result = validate_dataset_root(root)
        assert result.ok is True
        assert any("video" in w.lower() for w in result.warnings)

    def test_does_not_import_lerobot(self):
        # Regression guard: this module must stay a cheap filesystem check.
        import mcap_converter.viz.dataset_check as mod

        source = Path(mod.__file__).read_text()
        assert "import lerobot" not in source
        assert "from lerobot" not in source


class TestRenderNginxConf:
    def test_contains_listen_directive_with_correct_port(self):
        conf = render_nginx_conf(8080)
        assert "listen 8080;" in conf

    def test_contains_resolve_main_location_regex(self):
        conf = render_nginx_conf(8080)
        assert "resolve/main" in conf
        assert "/srv/dataset" in conf

    def test_contains_range_and_cors_headers(self):
        conf = render_nginx_conf(8080)
        assert "Accept-Ranges bytes" in conf
        assert "Access-Control-Allow-Origin" in conf
        assert '"*"' in conf or "'*'" in conf

    def test_different_ports_produce_different_output(self):
        conf_a = render_nginx_conf(8080)
        conf_b = render_nginx_conf(9090)
        assert "listen 8080;" in conf_a
        assert "listen 9090;" in conf_b
        assert conf_a != conf_b

    def test_output_is_syntactically_plausible_nginx(self):
        # Not a full nginx parse, but a basic brace-balance sanity check —
        # catches f-string escaping mistakes (unbalanced { } from bad
        # double-brace escaping).
        conf = render_nginx_conf(8080)
        assert conf.count("{") == conf.count("}")


class TestConstants:
    def test_pinned_sha_is_a_full_40_char_commit_sha(self):
        assert len(VISUALIZER_PINNED_SHA) == 40
        assert all(c in "0123456789abcdef" for c in VISUALIZER_PINNED_SHA)

    def test_repo_url_points_at_the_official_upstream(self):
        assert (
            VISUALIZER_REPO_URL == "https://github.com/huggingface/lerobot-dataset-visualizer.git"
        )

    def test_default_ports_are_distinct(self):
        assert DEFAULT_FRONTEND_PORT != DEFAULT_STATIC_PORT

    def test_cache_dir_env_var_name(self):
        assert CACHE_DIR_ENV_VAR == "MCAP_VIZ_CACHE_DIR"


class TestResolveCacheDir:
    def test_explicit_override_wins(self, tmp_path):
        override = tmp_path / "custom-cache"
        assert resolve_cache_dir(override) == override

    def test_uses_xdg_cache_home_when_set(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
        result = resolve_cache_dir()
        assert result == tmp_path / "anvil-mcap-viz"

    def test_falls_back_to_home_cache_when_xdg_unset(self, monkeypatch):
        monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
        result = resolve_cache_dir()
        assert result == Path.home() / ".cache" / "anvil-mcap-viz"

    def test_does_not_create_the_directory(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
        result = resolve_cache_dir()
        assert not result.exists()

    def test_empty_xdg_cache_home_falls_back_to_home_cache(self, monkeypatch):
        # Per the XDG Base Directory spec, an empty value is equivalent to
        # unset — must not be treated as a valid (empty-string-rooted) path.
        monkeypatch.setenv("XDG_CACHE_HOME", "")
        result = resolve_cache_dir()
        assert result == Path.home() / ".cache" / "anvil-mcap-viz"


class TestDefaultRepoId:
    def test_uses_local_prefix_and_directory_basename(self, tmp_path):
        dataset_dir = tmp_path / "my-session"
        dataset_dir.mkdir()
        assert default_repo_id(dataset_dir) == "local/my-session"

    def test_resolves_relative_paths(self, tmp_path, monkeypatch):
        dataset_dir = tmp_path / "another-session"
        dataset_dir.mkdir()
        monkeypatch.chdir(tmp_path)
        assert default_repo_id(Path("another-session")) == "local/another-session"

    def test_strips_trailing_slash(self, tmp_path):
        dataset_dir = tmp_path / "trailing-slash-session"
        dataset_dir.mkdir()
        assert default_repo_id(Path(str(dataset_dir) + "/")) == "local/trailing-slash-session"


class TestUrlBuilders:
    def test_browse_url(self):
        assert (
            browse_url(frontend_port=7860, repo_id="local/my-session", episode=0)
            == "http://localhost:7860/local/my-session/0"
        )

    def test_browse_url_with_nonzero_episode(self):
        assert (
            browse_url(frontend_port=3000, repo_id="anvil/foo", episode=42)
            == "http://localhost:3000/anvil/foo/42"
        )

    def test_static_probe_url(self):
        assert (
            static_probe_url(static_port=8080, repo_id="local/my-session")
            == "http://localhost:8080/local/my-session/resolve/main/meta/info.json"
        )

    def test_frontend_probe_url(self):
        assert frontend_probe_url(frontend_port=7860) == "http://localhost:7860/"


class TestRenderRunEnv:
    def test_contains_all_required_keys(self, tmp_path):
        content = render_run_env(
            dataset_root=tmp_path / "ds",
            nginx_conf_path=tmp_path / "nginx.conf",
            visualizer_dir=tmp_path / "viz-src",
            static_port=8080,
            frontend_port=7860,
        )
        for key in (
            "MCAP_VIZ_DATASET_ROOT",
            "MCAP_VIZ_NGINX_CONF",
            "MCAP_VIZ_VISUALIZER_DIR",
            "STATIC_PORT=8080",
            "FRONTEND_PORT=7860",
        ):
            assert key in content

    def test_paths_are_rendered_as_absolute(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        content = render_run_env(
            dataset_root=Path("relative-ds"),
            nginx_conf_path=Path("relative-nginx.conf"),
            visualizer_dir=Path("relative-viz-src"),
            static_port=8080,
            frontend_port=7860,
        )
        first_line = content.split("\n")[0]
        # Should be absolute, not literally the bare relative string alone.
        assert not first_line.endswith("=relative-ds")
        assert str(tmp_path) in content

    def test_is_valid_dotenv_shaped_lines(self, tmp_path):
        content = render_run_env(
            dataset_root=tmp_path / "ds",
            nginx_conf_path=tmp_path / "nginx.conf",
            visualizer_dir=tmp_path / "viz-src",
            static_port=8080,
            frontend_port=7860,
        )
        lines = [line for line in content.strip().split("\n") if line]
        for line in lines:
            assert "=" in line, f"line {line!r} doesn't look like KEY=VALUE"


class TestComposeFilePath:
    def test_points_at_checked_in_compose_file(self):
        path = compose_file_path()
        assert path.name == "docker-compose.yml"
        assert path.is_file()  # this task creates the real file, so it must exist


class TestComposeArgvBuilders:
    def test_up_argv_without_rebuild(self):
        argv = build_compose_up_argv(Path("/x/docker-compose.yml"), Path("/x/run.env"))
        assert argv == [
            "docker",
            "compose",
            "-p",
            "mcap-viz",
            "-f",
            "/x/docker-compose.yml",
            "--env-file",
            "/x/run.env",
            "up",
            "-d",
        ]

    def test_up_argv_with_rebuild(self):
        argv = build_compose_up_argv(
            Path("/x/docker-compose.yml"), Path("/x/run.env"), rebuild=True
        )
        assert argv[-1] == "--build"
        assert (
            argv[-2:] == ["-d", "--build"] or "--build" in argv
        )  # exact position isn't load-bearing, presence is

    def test_down_argv(self):
        argv = build_compose_down_argv(Path("/x/docker-compose.yml"), Path("/x/run.env"))
        assert argv == [
            "docker",
            "compose",
            "-p",
            "mcap-viz",
            "-f",
            "/x/docker-compose.yml",
            "--env-file",
            "/x/run.env",
            "down",
        ]

    def test_logs_argv(self):
        argv = build_compose_logs_argv(Path("/x/docker-compose.yml"), Path("/x/run.env"))
        assert argv == [
            "docker",
            "compose",
            "-p",
            "mcap-viz",
            "-f",
            "/x/docker-compose.yml",
            "--env-file",
            "/x/run.env",
            "logs",
            "--no-color",
        ]


class TestIsPortAvailable:
    def test_free_port_reports_available(self):
        # Bind to port 0 to let the OS pick a genuinely free ephemeral port,
        # close it, then immediately check availability (small TOCTOU risk
        # is acceptable in a test).
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            free_port = s.getsockname()[1]
        assert is_port_available(free_port) is True

    def test_bound_port_reports_unavailable(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            s.listen(1)
            bound_port = s.getsockname()[1]
            assert is_port_available(bound_port) is False


class TestCheckDockerAvailable:
    def test_docker_missing_from_path(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda _name: None)
        ok, message = check_docker_available()
        assert ok is False
        assert "not installed" in message.lower() or "not on path" in message.lower()

    def test_daemon_not_running(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/docker")

        def fake_runner(argv, **kwargs):
            return subprocess.CompletedProcess(argv, returncode=1, stdout="", stderr="daemon down")

        ok, message = check_docker_available(runner=fake_runner)
        assert ok is False
        assert "daemon" in message.lower()

    def test_compose_plugin_missing(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/docker")
        call_count = {"n": 0}

        def fake_runner(argv, **kwargs):
            call_count["n"] += 1
            if "version" in argv and "compose" not in argv:
                return subprocess.CompletedProcess(argv, returncode=0, stdout="ok", stderr="")
            return subprocess.CompletedProcess(argv, returncode=1, stdout="", stderr="no compose")

        ok, message = check_docker_available(runner=fake_runner)
        assert ok is False
        assert "compose" in message.lower()

    def test_all_available(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/docker")

        def fake_runner(argv, **kwargs):
            return subprocess.CompletedProcess(argv, returncode=0, stdout="ok", stderr="")

        ok, message = check_docker_available(runner=fake_runner)
        assert ok is True
        assert message == ""


class TestEnsureVisualizerSource:
    def test_clones_and_checks_out_when_absent(self, tmp_path):
        calls = []

        def fake_runner(argv, **kwargs):
            calls.append(argv)
            if argv[:2] == ["git", "clone"]:
                # simulate git actually creating the target directory
                Path(argv[-1]).mkdir(parents=True, exist_ok=True)
            return subprocess.CompletedProcess(argv, returncode=0, stdout="", stderr="")

        ok, source_dir, err = ensure_visualizer_source(tmp_path, runner=fake_runner)
        assert ok is True
        assert err == ""
        assert source_dir == tmp_path / "lerobot-dataset-visualizer"
        assert any(c[:2] == ["git", "clone"] for c in calls)
        assert any(c[:3] == ["git", "-C", str(source_dir)] and "checkout" in c for c in calls)

    def test_reuses_existing_checkout_at_correct_sha_without_fetching(self, tmp_path):
        source_dir = tmp_path / "lerobot-dataset-visualizer"
        source_dir.mkdir()
        calls = []

        def fake_runner(argv, **kwargs):
            calls.append(argv)
            if "rev-parse" in argv:
                return subprocess.CompletedProcess(
                    argv, returncode=0, stdout=VISUALIZER_PINNED_SHA + "\n", stderr=""
                )
            return subprocess.CompletedProcess(argv, returncode=0, stdout="", stderr="")

        ok, result_dir, err = ensure_visualizer_source(tmp_path, runner=fake_runner)
        assert ok is True
        assert result_dir == source_dir
        assert not any(c[:2] == ["git", "clone"] for c in calls)
        assert not any("fetch" in c for c in calls)  # correct SHA already -> no network needed

    def test_refetches_when_sha_mismatch(self, tmp_path):
        source_dir = tmp_path / "lerobot-dataset-visualizer"
        source_dir.mkdir()
        calls = []

        def fake_runner(argv, **kwargs):
            calls.append(argv)
            if "rev-parse" in argv:
                return subprocess.CompletedProcess(
                    argv, returncode=0, stdout="deadbeef" * 5, stderr=""
                )
            return subprocess.CompletedProcess(argv, returncode=0, stdout="", stderr="")

        ok, result_dir, err = ensure_visualizer_source(tmp_path, runner=fake_runner)
        assert ok is True
        assert any("fetch" in c for c in calls)
        assert any("checkout" in c for c in calls)

    def test_refresh_deletes_existing_before_cloning(self, tmp_path):
        source_dir = tmp_path / "lerobot-dataset-visualizer"
        source_dir.mkdir()
        (source_dir / "marker.txt").write_text("stale")
        calls = []

        def fake_runner(argv, **kwargs):
            calls.append(argv)
            if argv[:2] == ["git", "clone"]:
                Path(argv[-1]).mkdir(parents=True, exist_ok=True)
            return subprocess.CompletedProcess(argv, returncode=0, stdout="", stderr="")

        ok, result_dir, err = ensure_visualizer_source(tmp_path, refresh=True, runner=fake_runner)
        assert ok is True
        assert not (source_dir / "marker.txt").exists()  # old checkout was wiped
        assert any(c[:2] == ["git", "clone"] for c in calls)

    def test_clone_failure_reports_error(self, tmp_path):
        def fake_runner(argv, **kwargs):
            return subprocess.CompletedProcess(
                argv, returncode=1, stdout="", stderr="network unreachable"
            )

        ok, source_dir, err = ensure_visualizer_source(tmp_path, runner=fake_runner)
        assert ok is False
        assert "clone" in err.lower()
        assert "network unreachable" in err


class TestWaitForReady:
    def test_returns_true_when_all_urls_ready_immediately(self):
        class FakeResponse:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def getcode(self):
                return 200

        result = wait_for_ready(
            ["http://a", "http://b"],
            opener=lambda _url: FakeResponse(),
            sleep=lambda _s: None,
        )
        assert result is True

    def test_returns_false_on_persistent_failure_within_timeout(self):
        def always_fails(url):
            raise urllib.error.URLError("connection refused")

        calls = {"clock": 0}

        def fake_clock():
            calls["clock"] += 1
            return calls["clock"] * 10  # advances fast past any timeout

        result = wait_for_ready(
            ["http://a"],
            timeout_s=5.0,
            opener=always_fails,
            sleep=lambda _s: None,
            clock=fake_clock,
        )
        assert result is False

    def test_succeeds_after_a_few_failed_attempts(self):
        attempt = {"n": 0}

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def getcode(self):
                return 200

        def flaky_opener(url):
            attempt["n"] += 1
            if attempt["n"] < 3:
                raise urllib.error.URLError("not ready yet")
            return FakeResponse()

        result = wait_for_ready(
            ["http://a"],
            timeout_s=100.0,
            opener=flaky_opener,
            sleep=lambda _s: None,
        )
        assert result is True
        assert attempt["n"] >= 3
