"""Tests for the dataset-viz CLI's pure helper functions (Task 1: dataset_check + nginx_config;
Task 2: config)."""

import json
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
