"""Tests for the dataset-viz CLI's pure helper functions: dataset_check +
config (default_repo_id). CLI-level tests for the thin `visualize_dataset`
wrapper live in test_dataset_viz_cli.py.
"""

import json
from pathlib import Path

from mcap_converter.viz.config import default_repo_id
from mcap_converter.viz.dataset_check import validate_dataset_root


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
