"""Tests for ConfigLoader and DataConfig public API.

Tests the user-facing config loading contract, including:
- DataConfig defaults
- ConfigLoader.from_dict()
- ConfigLoader.from_yaml()
- Legacy field migration (DeprecationWarning behavior)
- Exception hierarchy
"""

import warnings

import pytest
import yaml

from mcap_converter import (
    ConfigLoader,
    DataConfig,
    ConfigurationError,
    DataExtractionError,
    McapConverterError,
    McapReadError,
    TimeAlignmentError,
    DatasetWriteError,
)


class TestDataConfigDefaults:
    def test_default_robot_state_topic(self):
        assert DataConfig().robot_state_topic == "/joint_states"

    def test_default_image_resolution(self):
        assert DataConfig().image_resolution == [640, 480]

    def test_default_joint_name_pattern_has_leader_follower(self):
        pattern = DataConfig().joint_name_pattern
        assert "leader" in pattern.source
        assert "follower" in pattern.source


class TestConfigLoaderFromDict:
    def test_empty_dict_returns_dataconfig(self):
        result = ConfigLoader.from_dict({})
        assert isinstance(result, DataConfig)

    def test_robot_state_topic_set(self):
        result = ConfigLoader.from_dict({"robot_state_topic": "/my/topic"})
        assert result.robot_state_topic == "/my/topic"

    def test_camera_topics_set(self):
        result = ConfigLoader.from_dict({"camera_topics": ["/cam/a"]})
        assert result.camera_topics == ["/cam/a"]

    def test_image_resolution_set(self):
        result = ConfigLoader.from_dict({"image_resolution": [320, 240]})
        assert result.image_resolution == [320, 240]

    def test_unrecognised_keys_ignored(self):
        # Extra keys in dict should not raise
        result = ConfigLoader.from_dict({"unknown_key_xyz": "value"})
        assert isinstance(result, DataConfig)


class TestConfigLoaderLegacyMigration:
    def test_robot_state_topics_plural_uses_first(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = ConfigLoader.from_dict({"robot_state_topics": ["/a", "/b"]})
        assert result.robot_state_topic == "/a"

    def test_robot_state_topics_plural_warns(self):
        with pytest.warns(DeprecationWarning):
            ConfigLoader.from_dict({"robot_state_topics": ["/a", "/b"]})

    def test_motor_feature_mapping_warns(self):
        with pytest.warns(DeprecationWarning):
            ConfigLoader.from_dict({"motor_feature_mapping": {"state": "position"}})

    def test_robot_state_topic_singular_takes_precedence(self):
        # If both old and new keys present, singular wins (no migration triggered)
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            result = ConfigLoader.from_dict({
                "robot_state_topic": "/new",
                "robot_state_topics": ["/old"],
            })
        assert result.robot_state_topic == "/new"


class TestConfigLoaderFromYaml:
    def test_load_valid_yaml(self, tmp_path):
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text("robot_state_topic: /test\n")
        result = ConfigLoader.from_yaml(str(yaml_file))
        assert isinstance(result, DataConfig)

    def test_yaml_values_preserved(self, tmp_path):
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text("robot_state_topic: /my_robot/joints\n")
        result = ConfigLoader.from_yaml(str(yaml_file))
        assert result.robot_state_topic == "/my_robot/joints"

    def test_yaml_camera_topics_list(self, tmp_path):
        content = yaml.dump({"camera_topics": ["/cam/left", "/cam/right"]})
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(content)
        result = ConfigLoader.from_yaml(str(yaml_file))
        assert result.camera_topics == ["/cam/left", "/cam/right"]

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            ConfigLoader.from_yaml("/nonexistent/config.yaml")


class TestExceptionHierarchy:
    def test_all_subclasses_catchable_as_base(self):
        subclasses = [
            ConfigurationError,
            McapReadError,
            DataExtractionError,
            TimeAlignmentError,
            DatasetWriteError,
        ]
        for exc_class in subclasses:
            exc = exc_class("test message")
            assert isinstance(exc, McapConverterError), (
                f"{exc_class.__name__} must inherit from McapConverterError"
            )

    def test_base_exception_is_exception(self):
        assert issubclass(McapConverterError, Exception)
