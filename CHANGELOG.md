# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial repository structure
- `mcap_converter` package for converting MCAP recordings to LeRobot format
- `lerobot_training` package for training imitation learning models
- `lerobot_control` ROS2 package for real-time inference
- Docker infrastructure for deployment
- CI/CD workflows for testing and releases
- Comprehensive documentation

### Changed
- Migrated from internal e2e_ml_pipeline repository
- Restructured for public release

### Removed
- Deprecated yam_teleop package
- Internal dataset files (users download from HuggingFace)
