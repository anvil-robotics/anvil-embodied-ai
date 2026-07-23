import json
from pathlib import Path

import yaml

from anvil_eval.contract import audit_policy_contract


def _write_contract_fixture(root: Path, arm_mapping: dict[str, str]) -> tuple[Path, Path, Path]:
    checkpoint = root / "checkpoint" / "pretrained_model"
    dataset = root / "dataset"
    checkpoint.mkdir(parents=True)
    (dataset / "meta").mkdir(parents=True)
    names = ["right_joint_1.pos", "left_joint_1.pos"]
    (checkpoint / "config.json").write_text(
        json.dumps(
            {
                "type": "pi05",
                "input_features": {"observation.state": {"shape": [2]}},
                "output_features": {"action": {"shape": [2]}},
                "action_feature_names": names,
                "use_relative_actions": True,
                "chunk_size": 30,
                "n_action_steps": 30,
            }
        )
    )
    (checkpoint / "policy_preprocessor.json").write_text(
        json.dumps(
            {
                "steps": [
                    {
                        "registry_name": "relative_actions_processor",
                        "config": {"enabled": True, "action_names": names, "exclude_joints": []},
                    }
                ]
            }
        )
    )
    (checkpoint / "policy_postprocessor.json").write_text(
        json.dumps(
            {
                "steps": [
                    {
                        "registry_name": "absolute_actions_processor",
                        "config": {"enabled": True},
                    }
                ]
            }
        )
    )
    (dataset / "meta" / "info.json").write_text(
        json.dumps(
            {
                "features": {
                    "observation.state": {"shape": [2], "names": names},
                    "action": {"shape": [2], "names": names},
                }
            }
        )
    )
    config = root / "inference.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "joint_names": {
                    "arm_mapping": arm_mapping,
                    "model_joint_order": ["joint1"],
                },
                "arms": {
                    "right": {"action_start": 0, "action_end": 1},
                    "left": {"action_start": 1, "action_end": 2},
                },
                "cameras": {"mapping": {}},
                "inference_tuning": {
                    "sync": {
                        "async_prefetch": True,
                        "prefetch_threshold": 20,
                        "replace_pending_actions": True,
                    },
                    "rtc": {"enabled": False},
                },
                "safety": {"max_position_delta": 0.05},
            },
            sort_keys=False,
        )
    )
    return checkpoint.parent, dataset, config


def test_contract_audit_accepts_matching_right_first_vector(tmp_path: Path) -> None:
    checkpoint, dataset, config = _write_contract_fixture(tmp_path, {"r": "right", "l": "left"})

    report = audit_policy_contract(checkpoint, dataset, config)

    assert report["errors"] == []
    assert report["checks"]["runtime_arm_order"] == ["right", "left"]


def test_contract_audit_rejects_left_first_runtime_vector(tmp_path: Path) -> None:
    checkpoint, dataset, config = _write_contract_fixture(tmp_path, {"l": "left", "r": "right"})

    report = audit_policy_contract(checkpoint, dataset, config)

    assert any("runtime/checkpoint arm order" in error for error in report["errors"])
