import json
from pathlib import Path

import numpy as np
import pytest
import torch
from anvil_embodiment.artifact import sha256_file, verify_base_policy
from anvil_embodiment.bridge import TARGET_TO_REFERENCE_TCP_ROTATION, BridgeError
from anvil_embodiment.kinematics import (
    MujocoArmKinematics,
    get_model_spec,
    model_spec_hash,
    torch_forward_kinematics,
)
from anvil_embodiment.residual import (
    AdapterLossWeights,
    compute_adapter_loss,
)
from anvil_shared.embodiment import EmbodimentAdapterSpec, EmbodimentError

REPO_ROOT = Path(__file__).resolve().parents[3]
MANIFEST = REPO_ROOT / "configs" / "embodiment_adapters" / "hf_folding_to_anvil_openarm2.json"


def test_manifest_pins_units_processors_and_models() -> None:
    spec = EmbodimentAdapterSpec.load(MANIFEST)

    assert spec.reference_vector.joint_unit == "degree"
    assert spec.target_vector.joint_unit == "radian"
    assert len(spec.base_policy_processor_sha256) == 4
    assert spec.deployment_status == "offline_only"
    assert model_spec_hash(get_model_spec(spec.reference_model.model_id)) == (
        spec.reference_model.sha256
    )
    assert model_spec_hash(get_model_spec(spec.target_model.model_id)) == (spec.target_model.sha256)


@pytest.mark.parametrize("model_id", ["hf_openarm_v1_extended", "anvil_openarm_v2"])
@pytest.mark.parametrize("side", ["right", "left"])
def test_torch_fk_matches_mujoco(model_id: str, side: str) -> None:
    spec = get_model_spec(model_id)
    kinematics = MujocoArmKinematics(spec, side)
    limits = kinematics.limits
    joints = limits[:, 0] * 0.35 + limits[:, 1] * 0.65

    expected_position, expected_rotation = kinematics.pose(joints)
    actual_position, actual_rotation = torch_forward_kinematics(
        spec, side, torch.tensor(joints, dtype=torch.float64)
    )

    np.testing.assert_allclose(actual_position.numpy(), expected_position, atol=1e-9)
    np.testing.assert_allclose(actual_rotation.numpy(), expected_rotation, atol=1e-9)


def test_tcp_alignment_is_a_proper_rotation() -> None:
    np.testing.assert_allclose(
        TARGET_TO_REFERENCE_TCP_ROTATION.T @ TARGET_TO_REFERENCE_TCP_ROTATION,
        np.eye(3),
        atol=1e-12,
    )
    assert np.linalg.det(TARGET_TO_REFERENCE_TCP_ROTATION) == pytest.approx(1.0)


def test_bridge_converts_degrees_and_round_trips_current_pose() -> None:
    from anvil_embodiment.artifact import load_adapter_artifact

    artifact = load_adapter_artifact(MANIFEST, require_weights=False)
    state = np.asarray(
        [0.0, 0.2, 0.01, 1.55, 0.0, -0.02, 0.01, 0.026]
        + [0.0, -0.2, 0.01, 1.55, 0.0, -0.02, 0.01, 0.026],
        dtype=np.float64,
    )
    reference = artifact.bridge.target_state_to_policy(state)

    assert np.max(np.abs(reference.values[[0, 1, 2, 3, 4, 5, 6]])) > 30.0
    round_trip = artifact.bridge.policy_chunk_to_target(reference.values[None], state)
    np.testing.assert_allclose(round_trip.values[0, :7], state[:7], atol=1e-12)
    np.testing.assert_allclose(round_trip.values[0, 8:15], state[8:15], atol=1e-12)


def test_bridge_rejects_nonfinite_input() -> None:
    from anvil_embodiment.artifact import load_adapter_artifact

    artifact = load_adapter_artifact(MANIFEST, require_weights=False)
    state = np.zeros(16)
    state[3] = np.nan
    with pytest.raises(BridgeError, match="finite"):
        artifact.bridge.target_state_to_policy(state)


def test_zero_initialized_residual_is_identity_and_bounded() -> None:
    from anvil_embodiment.artifact import load_adapter_artifact

    artifact = load_adapter_artifact(MANIFEST, require_weights=False)
    model = artifact.residual
    current = torch.zeros(2, 16)
    bridge = model.target_lower + (model.target_upper - model.target_lower) * torch.rand(
        2, artifact.spec.residual.chunk_size, 16
    )
    corrected, residual = model(current, bridge)

    torch.testing.assert_close(corrected, bridge)
    torch.testing.assert_close(residual, torch.zeros_like(residual))
    assert torch.all(model.correction_bounds[[7, 15]] == 0)

    with torch.no_grad():
        model.output.bias.fill_(100.0)
    saturated_command, saturated = model(current, bridge)
    assert torch.all(torch.abs(saturated) <= model.correction_bounds[None, None, :] + 1e-6)
    assert torch.all(saturated_command >= model.target_lower)
    assert torch.all(saturated_command <= model.target_upper)


def test_adapter_loss_is_zero_for_exact_target() -> None:
    from anvil_embodiment.artifact import load_adapter_artifact

    artifact = load_adapter_artifact(MANIFEST, require_weights=False)
    target = torch.zeros(2, 4, 16, dtype=torch.float64)
    target[..., 3] = 1.2
    target[..., 11] = 1.2
    residual = torch.zeros_like(target)
    loss, terms = compute_adapter_loss(
        corrected=target,
        residual=residual,
        target=target,
        target_ranges=torch.tensor(artifact.bridge.target_joint_ranges, dtype=torch.float64),
        target_model=artifact.bridge.target_spec,
        correction_bounds=torch.tensor(artifact.bridge.residual_bounds, dtype=torch.float64),
        weights=AdapterLossWeights(),
    )

    assert float(loss) == pytest.approx(0.0, abs=1e-12)
    assert all(float(value) == pytest.approx(0.0, abs=1e-12) for value in terms.values())


def test_base_policy_verification_pins_processor_state(tmp_path: Path) -> None:
    raw = json.loads(MANIFEST.read_text())
    base = tmp_path / "base"
    base.mkdir()
    processor_names = tuple(raw["base_policy"]["processor_sha256"])
    for index, name in enumerate(processor_names):
        (base / name).write_bytes(f"processor-{index}".encode())
        raw["base_policy"]["processor_sha256"][name] = sha256_file(base / name)
    (base / "config.json").write_text(
        json.dumps(
            {
                "type": "pi05",
                "chunk_size": 30,
                "output_features": {"action": {"shape": [16]}},
            }
        )
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(raw))
    spec = EmbodimentAdapterSpec.load(manifest)

    verify_base_policy(spec, base)
    (base / processor_names[-1]).write_bytes(b"tampered")
    with pytest.raises(EmbodimentError, match="integrity"):
        verify_base_policy(spec, base)
