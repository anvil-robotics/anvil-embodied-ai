"""Portable adapter artifact loading and integrity checks."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from anvil_shared.embodiment import EmbodimentAdapterSpec, EmbodimentError
from safetensors.torch import load_file, save_file

from .bridge import KinematicEmbodimentBridge
from .residual import ResidualChunkAdapter

MANIFEST_NAME = "adapter_manifest.json"
WEIGHTS_NAME = "adapter.safetensors"
TRAINING_NAME = "training_provenance.json"
REPORT_NAME = "eval_report.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_base_policy(spec: EmbodimentAdapterSpec, base_policy_dir: Path) -> None:
    missing = []
    mismatched = []
    for name, expected in spec.base_policy_processor_sha256.items():
        path = base_policy_dir / name
        if not path.is_file():
            missing.append(name)
        elif sha256_file(path) != expected:
            mismatched.append(name)
    if missing or mismatched:
        raise EmbodimentError(
            f"base policy processor integrity failure; missing={missing}, mismatched={mismatched}"
        )
    config_path = base_policy_dir / "config.json"
    if not config_path.is_file():
        raise EmbodimentError(f"missing base policy config: {config_path}")
    config = json.loads(config_path.read_text())
    if config.get("type") != "pi05":
        raise EmbodimentError(f"adapter requires pi05, got {config.get('type')!r}")
    action_shape = config.get("output_features", {}).get("action", {}).get("shape")
    if action_shape != [16]:
        raise EmbodimentError(f"adapter requires a 16-D action, got {action_shape}")
    if int(config.get("chunk_size", 0)) != spec.residual.chunk_size:
        raise EmbodimentError(
            f"base chunk_size={config.get('chunk_size')} does not match adapter "
            f"chunk_size={spec.residual.chunk_size}"
        )


@dataclass
class AdapterArtifact:
    path: Path
    spec: EmbodimentAdapterSpec
    bridge: KinematicEmbodimentBridge
    residual: ResidualChunkAdapter

    def save(
        self,
        output_dir: Path,
        *,
        training_provenance: dict[str, Any] | None = None,
        eval_report: dict[str, Any] | None = None,
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        destination_manifest = output_dir / MANIFEST_NAME
        if self.spec.path.resolve() != destination_manifest.resolve():
            shutil.copyfile(self.spec.path, destination_manifest)
        state = {
            name: value.detach().cpu().contiguous()
            for name, value in self.residual.state_dict().items()
        }
        save_file(state, str(output_dir / WEIGHTS_NAME))
        if training_provenance is not None:
            (output_dir / TRAINING_NAME).write_text(
                json.dumps(training_provenance, indent=2, sort_keys=True)
            )
        if eval_report is not None:
            (output_dir / REPORT_NAME).write_text(json.dumps(eval_report, indent=2, sort_keys=True))


def load_adapter_artifact(
    path: str | Path,
    *,
    base_policy_dir: str | Path | None = None,
    device: str | torch.device = "cpu",
    live: bool = False,
    require_weights: bool = True,
) -> AdapterArtifact:
    artifact_dir = Path(path)
    manifest_path = artifact_dir / MANIFEST_NAME if artifact_dir.is_dir() else artifact_dir
    spec = EmbodimentAdapterSpec.load(manifest_path)
    if live:
        spec.require_live_approved()
    if base_policy_dir is not None:
        verify_base_policy(spec, Path(base_policy_dir))
    bridge = KinematicEmbodimentBridge(spec)
    residual = ResidualChunkAdapter(
        spec.residual,
        torch.as_tensor(bridge.residual_bounds, dtype=torch.float32),
        torch.as_tensor(bridge.target_joint_ranges, dtype=torch.float32),
    ).to(device)
    weights_path = manifest_path.parent / WEIGHTS_NAME
    if weights_path.is_file():
        residual.load_state_dict(load_file(str(weights_path), device=str(device)))
    elif require_weights:
        raise EmbodimentError(f"missing adapter weights: {weights_path}")
    residual.eval()
    return AdapterArtifact(
        path=manifest_path.parent,
        spec=spec,
        bridge=bridge,
        residual=residual,
    )
