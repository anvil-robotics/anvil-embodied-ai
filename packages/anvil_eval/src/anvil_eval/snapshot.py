"""Saved-camera snapshot inference for the offline sanity workflow."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image

from .sanity import evaluate_native_relative_episode


class _SnapshotDataset:
    def __init__(self, item: dict[str, Any]) -> None:
        self.item = item
        self.hf_dataset = self

    def __getitem__(self, _index: int) -> dict[str, Any]:
        return self.item


def evaluate_saved_snapshot(
    *,
    model: Any,
    preprocessor: Any,
    postprocessor: Any,
    camera_report: dict[str, Any],
    state: torch.Tensor,
    state_source: str,
    device: str,
    task_description: str | None,
    joint_names: list[str],
    max_position_delta: float,
) -> dict[str, Any]:
    """Run one chunk using saved live images and an explicitly identified recorded state.

    Image and state timestamps are unrelated, so this validates preprocessing, output
    shape, relative restoration, and timing only. It is not a behavioral score.
    """
    item: dict[str, Any] = {
        "observation.state": state.clone(),
        "action": state.clone(),
    }
    image_sources: dict[str, str] = {}
    image_shapes: dict[str, list[int]] = {}
    for camera, details in camera_report.items():
        if details.get("status") != "found":
            raise FileNotFoundError(f"No saved image for required camera {camera}")
        path = details["path"]
        with Image.open(path) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        tensor = torch.from_numpy(array).float().div_(255.0).permute(2, 0, 1)
        item[f"observation.images.{camera}"] = tensor
        image_sources[camera] = path
        image_shapes[camera] = list(tensor.shape)

    result = evaluate_native_relative_episode(
        model=model,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        dataset=_SnapshotDataset(item),
        frame_indices=[0],
        episode_idx=-1,
        split_label="saved_snapshot",
        device=device,
        task_description=task_description,
        joint_names=joint_names,
    )
    predicted = result.predicted[0]
    state_array = state.detach().cpu().numpy().reshape(-1)
    limited = state_array + np.clip(
        predicted - state_array,
        -max_position_delta,
        max_position_delta,
    )
    return {
        "purpose": "preprocessing/output/timing only; images and state are not synchronized",
        "state_source": state_source,
        "camera_paths": image_sources,
        "input_shapes_chw": image_shapes,
        "normalized_first_action": result.normalized_output[0].tolist(),
        "relative_first_action": result.relative_output[0].tolist(),
        "absolute_first_action": predicted.tolist(),
        "limited_first_action": limited.tolist(),
        "clamped_indices": np.flatnonzero(
            np.abs(predicted - state_array) > max_position_delta
        ).tolist(),
        "timing_ms": {
            "preprocess": result.preprocess_latencies[0] * 1000.0,
            "model": result.inference_latencies[0] * 1000.0,
            "postprocess": result.postprocess_latencies[0] * 1000.0,
        },
    }
