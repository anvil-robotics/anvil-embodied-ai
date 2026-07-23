from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from anvil_eval import snapshot


def test_saved_snapshot_reports_live_image_shape_and_marks_state_as_unsynchronized(
    tmp_path, monkeypatch
) -> None:
    image_path = tmp_path / "frame_0000.png"
    Image.fromarray(np.zeros((6, 8, 3), dtype=np.uint8)).save(image_path)

    def fake_evaluate(**kwargs):
        item = kwargs["dataset"][0]
        assert item["observation.images.base"].shape == (3, 6, 8)
        return SimpleNamespace(
            predicted=np.array([[1.2, 2.0]], dtype=np.float32),
            normalized_output=np.array([[0.1, 0.2]], dtype=np.float32),
            relative_output=np.array([[0.2, 2.0]], dtype=np.float32),
            preprocess_latencies=[0.01],
            inference_latencies=[0.4],
            postprocess_latencies=[0.02],
        )

    monkeypatch.setattr(snapshot, "evaluate_native_relative_episode", fake_evaluate)
    report = snapshot.evaluate_saved_snapshot(
        model=object(),
        preprocessor=object(),
        postprocessor=object(),
        camera_report={"base": {"status": "found", "path": str(image_path)}},
        state=torch.tensor([1.0, 2.0]),
        state_source="episode 0 frame 0",
        device="cpu",
        task_description="fold",
        joint_names=["joint1", "gripper"],
        max_position_delta=0.05,
    )

    assert report["input_shapes_chw"]["base"] == [3, 6, 8]
    assert report["clamped_indices"] == [0]
    assert report["timing_ms"]["model"] == 400.0
    assert "not synchronized" in report["purpose"]
