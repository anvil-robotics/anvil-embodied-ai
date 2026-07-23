from __future__ import annotations

import torch

from anvil_eval.evaluator import EpisodeEvaluator


class CapturePreprocessor:
    def __init__(self) -> None:
        self.last_batch = None

    def __call__(self, batch: dict) -> dict:
        self.last_batch = dict(batch)
        return batch


def _make_evaluator(model_type: str, preprocessor: CapturePreprocessor) -> EpisodeEvaluator:
    return EpisodeEvaluator(
        model=object(),
        preprocessor=preprocessor,
        postprocessor=None,
        model_type=model_type,
        device="cpu",
        anvil_cfg={},
        task_description="place the block on the plate",
        joint_names=["joint1"],
    )


def test_language_conditioned_sync_policy_gets_task_prompt():
    preprocessor = CapturePreprocessor()
    evaluator = _make_evaluator("multi_task_dit", preprocessor)

    processed = evaluator._preprocess_policy_observation(
        {"observation.state": torch.tensor([1.0])}
    )

    assert processed["task"] == ["place the block on the plate"]
    assert preprocessor.last_batch["task"] == ["place the block on the plate"]


def test_act_does_not_get_task_prompt():
    preprocessor = CapturePreprocessor()
    evaluator = _make_evaluator("act", preprocessor)

    processed = evaluator._preprocess_policy_observation(
        {"observation.state": torch.tensor([1.0])}
    )

    assert "task" not in processed
    assert "task" not in preprocessor.last_batch
