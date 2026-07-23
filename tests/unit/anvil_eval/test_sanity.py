from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from anvil_eval.sanity import (
    compute_episode_sanity,
    evaluate_native_relative_episode,
    simulate_sync_prefetch,
)


class RelativeActionsProcessorStep:
    enabled = True

    def __init__(self) -> None:
        self.last_state = None

    def _build_mask(self, action_dim: int) -> list[bool]:
        return [True, False][:action_dim]


class FakePreprocessor:
    def __init__(self) -> None:
        self.relative_step = RelativeActionsProcessorStep()
        self.steps = [self.relative_step]
        self.states_seen: list[torch.Tensor] = []

    def __call__(self, batch: dict) -> dict:
        state = batch["observation.state"].reshape(1, -1).clone()
        self.relative_step.last_state = state
        self.states_seen.append(state)
        return batch


class FakePostprocessor:
    def __init__(self, relative_step: RelativeActionsProcessorStep) -> None:
        self.relative_step = relative_step

    def process_action(self, action: torch.Tensor) -> torch.Tensor:
        result = action.clone()
        result[:, 0] += self.relative_step.last_state[0, 0]
        return result


class FakeModel:
    def __init__(self) -> None:
        self.config = SimpleNamespace(n_action_steps=2, chunk_size=2)
        self.calls = 0

    def reset(self) -> None:
        self.calls = 0

    def predict_action_chunk(self, _batch: dict) -> torch.Tensor:
        outputs = [
            torch.tensor([[[0.5, 20.0], [1.0, 21.0]]]),
            torch.tensor([[[0.5, 40.0], [1.0, 41.0]]]),
        ]
        output = outputs[self.calls]
        self.calls += 1
        return output


class FakeDataset:
    def __init__(self) -> None:
        self.hf_dataset = self
        self.states = [
            torch.tensor([1.0, 10.0]),
            torch.tensor([2.0, 11.0]),
            torch.tensor([3.0, 30.0]),
            torch.tensor([4.0, 31.0]),
        ]
        self.actions = [
            torch.tensor([1.5, 20.0]),
            torch.tensor([2.0, 21.0]),
            torch.tensor([3.5, 40.0]),
            torch.tensor([4.0, 41.0]),
        ]

    def __getitem__(self, index: int) -> dict:
        return {
            "observation.state": self.states[index],
            "action": self.actions[index],
        }


def test_native_relative_replay_postprocesses_each_chunk_against_capture_state() -> None:
    dataset = FakeDataset()
    model = FakeModel()
    preprocessor = FakePreprocessor()
    postprocessor = FakePostprocessor(preprocessor.relative_step)

    result = evaluate_native_relative_episode(
        model=model,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        dataset=dataset,
        frame_indices=[0, 1, 2, 3],
        episode_idx=0,
        split_label="train",
        device="cpu",
        task_description="fold the shirt",
        joint_names=["right_joint_1.pos", "right_gripper.pos"],
    )

    np.testing.assert_allclose(result.predicted, np.stack(dataset.actions))
    np.testing.assert_allclose(
        result.relative_output,
        np.array([[0.5, 20.0], [1.0, 21.0], [0.5, 40.0], [1.0, 41.0]]),
    )
    assert result.chunk_starts == [0, 2]
    assert model.calls == 2
    assert [state.tolist() for state in preprocessor.states_seen] == [
        [[1.0, 10.0]],
        [[3.0, 30.0]],
    ]

    metrics, _limited = compute_episode_sanity(result, max_position_delta=100.0)
    assert metrics["model_mae"] == pytest.approx(0.0)
    assert metrics["model_beats_hold"] is True


def test_sync_prefetch_simulation_separates_prediction_and_action_rates() -> None:
    report = simulate_sync_prefetch(
        [0.46, 0.48, 0.47],
        total_steps=900,
        control_hz=30.0,
        chunk_size=30,
        refill_threshold=20,
        replace_pending=True,
    )

    assert report["raw_model_capacity_hz"] == pytest.approx(2.1277, rel=1e-3)
    assert 1.1 < report["prediction_rate_hz"] < 1.4
    assert report["action_publication_hz"] > 29.0
    assert report["post_warmup_starved_steps"] == 0
    assert report["replaced_actions"] > 0


def test_sync_prefetch_reports_starvation_when_latency_exceeds_refill_budget() -> None:
    report = simulate_sync_prefetch(
        [0.8],
        total_steps=600,
        control_hz=30.0,
        chunk_size=30,
        refill_threshold=20,
        replace_pending=True,
    )

    assert report["model_latency_p95_ms"] > report["refill_budget_ms"]
    assert report["post_warmup_starved_steps"] > 0
