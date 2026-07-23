from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch
from lerobot.policies.rtc.action_queue import ActionQueue
from lerobot.policies.rtc.configuration_rtc import RTCConfig


def _load_adapter():
    repo = Path(__file__).resolve().parents[3]
    path = repo / "ros2" / "src" / "lerobot_control" / "lerobot_control" / "vla_jepa_rtc.py"
    spec = importlib.util.spec_from_file_location("vla_jepa_rtc", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _IdentityDenoiser:
    def __call__(self, *, hidden_states, encoder_hidden_states, timestep):
        del encoder_hidden_states, timestep
        return hidden_states


class _FakeActionHead:
    action_horizon = 4
    num_inference_timesteps = 2
    config = SimpleNamespace(action_dim=2, action_num_timestep_buckets=100)
    model = _IdentityDenoiser()
    action_decoder = torch.nn.Identity()

    def _build_inputs(self, conditioning_tokens, actions, state, timesteps):
        del conditioning_tokens, state, timesteps
        return actions


class _RecordingRTC:
    def __init__(self):
        self.calls = []

    def denoise_step(self, **kwargs):
        self.calls.append(kwargs)
        return kwargs["original_denoise_step_partial"](kwargs["x_t"])


def test_vla_jepa_sampler_maps_flow_convention_for_rtc() -> None:
    adapter = _load_adapter()
    head = _FakeActionHead()
    conditioning = torch.zeros(1, 1, 2)
    noise = torch.ones(1, 4, 2)
    leftovers = torch.full((4, 2), 3.0)
    rtc = _RecordingRTC()

    ordinary = adapter.sample_vla_jepa_actions(head, conditioning, noise=noise)
    guided = adapter.sample_vla_jepa_actions(
        head,
        conditioning,
        noise=noise,
        rtc_processor=rtc,
        inference_delay=3,
        prev_chunk_left_over=leftovers,
        execution_horizon=4,
    )

    assert torch.equal(ordinary, torch.full_like(ordinary, 2.25))
    assert torch.equal(guided, ordinary)
    assert [call["time"] for call in rtc.calls] == [1.0, 0.5]
    assert all(call["inference_delay"] == 3 for call in rtc.calls)
    assert all(call["prev_chunk_left_over"] is leftovers for call in rtc.calls)
    assert all(call["execution_horizon"] == 4 for call in rtc.calls)


def test_rtc_queue_publishes_the_latency_aligned_action() -> None:
    queue = ActionQueue(RTCConfig(enabled=True))
    chunk = torch.arange(6, dtype=torch.float32).unsqueeze(-1)

    queue.merge(chunk, chunk, real_delay=3, action_index_before_inference=0)

    action = queue.get()
    assert action is not None
    assert action.item() == 3.0


def test_rtc_queue_holds_when_latency_consumes_full_chunk() -> None:
    queue = ActionQueue(RTCConfig(enabled=True))
    chunk = torch.arange(4, dtype=torch.float32).unsqueeze(-1)

    queue.merge(chunk, chunk, real_delay=4, action_index_before_inference=0)

    assert queue.get() is None


def test_install_vla_jepa_rtc_is_instance_scoped_and_idempotent() -> None:
    adapter = _load_adapter()

    def prepare_model_inputs(batch, training):
        del batch, training
        return {}

    policy = SimpleNamespace(
        config=SimpleNamespace(rtc_config=RTCConfig(enabled=True), device="cpu"),
        model=SimpleNamespace(action_model=_FakeActionHead()),
        _queues={},
        eval=lambda: None,
        _prepare_model_inputs=prepare_model_inputs,
    )

    adapter.install_vla_jepa_rtc(policy)
    installed_method = policy.predict_action_chunk
    adapter.install_vla_jepa_rtc(policy)

    assert policy.predict_action_chunk is installed_method
    assert policy._anvil_vla_jepa_rtc_installed is True
    assert policy.rtc_processor.rtc_config.enabled is True
