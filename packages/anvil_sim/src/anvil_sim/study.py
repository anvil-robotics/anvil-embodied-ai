"""Study-plugin interface for the generic sim-env validation harness.

The harness (``bench_runner`` / ``bench_spec`` / ``eval_replay``) is a
study-agnostic gated pipeline: it sequences stages, gates on GT-replay, and
records a ledger. Everything about a *particular* research study — its
dataset groups, action-type registry, math identities, command builders, and
the exact processor/anchor wiring the GT replay must reproduce — lives behind
this interface, so a second study can be added without touching the harness.

A study is a :class:`Study` value assembled by a ``build_*_study()`` factory
(e.g. :func:`anvil_sim.studies.libero_ee.study.build_libero_ee_study`) and
registered under a short name. A spec selects its study via the optional
``study:`` YAML field (default ``"libero_ee"``); the harness resolves it
through :func:`get_study`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

    from anvil_sim.bench_spec import BenchSpec


@dataclass(frozen=True)
class GtReplayConfig:
    """How the study's ``gt-replay`` baseline is produced and detected.

    The baseline is a native GT replay computed once per task; a treatment's
    replay success is gated against it. ``is_baseline(spec)`` tells the runner
    when a spec IS the baseline (so it need not be re-gated against itself).
    """

    baseline_action_type: str
    baseline_control_mode: str
    n_action_steps: int
    is_baseline: Callable[[BenchSpec], bool]


@dataclass(frozen=True)
class ReplayAdapter:
    """The study-specific wiring :func:`anvil_sim.eval_replay.replay` needs.

    The harness owns the LIBERO env rollout loop and the generic
    chunk-anchor state machine (:class:`~anvil_sim.eval_replay.GtActionProvider`);
    the study owns everything that touches its own encoding/processors:

    - ``make_processors(action_type, n_action_steps)`` -> ``(env_pre,
      env_post, obs_step_or_None)`` — the exact pre/post pipelines a real
      policy eval uses for this treatment (this is what keeps GT replay
      byte-equivalent to the closed-loop path, incl. the chunk-anchor
      timing).
    - ``make_state_probe()`` -> a diagnostics-only obs step (independent of
      the treatment pipeline) exposing ``.observation(dict)`` and
      ``.last_anvil_state`` for the init-state alignment metric.
    - ``provider_mode(action_type)`` -> ``"direct" | "rel_hand" |
      "rel_world"``.
    - ``action_encoding(action_type)`` -> ``"rot6d" | "axis_angle"``.
    - ``encode_to_rot6d`` / ``decode_from_rot6d`` — the codec the provider
      applies around the shared rot6d n-0 machinery for axis-angle families.
    """

    make_processors: Callable[[str, int], tuple[Any, Any, Any]]
    make_state_probe: Callable[[], Any]
    provider_mode: Callable[[str], str]
    action_encoding: Callable[[str], str]
    encode_to_rot6d: Callable[[np.ndarray], np.ndarray]
    decode_from_rot6d: Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class Study:
    """A registered study plugin. See the module docstring for the split."""

    # The group used as the GT-replay baseline.
    baseline_group: str
    # Every eval action type this study can evaluate/replay.
    eval_action_types: tuple[str, ...]
    # (group, task_index) -> on-disk dataset directory.
    dataset_root: Callable[[str, int], Path]
    # dataset_group -> a validate-math identity check (raises on failure).
    math_validators: Mapping[str, Callable[[BenchSpec], dict]]
    # Study-specific spec legality errors (empty list == legal).
    legality: Callable[[BenchSpec], list[str]]
    # Subprocess command builders (the harness runs them; the study shapes them).
    convert_command: Callable[[int, list[str]], list[str]]
    train_command: Callable[..., list[str]]
    eval_command: Callable[..., list[str]]
    # Whether the generic mcap dataset-validate stage applies to this spec.
    dataset_validate_skip: Callable[[BenchSpec], bool]
    gt_replay: GtReplayConfig
    replay_adapter: ReplayAdapter
    # Fill any study-specific fields a spec left unset (mutates spec in
    # place), called by load_spec after study resolution but before
    # validate() — e.g. libero_ee derives env_suite/env_task_id from
    # task_index and eval.control_mode from eval.action_type, so a spec
    # need not repeat them explicitly.
    fill_defaults: Callable[[BenchSpec], None]


# One-line registry: study name -> zero-arg factory. Factories import their
# study package lazily so merely importing the harness pulls in no study.
_STUDIES: dict[str, Callable[[], Study]] = {}
_CACHE: dict[str, Study] = {}


def register_study(name: str, factory: Callable[[], Study]) -> None:
    _STUDIES[name] = factory


def get_study(name: str) -> Study:
    """Resolve (and memoize) a registered study by name."""
    if name not in _CACHE:
        if name not in _STUDIES:
            raise ValueError(f"Unknown study {name!r}; registered: {sorted(_STUDIES)}")
        _CACHE[name] = _STUDIES[name]()
    return _CACHE[name]


def _build_libero_ee() -> Study:
    from anvil_sim.studies.libero_ee.study import build_libero_ee_study

    return build_libero_ee_study()


register_study("libero_ee", _build_libero_ee)
