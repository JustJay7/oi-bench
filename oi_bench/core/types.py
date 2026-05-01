"""
Shared types and dataclasses for OI-Bench.

All data flowing between the benchmark harness, tasks, and models
is typed using these structures. No simulation logic here — pure
data contracts.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import jax.numpy as jnp
from jax import Array
import numpy as np


@dataclass
class Stimulus:
    """
    Stimulus delivered to a model at each timestep.

    Attributes
    ----------
    current : Array
        External current injection, shape (n_neurons,), units pA.
        Zero for neurons not receiving direct stimulation.
    spike_train : Array
        Binary spike input, shape (n_neurons,), values {0, 1}.
        Used for pattern-based stimuli (e.g. CS population bursts).
    t : float
        Current simulation time in ms.
    label : int
        Trial label for supervised scoring. -1 = unlabelled.
    """
    current:     Array
    spike_train: Array
    t:           float
    label:       int = -1


@dataclass
class ModelState:
    """
    Minimal state snapshot the benchmark reads from a model each timestep.

    The benchmark never reaches inside a model — it only reads ModelState.
    This keeps the adapter interface clean and substrate-agnostic.

    Attributes
    ----------
    spikes : Array
        Binary spike output, shape (n_neurons,).
    membrane : Array
        Membrane potential, shape (n_neurons,), mV.
        May be zeros for models that don't expose membrane voltage.
    weights : Array or None
        Synaptic weight matrix, shape (n_pre, n_post).
        None for models that hide internal weights (e.g. reservoirs).
    extras : dict
        Model-specific state. Use for calcium concentration, adaptation
        variable, resource pools, etc. Does not affect scoring.
    """
    spikes:   Array
    membrane: Array
    weights:  Array | None = None
    extras:   dict[str, Any] = field(default_factory=dict)


@dataclass
class TrialResult:
    """
    Result of one benchmark trial.

    Returned by BenchmarkTask.compute_score() after each trial.

    Attributes
    ----------
    trial_id : int
        Zero-indexed trial number within the task run.
    correct : bool or None
        Whether the model produced the correct response.
        None for unsupervised tasks where correctness is not defined.
    response : Array
        Raw model output used for scoring — typically mean firing rate
        of the output population over the response window (pA or Hz).
    scores : dict
        Scalar performance metrics. Must include at least one of
        'accuracy' or 'performance'. All values must be float.
    state_trace : list[ModelState]
        Full per-step ModelState trace for the trial.
        Empty list if record=False was passed to the runner.
    metadata : dict
        Task-specific metadata (e.g. stimulus identity, delay duration).
    """
    trial_id:    int
    correct:     bool | None
    response:    Array
    scores:      dict[str, float]
    state_trace: list[ModelState] = field(default_factory=list)
    metadata:    dict[str, Any]   = field(default_factory=dict)
