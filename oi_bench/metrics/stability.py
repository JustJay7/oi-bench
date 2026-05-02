"""
Stability Metrics

Measures homeostatic regulation quality and network stability.
These metrics quantify whether the network maintains a useful
operating regime during learning.

References:
  Spec Section 6.3
  Turrigiano (2008) Cell 135:422-435 — homeostatic efficacy
"""

from __future__ import annotations
import numpy as np


def firing_rate_cv(
    spike_counts_per_trial: list[np.ndarray],
    trial_dur_ms: float,
    window: int = 10,
) -> float:
    """
    Coefficient of variation of population firing rate across trials.

    Low CV = stable firing rate (homeostasis working).
    High CV = unstable, fluctuating activity.

    Parameters
    ----------
    spike_counts_per_trial : list of (n_neurons,) arrays
        Spike counts per neuron per trial.
    trial_dur_ms : float
        Trial duration in ms.
    window : int
        Number of trials to compute CV over. Default 10 (last 10).

    Returns
    -------
    float
        CV = std / mean of population mean firing rate over window.
    """
    if not spike_counts_per_trial:
        return 0.0

    trial_dur_s = trial_dur_ms / 1000.0
    rates = [
        float(np.mean(counts) / trial_dur_s)
        for counts in spike_counts_per_trial
    ]

    last = rates[-window:] if len(rates) >= window else rates
    mean = np.mean(last)
    if mean < 1e-6:
        return 0.0
    return float(np.std(last) / mean)


def homeostatic_efficacy(
    r_mean: np.ndarray,
    r_target: float,
) -> float:
    """
    Normalised distance from homeostatic target rate.

    0.0 = perfect regulation (r_mean == r_target for all neurons)
    1.0 = completely off target

    Parameters
    ----------
    r_mean : np.ndarray, shape (n_neurons,)
        Current mean firing rate estimate per neuron (Hz).
    r_target : float
        Target firing rate (Hz).
    """
    if r_target < 1e-6:
        return 0.0
    return float(np.mean(np.abs(r_mean - r_target)) / r_target)


def weight_drift_stability(
    W_history: list[np.ndarray],
    last_fraction: float = 0.2,
) -> float:
    """
    Mean absolute weight change per trial in post-learning phase.

    Near 0.0 = weights consolidated, memory stable.
    High = weights still drifting (ongoing plasticity or instability).
    """
    n = len(W_history)
    if n < 2:
        return 0.0
    start = max(1, int(n * (1.0 - last_fraction)))
    drifts = [
        float(np.mean(np.abs(W_history[i] - W_history[i - 1])))
        for i in range(start, n)
    ]
    return float(np.mean(drifts)) if drifts else 0.0


def stability_stats(
    spike_counts_per_trial: list[np.ndarray],
    W_history: list[np.ndarray],
    r_mean: np.ndarray,
    r_target: float,
    trial_dur_ms: float,
) -> dict[str, float]:
    """
    Compute all stability metrics in one call.

    Returns
    -------
    dict with keys:
        firing_rate_cv        : float — CV of population rate (last 10 trials)
        homeostatic_efficacy  : float — normalised distance from r_target
        weight_drift_stability: float — mean |ΔW| in post-learning phase
    """
    return {
        'firing_rate_cv':         firing_rate_cv(
            spike_counts_per_trial, trial_dur_ms),
        'homeostatic_efficacy':   homeostatic_efficacy(r_mean, r_target),
        'weight_drift_stability': weight_drift_stability(W_history),
    }
