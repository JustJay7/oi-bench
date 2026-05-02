"""
Plasticity Metrics

Computes weight-based statistics from RunResult weight history.
Tracks synaptic specialization, LTP/LTD balance, and stability.

References:
  Spec Section 6.1
  Song et al. (2000) Nature Neuroscience 3:919-926 — weight entropy
"""

from __future__ import annotations
import numpy as np


def stdp_potentiation_ratio(
    n_ltp: float,
    n_ltd: float,
) -> float:
    """
    Ratio of cumulative LTP to LTD events.

    > 1.0 : net potentiation (learning)
    < 1.0 : net depression (forgetting)
    = 1.0 : balanced
    """
    return float(n_ltp / (n_ltd + 1e-10))


def weight_entropy(W: np.ndarray) -> float:
    """
    Shannon entropy of the synaptic weight distribution (bits).

    High entropy = uniform weights (unspecialized).
    Low entropy  = concentrated weights (specialized).

    Parameters
    ----------
    W : np.ndarray
        Weight matrix, shape (n_pre, n_post). Only connected weights used.
    """
    w_flat = W.flatten()
    w_flat = w_flat[w_flat > 1e-6]   # exclude silent synapses
    if len(w_flat) == 0:
        return 0.0
    # Normalise to probability distribution
    p = w_flat / w_flat.sum()
    return float(-np.sum(p * np.log2(p + 1e-10)))


def effective_connectivity(
    W: np.ndarray,
    threshold: float = 0.1,
) -> float:
    """
    Fraction of connected weights above threshold * w_max.

    Tracks how many synapses are functionally active.
    Low effective connectivity = sparse, specialised network.

    Parameters
    ----------
    threshold : float
        Fraction of w_max to use as activity threshold. Default 0.1.
    """
    w_max = W.max()
    if w_max < 1e-6:
        return 0.0
    return float(np.mean(W > threshold * w_max))


def weight_drift_rate(
    W_history: list[np.ndarray],
    last_fraction: float = 0.2,
) -> float:
    """
    Mean absolute weight change per trial during post-learning phase.

    Low drift = consolidated weights (stable memory).
    High drift = weights still changing (ongoing plasticity).

    Parameters
    ----------
    W_history : list[np.ndarray]
        Weight matrix snapshots, one per trial.
    last_fraction : float
        Fraction of trials considered post-learning. Default 0.2.
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


def plasticity_stats(
    W_history: list[np.ndarray],
    n_ltp: float,
    n_ltd: float,
) -> dict[str, float]:
    """
    Compute all plasticity metrics in one call.

    Parameters
    ----------
    W_history : list[np.ndarray]
        Weight snapshots per trial from RunResult.weight_stats_per_trial.
    n_ltp, n_ltd : float
        Cumulative LTP and LTD event counts from synapse.

    Returns
    -------
    dict with keys:
        stdp_potentiation_ratio : float
        weight_entropy_initial  : float  — bits at trial 0
        weight_entropy_final    : float  — bits at last trial
        effective_connectivity  : float  — fraction of active synapses
        weight_drift_rate       : float  — mean |ΔW| in post-learning phase
    """
    W_init  = W_history[0]  if W_history else np.array([])
    W_final = W_history[-1] if W_history else np.array([])

    return {
        'stdp_potentiation_ratio': stdp_potentiation_ratio(n_ltp, n_ltd),
        'weight_entropy_initial':  weight_entropy(W_init),
        'weight_entropy_final':    weight_entropy(W_final),
        'effective_connectivity':  effective_connectivity(W_final),
        'weight_drift_rate':       weight_drift_rate(W_history),
    }
