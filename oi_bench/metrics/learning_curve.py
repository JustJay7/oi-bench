"""
Learning Curve Metrics

Computes trial-by-trial performance statistics from a list of TrialResults.
All metrics are scalar floats for direct use in paper tables and figures.

References:
  Spec Section 6.1
"""

from __future__ import annotations
import numpy as np
from oi_bench.core.types import TrialResult


def convergence_trial(
    trial_results: list[TrialResult],
    criterion: float = 0.7,
    window: int      = 10,
    score_key: str   = 'accuracy',
) -> int:
    """
    First trial where rolling window mean exceeds criterion.

    Parameters
    ----------
    trial_results : list[TrialResult]
    criterion : float
        Performance threshold. Default 0.7.
    window : int
        Rolling window size in trials. Default 10.
    score_key : str
        Key to extract from TrialResult.scores. Default 'accuracy'.

    Returns
    -------
    int
        Trial index (0-based) of convergence. Returns len(trial_results)
        if criterion never reached (did not converge).
    """
    scores = [
        r.scores.get(score_key, r.scores.get('performance', 0.0))
        for r in trial_results
    ]
    n = len(scores)
    for i in range(window - 1, n):
        if np.mean(scores[i - window + 1:i + 1]) >= criterion:
            return i
    return n   # did not converge


def asymptotic_performance(
    trial_results: list[TrialResult],
    last_fraction: float = 0.2,
    score_key: str       = 'accuracy',
) -> float:
    """
    Mean performance over the last fraction of trials.

    Parameters
    ----------
    last_fraction : float
        Fraction of trials to average over. Default 0.2 (last 20%).
    """
    scores = [
        r.scores.get(score_key, r.scores.get('performance', 0.0))
        for r in trial_results
    ]
    n = len(scores)
    last_n = max(1, int(n * last_fraction))
    return float(np.mean(scores[-last_n:]))


def sample_efficiency(
    trial_results: list[TrialResult],
    score_key: str = 'accuracy',
) -> float:
    """
    Area under the learning curve, normalised to [0, 1].

    AUC / n_trials gives the time-averaged performance. A model that
    learns immediately scores 1.0; one that never learns scores 0.0.
    """
    scores = [
        r.scores.get(score_key, r.scores.get('performance', 0.0))
        for r in trial_results
    ]
    return float(np.mean(scores))


def learning_curve_stats(
    trial_results: list[TrialResult],
    score_key: str       = 'accuracy',
    criterion: float     = 0.7,
    window: int          = 10,
    last_fraction: float = 0.2,
) -> dict[str, float]:
    """
    Compute all learning curve metrics in one call.

    Returns
    -------
    dict with keys:
        convergence_trial     : int — first trial exceeding criterion
        asymptotic_performance: float — mean over last fraction of trials
        sample_efficiency     : float — normalised AUC
        n_trials              : int
    """
    n = len(trial_results)
    return {
        'convergence_trial':      convergence_trial(
            trial_results, criterion, window, score_key),
        'asymptotic_performance': asymptotic_performance(
            trial_results, last_fraction, score_key),
        'sample_efficiency':      sample_efficiency(
            trial_results, score_key),
        'n_trials':               n,
    }
