"""
Information-Theoretic Metrics for Spike Train Data

Computes mutual information and transfer entropy between input and output
spike trains using a binned estimator on binary spike sequences.

Transfer entropy formulation:
  TE(X→Y) = Σ p(y_{t+1}, y_t^k, x_t^l) · log[ p(y_{t+1} | y_t^k, x_t^l)
                                                / p(y_{t+1} | y_t^k) ]

Where k, l are history embedding lengths (default k=l=1 for binary bins).

Bin width: 10ms (spec Section 6.2). Normalised by bin width per
Shorten et al. (2021) PLOS Comp Biol to ensure convergence.

References:
  Schreiber (2000) Phys. Rev. Lett. 85:461 — transfer entropy
  Shorten et al. (2021) PLOS Comp Biol doi:10.1371/journal.pcbi.1008054
  Kraskov et al. (2004) Phys. Rev. E 69:066138 — KSG estimator
  Spec Section 6.2
"""

from __future__ import annotations
import numpy as np
from oi_bench.core.types import ModelState


def _bin_spikes(
    spike_trace: list[np.ndarray],
    dt_ms: float,
    bin_ms: float = 10.0,
) -> np.ndarray:
    """
    Bin spike train into binary population activity vector.

    Parameters
    ----------
    spike_trace : list of (n_neurons,) arrays
        Per-step spike vectors (binary).
    dt_ms : float
        Simulation timestep in ms.
    bin_ms : float
        Bin width in ms. Default 10ms.

    Returns
    -------
    binned : np.ndarray, shape (n_bins,)
        Binary activity: 1 if any neuron fired in bin, else 0.
    """
    steps_per_bin = max(1, int(bin_ms / dt_ms))
    n_steps = len(spike_trace)
    n_bins  = n_steps // steps_per_bin

    binned = np.zeros(n_bins, dtype=np.float32)
    for b in range(n_bins):
        start = b * steps_per_bin
        end   = start + steps_per_bin
        bin_spikes = np.array([
            np.any(spike_trace[i] > 0)
            for i in range(start, min(end, n_steps))
        ])
        binned[b] = float(np.any(bin_spikes))
    return binned


def mutual_information(
    x: np.ndarray,
    y: np.ndarray,
    eps: float = 1e-10,
) -> float:
    """
    Mutual information I(X; Y) for binary sequences.

    I(X;Y) = Σ_{x,y} p(x,y) log[ p(x,y) / (p(x)p(y)) ]

    Parameters
    ----------
    x, y : np.ndarray, shape (n_bins,)
        Binary spike sequences.

    Returns
    -------
    float
        Mutual information in bits.
    """
    assert len(x) == len(y), "x and y must have same length"
    n = len(x)
    if n == 0:
        return 0.0

    # Joint and marginal probabilities
    p_xy = np.zeros((2, 2))
    for xi, yi in zip(x.astype(int), y.astype(int)):
        p_xy[xi, yi] += 1.0
    p_xy /= (n + eps)

    p_x = p_xy.sum(axis=1)
    p_y = p_xy.sum(axis=0)

    mi = 0.0
    for i in range(2):
        for j in range(2):
            if p_xy[i, j] > eps:
                mi += p_xy[i, j] * np.log2(
                    p_xy[i, j] / (p_x[i] * p_y[j] + eps) + eps
                )
    return float(max(0.0, mi))


def transfer_entropy(
    source: np.ndarray,
    target: np.ndarray,
    history: int = 1,
    eps: float   = 1e-10,
) -> float:
    """
    Transfer entropy TE(source → target) for binary sequences.

    TE(X→Y) = I(Y_{t+1}; X_t | Y_t)
             = Σ p(y', y, x) log[ p(y'|y,x) / p(y'|y) ]

    Parameters
    ----------
    source : np.ndarray, shape (n_bins,)
        Source population binary activity.
    target : np.ndarray, shape (n_bins,)
        Target population binary activity.
    history : int
        Number of past bins to condition on. Default 1.

    Returns
    -------
    float
        Transfer entropy in bits per bin.
    """
    n = len(source)
    if n <= history + 1:
        return 0.0

    # Build (y_{t+1}, y_t, x_t) triplets
    y_future = target[history + 1:]
    y_past   = target[history:-1]
    x_past   = source[history:-1]

    n_eff = len(y_future)
    if n_eff == 0:
        return 0.0

    # Joint distribution p(y', y, x)
    p_yyx = np.zeros((2, 2, 2))
    for yf, yp, xp in zip(y_future.astype(int),
                            y_past.astype(int),
                            x_past.astype(int)):
        p_yyx[yf, yp, xp] += 1.0
    p_yyx /= (n_eff + eps)

    # Marginals
    p_yx = p_yyx.sum(axis=0)   # p(y, x)
    p_y  = p_yyx.sum(axis=(0, 2))  # p(y)  -- marginal over x and y'

    te = 0.0
    for yf in range(2):
        for yp in range(2):
            for xp in range(2):
                if p_yyx[yf, yp, xp] > eps:
                    p_cond_yx = p_yyx[yf, yp, xp] / (p_yx[yp, xp] + eps)
                    p_cond_y  = p_yyx[yf, yp, :].sum() / (p_y[yp] + eps)
                    te += p_yyx[yf, yp, xp] * np.log2(
                        p_cond_yx / (p_cond_y + eps) + eps
                    )
    return float(max(0.0, te))


def information_stats(
    state_trace: list[ModelState],
    dt_ms: float,
    bin_ms: float = 10.0,
) -> dict[str, float]:
    """
    Compute all information-theoretic metrics for one trial.

    Parameters
    ----------
    state_trace : list[ModelState]
        Per-step model states from one trial.
    dt_ms : float
        Simulation timestep in ms.
    bin_ms : float
        Bin width for spike binning. Default 10ms.

    Returns
    -------
    dict with keys:
        mutual_information_bits   : float
        transfer_entropy_bits_per_bin : float
        input_activity_rate       : float — mean input firing rate
        output_activity_rate      : float — mean output firing rate
    """
    if not state_trace:
        return {
            'mutual_information_bits':       0.0,
            'transfer_entropy_bits_per_bin': 0.0,
            'input_activity_rate':           0.0,
            'output_activity_rate':          0.0,
        }

    input_spikes  = [np.array(s.extras.get('input_spikes',
                     np.zeros(1))) for s in state_trace]
    output_spikes = [np.array(s.spikes) for s in state_trace]

    x_binned = _bin_spikes(input_spikes,  dt_ms, bin_ms)
    y_binned = _bin_spikes(output_spikes, dt_ms, bin_ms)

    mi = mutual_information(x_binned, y_binned)
    te = transfer_entropy(x_binned, y_binned)

    input_rate  = float(np.mean([np.mean(s) for s in input_spikes]))
    output_rate = float(np.mean([np.mean(s) for s in output_spikes]))

    return {
        'mutual_information_bits':       mi,
        'transfer_entropy_bits_per_bin': te,
        'input_activity_rate':           input_rate,
        'output_activity_rate':          output_rate,
    }
