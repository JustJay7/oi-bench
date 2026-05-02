"""
OI-Bench metrics package.

All metrics consume RunResult objects and produce scalar floats
for direct use in paper tables and figures.

  learning_curve.py  — convergence_trial, asymptotic_performance, sample_efficiency
  plasticity.py      — stdp_potentiation_ratio, weight_entropy, effective_connectivity
  information.py     — mutual_information, transfer_entropy
  stability.py       — firing_rate_cv, homeostatic_efficacy, weight_drift_stability
"""

from .learning_curve import learning_curve_stats
from .plasticity import plasticity_stats
from .information import information_stats
from .stability import stability_stats

__all__ = [
    "learning_curve_stats",
    "plasticity_stats",
    "information_stats",
    "stability_stats",
]
