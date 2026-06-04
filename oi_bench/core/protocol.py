"""
BenchmarkTask — Abstract base class for all benchmark tasks in OI-Bench.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from jax import Array
from .types import Stimulus, ModelState, TrialResult
from .adapter import OIModel


class BenchmarkTask(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def n_trials(self) -> int:
        ...

    @property
    @abstractmethod
    def trial_duration_ms(self) -> float:
        ...

    @property
    @abstractmethod
    def learning_axis(self) -> str:
        ...

    @property
    def requires_spike_times(self) -> bool:
        return False

    def is_learning_trial(self, trial_id: int) -> bool:
        """
        Return True if this trial is part of the learning phase.

        The runner calls this before each trial to decide whether to
        enable or freeze plasticity. Freezing weights during test trials
        prevents STDP from modifying the learned representation during
        evaluation.

        Default: all trials are learning trials (always plastic).
        Tasks with distinct train/test phases override this.
        """
        return True

    @abstractmethod
    def generate_trial(self, trial_id: int, rng_key) -> list[Stimulus]:
        ...

    @abstractmethod
    def compute_score(
        self,
        response: Array,
        trial_id: int,
        state_trace: list[ModelState],
        metadata: dict | None = None,
    ) -> dict[str, float]:
        ...

    def setup(self, model: OIModel) -> None:
        pass

    def learning_index(self, trial_results: list[TrialResult]) -> float:
        if not trial_results:
            return 0.0

        w1, w2, w3 = 0.5, 0.3, 0.2
        n = len(trial_results)

        last_20 = trial_results[int(0.8 * n):]
        accuracies = [
            r.scores.get('accuracy', r.scores.get('performance', 0.0))
            for r in last_20
        ]
        norm_accuracy = float(sum(accuracies)/len(accuracies)) if accuracies else 0.0

        criterion = 0.7
        convergence_trial = n
        all_scores = [
            r.scores.get('accuracy', r.scores.get('performance', 0.0))
            for r in trial_results
        ]
        for i in range(9, n):
            if sum(all_scores[i-9:i+1]) / 10.0 >= criterion:
                convergence_trial = i
                break

        norm_convergence  = 1.0 - convergence_trial / n
        sample_efficiency = sum(all_scores) / n

        # NOTE: chance-level performance maps to LI ≈ 0.35, not 0.0.
        # A model that never learns (accuracy=0.5) scores LI > 0 due to
        # the convergence and sample_efficiency terms. When interpreting
        # results, treat LI < 0.4 as near-chance performance.
        li = w1*norm_accuracy + w2*norm_convergence + w3*sample_efficiency
        return float(max(0.0, min(1.0, li)))
