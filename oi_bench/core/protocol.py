"""
BenchmarkTask — Abstract base class for all benchmark tasks in OI-Bench.

Tasks are responsible for:
  - Generating stimulus sequences (one per trial)
  - Defining episode structure (n_trials, trial_duration_ms)
  - Scoring model responses
  - Returning structured TrialResults

Tasks do NOT know about model internals. They only emit Stimulus objects
and receive ModelState objects back via the runner loop. This separation
ensures the benchmark is genuinely substrate-agnostic.

Task taxonomy (from spec Section 5):
  Axis 1 — Associative:   T1 ClassicalConditioning, T2 PatternCompletion
  Axis 2 — Temporal:      T3 SequencePrediction,    T4 IntervalTiming
  Axis 3 — WorkingMemory: T5 DelayMatchToSample,    T6 NBack
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from jax import Array
from .types import Stimulus, ModelState, TrialResult
from .adapter import OIModel


class BenchmarkTask(ABC):
    """
    Abstract base class for all benchmark tasks.

    Implementors must provide:
      name             : str property
      n_trials         : int property
      trial_duration_ms: float property
      learning_axis    : str property — 'associative', 'temporal', or 'working_memory'
      generate_trial() : produce stimulus sequence for one trial
      compute_score()  : score model output for one trial

    Optionally override:
      setup()               : called once before the full task run
      requires_spike_times  : bool property — set True for tasks needing burst timing (T4)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique task identifier, e.g. 'T1_ClassicalConditioning'."""
        ...

    @property
    @abstractmethod
    def n_trials(self) -> int:
        """Total number of trials in one task run."""
        ...

    @property
    @abstractmethod
    def trial_duration_ms(self) -> float:
        """Duration of each trial in ms."""
        ...

    @property
    @abstractmethod
    def learning_axis(self) -> str:
        """
        Learning axis this task evaluates.
        One of: 'associative', 'temporal', 'working_memory'.
        """
        ...

    @property
    def requires_spike_times(self) -> bool:
        """
        If True, the runner records a per-timestep population spike count
        array and passes it in compute_score() via
        metadata['spike_counts_timeseries'] — shape (n_steps,).

        Only tasks that need burst-timing detection (T4) should return True.
        Default: False.
        """
        return False

    @abstractmethod
    def generate_trial(self, trial_id: int, rng_key) -> list[Stimulus]:
        """
        Generate the complete stimulus sequence for one trial.

        Parameters
        ----------
        trial_id : int
        rng_key  : JAX PRNGKey

        Returns
        -------
        list[Stimulus]
            One Stimulus per timestep. Length must equal:
            int(trial_duration_ms / model.dt)
        """
        ...

    @abstractmethod
    def compute_score(
        self,
        response: Array,
        trial_id: int,
        state_trace: list[ModelState],
        metadata: dict | None = None,
    ) -> dict[str, float]:
        """
        Score the model's output for one trial.

        Parameters
        ----------
        response : Array
            Mean firing rate of the output population over the full trial.
            Shape (n_output,), units Hz.
        trial_id : int
        state_trace : list[ModelState]
            Full per-step state trace. Empty if record_traces=False.
        metadata : dict or None
            Runner-injected per-trial metadata. When requires_spike_times=True,
            contains 'spike_counts_timeseries': np.ndarray shape (n_steps,)
            with per-timestep population spike count.

        Returns
        -------
        dict[str, float]
            Must include at least one of 'accuracy' or 'performance'.
        """
        ...

    def setup(self, model: OIModel) -> None:
        """
        Optional hook called once before the full task run begins.
        Default: no-op.
        """
        pass

    def learning_index(self, trial_results: list[TrialResult]) -> float:
        """
        Compute the Learning Index (LI) for this task from trial results.

        LI ∈ [0, 1] per spec Section 6.4:
          LI = w1 · norm_accuracy + w2 · (1 - convergence_trial/n_trials)
             + w3 · sample_efficiency

        Weights: w1=0.5, w2=0.3, w3=0.2 (fixed).
        """
        if not trial_results:
            return 0.0

        w1, w2, w3 = 0.5, 0.3, 0.2
        n = len(trial_results)

        last_20 = trial_results[int(0.8 * n):]
        accuracies = [
            r.scores.get('accuracy', r.scores.get('performance', 0.0))
            for r in last_20
        ]
        norm_accuracy = float(sum(accuracies) / len(accuracies)) if accuracies else 0.0

        criterion = 0.7
        convergence_trial = n
        all_scores = [
            r.scores.get('accuracy', r.scores.get('performance', 0.0))
            for r in trial_results
        ]
        for i in range(9, n):
            window_mean = sum(all_scores[i-9:i+1]) / 10.0
            if window_mean >= criterion:
                convergence_trial = i
                break

        norm_convergence = 1.0 - convergence_trial / n
        sample_efficiency = sum(all_scores) / n

        li = w1 * norm_accuracy + w2 * norm_convergence + w3 * sample_efficiency
        return float(max(0.0, min(1.0, li)))
