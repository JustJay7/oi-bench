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
      setup()          : called once before the full task run

    The runner loop that calls these methods is in harness/runner.py.
    Tasks never call model methods directly.
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
        Used for grouping in benchmark reports and per-axis subscores.
        """
        ...

    @abstractmethod
    def generate_trial(self, trial_id: int, rng_key) -> list[Stimulus]:
        """
        Generate the complete stimulus sequence for one trial.

        Parameters
        ----------
        trial_id : int
            Zero-indexed trial number. Tasks may use this to schedule
            different stimulus conditions across trials (e.g. CS-only
            trials vs CS+US trials in classical conditioning).
        rng_key : JAX PRNGKey
            Randomness source. Use jax.random.split for sub-keys.
            All randomness must flow through this key for reproducibility.

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
    ) -> dict[str, float]:
        """
        Score the model's output for one trial.

        Parameters
        ----------
        response : Array
            Mean firing rate of the output population over the response
            window. Shape (n_output,), units Hz.
            This is the population rate code readout — see spec Section 12 Q1.
        trial_id : int
            Zero-indexed trial number.
        state_trace : list[ModelState]
            Full per-step state trace. May be empty if record=False.

        Returns
        -------
        dict[str, float]
            Scalar scores for this trial. Must include at least one of:
              'accuracy'    (float in [0, 1]) for classification tasks
              'performance' (float in [0, 1]) for continuous tasks
            May also include task-specific metrics.
        """
        ...

    def setup(self, model: OIModel) -> None:
        """
        Optional hook called once before the full task run begins.

        Use to validate model compatibility (check n_input, n_output),
        precompute stimulus patterns, or initialise task state.

        Parameters
        ----------
        model : OIModel
            The model that will be evaluated on this task.

        Default: no-op.
        """
        pass

    def learning_index(self, trial_results: list[TrialResult]) -> float:
        """
        Compute the Learning Index (LI) for this task from trial results.

        LI ∈ [0, 1] per spec Section 6.4:
          LI = w1 · norm_accuracy + w2 · (1 - convergence_trial/n_trials)
             + w3 · sample_efficiency

        Weights: w1=0.5, w2=0.3, w3=0.2 (fixed, not tunable).

        Parameters
        ----------
        trial_results : list[TrialResult]
            All trial results from one complete task run.

        Returns
        -------
        float
            Learning Index in [0, 1].
        """
        if not trial_results:
            return 0.0

        w1, w2, w3 = 0.5, 0.3, 0.2
        n = len(trial_results)

        # Asymptotic accuracy: mean over last 20% of trials
        last_20 = trial_results[int(0.8 * n):]
        accuracies = [
            r.scores.get('accuracy', r.scores.get('performance', 0.0))
            for r in last_20
        ]
        norm_accuracy = float(sum(accuracies) / len(accuracies)) if accuracies else 0.0

        # Convergence trial: first trial where rolling 10-trial mean > 0.7
        criterion = 0.7
        convergence_trial = n  # default: never converged
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

        # Sample efficiency: area under learning curve, normalised
        sample_efficiency = sum(all_scores) / n

        li = w1 * norm_accuracy + w2 * norm_convergence + w3 * sample_efficiency
        return float(max(0.0, min(1.0, li)))
