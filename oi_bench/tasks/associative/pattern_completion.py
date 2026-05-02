"""
Task T2 — Pattern Completion

Benchmark axis: Associative Learning
Biological analog: CA3 hippocampal attractor dynamics (Rolls 2007, Hippocampus)

PROTOCOL
--------
The network learns N=10 spatial patterns during a learning phase.
Each pattern is a random binary activation of 30% of input neurons.
During test, degraded versions (70% correct + 30% noise) are presented.
The model must reconstruct the full stored pattern from the partial cue.

Trial structure over n_trials=300:
  Trials 0-199   : Learning phase — full patterns presented repeatedly
                   (each of 10 patterns presented 20 times, interleaved)
  Trials 200-299 : Test phase — degraded patterns (70% signal + 30% noise)

SCORING
-------
Hamming similarity between model output and stored pattern:
  similarity = 1 - (hamming_distance / n_output)

A similarity of 1.0 = perfect completion.
Chance level = 0.3 (random 30% active baseline).

Score per trial:
  'accuracy'    : normalised Hamming similarity ∈ [0, 1]
  'similarity'  : raw Hamming similarity
  'pattern_id'  : which of the 10 patterns was presented

Learning Index uses accuracy over test trials only (last 100 trials).

References:
  Rolls (2007) Hippocampus 17:1153-1173
  Hopfield (1982) PNAS 79:2554-2558
  Spec Section 5, Task T2
"""

from __future__ import annotations
import jax
import jax.numpy as jnp
import numpy as np
from typing import List

from oi_bench.core.protocol import BenchmarkTask
from oi_bench.core.types import Stimulus, ModelState, TrialResult
from oi_bench.core.adapter import OIModel


class PatternCompletionTask(BenchmarkTask):
    """
    T2 — Pattern Completion.

    Parameters
    ----------
    n_patterns : int
        Number of distinct patterns to store. Default 10.
    pattern_sparsity : float
        Fraction of input neurons active in each pattern. Default 0.3.
    noise_level : float
        Fraction of pattern bits flipped during test. Default 0.3.
    n_learning_reps : int
        Number of times each pattern is presented during learning. Default 20.
    pattern_duration_ms : float
        Duration of each pattern presentation. Default 200ms.
    response_onset_ms : float
        Start of response window within trial. Default 100ms.
    response_duration_ms : float
        Duration of response window. Default 100ms.
    pattern_current : float
        Current injected to active input neurons (pA). Default 400.0.
    similarity_threshold : float
        Minimum similarity for a trial to count as correct. Default 0.7.
    """

    def __init__(
        self,
        n_patterns: int              = 10,
        pattern_sparsity: float      = 0.3,
        noise_level: float           = 0.3,
        n_learning_reps: int         = 20,
        pattern_duration_ms: float   = 200.0,
        response_onset_ms: float     = 100.0,
        response_duration_ms: float  = 100.0,
        pattern_current: float       = 400.0,
        similarity_threshold: float  = 0.7,
        dt: float                    = 0.1,
        seed: int                    = 42,
    ):
        self._n_patterns          = n_patterns
        self._pattern_sparsity    = pattern_sparsity
        self._noise_level         = noise_level
        self._n_learning_reps     = n_learning_reps
        self._pattern_duration_ms = pattern_duration_ms
        self._response_onset_ms   = response_onset_ms
        self._response_duration_ms = response_duration_ms
        self._pattern_current     = pattern_current
        self._similarity_threshold = similarity_threshold
        self._dt                  = dt
        self._seed                = seed

        # Set during setup()
        self._n_input    = None
        self._n_output   = None
        self._patterns   = None   # shape (n_patterns, n_input) binary
        self._trial_schedule = None   # list of (pattern_id, is_test) per trial

    @property
    def name(self) -> str:
        return "T2_PatternCompletion"

    @property
    def n_trials(self) -> int:
        if self._trial_schedule is None:
            return self._n_patterns * self._n_learning_reps + \
                   self._n_patterns * 10   # 10 test trials per pattern
        return len(self._trial_schedule)

    @property
    def trial_duration_ms(self) -> float:
        return self._pattern_duration_ms

    @property
    def learning_axis(self) -> str:
        return "associative"

    def setup(self, model: OIModel) -> None:
        """
        Generate patterns and build trial schedule.
        Patterns are fixed across all runs for reproducibility.
        """
        self._n_input  = model.n_input
        self._n_output = model.n_output
        self._dt       = model.dt

        rng = np.random.RandomState(self._seed)

        # Generate n_patterns sparse binary patterns over input population
        n_active = max(1, int(self._n_input * self._pattern_sparsity))
        self._patterns = np.zeros((self._n_patterns, self._n_input), dtype=np.float32)
        for i in range(self._n_patterns):
            active = rng.choice(self._n_input, size=n_active, replace=False)
            self._patterns[i, active] = 1.0

        # Build trial schedule: learning phase then test phase
        # Learning: each pattern n_learning_reps times, interleaved
        schedule = []
        pattern_ids = list(range(self._n_patterns)) * self._n_learning_reps
        rng.shuffle(pattern_ids)
        for pid in pattern_ids:
            schedule.append((pid, False))   # (pattern_id, is_test)

        # Test: each pattern 10 times with noise
        test_ids = list(range(self._n_patterns)) * 10
        rng.shuffle(test_ids)
        for pid in test_ids:
            schedule.append((pid, True))

        self._trial_schedule = schedule

        n_learning = sum(1 for _, t in schedule if not t)
        n_test     = sum(1 for _, t in schedule if t)
        print(f"  T2 setup: {self._n_patterns} patterns | "
              f"sparsity={self._pattern_sparsity} | "
              f"noise={self._noise_level}")
        print(f"  Learning trials: {n_learning} | Test trials: {n_test}")

    def generate_trial(self, trial_id: int, rng_key) -> List[Stimulus]:
        """Generate stimulus for one trial."""
        assert self._patterns is not None, "Call setup() first"

        pattern_id, is_test = self._trial_schedule[trial_id]
        pattern = self._patterns[pattern_id].copy()

        if is_test:
            # Add noise: flip noise_level fraction of bits
            rng = np.random.RandomState(
                int(jax.random.fold_in(rng_key, trial_id)[0]) % (2**31)
            )
            n_flip = max(1, int(self._n_input * self._noise_level))
            flip_idx = rng.choice(self._n_input, size=n_flip, replace=False)
            pattern[flip_idx] = 1.0 - pattern[flip_idx]

        n_steps = int(self._pattern_duration_ms / self._dt)
        stimuli = []
        for step in range(n_steps):
            t_ms = step * self._dt
            current     = np.zeros(self._n_input, dtype=np.float32)
            spike_train = np.zeros(self._n_input, dtype=np.float32)

            # Present pattern during full trial
            current = pattern * self._pattern_current

            stimuli.append(Stimulus(
                current     = current,
                spike_train = spike_train,
                t           = t_ms,
                label       = pattern_id,
            ))

        return stimuli

    def compute_score(
        self,
        response: "jnp.ndarray",
        trial_id: int,
        state_trace: List[ModelState],
    ) -> dict:
        """
        Score via Hamming similarity between model output and stored pattern.

        Model output = binary activity of output population during response window.
        Target = the stored pattern (projected to output size if needed).
        """
        pattern_id, is_test = self._trial_schedule[trial_id]
        pattern = self._patterns[pattern_id]

        # Project pattern to output size for comparison
        # Use first n_output elements of input pattern as target
        n_compare = min(self._n_output, self._n_input)
        target = pattern[:n_compare]

        if state_trace:
            # Use response window activity
            resp_start = int(self._response_onset_ms / self._dt)
            resp_end   = int(
                (self._response_onset_ms + self._response_duration_ms) / self._dt
            )
            resp_end = min(resp_end, len(state_trace))

            if resp_end > resp_start:
                # Mean activity during response window → binarise
                mean_activity = np.mean([
                    np.array(state_trace[i].spikes[:n_compare])
                    for i in range(resp_start, resp_end)
                ], axis=0)
                output_binary = (mean_activity > 0).astype(np.float32)
            else:
                output_binary = (np.array(response[:n_compare]) > 0).astype(np.float32)
        else:
            output_binary = (np.array(response[:n_compare]) > 0).astype(np.float32)

        # Hamming similarity
        matches    = float(np.sum(output_binary == target))
        similarity = matches / n_compare

        # Normalise: chance = pattern_sparsity, perfect = 1.0
        chance = self._pattern_sparsity
        norm_similarity = max(0.0, (similarity - chance) / (1.0 - chance))

        return {
            'accuracy':    float(norm_similarity),
            'similarity':  float(similarity),
            'pattern_id':  pattern_id,
            'is_test':     float(is_test),
        }

    def learning_index(self, trial_results: list) -> float:
        """Override: compute LI on test trials only."""
        test_results = [
            r for r in trial_results
            if r.scores.get('is_test', 0.0) > 0.5
        ]
        if not test_results:
            return 0.0
        return super().learning_index(test_results)
