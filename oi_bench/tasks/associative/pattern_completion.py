"""
Task T2 — Pattern Completion

Benchmark axis: Associative Learning
Biological analog: CA3 hippocampal attractor dynamics (Rolls 2007, Hippocampus)

PROTOCOL
--------
The network learns N=5 spatial patterns during a learning phase.
Each pattern is a random binary activation of 30% of input neurons.
During test, degraded versions (70% correct + 30% noise) are presented.
The model must reconstruct the full stored pattern from the partial cue.

Trial structure:
  Trials 0 to n_learning_trials-1  : Learning phase — full patterns
  Trials n_learning_trials to end   : Test phase — degraded patterns

SCORING
-------
During learning, we record the mean output population firing rate vector
for each pattern via exponential moving average (EMA, alpha=0.3).

alpha=0.3 is chosen deliberately:
  - Low alpha (0.1) tracks historical average — dominated by early
    learning responses before weights have converged. Centroids
    represent an average over the learning trajectory, not the
    converged representation.
  - High alpha (0.3) weights recent responses 3× more than 10 trials
    ago. Centroids track the network's current converged state, which
    is what the test phase will encounter.

During test, score by cosine similarity to stored centroids.
Correct = closest centroid matches presented pattern identity.

References:
  Rolls (2007) Hippocampus 17:1153-1173
  Hopfield (1982) PNAS 79:2554-2558
  Spec Section 5, Task T2
"""

from __future__ import annotations
import jax
import numpy as np
from typing import List

from oi_bench.core.protocol import BenchmarkTask
from oi_bench.core.types import Stimulus, ModelState
from oi_bench.core.adapter import OIModel


class PatternCompletionTask(BenchmarkTask):

    def __init__(
        self,
        n_patterns: int              = 5,
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
        self._n_patterns           = n_patterns
        self._pattern_sparsity     = pattern_sparsity
        self._noise_level          = noise_level
        self._n_learning_reps      = n_learning_reps
        self._pattern_duration_ms  = pattern_duration_ms
        self._response_onset_ms    = response_onset_ms
        self._response_duration_ms = response_duration_ms
        self._pattern_current      = pattern_current
        self._similarity_threshold = similarity_threshold
        self._dt                   = dt
        self._seed                 = seed
        self._n_input              = None
        self._n_output             = None
        self._patterns             = None
        self._trial_schedule       = None
        self._output_centroids     = None
        self._centroid_initialized = None

        # EMA alpha=0.3: weights recent responses 3× more than ~10 trials ago.
        # Centroids track converged late-learning state, not historical average.
        self._centroid_alpha = 0.3

    @property
    def name(self) -> str:
        return "T2_PatternCompletion"

    @property
    def n_trials(self) -> int:
        if self._trial_schedule is None:
            return (self._n_patterns * self._n_learning_reps +
                    self._n_patterns * 10)
        return len(self._trial_schedule)

    @property
    def trial_duration_ms(self) -> float:
        return self._pattern_duration_ms

    @property
    def learning_axis(self) -> str:
        return "associative"

    def is_learning_trial(self, trial_id: int) -> bool:
        if self._trial_schedule is None:
            return True
        _, is_test = self._trial_schedule[trial_id]
        return not is_test

    def setup(self, model: OIModel) -> None:
        self._n_input  = model.n_input
        self._n_output = model.n_output
        self._dt       = model.dt

        rng = np.random.RandomState(self._seed)
        n_active = max(1, int(self._n_input * self._pattern_sparsity))
        self._patterns = np.zeros(
            (self._n_patterns, self._n_input), dtype=np.float32
        )
        for i in range(self._n_patterns):
            active = rng.choice(self._n_input, size=n_active, replace=False)
            self._patterns[i, active] = 1.0

        # Build trial schedule: learning then test
        schedule = []
        pattern_ids = list(range(self._n_patterns)) * self._n_learning_reps
        rng.shuffle(pattern_ids)
        for pid in pattern_ids:
            schedule.append((pid, False))
        test_ids = list(range(self._n_patterns)) * 10
        rng.shuffle(test_ids)
        for pid in test_ids:
            schedule.append((pid, True))
        self._trial_schedule = schedule

        # Centroid accumulators — initialised to zero, first update sets value
        self._output_centroids     = np.zeros(
            (self._n_patterns, self._n_output), dtype=np.float64
        )
        self._centroid_initialized = np.zeros(self._n_patterns, dtype=bool)

        n_learning = sum(1 for _, t in schedule if not t)
        n_test     = sum(1 for _, t in schedule if t)
        print(f"  T2 setup: {self._n_patterns} patterns | "
              f"sparsity={self._pattern_sparsity} | noise={self._noise_level}")
        print(f"  Learning trials: {n_learning} | Test trials: {n_test}")

    def generate_trial(self, trial_id: int, rng_key) -> List[Stimulus]:
        assert self._patterns is not None, "Call setup() first"
        pattern_id, is_test = self._trial_schedule[trial_id]
        pattern = self._patterns[pattern_id].copy()

        if is_test:
            rng = np.random.RandomState(
                int(jax.random.fold_in(rng_key, trial_id)[0]) % (2**31)
            )
            n_flip   = max(1, int(self._n_input * self._noise_level))
            flip_idx = rng.choice(self._n_input, size=n_flip, replace=False)
            pattern[flip_idx] = 1.0 - pattern[flip_idx]

        n_steps = int(self._pattern_duration_ms / self._dt)
        stimuli = []
        for step in range(n_steps):
            stimuli.append(Stimulus(
                current     = pattern * self._pattern_current,
                spike_train = np.zeros(self._n_input, dtype=np.float32),
                t           = step * self._dt,
                label       = pattern_id,
            ))
        return stimuli

    def compute_score(
        self,
        response,
        trial_id: int,
        state_trace: List[ModelState],
        metadata: dict | None = None,
    ) -> dict:
        pattern_id, is_test = self._trial_schedule[trial_id]
        response_np = np.array(response, dtype=np.float64)

        if not is_test:
            # Update centroid via EMA — first update initialises directly
            if not self._centroid_initialized[pattern_id]:
                self._output_centroids[pattern_id] = response_np.copy()
                self._centroid_initialized[pattern_id] = True
            else:
                a = self._centroid_alpha
                self._output_centroids[pattern_id] = (
                    (1.0 - a) * self._output_centroids[pattern_id]
                    + a * response_np
                )
            return {
                'accuracy':   1.0,
                'confidence': 1.0,
                'pattern_id': pattern_id,
                'is_test':    0.0,
            }

        # Test trial: score by closest centroid (cosine similarity)
        if not np.any(self._centroid_initialized):
            return {
                'accuracy':   0.0,
                'confidence': 0.0,
                'pattern_id': pattern_id,
                'is_test':    1.0,
            }

        def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
            na = np.linalg.norm(a)
            nb = np.linalg.norm(b)
            if na < 1e-8 or nb < 1e-8:
                return 0.0
            return float(np.dot(a, b) / (na * nb))

        similarities = np.array([
            cosine_sim(response_np, self._output_centroids[i])
            if self._centroid_initialized[i] else -1.0
            for i in range(self._n_patterns)
        ])

        predicted = int(np.argmax(similarities))
        correct   = (predicted == pattern_id)

        return {
            'accuracy':   float(correct),
            'confidence': float(similarities[predicted]),
            'pattern_id': pattern_id,
            'is_test':    1.0,
        }

    def learning_index(self, trial_results: list) -> float:
        """Override: LI computed on test trials only."""
        test_results = [
            r for r in trial_results
            if r.scores.get('is_test', 0.0) > 0.5
        ]
        if not test_results:
            return 0.0
        return super().learning_index(test_results)
