"""
Task T2 — Pattern Completion

Benchmark axis: Associative Learning
Biological analog: CA3 hippocampal attractor dynamics (Rolls 2007, Hippocampus)

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

    @property
    def name(self) -> str:
        return "T2_PatternCompletion"

    @property
    def n_trials(self) -> int:
        if self._trial_schedule is None:
            return self._n_patterns * self._n_learning_reps + self._n_patterns * 10
        return len(self._trial_schedule)

    @property
    def trial_duration_ms(self) -> float:
        return self._pattern_duration_ms

    @property
    def learning_axis(self) -> str:
        return "associative"

    def setup(self, model: OIModel) -> None:
        self._n_input  = model.n_input
        self._n_output = model.n_output
        self._dt       = model.dt
        rng = np.random.RandomState(self._seed)
        n_active = max(1, int(self._n_input * self._pattern_sparsity))
        self._patterns = np.zeros((self._n_patterns, self._n_input), dtype=np.float32)
        for i in range(self._n_patterns):
            active = rng.choice(self._n_input, size=n_active, replace=False)
            self._patterns[i, active] = 1.0
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
            t_ms = step * self._dt
            stimuli.append(Stimulus(
                current     = pattern * self._pattern_current,
                spike_train = np.zeros(self._n_input, dtype=np.float32),
                t           = t_ms,
                label       = pattern_id,
            ))
        return stimuli

    def compute_score(
        self,
        response: "jnp.ndarray",
        trial_id: int,
        state_trace: List[ModelState],
        metadata: dict | None = None,
    ) -> dict:
        pattern_id, is_test = self._trial_schedule[trial_id]
        pattern   = self._patterns[pattern_id]
        n_compare = min(self._n_output, self._n_input)
        target    = pattern[:n_compare]
        if state_trace:
            resp_start = int(self._response_onset_ms / self._dt)
            resp_end   = int(
                (self._response_onset_ms + self._response_duration_ms) / self._dt
            )
            resp_end = min(resp_end, len(state_trace))
            if resp_end > resp_start:
                mean_activity = np.mean([
                    np.array(state_trace[i].spikes[:n_compare])
                    for i in range(resp_start, resp_end)
                ], axis=0)
                output_binary = (mean_activity > 0).astype(np.float32)
            else:
                output_binary = (np.array(response[:n_compare]) > 0).astype(np.float32)
        else:
            output_binary = (np.array(response[:n_compare]) > 0).astype(np.float32)
        matches         = float(np.sum(output_binary == target))
        similarity      = matches / n_compare
        chance          = self._pattern_sparsity
        norm_similarity = max(0.0, (similarity - chance) / (1.0 - chance))
        return {
            'accuracy':   float(norm_similarity),
            'similarity': float(similarity),
            'pattern_id': pattern_id,
            'is_test':    float(is_test),
        }

    def learning_index(self, trial_results: list) -> float:
        test_results = [r for r in trial_results if r.scores.get('is_test', 0.0) > 0.5]
        if not test_results:
            return 0.0
        return super().learning_index(test_results)
