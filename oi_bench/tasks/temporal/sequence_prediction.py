"""
Task T3 — Sequence Prediction

Benchmark axis: Temporal Learning
Biological analog: Hippocampal sequence replay, cerebellar timing
(Dragoi & Tonegawa 2011, Nature; Leutgeb et al. 2005, Science)

References:
  Dragoi & Tonegawa (2011) Nature 469:397-401
  Drew & Abbott (2006) PNAS 103:8876-8881
  Spec Section 5, Task T3
"""

from __future__ import annotations
import jax
import jax.numpy as jnp
import numpy as np
from typing import List

from oi_bench.core.protocol import BenchmarkTask
from oi_bench.core.types import Stimulus, ModelState, TrialResult
from oi_bench.core.adapter import OIModel


class SequencePredictionTask(BenchmarkTask):

    def __init__(
        self,
        n_elements: int             = 5,
        element_duration_ms: float  = 20.0,
        isi_ms: float               = 30.0,
        n_learning_trials: int      = 200,
        n_test_trials: int          = 100,
        truncate_at: int            = 2,
        element_current: float      = 400.0,
        element_fraction: float     = 0.15,
        timing_tolerance_ms: float  = 10.0,
        dt: float                   = 0.1,
        seed: int                   = 42,
    ):
        self._n_elements       = n_elements
        self._element_dur      = element_duration_ms
        self._isi              = isi_ms
        self._n_learning       = n_learning_trials
        self._n_test           = n_test_trials
        self._truncate_at      = truncate_at
        self._element_current  = element_current
        self._element_fraction = element_fraction
        self._timing_tolerance = timing_tolerance_ms
        self._dt               = dt
        self._seed             = seed
        self._full_dur         = n_elements * (element_duration_ms + isi_ms)
        self._trunc_dur        = (truncate_at + 1) * (element_duration_ms + isi_ms)
        self._n_input          = None
        self._element_neurons  = None

    @property
    def name(self) -> str:
        return "T3_SequencePrediction"

    @property
    def n_trials(self) -> int:
        return self._n_learning + self._n_test

    @property
    def trial_duration_ms(self) -> float:
        return self._full_dur

    @property
    def learning_axis(self) -> str:
        return "temporal"

    def setup(self, model: OIModel) -> None:
        self._n_input  = model.n_input
        self._n_output = model.n_output
        self._dt       = model.dt
        rng = np.random.RandomState(self._seed)
        n_per_element = max(1, int(self._n_input * self._element_fraction))
        all_idx = np.arange(self._n_input)
        rng.shuffle(all_idx)
        self._element_neurons = []
        for i in range(self._n_elements):
            start = (i * n_per_element) % self._n_input
            end   = start + n_per_element
            if end <= self._n_input:
                self._element_neurons.append(all_idx[start:end])
            else:
                idx = np.concatenate([all_idx[start:], all_idx[:end - self._n_input]])
                self._element_neurons.append(idx)
        print(f"  T3 setup: {self._n_elements}-element sequence | "
              f"each element: {n_per_element} neurons | "
              f"truncate at element {self._truncate_at}")
        print(f"  Learning: {self._n_learning} | Test: {self._n_test} trials")

    def generate_trial(self, trial_id: int, rng_key) -> List[Stimulus]:
        is_test   = trial_id >= self._n_learning
        n_present = self._truncate_at + 1 if is_test else self._n_elements
        n_steps   = int(self._full_dur / self._dt)
        stimuli   = []
        for step in range(n_steps):
            t_ms    = step * self._dt
            current = np.zeros(self._n_input, dtype=np.float32)
            for elem_idx in range(n_present):
                onset  = elem_idx * (self._element_dur + self._isi)
                offset = onset + self._element_dur
                if onset <= t_ms < offset:
                    current[self._element_neurons[elem_idx]] = self._element_current
                    break
            stimuli.append(Stimulus(
                current     = current,
                spike_train = np.zeros(self._n_input, dtype=np.float32),
                t           = t_ms,
                label       = int(is_test),
            ))
        return stimuli

    def compute_score(
        self,
        response: "jnp.ndarray",
        trial_id: int,
        state_trace: List[ModelState],
        metadata: dict | None = None,
    ) -> dict:
        is_test = trial_id >= self._n_learning
        if not is_test:
            return {'accuracy': 1.0, 'timing_precision': 1.0,
                    'd_correct': 1.0, 'e_correct': 1.0}
        if not state_trace:
            return {'accuracy': 0.0, 'timing_precision': 0.0,
                    'd_correct': 0.0, 'e_correct': 0.0}
        d_idx = self._truncate_at + 1
        e_idx = self._truncate_at + 2

        def element_window_active(elem_idx: int) -> float:
            if elem_idx >= self._n_elements:
                return 0.0
            onset   = elem_idx * (self._element_dur + self._isi)
            offset  = onset + self._element_dur
            s_start = int(onset  / self._dt)
            s_end   = int(offset / self._dt)
            s_end   = min(s_end, len(state_trace))
            if s_end <= s_start:
                return 0.0
            spikes_in_window = sum(
                float(np.any(np.array(state_trace[i].spikes) > 0))
                for i in range(s_start, s_end)
            )
            return float(spikes_in_window > 0)

        d_correct = element_window_active(d_idx)
        e_correct = element_window_active(e_idx)
        accuracy  = float((d_correct + e_correct) / 2.0)
        return {
            'accuracy':         accuracy,
            'timing_precision': accuracy,
            'd_correct':        d_correct,
            'e_correct':        e_correct,
        }

    def learning_index(self, trial_results: list) -> float:
        test_results = [r for r in trial_results if r.trial_id >= self._n_learning]
        if not test_results:
            return 0.0
        return super().learning_index(test_results)
