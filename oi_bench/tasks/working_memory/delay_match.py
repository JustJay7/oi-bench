"""
Task T5 — Delay Match-to-Sample (DMS)

Benchmark axis: Working Memory
Biological analog: PFC delay-period activity, object working memory
(Funahashi et al. 1989 J Neurophysiol; Goldman-Rakic 1995 Neuron)

PROTOCOL
--------
Sample stimulus (one of 4 spatial patterns, 100ms)
→ Delay period (500ms, no input)
→ Test stimulus (same or different pattern, 100ms)
→ Response window (100ms): match or non-match

SCORING
-------
The runner passes response = mean firing rate over the full trial (Hz),
shape (n_output,). We split the output population in half:
  - First half  → match  detector
  - Second half → non-match detector

The half with higher mean firing rate is the model's response.
This is a population rate-code readout — substrate-agnostic and consistent
with the population rate code readout defined in spec Section 12 Q1.

Score per trial:
  accuracy    : 1.0 if correct response, 0.0 otherwise
  is_match    : 1.0 if trial is a match trial
  hit         : 1.0 if match trial and model responded match
  false_alarm : 1.0 if non-match trial and model responded match

References:
  Funahashi et al. (1989) J Neurophysiol 61:331-349
  Goldman-Rakic (1995) Neuron 14:477-485
  Spec Section 5, Task T5
"""

from __future__ import annotations
import jax
import jax.numpy as jnp
import numpy as np
from typing import List

from oi_bench.core.protocol import BenchmarkTask
from oi_bench.core.types import Stimulus, ModelState
from oi_bench.core.adapter import OIModel


class DelayMatchTask(BenchmarkTask):

    def __init__(
        self,
        n_patterns: int             = 4,
        sample_duration_ms: float   = 100.0,
        delay_duration_ms: float    = 500.0,
        test_duration_ms: float     = 100.0,
        response_duration_ms: float = 100.0,
        pattern_current: float      = 400.0,
        pattern_fraction: float     = 0.25,
        n_trials: int               = 200,
        match_prob: float           = 0.5,
        dt: float                   = 0.1,
        seed: int                   = 42,
    ):
        self._n_patterns      = n_patterns
        self._sample_dur      = sample_duration_ms
        self._delay_dur       = delay_duration_ms
        self._test_dur        = test_duration_ms
        self._response_dur    = response_duration_ms
        self._pattern_current = pattern_current
        self._pattern_frac    = pattern_fraction
        self._n_trials_val    = n_trials
        self._match_prob      = match_prob
        self._dt              = dt
        self._seed            = seed
        self._trial_dur       = (sample_duration_ms + delay_duration_ms +
                                 test_duration_ms + response_duration_ms)
        self._n_input         = None
        self._n_output        = None
        self._patterns        = None
        self._trial_info      = None

    @property
    def name(self) -> str:
        return "T5_DelayMatchToSample"

    @property
    def n_trials(self) -> int:
        return self._n_trials_val

    @property
    def trial_duration_ms(self) -> float:
        return self._trial_dur

    @property
    def learning_axis(self) -> str:
        return "working_memory"

    def setup(self, model: OIModel) -> None:
        self._n_input  = model.n_input
        self._n_output = model.n_output
        self._dt       = model.dt
        rng = np.random.RandomState(self._seed)
        n_active = max(1, int(self._n_input * self._pattern_frac))
        self._patterns = np.zeros((self._n_patterns, self._n_input), dtype=np.float32)
        for i in range(self._n_patterns):
            idx = rng.choice(self._n_input, size=n_active, replace=False)
            self._patterns[i, idx] = 1.0
        self._trial_info = []
        for _ in range(self._n_trials_val):
            sample_id = rng.randint(0, self._n_patterns)
            is_match  = rng.rand() < self._match_prob
            if is_match:
                test_id = sample_id
            else:
                others  = [i for i in range(self._n_patterns) if i != sample_id]
                test_id = rng.choice(others)
            self._trial_info.append((sample_id, test_id, is_match))
        n_match = sum(1 for _, _, m in self._trial_info if m)
        print(f"  T5 setup: {self._n_patterns} patterns | "
              f"delay={self._delay_dur}ms | "
              f"match trials: {n_match}/{self._n_trials_val}")

    def generate_trial(self, trial_id: int, rng_key) -> List[Stimulus]:
        sample_id, test_id, is_match = self._trial_info[trial_id]
        n_steps    = int(self._trial_dur / self._dt)
        sample_end = self._sample_dur
        test_start = self._sample_dur + self._delay_dur
        test_end   = test_start + self._test_dur
        stimuli    = []
        for step in range(n_steps):
            t_ms    = step * self._dt
            current = np.zeros(self._n_input, dtype=np.float32)
            if t_ms < sample_end:
                current = self._patterns[sample_id] * self._pattern_current
            elif test_start <= t_ms < test_end:
                current = self._patterns[test_id] * self._pattern_current
            stimuli.append(Stimulus(
                current     = current,
                spike_train = np.zeros(self._n_input, dtype=np.float32),
                t           = t_ms,
                label       = int(is_match),
            ))
        return stimuli

    def compute_score(
        self,
        response,
        trial_id: int,
        state_trace: list,
        metadata: dict | None = None,
    ) -> dict:
        """
        Score using population rate-code readout from response array.

        response = mean firing rate over full trial, shape (n_output,).
        First half of output = match detector.
        Second half of output = non-match detector.
        Whichever half has higher mean rate is the model's decision.

        This is always populated by the runner regardless of record_traces.
        """
        _, _, is_match = self._trial_info[trial_id]

        response_np = np.array(response)
        half        = self._n_output // 2

        match_rate    = float(np.mean(response_np[:half]))
        nonmatch_rate = float(np.mean(response_np[half:]))

        responded_match = match_rate > nonmatch_rate
        correct         = (responded_match == is_match)
        hit             = float(is_match and responded_match)
        false_alarm     = float((not is_match) and responded_match)

        return {
            'accuracy':    float(correct),
            'is_match':    float(is_match),
            'hit':         hit,
            'false_alarm': false_alarm,
        }
