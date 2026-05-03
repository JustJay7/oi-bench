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
→ Response window (100ms): match or non-match scored here

SCORING
-------
requires_spike_times=True. Runner provides spike_counts_timeseries
shape (n_steps,) in metadata. We read only the response window
(last 100ms of the 800ms trial).

Response window scoring: first half of output = match detector,
second half = non-match detector. We use the fraction of trial
spikes that fell in the response window to weight the per-neuron
mean firing rate, isolating decision-period activity from
sample+delay activity.

References:
  Funahashi et al. (1989) J Neurophysiol 61:331-349
  Goldman-Rakic (1995) Neuron 14:477-485
  Spec Section 5, Task T5
"""

from __future__ import annotations
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

    @property
    def requires_spike_times(self) -> bool:
        """
        T5 scoring must read only the 100ms response window.
        Full-trial mean rate is dominated by 700ms of sample+delay
        activity unrelated to the match/non-match decision.
        """
        return True

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
        Score using response-window spike fraction weighting.

        spike_counts_timeseries: (n_steps,) population total per step.
        response: (n_output,) mean Hz over full trial.

        We compute what fraction of all trial spikes occurred in the
        response window, then weight the per-neuron response array by
        that fraction. This approximates reading only the response window
        without requiring per-neuron per-timestep traces.
        """
        _, _, is_match = self._trial_info[trial_id]
        spike_ts    = (metadata.get('spike_counts_timeseries')
                       if metadata is not None else None)
        response_np = np.array(response)
        half        = max(1, self._n_output // 2)

        if spike_ts is not None and len(spike_ts) > 0:
            total_steps  = len(spike_ts)
            resp_steps   = int(self._response_dur / self._dt)
            resp_start   = max(0, total_steps - resp_steps)
            total_spikes = float(np.sum(spike_ts)) + 1e-10
            resp_spikes  = float(np.sum(spike_ts[resp_start:])) + 1e-10
            resp_frac    = resp_spikes / total_spikes
            # Weight per-neuron rate by fraction of activity in response window
            weighted     = response_np * resp_frac
            match_rate    = float(np.mean(weighted[:half]))
            nonmatch_rate = float(np.mean(weighted[half:]))
        else:
            match_rate    = float(np.mean(response_np[:half]))
            nonmatch_rate = float(np.mean(response_np[half:]))

        responded_match = match_rate > nonmatch_rate
        correct         = (responded_match == is_match)

        return {
            'accuracy':    float(correct),
            'is_match':    float(is_match),
            'hit':         float(is_match and responded_match),
            'false_alarm': float((not is_match) and responded_match),
        }
