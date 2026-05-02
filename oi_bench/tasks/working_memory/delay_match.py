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

Match  → output population A should be active
Non-match → output population B should be active

This is the hardest task and most sensitive to fractional memory depth.
The delay period requires sustained activity to bridge the gap.

SCORING
-------
d-prime (signal detection theory):
  d' = Z(hit_rate) - Z(false_alarm_rate)

Score per trial:
  'accuracy'    : 1.0 if correct response, 0.0 otherwise
  'is_match'    : 1.0 if trial is a match trial
  'hit'         : 1.0 if match trial and model responded match
  'false_alarm' : 1.0 if non-match trial and model responded match

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
        self._patterns        = None
        self._trial_info      = None   # list of (sample_id, test_id, is_match)

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

        # Pre-generate trial info
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
        n_steps = int(self._trial_dur / self._dt)
        sample_end  = self._sample_dur
        test_start  = self._sample_dur + self._delay_dur
        test_end    = test_start + self._test_dur

        stimuli = []
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

    def compute_score(self, response, trial_id, state_trace) -> dict:
        _, _, is_match = self._trial_info[trial_id]

        if not state_trace:
            return {'accuracy': 0.5, 'is_match': float(is_match),
                    'hit': 0.0, 'false_alarm': 0.0}

        # Response window: last response_dur ms
        resp_start = int((self._trial_dur - self._response_dur) / self._dt)
        resp_end   = len(state_trace)

        # Model output: mean activity of first half (match) vs second half (non-match)
        half = self._n_output // 2
        match_activity    = 0.0
        nonmatch_activity = 0.0
        n_resp = 0

        for i in range(resp_start, resp_end):
            spikes = np.array(state_trace[i].spikes)
            match_activity    += float(np.mean(spikes[:half]))
            nonmatch_activity += float(np.mean(spikes[half:]))
            n_resp += 1

        if n_resp > 0:
            match_activity    /= n_resp
            nonmatch_activity /= n_resp

        responded_match = match_activity > nonmatch_activity
        correct = (responded_match == is_match)

        hit          = float(is_match and responded_match)
        false_alarm  = float((not is_match) and responded_match)

        return {
            'accuracy':     float(correct),
            'is_match':     float(is_match),
            'hit':          hit,
            'false_alarm':  false_alarm,
        }
