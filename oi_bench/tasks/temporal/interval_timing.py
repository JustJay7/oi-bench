"""
Task T4 — Interval Timing

Benchmark axis: Temporal Learning
Biological analog: Scalar timing, Weber's law in neural circuits
(Gibbon 1977 Psych Review; Mello et al. 2015 Current Biology)

PROTOCOL
--------
Model receives a brief stimulus (onset cue), must maintain activity for
a target interval T_target, then produce an output burst.

Three target intervals tested: 200ms, 500ms, 1000ms.
Each interval gets n_trials_per_interval=50 trials.

SCORING
-------
Reproduced interval = time from cue offset to first output population burst.
Weber fraction = std(reproduced) / mean(reproduced).
Lower Weber fraction = more precise timing = better score.

Score per trial:
  'accuracy'     : 1 - |reproduced - T_target| / T_target  (clamped to [0,1])
  'weber_error'  : |reproduced - T_target| / T_target

References:
  Gibbon (1977) Psychological Review 84:279-325
  Mello et al. (2015) Current Biology 25:2913-2919
  Spec Section 5, Task T4
"""

from __future__ import annotations
import jax.numpy as jnp
import numpy as np
from typing import List

from oi_bench.core.protocol import BenchmarkTask
from oi_bench.core.types import Stimulus, ModelState
from oi_bench.core.adapter import OIModel


class IntervalTimingTask(BenchmarkTask):

    def __init__(
        self,
        target_intervals_ms: List[float] = [200.0, 500.0, 1000.0],
        n_trials_per_interval: int       = 50,
        cue_duration_ms: float           = 20.0,
        cue_current: float               = 400.0,
        cue_fraction: float              = 0.3,
        dt: float                        = 0.1,
    ):
        self._intervals           = target_intervals_ms
        self._n_per_interval      = n_trials_per_interval
        self._cue_dur             = cue_duration_ms
        self._cue_current         = cue_current
        self._cue_fraction        = cue_fraction
        self._dt                  = dt
        self._max_interval        = max(target_intervals_ms)
        self._trial_dur           = cue_duration_ms + self._max_interval + 200.0
        self._n_input             = None
        self._cue_neurons         = None

    @property
    def name(self) -> str:
        return "T4_IntervalTiming"

    @property
    def n_trials(self) -> int:
        return len(self._intervals) * self._n_per_interval

    @property
    def trial_duration_ms(self) -> float:
        return self._trial_dur

    @property
    def learning_axis(self) -> str:
        return "temporal"

    def setup(self, model: OIModel) -> None:
        self._n_input  = model.n_input
        self._n_output = model.n_output
        self._dt       = model.dt
        n_cue = max(1, int(self._n_input * self._cue_fraction))
        self._cue_neurons = np.arange(n_cue)
        print(f"  T4 setup: intervals={self._intervals}ms | "
              f"{self._n_per_interval} trials each")

    def _get_target(self, trial_id: int) -> float:
        interval_idx = trial_id // self._n_per_interval
        return self._intervals[interval_idx % len(self._intervals)]

    def generate_trial(self, trial_id: int, rng_key) -> List[Stimulus]:
        n_steps = int(self._trial_dur / self._dt)
        stimuli = []
        for step in range(n_steps):
            t_ms    = step * self._dt
            current = np.zeros(self._n_input, dtype=np.float32)
            if t_ms < self._cue_dur:
                current[self._cue_neurons] = self._cue_current
            stimuli.append(Stimulus(
                current     = current,
                spike_train = np.zeros(self._n_input, dtype=np.float32),
                t           = t_ms,
                label       = trial_id // self._n_per_interval,
            ))
        return stimuli

    def compute_score(self, response, trial_id, state_trace) -> dict:
        target = self._get_target(trial_id)
        if not state_trace:
            return {'accuracy': 0.0, 'weber_error': 1.0, 'target_ms': target}

        # Find first output burst after cue offset
        cue_end_step = int(self._cue_dur / self._dt)
        reproduced_ms = target   # default: assume no burst = max error

        for i in range(cue_end_step, len(state_trace)):
            if float(np.any(np.array(state_trace[i].spikes) > 0)):
                reproduced_ms = i * self._dt - self._cue_dur
                break

        weber_error = abs(reproduced_ms - target) / target
        accuracy    = float(max(0.0, 1.0 - weber_error))

        return {
            'accuracy':      accuracy,
            'weber_error':   float(weber_error),
            'reproduced_ms': float(reproduced_ms),
            'target_ms':     float(target),
        }
