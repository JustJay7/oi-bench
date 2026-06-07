"""
Task T1 — Trace Conditioning

Benchmark axis: Associative Learning
Biological analog: Eyeblink trace conditioning (Solomon et al. 1986, Science)

PROTOCOL
--------
Each trial is 400ms:

  [0,   50ms)  — baseline, no stimulus
  [50, 100ms)  — CS burst (50ms) to input population
  [100,200ms)  — trace gap (100ms silence)
  [200,250ms)  — US current to output (conditioning trials only)
  [250,400ms)  — response window (150ms), no input

Trial structure over 200 trials:
  Trials 0-149   : conditioning (CS + US paired)
  Trials 150-174 : CS-alone test trials (25 trials)
  Trials 175-199 : blank catch trials (25 trials, no CS, no US)
  Interleaved randomly within the test phase.

Trace conditioning is biologically harder than delay conditioning:
  CAdEx: adaptation current w maintains a sub-threshold trace during the
         100ms gap, allowing CS-US association across the silence.
  LIF:   no adaptation → no persistent trace → weaker association.

SCORING
-------
d′ = z(hit_rate) − z(false_alarm_rate), log-linear corrected.
Final LI = d′ / 4.65, clipped [0,1]. (d′ max ≈ 4.65 at HR=0.99, FA=0.01.)

References:
  Solomon et al. (1986) Science 233:534-537
  Kishimoto et al. (2001) J Neurophysiol 86:1867-1875
  Spec Section 5, Task T1
"""

from __future__ import annotations
import numpy as np
from typing import List

from oi_bench.core.protocol import BenchmarkTask
from oi_bench.core.types import Stimulus, ModelState, TrialResult
from oi_bench.core.adapter import OIModel


def _probit(p: float) -> float:
    """Inverse normal CDF — Abramowitz & Stegun rational approximation."""
    p = float(np.clip(p, 0.001, 0.999))
    sign = 1.0 if p >= 0.5 else -1.0
    q = p if p >= 0.5 else 1.0 - p
    t = float(np.sqrt(-2.0 * np.log(1.0 - q)))
    c = [2.515517, 0.802853, 0.010328]
    d = [1.432788, 0.189269, 0.001308]
    num = c[0] + c[1] * t + c[2] * t * t
    den = 1.0 + d[0] * t + d[1] * t * t + d[2] * t * t * t
    return sign * (t - num / den)


class ClassicalConditioningTask(BenchmarkTask):

    def __init__(
        self,
        n_trials: int               = 200,
        trial_duration_ms: float    = 400.0,
        cs_onset_ms: float          = 50.0,
        cs_duration_ms: float       = 50.0,
        trace_gap_ms: float         = 75.0,
        us_onset_ms: float          = 175.0,
        us_duration_ms: float       = 50.0,
        response_onset_ms: float    = 225.0,
        response_duration_ms: float = 175.0,
        cs_current: float           = 400.0,
        us_current: float           = 500.0,
        cs_fraction: float          = 0.3,
        cr_threshold: float         = 10.0,
        n_conditioning_trials: int  = 50,
        dt: float                   = 0.1,
        seed: int                   = 42,
    ):
        self._n_trials              = n_trials
        self._trial_duration_ms     = trial_duration_ms
        self._cs_onset_ms           = cs_onset_ms
        self._cs_duration_ms        = cs_duration_ms
        self._trace_gap_ms          = trace_gap_ms
        self._us_onset_ms           = us_onset_ms
        self._us_duration_ms        = us_duration_ms
        self._response_onset_ms     = response_onset_ms
        self._response_duration_ms  = response_duration_ms
        self._cs_current            = cs_current
        self._us_current            = us_current
        self._cs_fraction           = cs_fraction
        self._cr_threshold          = cr_threshold
        self._n_conditioning        = n_conditioning_trials
        self._dt                    = dt
        self._seed                  = seed

        n_test  = n_trials - n_conditioning_trials
        n_cs    = (n_test * 2) // 3   # 100 CS-test
        n_blank = n_test - n_cs        # 50 blank
        self._n_cs_test  = n_cs
        self._n_blank    = n_blank

        self._n_input       = None
        self._n_output      = None
        self._cs_neurons    = None
        self._test_schedule = None  # list of 'cs' or 'blank' for test trials
        self._hits = []
        self._fas  = []

    @property
    def name(self) -> str:
        return "T1_ClassicalConditioning"

    @property
    def n_trials(self) -> int:
        return self._n_trials

    @property
    def trial_duration_ms(self) -> float:
        return self._trial_duration_ms

    @property
    def learning_axis(self) -> str:
        return "associative"

    @property
    def requires_spike_times(self) -> bool:
        return True

    def is_learning_trial(self, trial_id: int) -> bool:
        return trial_id < self._n_conditioning

    def setup(self, model: OIModel) -> None:
        self._n_input  = model.n_input
        self._n_output = model.n_output
        self._dt       = model.dt
        n_cs = max(1, int(self._n_input * self._cs_fraction))
        self._cs_neurons = np.arange(n_cs)

        rng = np.random.RandomState(self._seed)
        schedule = (['cs'] * self._n_cs_test + ['blank'] * self._n_blank)
        rng.shuffle(schedule)
        self._test_schedule = schedule
        self._hits = []
        self._fas  = []

        print(f"  T1 setup: {n_cs} CS neurons | "
              f"{self._n_conditioning} conditioning + "
              f"{self._n_cs_test} CS-test + {self._n_blank} blank trials")

    def generate_trial(self, trial_id: int, rng_key) -> List[Stimulus]:
        assert self._n_input is not None, "Call setup() first"
        is_conditioning = trial_id < self._n_conditioning
        if not is_conditioning:
            trial_type = self._test_schedule[trial_id - self._n_conditioning]
        else:
            trial_type = 'conditioning'

        n_steps = int(self._trial_duration_ms / self._dt)
        cs_on  = self._cs_onset_ms
        cs_off = cs_on + self._cs_duration_ms
        stimuli = []
        for step in range(n_steps):
            t_ms    = step * self._dt
            current = np.zeros(self._n_input, dtype=np.float32)
            if trial_type != 'blank' and cs_on <= t_ms < cs_off:
                current[self._cs_neurons] = self._cs_current
            stimuli.append(Stimulus(
                current     = current,
                spike_train = np.zeros(self._n_input, dtype=np.float32),
                t           = t_ms,
                label       = int(is_conditioning),
            ))
        return stimuli

    def us_current_for_step(self, t_ms: float, trial_id: int) -> np.ndarray:
        if trial_id >= self._n_conditioning:
            return np.zeros(self._n_output, dtype=np.float32)
        us_on  = self._us_onset_ms
        us_off = us_on + self._us_duration_ms
        if us_on <= t_ms < us_off:
            return np.full(self._n_output, self._us_current, dtype=np.float32)
        return np.zeros(self._n_output, dtype=np.float32)

    def compute_score(
        self,
        response,
        trial_id: int,
        state_trace: list,
        metadata: dict | None = None,
    ) -> dict:
        is_conditioning = trial_id < self._n_conditioning

        spike_ts = (metadata.get('spike_counts_timeseries')
                    if metadata is not None else None)
        if spike_ts is not None and len(spike_ts) > 0:
            resp_start = int(self._response_onset_ms / self._dt)
            resp_end   = min(int((self._response_onset_ms
                                  + self._response_duration_ms) / self._dt),
                             len(spike_ts))
            n_steps = resp_end - resp_start
            if n_steps > 0:
                window_dur_s = n_steps * self._dt / 1000.0
                cr_rate = (float(np.sum(spike_ts[resp_start:resp_end]))
                           / (self._n_output * window_dur_s))
            else:
                cr_rate = float(np.mean(np.array(response)))
        else:
            cr_rate = float(np.mean(np.array(response)))

        cr_detected = float(cr_rate >= self._cr_threshold)

        if is_conditioning:
            accuracy = 1.0
        else:
            test_idx   = trial_id - self._n_conditioning
            trial_type = self._test_schedule[test_idx]
            if trial_type == 'cs':
                self._hits.append(cr_detected)
                accuracy = cr_detected
            else:  # blank
                self._fas.append(cr_detected)
                accuracy = 1.0 - cr_detected
            if test_idx < 20:
                print(f"  T1 test[{test_idx:3d}] trial={trial_id} type={trial_type:5s} "
                      f"cr_rate={cr_rate:.1f}Hz threshold={self._cr_threshold:.1f}Hz "
                      f"detected={bool(cr_detected)} hits={len(self._hits)} fas={len(self._fas)}")

        return {
            'accuracy':    float(accuracy),
            'cr_rate':     float(cr_rate),
            'cr_detected': float(cr_detected),
        }

    def learning_index(self, trial_results: list) -> float:
        if not self._hits and not self._fas:
            return super().learning_index(trial_results)
        hit_rate = float(np.mean(self._hits)) if self._hits else 0.0
        fa_rate  = float(np.mean(self._fas))  if self._fas  else 0.0
        dprime   = _probit(hit_rate) - _probit(fa_rate)
        return float(np.clip(dprime / 4.65, 0.0, 1.0))
