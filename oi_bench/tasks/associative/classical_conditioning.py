"""
Task T1 — Classical Conditioning

Benchmark axis: Associative Learning
Biological analog: Eyeblink conditioning (Kagan et al. 2022, Neuron)

PROTOCOL
--------
Each trial is 400ms and consists of three phases:

  [0, 50ms)     — Baseline: no stimulus
  [50, 200ms)   — CS window: conditioned stimulus burst to input population A
  [150, 200ms)  — CS+US overlap window (conditioning trials only):
                    CS continues + US direct current to output population
  [200, 400ms)  — Response window: model output scored here

Trial structure over n_trials=200:
  Trials 0-149   : Conditioning trials (CS + US paired)
  Trials 150-199 : Test trials (CS alone, no US)

SCORING
-------
Conditioned Response (CR) = output population mean firing rate during
response window exceeds CR_threshold (default 20 Hz).

Score per trial:
  'accuracy'    : 1.0 if CR present on test trial, 0.0 otherwise
  'cr_rate'     : mean firing rate of output during response window (Hz)
  'cr_detected' : binary float {0.0, 1.0}

References:
  Kagan et al. (2022) Neuron 110:3952-3960
  Drew & Abbott (2006) PNAS 103:8876-8881
"""

from __future__ import annotations
import jax
import jax.numpy as jnp
import numpy as np
from typing import List

from oi_bench.core.protocol import BenchmarkTask
from oi_bench.core.types import Stimulus, ModelState, TrialResult
from oi_bench.core.adapter import OIModel


class ClassicalConditioningTask(BenchmarkTask):

    def __init__(
        self,
        n_trials: int               = 200,
        trial_duration_ms: float    = 400.0,
        cs_onset_ms: float          = 50.0,
        cs_duration_ms: float       = 150.0,
        us_onset_ms: float          = 150.0,
        us_duration_ms: float       = 50.0,
        response_onset_ms: float    = 200.0,
        response_duration_ms: float = 200.0,
        cs_current: float           = 400.0,
        us_current: float           = 500.0,
        cs_fraction: float          = 0.3,
        cr_threshold: float         = 20.0,
        n_conditioning_trials: int  = 150,
        dt: float                   = 0.1,
    ):
        self._n_trials              = n_trials
        self._trial_duration_ms     = trial_duration_ms
        self._cs_onset_ms           = cs_onset_ms
        self._cs_duration_ms        = cs_duration_ms
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
        self._n_input               = None
        self._n_output              = None
        self._cs_neurons            = None

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

    def setup(self, model: OIModel) -> None:
        self._n_input  = model.n_input
        self._n_output = model.n_output
        self._dt       = model.dt
        n_cs = max(1, int(self._n_input * self._cs_fraction))
        self._cs_neurons = np.arange(n_cs)
        print(f"  T1 setup: n_input={self._n_input} n_output={self._n_output}")
        print(f"  CS population: {n_cs} neurons ({self._cs_fraction*100:.0f}% of input)")
        print(f"  Conditioning trials: {self._n_conditioning} / {self._n_trials}")
        print(f"  CR threshold: {self._cr_threshold} Hz")

    def generate_trial(self, trial_id: int, rng_key) -> List[Stimulus]:
        assert self._n_input is not None, "Call setup() before generate_trial()"
        is_conditioning = trial_id < self._n_conditioning
        n_steps = int(self._trial_duration_ms / self._dt)
        stimuli = []
        for step in range(n_steps):
            t_ms        = step * self._dt
            current     = np.zeros(self._n_input, dtype=np.float32)
            spike_train = np.zeros(self._n_input, dtype=np.float32)
            cs_on  = self._cs_onset_ms
            cs_off = self._cs_onset_ms + self._cs_duration_ms
            if cs_on <= t_ms < cs_off:
                current[self._cs_neurons] = self._cs_current
            stimuli.append(Stimulus(
                current     = current,
                spike_train = spike_train,
                t           = t_ms,
                label       = int(is_conditioning),
            ))
        return stimuli

    def compute_score(
        self,
        response: "jnp.ndarray",
        trial_id: int,
        state_trace: List[ModelState],
        metadata: dict | None = None,
    ) -> dict:
        is_test_trial = trial_id >= self._n_conditioning

        if state_trace:
            resp_start   = int(self._response_onset_ms / self._dt)
            resp_end     = int(
                (self._response_onset_ms + self._response_duration_ms) / self._dt
            )
            resp_end     = min(resp_end, len(state_trace))
            window_dur_s = (resp_end - resp_start) * self._dt / 1000.0
            if window_dur_s > 0 and resp_end > resp_start:
                spike_sum = sum(
                    np.array(state_trace[i].spikes)
                    for i in range(resp_start, resp_end)
                )
                cr_rate = float(np.mean(spike_sum) / window_dur_s)
            else:
                cr_rate = float(np.mean(response))
        else:
            cr_rate = float(np.mean(response))

        cr_detected = float(cr_rate >= self._cr_threshold)
        accuracy    = cr_detected if is_test_trial else 1.0

        return {
            'accuracy':    float(accuracy),
            'cr_rate':     cr_rate,
            'cr_detected': cr_detected,
        }

    def us_current_for_step(self, t_ms: float, trial_id: int) -> np.ndarray:
        is_conditioning = trial_id < self._n_conditioning
        us_on  = self._us_onset_ms
        us_off = self._us_onset_ms + self._us_duration_ms
        if is_conditioning and us_on <= t_ms < us_off:
            return np.full(self._n_output, self._us_current, dtype=np.float32)
        return np.zeros(self._n_output, dtype=np.float32)
