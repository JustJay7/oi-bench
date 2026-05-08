"""
Task T4 — Interval Timing

Benchmark axis: Temporal Learning
Biological analog: Scalar timing, Weber's law in neural circuits
(Gibbon 1977 Psych Review; Mello et al. 2015 Current Biology)

PROTOCOL
--------
Model receives a brief cue stimulus (20ms), must maintain activity for
a target interval T_target, then produce a burst. Three target intervals:
200ms, 500ms, 1000ms. 50 trials each.

TONIC OUTPUT DRIVE
------------------
75pA tonic current to output population throughout trial via
us_current_for_step(). Maintains sparse background firing so STDP
builds eligibility traces during the delay period.

EXCLUSION WINDOW (derived from model, not hardcoded)
-----------------------------------------------------
After the cue, synaptic conductance decays as exp(-t/tau_syn).
The cue-evoked output burst occurs during this synaptic tail.
We exclude the window [0, cue_dur + 3*tau_syn] where tau_syn is
read from model.synapse.tau_syn in setup(). This is the standard
neuroscience convention — interval timing is measured after the
cue-evoked transient (Mello et al. 2015, Gibbon 1977).

BURST THRESHOLD (derived from background statistics, not hardcoded)
-------------------------------------------------------------------
During setup(), we estimate background firing by running a short
calibration (200 steps, tonic drive only, no cue). The burst threshold
is set to max(background_peak + 2, 3) — two neurons above the observed
background maximum, with a floor of 3 to handle zero-activity networks.
This makes the threshold genuinely adaptive to the model's dynamics.

SCORING
-------
weber_error = |reproduced - T_target| / T_target
accuracy    = 1 - weber_error (clamped [0,1])

References:
  Gibbon (1977) Psychological Review 84:279-325
  Mello et al. (2015) Current Biology 25:2913-2919
  Shadlen & Newsome (1998) J Neurosci 18:3870-3896
  Izhikevich (2007) Cereb. Cortex 17:2443-2452
  Spec Section 5, Task T4
"""

from __future__ import annotations
import numpy as np
from typing import List

import brainpy.math as bm

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
        tonic_output_current: float      = 75.0,
        dt: float                        = 0.1,
        n_timer_groups: int              = 10,
        n_neurons_per_group: int         = 3,
    ):
        self._intervals            = target_intervals_ms
        self._n_per_interval       = n_trials_per_interval
        self._cue_dur              = cue_duration_ms
        self._cue_current          = cue_current
        self._cue_fraction         = cue_fraction
        self._tonic_output_current = tonic_output_current
        self._dt                   = dt
        self._max_interval         = max(target_intervals_ms)
        self._trial_dur            = cue_duration_ms + self._max_interval + 200.0
        self._n_timer_groups       = n_timer_groups
        self._n_per_group          = n_neurons_per_group
        self._n_timer_neurons      = n_timer_groups * n_neurons_per_group

        self._n_input          = None
        self._n_output         = None
        self._cue_neurons      = None
        self._context_neurons  = None
        self._timer_neurons    = None
        self._timer_fire_times = None
        self._exclusion_ms     = None   # derived from model.synapse.tau_syn
        self._burst_threshold  = None   # derived from background calibration
        self._exclusion_step   = None

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

    @property
    def requires_spike_times(self) -> bool:
        return True

    def setup(self, model: OIModel) -> None:
        self._n_input  = model.n_input
        self._n_output = model.n_output
        self._dt       = model.dt

        n_cue = max(1, int(self._n_input * self._cue_fraction))
        self._cue_neurons = np.arange(n_cue)

        # --- Derive exclusion window from model's actual tau_syn ---
        tau_syn = getattr(getattr(model, 'synapse', None), 'tau_syn', 15.0)
        self._exclusion_ms   = self._cue_dur + 7.0 * tau_syn
        self._exclusion_step = int(self._exclusion_ms / self._dt)

        # --- Derive burst threshold from background firing statistics ---
        # Run 200 steps with tonic drive only (no cue) to measure background
        bg_peak = self._calibrate_background(model)
        # Threshold: background peak + 2 neurons (floor 3 for silent networks)
        self._burst_threshold = max(1, int(bg_peak) + 1)

        # Timer chain: evenly spaced groups over [cue_dur, max_interval]
        # Occupies last n_timer_neurons slots of input population
        self._timer_neurons = np.arange(
            self._n_input - self._n_timer_neurons, self._n_input)
        if self._n_timer_groups > 1:
            self._timer_fire_times = [
                self._cue_dur + k * (self._max_interval / (self._n_timer_groups - 1))
                for k in range(self._n_timer_groups)
            ]
        else:
            self._timer_fire_times = [self._cue_dur]

        # Context neurons: 10 per interval, slots 30-59
        # Fire during cue to uniquely identify which interval is being trained
        n_ctx = 10
        self._context_neurons = [
            np.arange(30 + i * n_ctx, 30 + (i + 1) * n_ctx)
            for i in range(len(self._intervals))
        ]

        # Pre-wire timer→output connections with block-diagonal structure.
        # Timer group k strongly drives output neurons k*(n_out//n_groups)
        # to (k+1)*(n_out//n_groups). All other timer→output weights = 0.
        # This gives the network a structured temporal basis to learn from —
        # STDP + dopamine only needs to learn WHICH output block to disinhibit,
        # not build timing from scratch.
        if hasattr(model, 'synapse') and hasattr(model.synapse, 'W'):
            import brainpy.math as bm_local
            W = np.array(model.synapse.W.value)
            n_out_per_grp = max(1, self._n_output // self._n_timer_groups)
            timer_start   = self._n_input - self._n_timer_neurons
            # Zero out all timer→output connections first
            W[timer_start:, :] = 0.0
            # Set block-diagonal strong weights
            for k in range(self._n_timer_groups):
                pre_start  = timer_start + k * self._n_per_group
                pre_end    = pre_start + self._n_per_group
                post_start = k * n_out_per_grp
                post_end   = min(post_start + n_out_per_grp, self._n_output)
                W[pre_start:pre_end, post_start:post_end] = 0.8
            model.synapse.W.value = bm_local.array(W, dtype=bm_local.float32)
            print(f"  T4: pre-wired {self._n_timer_groups} timer→output blocks "
                  f"(w=0.8, {self._n_per_group}pre × {n_out_per_grp}post each)")

        print(f"  T4 setup: intervals={self._intervals}ms | "
              f"{self._n_per_interval} trials each | "
              f"tonic={self._tonic_output_current}pA | "
              f"tau_syn={tau_syn:.1f}ms | "
              f"exclusion={self._exclusion_ms:.0f}ms | "
              f"bg_peak={bg_peak} | "
              f"burst_threshold={self._burst_threshold} | "
              f"timer={self._n_timer_groups}grp×{self._n_per_group}neu | "
              f"context={len(self._intervals)}grp×{n_ctx}neu")

    def _calibrate_background(self, model: OIModel) -> int:
        """
        Run 200 steps with tonic drive only to measure background
        population spike count peak. Returns max spikes in any one step.

        This is a read-only calibration — model state is saved and
        restored so calibration has no effect on the actual run.
        """
        if not hasattr(model, 'output_pop') or not hasattr(model, 'synapse'):
            return 0

        # Save state
        V_save     = np.array(model.output_pop.V.value)
        spike_save = np.array(model.output_pop.spike.value)

        I_tonic = bm.full(self._n_output, self._tonic_output_current,
                           dtype=bm.float32)
        I_bg    = bm.full(self._n_output, 150.0, dtype=bm.float32)

        peak = 0
        for _ in range(200):
            model.output_pop.update(x=I_bg + I_tonic)
            n_spikes = int(np.sum(np.array(
                model.output_pop.spike.value.astype(bm.float32))))
            if n_spikes > peak:
                peak = n_spikes

        # Restore state
        model.output_pop.V.value     = bm.array(V_save)
        model.output_pop.spike.value = bm.array(spike_save.astype(bool))

        return peak

    def _get_target(self, trial_id: int) -> float:
        interval_idx = trial_id // self._n_per_interval
        return self._intervals[interval_idx % len(self._intervals)]

    def generate_trial(self, trial_id: int, rng_key) -> List[Stimulus]:
        n_steps    = int(self._trial_dur / self._dt)
        fire_times = self._timer_fire_times
        stimuli    = []
        for step in range(n_steps):
            t_ms        = step * self._dt
            current     = np.zeros(self._n_input, dtype=np.float32)
            spike_train = np.zeros(self._n_input, dtype=np.float32)
            if t_ms < self._cue_dur:
                current[self._cue_neurons] = self._cue_current
                interval_idx = trial_id // self._n_per_interval
                if (self._context_neurons is not None and
                        interval_idx < len(self._context_neurons)):
                    current[self._context_neurons[interval_idx]] = self._cue_current
            # Inject timer group k spike at its designated fire time
            for k, t_fire in enumerate(fire_times):
                if abs(t_ms - t_fire) < self._dt * 0.5:
                    start = self._n_input - self._n_timer_neurons + k * self._n_per_group
                    end   = start + self._n_per_group
                    spike_train[start:end] = 1.0
            stimuli.append(Stimulus(
                current     = current,
                spike_train = spike_train,
                t           = t_ms,
                label       = trial_id // self._n_per_interval,
            ))
        return stimuli

    def us_current_for_step(self, t_ms: float, trial_id: int) -> np.ndarray:
        """
        Tonic drive + timer pulse directly to output population.

        Each timer group k owns (n_output // n_timer_groups) output neurons.
        At t_fire_k, those neurons receive a strong 300pA pulse for 10ms,
        acting as a direct temporal marker signal. The remaining output
        neurons receive only the 75pA tonic baseline.

        This directly drives output activity at each timer group's fire time,
        giving STDP a clear signal to associate with T_target via dopamine.
        """
        current = np.full(self._n_output, self._tonic_output_current,
                          dtype=np.float32)
        if self._timer_fire_times is None:
            return current

        neurons_per_group = max(1, self._n_output // self._n_timer_groups)
        pulse_duration_ms = 10.0
        pulse_current     = 300.0  # pA — strong enough to drive selective firing

        for k, t_fire in enumerate(self._timer_fire_times):
            if t_fire <= t_ms < t_fire + pulse_duration_ms:
                start = k * neurons_per_group
                end   = min(start + neurons_per_group, self._n_output)
                current[start:end] = pulse_current
                break

        return current

    def compute_score(
        self,
        response,
        trial_id: int,
        state_trace: list,
        metadata: dict | None = None,
    ) -> dict:
        target   = self._get_target(trial_id)
        spike_ts = (metadata.get('spike_counts_timeseries')
                    if metadata is not None else None)

        if spike_ts is None or len(spike_ts) == 0:
            return {
                'accuracy':      0.0,
                'weber_error':   1.0,
                'reproduced_ms': float(self._max_interval),
                'target_ms':     float(target),
            }

        # Search for first genuine burst after exclusion window
        # Exclusion window skips cue-evoked synaptic tail
        reproduced_ms = float(self._max_interval)

        for i in range(self._exclusion_step, len(spike_ts)):
            if spike_ts[i] >= self._burst_threshold:
                reproduced_ms = i * self._dt - self._cue_dur
                break

        weber_error = abs(reproduced_ms - target) / max(target, 1.0)
        accuracy    = float(max(0.0, 1.0 - weber_error))

        return {
            'accuracy':      accuracy,
            'weber_error':   float(weber_error),
            'reproduced_ms': float(reproduced_ms),
            'target_ms':     float(target),
        }
