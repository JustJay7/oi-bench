"""
Task T4 — Interval Timing

Benchmark axis: Temporal Learning
Biological analog: Scalar timing, Weber's law, time cells
(Gibbon 1977 Psych Review; MacDonald et al. 2011 Science;
 Mello et al. 2015 Current Biology; Matell & Meck 2004 Cortex)

ARCHITECTURE — DISCRETE POPULATION PACKETS + THREE-FACTOR STDP
---------------------------------------------------------------
The input population is divided into n_groups sequential subgroups.
Each group fires for a brief window (group_dur_ms) at a fixed delay
post-cue, covering [0, max_interval] uniformly. Between activations,
input is silent.

This creates a sparse temporal code: at any moment only one group is
active, producing a clean population burst through the FF synapse.
The output population (with I_bg=0) only fires when driven by an
active group — no spontaneous background activity.

Three-factor STDP + dopamine (Izhikevich 2007):
  - Eligibility trace accumulates during group → output co-activation
  - Dopamine delivered when output bursts near T_target
  - Over trials, synapses from the T_target group are selectively
    potentiated; other groups are depressed or neutral

CRITICAL PARAMETERS
-------------------
I_background=0 in the T4 model (set in run.py):
  CAdEx fires at ~600pA from rest. With I_bg=150pA, spontaneous
  firing drowns the group signal. With I_bg=0 and I_us=75pA,
  output is silent between groups and only fires during group
  activations where I_syn peak > 600pA threshold.

group_current=3000pA, n_neurons_per_group=10, conn_prob=1.0:
  10 neurons × 3000pA → I_syn peak ~625pA > 600pA CAdEx threshold.
  All-to-all connectivity ensures every output neuron receives the
  full group signal reliably.

BURST DETECTION
---------------
burst_threshold calibrated from 500 steps of silent (inter-group)
activity. Burst = first step after exclusion where population spike
count exceeds baseline. With I_bg=0 this is typically threshold=1.

SCORING
-------
weber_error = |reproduced - T_target| / T_target
accuracy    = max(0, 1 - weber_error)

References:
  Gibbon (1977) Psychological Review 84:279-325
  MacDonald et al. (2011) Science 331:1117-1120
  Matell & Meck (2004) Cortex 40:247-273
  Izhikevich (2007) Cereb. Cortex 17:2443-2452
  Mello et al. (2015) Current Biology 25:2913-2919
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
        cue_current: float               = 800.0,
        tonic_output_current: float      = 75.0,
        n_groups: int                    = 10,
        group_current: float             = 800.0,
        group_dur_ms: float              = 30.0,
        dt: float                        = 0.1,
    ):
        """
        Parameters
        ----------
        target_intervals_ms : list of float
            Target intervals to learn (ms).
        n_trials_per_interval : int
            Training trials per interval.
        cue_duration_ms : float
            Duration of cue burst (ms).
        cue_current : float
            Current to ALL input neurons during cue (pA).
        tonic_output_current : float
            Tonic drive to output population (pA). Sub-threshold.
            With I_bg=0 keeps output near rest without spontaneous firing.
        n_groups : int
            Number of sequential input groups. Must divide n_input.
            Default 10 groups × 10 neurons for n_input=100.
        group_current : float
            Drive current to active group neurons (pA).
            Must produce I_syn > CAdEx threshold (~600pA from rest).
            Default 3000pA → I_syn peak ~625pA.
        group_dur_ms : float
            Duration each group is active (ms). Default 30ms.
        dt : float
            Simulation timestep (ms).
        """
        self._intervals            = target_intervals_ms
        self._n_per_interval       = n_trials_per_interval
        self._cue_dur              = cue_duration_ms
        self._cue_current          = cue_current
        self._tonic_output_current = tonic_output_current
        self._n_groups             = n_groups
        self._group_current        = group_current
        self._group_dur_ms         = group_dur_ms
        self._dt                   = dt

        self._max_interval = float(max(target_intervals_ms))
        self._trial_dur    = cue_duration_ms + self._max_interval + 200.0

        self._n_input          = None
        self._n_output         = None
        self._group_neurons    = None   # list of arrays, one per group
        self._group_on_steps   = None   # (n_groups,) int
        self._group_off_steps  = None   # (n_groups,) int
        self._group_times_ms   = None   # (n_groups,) float — centre time post-cue
        self._exclusion_ms     = None
        self._exclusion_step   = None
        self._burst_threshold  = None
        self._I_in_precomputed = None   # (n_steps, n_input) float32

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

        assert self._n_input % self._n_groups == 0, (
            f"n_input={self._n_input} must be divisible by n_groups={self._n_groups}"
        )
        npg = self._n_input // self._n_groups

        # Group neuron assignments: group k = neurons [k*npg : (k+1)*npg]
        self._group_neurons = [
            np.arange(k * npg, (k + 1) * npg) for k in range(self._n_groups)
        ]

        # Group fire times: evenly spaced across [0, max_interval]
        # Group k fires at: cue_dur + k * delta_t
        delta_t = self._max_interval / self._n_groups
        cue_step  = int(self._cue_dur / self._dt)
        dur_steps = int(self._group_dur_ms / self._dt)

        self._group_on_steps  = np.array([
            cue_step + int(k * delta_t / self._dt)
            for k in range(self._n_groups)
        ], dtype=np.int32)
        self._group_off_steps = self._group_on_steps + dur_steps
        self._group_times_ms  = np.array([
            k * delta_t + self._group_dur_ms / 2.0
            for k in range(self._n_groups)
        ], dtype=np.float32)

        # Pre-compute stimulus array
        n_steps = int(self._trial_dur / self._dt)
        self._I_in_precomputed = self._build_stimulus_array(n_steps)

        # Exclusion window
        tau_syn = getattr(getattr(model, 'synapse', None), 'tau_syn', 15.0)
        self._exclusion_ms   = self._cue_dur + 3.0 * tau_syn
        self._exclusion_step = int(self._exclusion_ms / self._dt)

        # Burst threshold from silent baseline
        self._burst_threshold = self._calibrate_threshold(model)

        if hasattr(model, 'disable_timer'):
            model.disable_timer()

        print(
            f"  T4 setup: intervals={self._intervals}ms | "
            f"{self._n_per_interval} trials each | "
            f"tonic={self._tonic_output_current}pA | "
            f"tau_syn={tau_syn:.1f}ms | "
            f"exclusion={self._exclusion_ms:.0f}ms | "
            f"burst_threshold={self._burst_threshold} | "
            f"n_groups={self._n_groups} | "
            f"delta_t={delta_t:.0f}ms | "
            f"group_current={self._group_current}pA | "
            f"group_dur={self._group_dur_ms}ms"
        )

    def _build_stimulus_array(self, n_steps: int) -> np.ndarray:
        """
        Pre-compute input current array.
        Cue: I_cue to all neurons for cue_dur.
        Post-cue: sequential group activations, silence otherwise.
        Shape: (n_steps, n_input), dtype float32.
        """
        I = np.zeros((n_steps, self._n_input), dtype=np.float32)
        cue_steps = int(self._cue_dur / self._dt)

        # Cue burst to all input neurons
        I[:cue_steps, :] = self._cue_current

        # Sequential group activations
        for k in range(self._n_groups):
            on  = self._group_on_steps[k]
            off = self._group_off_steps[k]
            if off <= n_steps:
                I[on:off, self._group_neurons[k]] = self._group_current
            elif on < n_steps:
                I[on:n_steps, self._group_neurons[k]] = self._group_current

        return I

    def _calibrate_threshold(self, model: OIModel) -> int:
        """
        Measure peak output spike count during a silent inter-group
        period (500 steps, no input drive, only tonic output current).
        burst_threshold = peak + 1.

        With I_bg=0 and I_us=75pA, CAdEx is silent → peak=0 → threshold=1.
        This ensures any group-driven burst is detected immediately.
        """
        if not hasattr(model, 'output_pop'):
            return 1

        V_save     = np.array(model.output_pop.V.value)
        spike_save = np.array(model.output_pop.spike.value)

        I_total = bm.full(
            self._n_output,
            getattr(model, '_I_bg', 0.0) + self._tonic_output_current,
            dtype=bm.float32,
        )
        peak = 0
        for _ in range(500):
            model.output_pop.update(x=I_total)
            n_sp = int(np.sum(np.array(
                model.output_pop.spike.value.astype(bm.float32))))
            if n_sp > peak:
                peak = n_sp

        model.output_pop.V.value     = bm.array(V_save)
        model.output_pop.spike.value = bm.array(spike_save.astype(bool))
        return max(1, peak + 1)

    def _get_target(self, trial_id: int) -> float:
        return self._intervals[
            (trial_id // self._n_per_interval) % len(self._intervals)]

    def _target_group(self, trial_id: int) -> int:
        """Return group index whose fire time is closest to T_target."""
        target  = self._get_target(trial_id)
        delta_t = self._max_interval / self._n_groups
        k_star  = int(round(target / delta_t)) - 1
        return min(max(k_star, 0), self._n_groups - 1)

    def generate_trial(self, trial_id: int, rng_key) -> List[Stimulus]:
        """Return pre-computed stimulus list. Same every trial."""
        assert self._I_in_precomputed is not None, "Call setup() first"
        n_steps = self._I_in_precomputed.shape[0]
        return [
            Stimulus(
                current     = self._I_in_precomputed[step],
                spike_train = np.zeros(self._n_input, dtype=np.float32),
                t           = step * self._dt,
                label       = trial_id // self._n_per_interval,
            )
            for step in range(n_steps)
        ]

    def us_current_for_step(self, t_ms: float, trial_id: int) -> np.ndarray:
        """Sub-threshold tonic drive to output. Keeps neurons near rest."""
        return np.full(self._n_output, self._tonic_output_current,
                       dtype=np.float32)

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
