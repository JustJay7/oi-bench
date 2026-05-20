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
        noise_std: float                 = 200.0,
        dt: float                        = 0.1,
    ):
        self._intervals            = target_intervals_ms
        self._n_per_interval       = n_trials_per_interval
        self._cue_dur              = cue_duration_ms
        self._cue_current          = cue_current
        self._tonic_output_current = tonic_output_current
        self._n_groups             = n_groups
        self._group_current        = group_current
        self._group_dur_ms         = group_dur_ms
        self._noise_std            = noise_std
        self._dt                   = dt
        self._noise_trial_id       = -1
        self._noise_array          = None

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

    @property
    def group_off_steps(self) -> 'np.ndarray':
        """Timestep index at the end of each group's burst window.
        Used by the eligibility scan to snapshot e_ff after each group's
        full synaptic contribution has accumulated (Izhikevich 2007: DA at
        burst time, not trial end).  Shape: (n_groups,) int32."""
        return self._group_off_steps

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
        # Skip cue + group 0 + group 1 + synaptic tail so group 1's
        # residual activity cannot win the max-window contest for
        # targets that should map to group 2+ (e.g. 200ms → group 2).
        self._exclusion_ms   = self._cue_dur + delta_t + self._group_dur_ms + 3.0 * tau_syn
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
        """Return group index whose center time is closest to T_target."""
        target = self._get_target(trial_id)
        return int(np.argmin(np.abs(self._group_times_ms - target)))

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
        """Tonic drive + per-trial Gaussian noise to break deterministic symmetry."""
        if self._noise_std > 0:
            if self._noise_trial_id != trial_id:
                n_steps = int(self._trial_dur / self._dt) + 1
                rng = np.random.default_rng(trial_id + 777)
                self._noise_array = rng.normal(
                    0.0, self._noise_std,
                    (n_steps, self._n_output)).astype(np.float32)
                self._noise_trial_id = trial_id
            step_idx = int(round(t_ms / self._dt))
            step_idx = min(step_idx, len(self._noise_array) - 1)
            return (np.full(self._n_output, self._tonic_output_current,
                            dtype=np.float32)
                    + self._noise_array[step_idx])
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

        # Find group window (after exclusion) with highest mean spike activity.
        # This makes reproduced_ms track which group has the strongest synapses,
        # enabling weight-based selectivity rather than first-spike detection.
        best_k    = -1
        best_rate = -1.0
        for k in range(self._n_groups):
            on  = int(self._group_on_steps[k])
            off = int(self._group_off_steps[k])
            if off <= self._exclusion_step:
                continue
            on = max(on, self._exclusion_step)
            window = spike_ts[on : min(off, len(spike_ts))]
            if len(window) > 0:
                rate = float(np.mean(window))
                if rate > best_rate:
                    best_rate = rate
                    best_k    = k
        reproduced_ms = (float(self._group_times_ms[best_k])
                         if best_k >= 0 else float(self._max_interval))

        weber_error = abs(reproduced_ms - target) / max(target, 1.0)
        accuracy    = float(max(0.0, 1.0 - weber_error))

        return {
            'accuracy':      accuracy,
            'weber_error':   float(weber_error),
            'reproduced_ms': float(reproduced_ms),
            'target_ms':     float(target),
        }

    def eligibility_for_target(
        self,
        e_ff: 'jnp.ndarray',
        target_ms: float,
        tau_e: float,
    ) -> 'jnp.ndarray':
        """
        Extract the naturally decayed eligibility trace for the target group.

        Implements the Izhikevich (2007) three-factor rule directly: the weight
        update uses e(T_trial), the trace value at trial end after natural
        exponential decay from each STDP event. No temporal un-decay correction
        is applied — per Frémaux & Gerstner (2016) and Gerstner et al. (2018),
        the rule is dw = e(t) · M_3rd(t), where e(t) is the currently decayed
        value. The natural decay gradient already assigns more credit to groups
        that fired recently (late groups retain more trace at trial end).

        On wrong trials the target group's natural trace is small because its
        synaptic weights are low (→ weak STDP coincidences → weak eligibility).
        The bootstrap LTP (da=0.5 applied by the caller) provides the upward
        signal needed to grow the target group out of its low-weight state.
        Only the target group rows are non-zero; all other rows are zeroed.
        """
        import jax.numpy as jnp
        import numpy as np

        npg = self._n_input // self._n_groups
        k_target = int(np.argmin(np.abs(self._group_times_ms - target_ms)))

        mask = np.zeros(self._n_input, dtype=np.float32)
        s = k_target * npg
        mask[s : s + npg] = 1.0

        mask_2d = jnp.array(mask)[:, None]
        return e_ff * mask_2d
