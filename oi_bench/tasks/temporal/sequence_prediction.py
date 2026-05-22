"""
Task T3 — Sequence Prediction

Benchmark axis: Temporal Learning
Biological analog: Hippocampal sequence replay, cerebellar timing
(Dragoi & Tonegawa 2011, Nature; Leutgeb et al. 2005, Science)

PROTOCOL
--------
A repeating 5-element spike sequence A→B→C→D→E is presented during learning.
Each element is a distinct 20ms burst to a different subset of input neurons,
separated by 30ms ISI. Total sequence duration: 5 × (20+30) = 250ms.

Trial structure:
  Trials 0 to n_learning_trials-1   : Learning — full sequence A→B→C→D→E
  Trials n_learning_trials to end    : Test — truncated A→B→C only,
                                       score whether output fires in D/E windows

PLASTICITY DURING TEST
----------------------
is_learning_trial() returns False for test trials (trial_id >= n_learning).
The runner passes plasticity_scale=0.0 into synapse.update() during test
trials, freezing weights. Without this, test-phase STDP causes weight
explosion (0.64 → 0.96 in 10 trials) as the truncated stimulus drives
unbalanced LTP, corrupting test scores.

SCORING
-------
requires_spike_times=True so runner provides per-timestep population spike
counts in metadata['spike_counts_timeseries']. On test trials we check for
output activity in the D and E element windows after the truncation point.

Only the E window counts toward accuracy. D fires by passive FF bleedthrough
(C→D gap=30ms, tau_syn=15ms → exp(−30/15)=13.5% residual conductance, which
is sufficient to drive output above threshold regardless of learning). E is
2 hops from the last presented element (C→D→E): residual from C decays to
~0.5% by t=200ms, so E firing requires the D→E recurrent chain built by STDP.
Scoring D would give acc=0.5 to any untrained network (D always fires, E never
fires), making the metric indistinguishable from no learning.

Biological basis: Dragoi & Tonegawa (2011) distinguish pre-play (no learning)
from post-play (after learning) sequences. The E window captures this contrast;
the D window does not.

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
        self._n_input          = None
        self._element_neurons  = None
        self._model            = None

        self.record_diagnostics          = False
        self._diag_DE_weight             = []
        self._diag_D_spikes_in_E_window  = []
        self._diag_E_spikes              = []
        self._diag_E_spikes_test         = []
        self._diag_DE_weight_init        = None   # captured at setup() before any STDP

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

    @property
    def requires_spike_times(self) -> bool:
        return True

    def is_learning_trial(self, trial_id: int) -> bool:
        """Freeze plasticity during test trials to prevent STDP corruption."""
        return trial_id < self._n_learning

    def setup(self, model: OIModel) -> None:
        self._model    = model
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

        if self.record_diagnostics and hasattr(model, 'synapse'):
            D_neurons = self._element_neurons[self._truncate_at + 1]
            W0 = np.array(model.synapse.W.value)
            self._diag_DE_weight_init = float(np.mean(W0[D_neurons, :]))

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

    def _record_learning_diagnostics(self, metadata: dict | None) -> None:
        """Record D→E weight, D-bridge spikes, and E-window spikes after a learning trial."""
        D_idx = self._truncate_at + 1   # element index for D
        E_idx = self._truncate_at + 2   # element index for E

        # Mean feedforward weight from D-input neurons to output
        D_neurons = self._element_neurons[D_idx]
        W = np.array(self._model.synapse.W.value)
        self._diag_DE_weight.append(float(np.mean(W[D_neurons, :])))

        spike_ts = (metadata.get('spike_counts_timeseries')
                    if metadata is not None else None)
        if spike_ts is not None and len(spike_ts) > 0:
            # D fires at [D_onset, D_offset); 50ms after D fires = [D_offset, D_offset+50ms]
            d_offset_step = int(
                (D_idx * (self._element_dur + self._isi) + self._element_dur) / self._dt)
            bridge_end_step = d_offset_step + int(50.0 / self._dt)
            e_onset_step  = int(E_idx * (self._element_dur + self._isi) / self._dt)
            e_offset_step = int(
                (E_idx * (self._element_dur + self._isi) + self._element_dur) / self._dt)
            n = len(spike_ts)
            self._diag_D_spikes_in_E_window.append(
                float(np.sum(spike_ts[min(d_offset_step, n) : min(bridge_end_step, n)])))
            self._diag_E_spikes.append(
                float(np.sum(spike_ts[min(e_onset_step, n) : min(e_offset_step, n)])))
        else:
            self._diag_D_spikes_in_E_window.append(0.0)
            self._diag_E_spikes.append(0.0)

    def _record_test_diagnostics(self, metadata: dict | None) -> None:
        """Record E-window output spikes on test trials (A→B→C only, no E input)."""
        E_idx = self._truncate_at + 2
        spike_ts = (metadata.get('spike_counts_timeseries')
                    if metadata is not None else None)
        if spike_ts is not None and len(spike_ts) > 0:
            e_onset_step  = int(E_idx * (self._element_dur + self._isi) / self._dt)
            e_offset_step = int(
                (E_idx * (self._element_dur + self._isi) + self._element_dur) / self._dt)
            n = len(spike_ts)
            self._diag_E_spikes_test.append(
                float(np.sum(spike_ts[min(e_onset_step, n) : min(e_offset_step, n)])))
        else:
            self._diag_E_spikes_test.append(0.0)

    def compute_score(
        self,
        response,
        trial_id: int,
        state_trace: List[ModelState],
        metadata: dict | None = None,
    ) -> dict:
        is_test = trial_id >= self._n_learning

        if not is_test:
            if (self.record_diagnostics
                    and self._model is not None
                    and self._element_neurons is not None):
                self._record_learning_diagnostics(metadata)
            return {'accuracy': 1.0, 'timing_precision': 1.0,
                    'd_correct': 1.0, 'e_correct': 1.0}

        spike_ts = metadata.get('spike_counts_timeseries') \
                   if metadata is not None else None

        if self.record_diagnostics and self._element_neurons is not None:
            self._record_test_diagnostics(metadata)

        if spike_ts is None or len(spike_ts) == 0:
            mean_rate = float(np.mean(np.array(response)))
            score     = float(mean_rate > 0) * 0.5
            return {'accuracy': score, 'timing_precision': 0.0,
                    'd_correct': score, 'e_correct': score}

        def window_has_activity(elem_idx: int) -> float:
            if elem_idx >= self._n_elements:
                return 0.0
            onset   = elem_idx * (self._element_dur + self._isi)
            offset  = onset + self._element_dur
            s_start = int(onset  / self._dt)
            s_end   = min(int(offset / self._dt), len(spike_ts))
            if s_end <= s_start:
                return 0.0
            return float(np.any(spike_ts[s_start:s_end] > 0))

        d_correct = window_has_activity(self._truncate_at + 1)
        e_correct = window_has_activity(self._truncate_at + 2)
        # D fires by FF bleedthrough regardless of learning; only E requires
        # the learned D→E recurrent chain and is a genuine prediction signal.
        accuracy  = float(e_correct)

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
