"""
Task T5 — SSA Oddball Detection

Benchmark axis: Stimulus-Specific Adaptation / Sensory Memory
Biological analog: Mismatch negativity, auditory oddball (Ulanovsky et al. 2003)

PROTOCOL
--------
20 blocks, each 2500ms (50 stimuli × 50ms SOA).
Each stimulus: 30ms active (400pA) + 20ms silence.

Two patterns — S1 (neurons 0-29) and S2 (neurons 30-59):
  Blocks 0-9:  S1=standard (80%), S2=deviant (20%)
  Blocks 10-19: S2=standard (80%), S1=deviant (20%)
Role swap ensures neither pattern has an intrinsic advantage.

MECHANISM
---------
CAdEx: adaptation current (w) builds up during repeated standard stimuli,
       reducing response amplitude. Deviants (less adapted) produce larger
       responses → SSA signal. LIF has no adaptation → flat response.

SCORING
-------
CSI = (mean_deviant_rate − mean_standard_rate) /
      (mean_deviant_rate + mean_standard_rate)

CSI ∈ [-1, 1]. Positive = suppression to standard (correct SSA).
Final accuracy = (CSI + 1) / 2, maps to [0, 1] with 0.5 = no adaptation.

References:
  Ulanovsky et al. (2003) Nature Neuroscience 6:391-398
  Farley et al. (2010) J Neurosci 30:2555-2566
  Spec Section 5, Task T5
"""

from __future__ import annotations
import numpy as np
from typing import List

from oi_bench.core.protocol import BenchmarkTask
from oi_bench.core.types import Stimulus
from oi_bench.core.adapter import OIModel


class OddballDetectionTask(BenchmarkTask):

    def __init__(
        self,
        n_blocks: int            = 20,
        n_stimuli_per_block: int = 50,
        stimulus_duration_ms: float = 30.0,
        isi_ms: float            = 20.0,
        standard_prob: float     = 0.8,
        pattern_neurons: int     = 30,
        stimulus_current: float  = 400.0,
        dt: float                = 0.1,
        seed: int                = 42,
    ):
        self._n_blocks          = n_blocks
        self._n_per_block       = n_stimuli_per_block
        self._stim_dur_ms       = stimulus_duration_ms
        self._isi_ms            = isi_ms
        self._soa_ms            = stimulus_duration_ms + isi_ms
        self._standard_prob     = standard_prob
        self._pattern_neurons   = pattern_neurons
        self._stim_current      = stimulus_current
        self._dt                = dt
        self._seed              = seed
        self._trial_duration_ms = n_stimuli_per_block * (stimulus_duration_ms + isi_ms)

        self._n_input           = None
        self._n_output          = None
        self._s1_neurons        = None
        self._s2_neurons        = None
        self._block_sequences   = None  # list of lists: [(pat_id, is_deviant), ...]

        self._all_standard_rates = []
        self._all_deviant_rates  = []

    @property
    def name(self) -> str:
        return "T5_OddballDetection"

    @property
    def n_trials(self) -> int:
        return self._n_blocks

    @property
    def trial_duration_ms(self) -> float:
        return self._trial_duration_ms

    @property
    def learning_axis(self) -> str:
        return "sensory_memory"

    @property
    def requires_spike_times(self) -> bool:
        return True

    def setup(self, model: OIModel) -> None:
        self._n_input  = model.n_input
        self._n_output = model.n_output
        self._dt       = model.dt
        n = self._pattern_neurons
        self._s1_neurons = np.arange(0, n)
        self._s2_neurons = np.arange(n, 2 * n)

        rng = np.random.RandomState(self._seed)
        self._block_sequences = []
        for block_id in range(self._n_blocks):
            # Swap roles halfway
            if block_id < self._n_blocks // 2:
                std_pat, dev_pat = 0, 1  # S1 standard, S2 deviant
            else:
                std_pat, dev_pat = 1, 0  # S2 standard, S1 deviant
            n_deviants  = max(1, round(self._n_per_block * (1.0 - self._standard_prob)))
            n_standards = self._n_per_block - n_deviants
            seq_labels  = [False] * n_standards + [True] * n_deviants
            rng.shuffle(seq_labels)
            seq = [(dev_pat if is_dev else std_pat, is_dev)
                   for is_dev in seq_labels]
            self._block_sequences.append(seq)

        self._all_standard_rates = []
        self._all_deviant_rates  = []

        n_dev_per_block = max(1, round(self._n_per_block * (1.0 - self._standard_prob)))
        print(f"  T5 setup: {self._n_blocks} blocks × {self._n_per_block} stimuli "
              f"({n_dev_per_block} deviant/block) | trial={self._trial_duration_ms:.0f}ms")

    def generate_trial(self, trial_id: int, rng_key) -> List[Stimulus]:
        assert self._n_input is not None, "Call setup() first"
        block_seq = self._block_sequences[trial_id]
        neurons_by_pat = [self._s1_neurons, self._s2_neurons]

        n_steps  = int(self._trial_duration_ms / self._dt)
        soa_steps = int(self._soa_ms / self._dt)
        act_steps = int(self._stim_dur_ms / self._dt)

        stimuli = []
        for step in range(n_steps):
            stim_idx = step // soa_steps
            step_in  = step % soa_steps
            current  = np.zeros(self._n_input, dtype=np.float32)
            if stim_idx < len(block_seq) and step_in < act_steps:
                pat_id, _ = block_seq[stim_idx]
                current[neurons_by_pat[pat_id]] = self._stim_current
            stimuli.append(Stimulus(
                current     = current,
                spike_train = np.zeros(self._n_input, dtype=np.float32),
                t           = step * self._dt,
                label       = trial_id,
            ))
        return stimuli

    def compute_score(
        self,
        response,
        trial_id: int,
        state_trace: list,
        metadata: dict | None = None,
    ) -> dict:
        block_seq   = self._block_sequences[trial_id]
        spike_ts    = (metadata.get('spike_counts_timeseries')
                       if metadata is not None else None)

        std_rates, dev_rates = [], []
        soa_steps = int(self._soa_ms / self._dt)
        act_steps = int(self._stim_dur_ms / self._dt)

        if spike_ts is not None and len(spike_ts) > 0:
            for i, (pat_id, is_deviant) in enumerate(block_seq):
                start = i * soa_steps
                end   = min(start + act_steps, len(spike_ts))
                if end > start:
                    window_dur_s = (end - start) * self._dt / 1000.0
                    rate = (float(np.sum(spike_ts[start:end]))
                            / (self._n_output * window_dur_s))
                    if is_deviant:
                        dev_rates.append(rate)
                    else:
                        std_rates.append(rate)

        self._all_standard_rates.extend(std_rates)
        self._all_deviant_rates.extend(dev_rates)

        if std_rates and dev_rates:
            mean_std = float(np.mean(std_rates))
            mean_dev = float(np.mean(dev_rates))
            csi  = (mean_dev - mean_std) / (mean_dev + mean_std + 1e-8)
            accuracy = float(np.clip((csi + 1.0) / 2.0, 0.0, 1.0))
        else:
            mean_std = mean_dev = 0.0
            accuracy = 0.5

        if trial_id < 3:
            n_std = len(std_rates)
            n_dev = len(dev_rates)
            csi_val = (mean_dev - mean_std) / (mean_dev + mean_std + 1e-8) if (std_rates and dev_rates) else float('nan')
            print(f"  [T5 debug block {trial_id}] "
                  f"n_std={n_std} n_dev={n_dev} | "
                  f"mean_std={mean_std:.2f} mean_dev={mean_dev:.2f} | "
                  f"CSI={csi_val:.4f} acc={accuracy:.4f} | "
                  f"spike_ts={'present' if spike_ts is not None else 'None'} "
                  f"len={len(spike_ts) if spike_ts is not None else 0}")

        return {
            'accuracy':      float(accuracy),
            'standard_rate': float(np.mean(std_rates)) if std_rates else 0.0,
            'deviant_rate':  float(np.mean(dev_rates)) if dev_rates else 0.0,
        }

    def learning_index(self, trial_results: list) -> float:
        if not self._all_standard_rates and not self._all_deviant_rates:
            return super().learning_index(trial_results)
        mean_std = float(np.mean(self._all_standard_rates)) if self._all_standard_rates else 0.0
        mean_dev = float(np.mean(self._all_deviant_rates))  if self._all_deviant_rates  else 0.0
        csi = (mean_dev - mean_std) / (mean_dev + mean_std + 1e-8)
        return float(np.clip((csi + 1.0) / 2.0, 0.0, 1.0))
