"""
Task T6 — 1-Back Change Detection

Benchmark axis: Working Memory / Online Change Detection
Biological analog: Change blindness, visual short-term memory
(Rensink 2002 Annual Review; Vogel et al. 2001 Nature)

PROTOCOL
--------
40 blocks, each 2500ms (50 stimuli × 50ms SOA).
Each stimulus: 30ms active (400pA) + 20ms silence.
3 patterns — A (neurons 0-32), B (neurons 33-65), C (neurons 66-98).
~50% of consecutive pairs are changes (different pattern from previous).

MECHANISM
---------
CAdEx: adaptation suppresses the current-pattern response after repeated
       presentations (SSA). A change = novel pattern = larger response →
       model can detect changes via elevated firing.
LIF:   no adaptation → flat response → weak change signal.

SCORING
-------
For each consecutive stimulus pair within a block:
  - record output firing rate at the second stimulus
  - label: change (1) or no-change (0)
AUC = P(rate_change > rate_nochange) for all pairs across all blocks.
AUC=0.5 is chance, AUC=1.0 is perfect discrimination.
Final LI = AUC directly [0,1], chance-corrected.

References:
  Rensink (2002) Annual Review of Psychology 53:245-277
  Vogel et al. (2001) Nature 412:751-756
  Spec Section 5, Task T6
"""

from __future__ import annotations
import numpy as np
from typing import List

from oi_bench.core.protocol import BenchmarkTask
from oi_bench.core.types import Stimulus
from oi_bench.core.adapter import OIModel


def _auc(pos_rates: list, neg_rates: list) -> float:
    """Mann-Whitney AUC: P(rate_change > rate_nochange)."""
    if not pos_rates or not neg_rates:
        return 0.5
    count = 0.0
    n = len(pos_rates) * len(neg_rates)
    for p in pos_rates:
        for q in neg_rates:
            if p > q:
                count += 1.0
            elif p == q:
                count += 0.5
    return count / n


class ChangeDetectionTask(BenchmarkTask):

    def __init__(
        self,
        n_blocks: int            = 40,
        n_stimuli_per_block: int = 50,
        stimulus_duration_ms: float = 30.0,
        isi_ms: float            = 20.0,
        change_prob: float       = 0.5,
        stimulus_current: float  = 400.0,
        dt: float                = 0.1,
        seed: int                = 42,
    ):
        self._n_blocks    = n_blocks
        self._n_per_block = n_stimuli_per_block
        self._stim_dur_ms = stimulus_duration_ms
        self._isi_ms      = isi_ms
        self._soa_ms      = stimulus_duration_ms + isi_ms
        self._change_prob = change_prob
        self._stim_current= stimulus_current
        self._dt          = dt
        self._seed        = seed
        self._trial_duration_ms = n_stimuli_per_block * (stimulus_duration_ms + isi_ms)

        self._n_input         = None
        self._n_output        = None
        self._pattern_neurons = None  # list of 3 arrays
        self._block_sequences = None  # list of lists: [pat_id, ...]
        self._block_labels    = None  # list of lists: [change_bool, ...] (len=n_per_block-1)

        self._all_change_rates   = []
        self._all_nochange_rates = []

    @property
    def name(self) -> str:
        return "T6_NBack"

    @property
    def n_trials(self) -> int:
        return self._n_blocks

    @property
    def trial_duration_ms(self) -> float:
        return self._trial_duration_ms

    @property
    def learning_axis(self) -> str:
        return "working_memory"

    @property
    def requires_spike_times(self) -> bool:
        return True

    def setup(self, model: OIModel) -> None:
        self._n_input  = model.n_input
        self._n_output = model.n_output
        self._dt       = model.dt
        self._pattern_neurons = [
            np.arange(0,  33),   # A
            np.arange(33, 66),   # B
            np.arange(66, 99),   # C
        ]

        rng = np.random.RandomState(self._seed)
        n_pairs = self._n_per_block - 1
        n_changes = round(n_pairs * self._change_prob)
        n_nochanges = n_pairs - n_changes

        self._block_sequences = []
        self._block_labels    = []
        for _ in range(self._n_blocks):
            labels = [True] * n_changes + [False] * n_nochanges
            rng.shuffle(labels)
            seq = [rng.randint(0, 3)]
            for is_change in labels:
                if is_change:
                    others = [p for p in range(3) if p != seq[-1]]
                    seq.append(int(rng.choice(others)))
                else:
                    seq.append(seq[-1])
            self._block_sequences.append(seq)
            self._block_labels.append(labels)

        self._all_change_rates   = []
        self._all_nochange_rates = []

        print(f"  T6 setup: {self._n_blocks} blocks × {self._n_per_block} stimuli "
              f"| ~{n_changes}/{n_pairs} change pairs/block "
              f"| trial={self._trial_duration_ms:.0f}ms")

    def generate_trial(self, trial_id: int, rng_key) -> List[Stimulus]:
        assert self._n_input is not None, "Call setup() first"
        block_seq = self._block_sequences[trial_id]
        n_steps   = int(self._trial_duration_ms / self._dt)
        soa_steps = int(self._soa_ms / self._dt)
        act_steps = int(self._stim_dur_ms / self._dt)

        stimuli = []
        for step in range(n_steps):
            stim_idx = step // soa_steps
            step_in  = step % soa_steps
            current  = np.zeros(self._n_input, dtype=np.float32)
            if stim_idx < len(block_seq) and step_in < act_steps:
                pat_id = block_seq[stim_idx]
                current[self._pattern_neurons[pat_id]] = self._stim_current
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
        labels   = self._block_labels[trial_id]    # n_per_block-1 booleans
        spike_ts = (metadata.get('spike_counts_timeseries')
                    if metadata is not None else None)

        change_rates, nochange_rates = [], []
        soa_steps = int(self._soa_ms / self._dt)
        act_steps = int(self._stim_dur_ms / self._dt)

        if spike_ts is not None and len(spike_ts) > 0:
            for pair_idx, is_change in enumerate(labels):
                # Rate at the SECOND stimulus of the pair (index pair_idx+1)
                stim_idx = pair_idx + 1
                start = stim_idx * soa_steps
                end   = min(start + act_steps, len(spike_ts))
                if end > start:
                    window_dur_s = (end - start) * self._dt / 1000.0
                    rate = (float(np.sum(spike_ts[start:end]))
                            / (self._n_output * window_dur_s))
                    if is_change:
                        change_rates.append(rate)
                    else:
                        nochange_rates.append(rate)

        self._all_change_rates.extend(change_rates)
        self._all_nochange_rates.extend(nochange_rates)

        block_auc = _auc(change_rates, nochange_rates)

        return {
            'accuracy':       float(block_auc),
            'change_rate':    float(np.mean(change_rates))   if change_rates   else 0.0,
            'nochange_rate':  float(np.mean(nochange_rates)) if nochange_rates else 0.0,
        }

    def learning_index(self, trial_results: list) -> float:
        return float(_auc(self._all_change_rates, self._all_nochange_rates))


# Backward-compat alias
NBackTask = ChangeDetectionTask
