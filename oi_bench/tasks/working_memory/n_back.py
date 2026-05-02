"""
Task T6 — N-Back (N=1 and N=2)

Benchmark axis: Working Memory
Biological analog: Continuous online memory updating, PFC-dependent
(Owen et al. 1999 J Cognitive Neuroscience; Kane & Engle 2002 Psych Sci)

PROTOCOL
--------
Continuous stream of stimuli, one every 500ms.
Model must signal when the current stimulus matches the one N steps back.

N=1: match if current == previous
N=2: match if current == two steps ago

SCORING
-------
The runner passes response = mean firing rate over the full trial (Hz),
shape (n_output,). Mean output population rate is used as the detection
signal — high activity signals a target (N-back match), low activity
signals a non-target. Threshold at mean of max possible rate.

This is consistent with the population rate code readout in spec Section 12 Q1
and works without state_trace.

Score per trial:
  accuracy         : 1.0 if correct, 0.0 otherwise
  n_back           : 1 or 2
  is_target        : 1.0 if this is an N-back match trial
  responded_target : 1.0 if model responded as if target

References:
  Owen et al. (1999) J Cognitive Neuroscience 11:567-581
  Kane & Engle (2002) Psychological Science 13:14-17
  Spec Section 5, Task T6
"""

from __future__ import annotations
import jax
import jax.numpy as jnp
import numpy as np
from typing import List

from oi_bench.core.protocol import BenchmarkTask
from oi_bench.core.types import Stimulus, ModelState
from oi_bench.core.adapter import OIModel


class NBackTask(BenchmarkTask):

    def __init__(
        self,
        n_back_values: List[int]    = [1, 2],
        n_stimuli: int              = 3,
        trials_per_block: int       = 200,
        stimulus_duration_ms: float = 200.0,
        isi_ms: float               = 300.0,
        stimulus_current: float     = 400.0,
        stimulus_fraction: float    = 0.3,
        dt: float                   = 0.1,
        seed: int                   = 42,
    ):
        self._n_back_values    = n_back_values
        self._n_stimuli        = n_stimuli
        self._trials_per_block = trials_per_block
        self._stim_dur         = stimulus_duration_ms
        self._isi              = isi_ms
        self._stim_current     = stimulus_current
        self._stim_frac        = stimulus_fraction
        self._dt               = dt
        self._seed             = seed
        self._trial_dur        = stimulus_duration_ms + isi_ms
        self._n_input          = None
        self._n_output         = None
        self._stim_patterns    = None
        self._trial_info       = None
        # Running mean output rate — updated trial-by-trial to set detection threshold
        self._rate_history: List[float] = []

    @property
    def name(self) -> str:
        return "T6_NBack"

    @property
    def n_trials(self) -> int:
        return len(self._n_back_values) * self._trials_per_block

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
        self._rate_history = []
        rng = np.random.RandomState(self._seed)
        n_active = max(1, int(self._n_input * self._stim_frac))
        self._stim_patterns = np.zeros(
            (self._n_stimuli, self._n_input), dtype=np.float32
        )
        for i in range(self._n_stimuli):
            idx = rng.choice(self._n_input, size=n_active, replace=False)
            self._stim_patterns[i, idx] = 1.0
        self._trial_info = []
        for n in self._n_back_values:
            history = []
            for _ in range(self._trials_per_block):
                stim_id   = rng.randint(0, self._n_stimuli)
                is_target = (len(history) >= n and history[-n] == stim_id)
                self._trial_info.append((stim_id, n, is_target))
                history.append(stim_id)
        n_targets = sum(1 for _, _, t in self._trial_info if t)
        print(f"  T6 setup: N-back={self._n_back_values} | "
              f"{self._n_stimuli} stimuli | "
              f"target rate: {n_targets/len(self._trial_info):.2f}")

    def generate_trial(self, trial_id: int, rng_key) -> List[Stimulus]:
        stim_id, n, _ = self._trial_info[trial_id]
        n_steps = int(self._trial_dur / self._dt)
        stimuli = []
        for step in range(n_steps):
            t_ms    = step * self._dt
            current = np.zeros(self._n_input, dtype=np.float32)
            if t_ms < self._stim_dur:
                current = self._stim_patterns[stim_id] * self._stim_current
            stimuli.append(Stimulus(
                current     = current,
                spike_train = np.zeros(self._n_input, dtype=np.float32),
                t           = t_ms,
                label       = int(n),
            ))
        return stimuli

    def compute_score(
        self,
        response,
        trial_id: int,
        state_trace: list,
        metadata: dict | None = None,
    ) -> dict:
        """
        Score using population mean firing rate from response array.

        response = mean firing rate over full trial, shape (n_output,).
        Mean population rate is the detection signal.
        Threshold: adaptive — median of recent trial rates to avoid
        fixed-threshold failure on untrained networks.
        """
        _, n, is_target = self._trial_info[trial_id]

        mean_rate = float(np.mean(np.array(response)))
        self._rate_history.append(mean_rate)

        # Adaptive threshold: median of all observed rates so far.
        # On an untrained network this correctly yields ~50% detection
        # (chance), not systematic 0% or 100%.
        threshold = float(np.median(self._rate_history))

        responded_target = mean_rate > threshold
        correct          = (responded_target == is_target)

        return {
            'accuracy':         float(correct),
            'n_back':           n,
            'is_target':        float(is_target),
            'responded_target': float(responded_target),
        }
