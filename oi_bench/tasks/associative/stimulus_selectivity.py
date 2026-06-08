"""
Task T1 — Stimulus Selectivity via Competitive STDP

Benchmark axis: Associative Learning / Sensory Discrimination
Biological analog: STDP-driven input selectivity (Song, Miller & Abbott 2000)

PROTOCOL
--------
Two non-overlapping input populations:
  Pop A: neurons   0-49  (50 neurons) — fires at 20 Hz Poisson
  Pop B: neurons  50-99  (50 neurons) — fires at 80 Hz Poisson

Each trial: 200ms active stimulus + 100ms silence (= 300ms total).
Trials alternate ABAB... throughout learning and test.

Learning phase: 200 trials (100 A, 100 B). STDP + homeostasis enabled.
Test phase:     100 trials  (50 A,  50 B). Weights frozen.

MECHANISM
---------
With Hebbian STDP and synaptic competition, weights from the high-rate
input population (B, 80 Hz) are potentiated while weights from the
low-rate population (A, 20 Hz) are depressed (Song et al. 2000).
After learning, output neurons respond selectively to the high-rate input.

CAdEx adaptation sharpens this selectivity beyond LIF: the spike-frequency
adaptation in output neurons creates a steeper competition between the
two inputs, amplifying the d-prime above the LIF baseline.

POISSON INPUT ENCODING
----------------------
Poisson spike events are encoded as brief large-current pulses
(_I_SPIKE_PA) into the input population. The amplitude is chosen to
reliably cause one spike per event in both CAdEx (V_peak ≈ +20mV,
requires ≥180 kpA from rest) and LIF (V_T ≈ -50mV, requires ≥40 kpA).
200 kpA covers both with margin.

SCORING
-------
d' = (μ_B − μ_A) / √(0.5·(σ_A² + σ_B²))

where μ, σ are the mean and std of each trial's mean output firing rate
(Hz) across all test trials for each class. LI = clip(d' / 2.0, 0, 1).

A sliding d' over the last 20 training trials per class is tracked during
learning and reported as a normalized per-trial accuracy for verbose output.
Convergence trial = first trial where sliding d' ≥ dprime_criterion (1.0).

References:
  Song, Miller & Abbott (2000) Nature Neuroscience 3:919-926
  Spec Section 5, Task T1
"""

from __future__ import annotations
import numpy as np
from typing import List

from oi_bench.core.protocol import BenchmarkTask
from oi_bench.core.types import Stimulus, ModelState
from oi_bench.core.adapter import OIModel

# Amplitude large enough to spike any input neuron (CAdEx or LIF) in one dt step.
# CAdEx V_peak≈+20mV from E_L=-70mV: needs ≥(90mV × 200pF / 0.1ms) = 180 kpA.
# LIF   V_T  ≈-50mV from E_L=-70mV: needs ≥( 20mV × 200pF / 0.1ms) =  40 kpA.
_I_SPIKE_PA: float = 200_000.0


class StimulusSelectivityTask(BenchmarkTask):

    def __init__(
        self,
        n_input_per_pop: int    = 50,
        rate_a: float           = 40.0,
        rate_b: float           = 60.0,
        stimulus_dur_ms: float  = 200.0,
        iti_ms: float           = 100.0,
        n_learning_trials: int  = 400,
        n_test_trials: int      = 100,
        dt: float               = 0.1,
        dprime_criterion: float = 1.0,
        dprime_ceiling: float   = 2.0,
        seed: int               = 42,
    ):
        self._n_per_pop      = n_input_per_pop
        self._rate_a         = rate_a
        self._rate_b         = rate_b
        self._stim_dur_ms    = stimulus_dur_ms
        self._iti_ms         = iti_ms
        self._n_learning     = n_learning_trials
        self._n_test         = n_test_trials
        self._dt             = dt
        self._dprime_crit    = dprime_criterion
        self._dprime_ceiling = dprime_ceiling
        self._seed           = seed

        self._n_input        = None
        self._n_output       = None
        self._pop_a          = None
        self._pop_b          = None
        self._trial_schedule = None   # list of (cls: 'A'|'B', is_test: bool)

        self._learn_a_rates: list[float] = []
        self._learn_b_rates: list[float] = []

    # ------------------------------------------------------------------
    # BenchmarkTask interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "T1_StimulusSelectivity"

    @property
    def n_trials(self) -> int:
        return self._n_learning + self._n_test

    @property
    def trial_duration_ms(self) -> float:
        return self._stim_dur_ms + self._iti_ms

    @property
    def learning_axis(self) -> str:
        return "associative"

    @property
    def requires_spike_times(self) -> bool:
        return True

    def is_learning_trial(self, trial_id: int) -> bool:
        return trial_id < self._n_learning

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self, model: OIModel) -> None:
        self._n_input  = model.n_input
        self._n_output = model.n_output
        self._dt       = model.dt

        n = min(self._n_per_pop, self._n_input // 2)
        self._pop_a = np.arange(0, n)
        self._pop_b = np.arange(n, 2 * n)

        # ABAB... alternating for both phases
        self._trial_schedule = []
        for i in range(self._n_learning + self._n_test):
            cls     = 'A' if i % 2 == 0 else 'B'
            is_test = (i >= self._n_learning)
            self._trial_schedule.append((cls, is_test))

        self._learn_a_rates = []
        self._learn_b_rates = []

        print(f"  T1 setup: {self._n_learning} learning + {self._n_test} test trials "
              f"| trial={self.trial_duration_ms:.0f}ms")
        print(f"  Pop A (0–{n-1}) @ {self._rate_a:.0f}Hz | "
              f"Pop B ({n}–{2*n-1}) @ {self._rate_b:.0f}Hz | "
              f"rate ratio={self._rate_b / self._rate_a:.1f}x")

    # ------------------------------------------------------------------
    # Trial generation
    # ------------------------------------------------------------------

    def generate_trial(self, trial_id: int, rng_key) -> List[Stimulus]:
        assert self._trial_schedule is not None, "Call setup() first"
        stim_class, _ = self._trial_schedule[trial_id]

        # Deterministic per-trial RNG — independent of JAX key scheduling
        rng = np.random.RandomState(self._seed * 1000 + trial_id)

        pop  = self._pop_a if stim_class == 'A' else self._pop_b
        rate = self._rate_a if stim_class == 'A' else self._rate_b
        p    = rate * self._dt / 1000.0   # Poisson p per neuron per dt step

        stim_steps  = int(self._stim_dur_ms    / self._dt)
        total_steps = int(self.trial_duration_ms / self._dt)

        stimuli = []
        for step in range(total_steps):
            current = np.zeros(self._n_input, dtype=np.float32)
            if step < stim_steps:
                fires = rng.rand(len(pop)) < p
                current[pop[fires]] = _I_SPIKE_PA
            stimuli.append(Stimulus(
                current     = current,
                spike_train = np.zeros(self._n_input, dtype=np.float32),
                t           = step * self._dt,
                label       = trial_id,
            ))
        return stimuli

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _sliding_dprime(self, window: int = 20) -> float:
        """d' from the last `window` learning trials of each class."""
        a = self._learn_a_rates[-window:]
        b = self._learn_b_rates[-window:]
        if len(a) < 2 or len(b) < 2:
            return 0.0
        mu_a, mu_b = float(np.mean(a)), float(np.mean(b))
        s_a,  s_b  = float(np.std(a)),  float(np.std(b))
        pooled = np.sqrt(0.5 * (s_a**2 + s_b**2) + 1e-8)
        return float((mu_b - mu_a) / pooled)

    def compute_score(
        self,
        response,
        trial_id: int,
        state_trace: List[ModelState],
        metadata: dict | None = None,
    ) -> dict:
        stim_class, is_test = self._trial_schedule[trial_id]
        rate = float(np.mean(np.array(response)))

        if not is_test:
            if stim_class == 'A':
                self._learn_a_rates.append(rate)
            else:
                self._learn_b_rates.append(rate)

            dp  = self._sliding_dprime()
            # Map d' ∈ [-ceiling, +ceiling] → [0, 1] for verbose progress display
            acc = float(np.clip(
                (dp + self._dprime_ceiling) / (2.0 * self._dprime_ceiling),
                0.0, 1.0,
            ))
            return {
                'accuracy':       acc,
                'rate':           rate,
                'sliding_dprime': float(dp),
                'stim_class':     0.0 if stim_class == 'A' else 1.0,
                'is_test':        0.0,
            }

        # Test trial — return rate for aggregation in learning_index()
        return {
            'accuracy':   0.5,           # placeholder; LI uses d' not per-trial acc
            'rate':       rate,
            'stim_class': 0.0 if stim_class == 'A' else 1.0,
            'is_test':    1.0,
        }

    def learning_index(self, trial_results: list) -> float:
        """d-prime from test trials, normalised against ceiling of 2.0."""
        test_a = [r.scores['rate'] for r in trial_results
                  if r.scores.get('is_test', 0.0) > 0.5
                  and r.scores.get('stim_class', 0.0) < 0.5]
        test_b = [r.scores['rate'] for r in trial_results
                  if r.scores.get('is_test', 0.0) > 0.5
                  and r.scores.get('stim_class', 0.0) > 0.5]

        if len(test_a) < 2 or len(test_b) < 2:
            return 0.0

        mu_a, mu_b = float(np.mean(test_a)), float(np.mean(test_b))
        s_a,  s_b  = float(np.std(test_a)),  float(np.std(test_b))
        pooled     = np.sqrt(0.5 * (s_a**2 + s_b**2) + 1e-8)
        dp         = (mu_b - mu_a) / pooled

        return float(np.clip(dp / self._dprime_ceiling, 0.0, 1.0))
