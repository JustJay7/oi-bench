"""
Task T1 — Classical Conditioning

Benchmark axis: Associative Learning
Biological analog: Eyeblink conditioning (Kagan et al. 2022, Neuron)

PROTOCOL
--------
Each trial is 500ms and consists of three phases:

  [0, 100ms)    — Baseline: no stimulus
  [100, 200ms)  — CS window: conditioned stimulus burst to input population A
  [200, 300ms)  — CS+US window (conditioning trials only):
                    CS continues + US direct current to output population
  [300, 500ms)  — Response window: model output scored here

Trial structure over n_trials=200:
  Trials 0-149   : Conditioning trials (CS + US paired)
  Trials 150-199 : Test trials (CS alone, no US)

Learning signal: CS→output pathway strengthens via STDP during CS+US pairing.
After sufficient conditioning, CS alone should drive output population above
the conditioned response (CR) threshold.

SCORING
-------
Conditioned Response (CR) = output population mean firing rate during
response window exceeds CR_threshold (default 20 Hz).

For conditioning trials: correct = True (CR or not, we're training)
For test trials: correct = (CR present) — this is the learning measure

Score per trial:
  'accuracy'      : 1.0 if CR present on test trial, 0.0 otherwise
  'cr_rate'       : mean firing rate of output during response window (Hz)
  'cr_detected'   : binary float {0.0, 1.0}

Learning Index uses accuracy over test trials only (last 50 trials).

References:
  Kagan et al. (2022) Neuron 110:3952-3960
  Drew & Abbott (2006) PNAS 103:8876-8881 — STDP and classical conditioning
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
    """
    T1 — Classical Conditioning.

    Parameters
    ----------
    n_trials : int
        Total trials. Default 200 (150 conditioning + 50 test).
    trial_duration_ms : float
        Duration of each trial. Default 500ms.
    cs_onset_ms : float
        CS onset time within trial. Default 100ms.
    cs_duration_ms : float
        CS stimulus duration. Default 100ms.
    us_onset_ms : float
        US onset time (must be >= cs_onset_ms). Default 200ms.
    us_duration_ms : float
        US stimulus duration. Default 100ms.
    response_onset_ms : float
        Start of response scoring window. Default 300ms.
    response_duration_ms : float
        Duration of response window. Default 200ms.
    cs_current : float
        Current injected to CS input neurons (pA). Default 400.0.
    us_current : float
        Current injected directly to output neurons (pA). Default 500.0.
    cs_fraction : float
        Fraction of input population that receives CS. Default 0.3.
    cr_threshold : float
        Firing rate threshold for conditioned response (Hz). Default 20.0.
    n_conditioning_trials : int
        Number of CS+US paired trials. Default 150.
    """

    def __init__(
        self,
        n_trials: int              = 200,
        trial_duration_ms: float   = 500.0,
        cs_onset_ms: float         = 100.0,
        cs_duration_ms: float      = 100.0,
        us_onset_ms: float         = 200.0,
        us_duration_ms: float      = 100.0,
        response_onset_ms: float   = 300.0,
        response_duration_ms: float = 200.0,
        cs_current: float          = 400.0,
        us_current: float          = 500.0,
        cs_fraction: float         = 0.3,
        cr_threshold: float        = 20.0,
        n_conditioning_trials: int = 150,
        dt: float                  = 0.1,
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

        # Set during setup()
        self._n_input  = None
        self._n_output = None
        self._cs_neurons = None   # indices of CS input neurons

    # ------------------------------------------------------------------
    # BenchmarkTask interface
    # ------------------------------------------------------------------

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
        """
        Validate model compatibility and fix CS neuron indices.
        CS neurons are fixed across all trials for reproducibility.
        """
        self._n_input  = model.n_input
        self._n_output = model.n_output
        self._dt       = model.dt

        n_cs = max(1, int(self._n_input * self._cs_fraction))
        self._cs_neurons = np.arange(n_cs)   # first n_cs neurons = CS population

        print(f"  T1 setup: n_input={self._n_input} n_output={self._n_output}")
        print(f"  CS population: {n_cs} neurons ({self._cs_fraction*100:.0f}% of input)")
        print(f"  Conditioning trials: {self._n_conditioning} / {self._n_trials}")
        print(f"  CR threshold: {self._cr_threshold} Hz")

    def generate_trial(self, trial_id: int, rng_key) -> List[Stimulus]:
        """
        Generate stimulus sequence for one trial.

        Conditioning trials (0 to n_conditioning-1): CS + US paired.
        Test trials (n_conditioning to n_trials-1): CS alone.
        """
        assert self._n_input is not None, "Call setup() before generate_trial()"

        is_conditioning = trial_id < self._n_conditioning
        n_steps = int(self._trial_duration_ms / self._dt)

        stimuli = []
        for step in range(n_steps):
            t_ms = step * self._dt

            # Build input current array (n_input,)
            current = np.zeros(self._n_input, dtype=np.float32)
            spike_train = np.zeros(self._n_input, dtype=np.float32)

            # CS window
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
    ) -> dict:
        """
        Score one trial.

        Extracts mean output firing rate during response window from
        state_trace. CR is detected if rate exceeds cr_threshold.

        Parameters
        ----------
        response : array (n_output,)
            Mean output firing rate over full trial (Hz). Used as fallback
            if state_trace is empty.
        trial_id : int
        state_trace : list[ModelState]
            Per-step model states. Used to extract response window firing.

        Returns
        -------
        dict with keys: 'accuracy', 'cr_rate', 'cr_detected'
        """
        is_test_trial = trial_id >= self._n_conditioning

        if state_trace:
            # Extract response window steps
            resp_start = int(self._response_onset_ms / self._dt)
            resp_end   = int(
                (self._response_onset_ms + self._response_duration_ms) / self._dt
            )
            resp_end   = min(resp_end, len(state_trace))

            # Mean firing rate = total spikes / window duration in seconds
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
            # Fallback: use pre-computed response array
            cr_rate = float(np.mean(response))

        cr_detected = float(cr_rate >= self._cr_threshold)

        # Accuracy only meaningful on test trials
        # On conditioning trials: always 1.0 (we're training, not scoring)
        accuracy = cr_detected if is_test_trial else 1.0

        return {
            'accuracy':    float(accuracy),
            'cr_rate':     cr_rate,
            'cr_detected': cr_detected,
        }

    def us_current_for_step(self, t_ms: float, trial_id: int) -> np.ndarray:
        """
        Returns US current to inject directly into output population.

        Called by the runner during conditioning trials.
        The US bypasses the input population and drives output directly,
        forcing post-synaptic firing to pair with CS-driven pre-synaptic
        activity — the STDP pairing condition.
        """
        is_conditioning = trial_id < self._n_conditioning
        us_on  = self._us_onset_ms
        us_off = self._us_onset_ms + self._us_duration_ms

        if is_conditioning and us_on <= t_ms < us_off:
            return np.full(self._n_output, self._us_current, dtype=np.float32)
        return np.zeros(self._n_output, dtype=np.float32)
