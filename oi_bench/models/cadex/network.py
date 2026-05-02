"""
CAdExNetwork — Reference OI Model implementing OIModel adapter interface.

Architecture (per spec Section 4):
  Input population  : CAdExNeuron (standard, n_input neurons)
  Output population : CAdExFractalNeuron (fractional membrane, n_output neurons)
  Synapse           : TripletSTDPSynapse (input → output, all-to-all)
  Homeostasis       : HomeostaticPlasticity (output population, ITI)

This is the reference model for OI-Bench. It implements OIModel so any
benchmark task can evaluate it without knowing its internals.

Network flow per timestep:
  1. Stimulus current injected into input population
  2. Input neurons fire → spikes propagate through synapse → I_syn to output
  3. Output neurons integrate I_syn + background → fire or not
  4. STDP updates weights based on pre/post spike timing
  5. ModelState snapshot returned to benchmark harness

Homeostasis is applied in post_trial() — the ITI hook — not during steps.
This preserves the timescale separation between STDP (ms) and homeostasis (ITI).
"""

import os
os.environ["JAX_PLATFORMS"] = "mps"

import numpy as np
import brainpy.math as bm
import jax.numpy as jnp

from oi_bench.core.adapter import OIModel
from oi_bench.core.types import Stimulus, ModelState
from oi_bench.models.cadex.neuron import CAdExNeuron
from oi_bench.models.cadex.fractal_neuron import CAdExFractalNeuron
from oi_bench.models.cadex.synapse import TripletSTDPSynapse
from oi_bench.models.cadex.homeostatic import HomeostaticPlasticity


class CAdExNetwork(OIModel):
    """
    CAdEx reference network for OI-Bench.

    Parameters
    ----------
    n_input : int
        Input population size. Default 100.
    n_output : int
        Output population size. Default 50.
    alpha : float
        Fractional membrane order for output population. Default 0.85.
        1.0 = integer-order (ablation baseline).
    conn_prob : float
        Input→output connection probability. Default 1.0 (all-to-all).
        Use <1.0 for sparse connectivity in larger networks.
    dt : float
        Simulation timestep (ms). Default 0.1.
    I_background : float
        Tonic background current to output population (pA). Default 150.0.
        Keeps output population near but below rheobase without input.
    plasticity : bool
        If False, STDP weights are frozen. Default True.
    homeostasis : bool
        If False, homeostatic plasticity is disabled. Default True.
        Used for ablation: model=cadex homeostasis=false.
    seed : int
        Random seed for weight initialisation. Default 0.
    """

    def __init__(
        self,
        n_input: int       = 100,
        n_output: int      = 50,
        alpha: float       = 0.85,
        conn_prob: float   = 1.0,
        dt: float          = 0.1,
        I_background: float = 150.0,
        plasticity: bool   = True,
        homeostasis: bool  = True,
        seed: int          = 0,
    ):
        np.random.seed(seed)

        self._n_input    = n_input
        self._n_output   = n_output
        self._alpha      = alpha
        self._dt         = dt
        self._I_bg       = I_background
        self._plasticity = plasticity

        # --- Input population: standard CAdEx ---
        self.input_pop = CAdExNeuron(size=n_input, dt=dt)

        # --- Output population: fractional CAdEx ---
        self.output_pop = CAdExFractalNeuron(size=n_output, alpha=alpha, dt=dt)

        # Add V_T as a Variable on output_pop for intrinsic excitability homeostasis
        self.output_pop.V_T = bm.Variable(
            bm.full(n_output, self.output_pop.V_T)
        )

        # --- Synapse: TripletSTDP input → output ---
        if conn_prob >= 1.0:
            conn = np.ones((n_input, n_output), dtype=bool)
        else:
            conn = {'prob': conn_prob}

        self.synapse = TripletSTDPSynapse(
            pre        = self.input_pop,
            post       = self.output_pop,
            conn       = conn,
            w_init     = 0.3,
            g_max      = 3.0,
            tau_syn    = 15.0,
            plasticity = plasticity,
        )

        # --- Homeostatic plasticity (ITI mechanism) ---
        self.homeo = HomeostaticPlasticity(
            neurons      = self.output_pop,
            synapse      = self.synapse,
            r_target     = 110.0,   # actual operating rate — homeostasis neutral
            trial_dur_ms = 400.0,
            gamma        = 0.5,
            eta_h        = 0.001,
            enabled      = homeostasis,
        )

        # Spike accumulator for homeostasis (reset each trial)
        self._trial_spike_counts = np.zeros(n_output, dtype=np.float32)

    # ------------------------------------------------------------------
    # OIModel interface
    # ------------------------------------------------------------------

    @property
    def n_input(self) -> int:
        return self._n_input

    @property
    def n_output(self) -> int:
        return self._n_output

    @property
    def dt(self) -> float:
        return self._dt

    def reset(self) -> None:
        """
        Full episode reset. Resets all dynamic state except weights.
        Weights persist across trials to allow learning to accumulate.
        """
        self.input_pop.V.value     = bm.full(self._n_input,  self.input_pop.E_L)
        self.input_pop.w.value     = bm.zeros(self._n_input)
        self.input_pop.Ca.value    = bm.zeros(self._n_input)
        self.input_pop.spike.value = bm.zeros(self._n_input, dtype=bool)

        self.output_pop.V.value     = bm.full(self._n_output, self.output_pop.E_L)
        self.output_pop.w.value     = bm.zeros(self._n_output)
        self.output_pop.Ca.value    = bm.zeros(self._n_output)
        self.output_pop.spike.value = bm.zeros(self._n_output, dtype=bool)

        self.synapse.r1.value = bm.zeros(self._n_input)
        self.synapse.r2.value = bm.zeros(self._n_input)
        self.synapse.o1.value = bm.zeros(self._n_output)
        self.synapse.o2.value = bm.zeros(self._n_output)
        self.synapse.g.value  = bm.zeros((self._n_input, self._n_output))

        self._trial_spike_counts[:] = 0.0

    def step(self, stimulus: Stimulus) -> ModelState:
        """
        Advance one timestep.

        Applies stimulus current to input population, propagates spikes
        through synapse, updates output population, runs STDP.

        Parameters
        ----------
        stimulus : Stimulus
            current: pA injected into input population, shape (n_input,)
            spike_train: binary input spikes, shape (n_input,) — added to current

        Returns
        -------
        ModelState
            spikes   : output population spikes, shape (n_output,)
            membrane : output population V, shape (n_output,)
            weights  : synapse weight matrix, shape (n_input, n_output)
            extras   : {'Ca': output Ca, 'w_adapt': output adaptation,
                        'input_spikes': input population spikes}
        """
        # Step input population
        I_input = stimulus.current + stimulus.spike_train * 100.0  # spike → 100pA pulse
        self.input_pop.update(x=bm.array(I_input, dtype=bm.float32))

        # Get input spikes, propagate through synapse
        S_pre  = self.input_pop.spike.value.astype(bm.float32)
        S_post = self.output_pop.spike.value.astype(bm.float32)
        I_syn  = self.synapse.update(S_pre, S_post, self.output_pop.V.value)

        # Step output population with background + synaptic drive
        I_total = bm.full(self._n_output, self._I_bg) + I_syn
        self.output_pop.update(x=I_total)

        # Accumulate spikes for homeostasis
        self._trial_spike_counts += np.array(
            self.output_pop.spike.value.astype(bm.float32)
        )

        return ModelState(
            spikes   = self.output_pop.spike.value.astype(bm.float32),
            membrane = self.output_pop.V.value,
            weights  = self.synapse.W.value,
            extras   = {
                'Ca':           self.output_pop.Ca.value,
                'w_adapt':      self.output_pop.w.value,
                'input_spikes': self.input_pop.spike.value.astype(bm.float32),
                'ltp_total':    float(self.synapse.n_ltp.value),
                'ltd_total':    float(self.synapse.n_ltd.value),
            }
        )

    def pre_trial(self, trial_id: int) -> None:
        """Reset per-trial accumulators and STDP traces."""
        self._trial_spike_counts[:] = 0.0
        self.synapse.r1.value = bm.zeros(self._n_input)
        self.synapse.r2.value = bm.zeros(self._n_input)
        self.synapse.o1.value = bm.zeros(self._n_output)
        self.synapse.o2.value = bm.zeros(self._n_output)
        self.synapse.g.value  = bm.zeros((self._n_input, self._n_output))

    def post_trial(
        self,
        trial_id: int,
        state_trace: list[ModelState],
        modulator: float = 1.0,
    ) -> None:
        """
        Apply homeostatic plasticity at end of trial (ITI).

        Also applies three-factor modulation if modulator != 1.0.
        """
        # Three-factor: scale weight updates by neuromodulatory signal
        self.synapse.modulator = modulator

        # Homeostasis uses spike counts accumulated during this trial
        self.homeo.update(self._trial_spike_counts.copy())

    @property
    def weight_stats(self) -> dict:
        """Convenience accessor for synapse weight statistics."""
        return self.synapse.weight_stats

    @property
    def homeo_stats(self) -> dict:
        """Convenience accessor for homeostatic state."""
        return self.homeo.stats
