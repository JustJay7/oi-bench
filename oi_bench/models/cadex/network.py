"""
CAdExNetwork — Reference OI Model implementing OIModel adapter interface.

Architecture (per spec Section 4):
  Input population  : CAdExNeuron (standard, n_input neurons)
  Output population : CAdExFractalNeuron (fractional membrane, n_output neurons)
  Synapse           : TripletSTDPSynapse (input → output, all-to-all)
  Homeostasis       : HomeostaticPlasticity (output population, ITI)

Network flow per timestep:
  1. Stimulus current injected into input population
  2. Input neurons fire → spikes propagate through synapse → I_syn to output
  3. Output neurons integrate I_syn + background → fire or not
  4. STDP updates weights based on pre/post spike timing
  5. ModelState snapshot returned to benchmark harness

Homeostasis applied in post_trial() — ITI hook — not during steps.
configure_homeostasis() must be called before each task to set the
correct r_target and trial_dur_ms for that task's operating point.
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
    conn_prob : float
        Input→output connection probability. Default 1.0.
    dt : float
        Simulation timestep (ms). Default 0.1.
    I_background : float
        Tonic background current to output population (pA). Default 150.0.
    plasticity : bool
        If False, STDP weights frozen. Default True.
    homeostasis : bool
        If False, homeostatic plasticity disabled. Default True.
    seed : int
        Random seed for weight initialisation. Default 0.
    """

    def __init__(
        self,
        n_input: int        = 100,
        n_output: int       = 50,
        alpha: float        = 0.85,
        conn_prob: float    = 1.0,
        dt: float           = 0.1,
        I_background: float = 150.0,
        plasticity: bool    = True,
        homeostasis: bool   = True,
        seed: int           = 0,
    ):
        np.random.seed(seed)

        self._n_input    = n_input
        self._n_output   = n_output
        self._alpha      = alpha
        self._dt         = dt
        self._I_bg       = I_background
        self._plasticity = plasticity

        # Input population: standard CAdEx
        self.input_pop = CAdExNeuron(size=n_input, dt=dt)

        # Output population: fractional CAdEx
        self.output_pop = CAdExFractalNeuron(size=n_output, alpha=alpha, dt=dt)

        # V_T as Variable for intrinsic excitability homeostasis
        self.output_pop.V_T = bm.Variable(
            bm.full(n_output, self.output_pop.V_T)
        )

        # Synapse: TripletSTDP input → output
        conn = (np.ones((n_input, n_output), dtype=bool)
                if conn_prob >= 1.0 else {'prob': conn_prob})

        self.synapse = TripletSTDPSynapse(
            pre        = self.input_pop,
            post       = self.output_pop,
            conn       = conn,
            w_init     = 0.3,
            g_max      = 3.0,
            tau_syn    = 15.0,
            plasticity = plasticity,
        )

        # Recurrent synapse: output → output (for attractor dynamics, T2)
        # Sparse 20% connectivity — prevents runaway activity
        # STDP strengthens co-active connections during pattern learning
        rec_conn = np.random.rand(n_output, n_output) < 0.2
        np.fill_diagonal(rec_conn, False)   # no self-connections

        self.rec_synapse = TripletSTDPSynapse(
            pre        = self.output_pop,
            post       = self.output_pop,
            conn       = rec_conn,
            w_init     = 0.1,          # start weak — STDP strengthens them
            g_max      = 1.5,          # weaker than feedforward
            tau_syn    = 15.0,
            plasticity = plasticity,
        )

        # Global inhibition: scales with mean population activity
        # Prevents runaway excitation in recurrent network
        self._inhibition_strength = 2.0   # pA per mean spike

        # Homeostatic plasticity — calibrated per task via configure_homeostasis()
        self.homeo = HomeostaticPlasticity(
            neurons      = self.output_pop,
            synapse      = self.synapse,
            r_target     = 110.0,    # default — override per task
            trial_dur_ms = 400.0,    # default — override per task
            gamma        = 0.5,
            eta_h        = 0.001,
            enabled      = homeostasis,
        )

        self._trial_spike_counts = np.zeros(n_output, dtype=np.float32)

    def configure_homeostasis(
        self,
        r_target: float,
        trial_dur_ms: float,
        enabled: bool = True,
    ) -> None:
        """
        Update homeostasis parameters for the current task.

        Parameters
        ----------
        r_target : float
            Natural output firing rate for this task (Hz).
        trial_dur_ms : float
            Actual trial duration for this task (ms).
        enabled : bool
            Set False for sparse-stimulus tasks (e.g. T4) where homeostasis
            causes instability due to long silent inter-stimulus windows.
        """
        self.homeo.r_target     = r_target
        self.homeo.trial_dur_ms = trial_dur_ms
        self.homeo.r_mean[:]    = r_target
        self.homeo.alpha        = 1.0 - np.exp(-1.0 / 5.0)
        self.homeo.enabled      = enabled

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
        """Full episode reset. Weights persist across trials."""
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

        self.rec_synapse.r1.value = bm.zeros(self._n_output)
        self.rec_synapse.r2.value = bm.zeros(self._n_output)
        self.rec_synapse.o1.value = bm.zeros(self._n_output)
        self.rec_synapse.o2.value = bm.zeros(self._n_output)
        self.rec_synapse.g.value  = bm.zeros((self._n_output, self._n_output))

        self._trial_spike_counts[:] = 0.0

    def step(self, stimulus: Stimulus) -> ModelState:
        """Advance one timestep."""
        I_input = stimulus.current + stimulus.spike_train * 100.0
        self.input_pop.update(x=bm.array(I_input, dtype=bm.float32))

        S_pre  = self.input_pop.spike.value.astype(bm.float32)
        S_post = self.output_pop.spike.value.astype(bm.float32)
        I_syn  = self.synapse.update(S_pre, S_post, self.output_pop.V.value)

        # Recurrent input (output → output) for attractor dynamics
        I_rec  = self.rec_synapse.update(S_post, S_post, self.output_pop.V.value)

        # Global inhibition — prevents runaway recurrent excitation
        mean_activity = jnp.mean(S_post)
        I_inh = -self._inhibition_strength * mean_activity * bm.ones(self._n_output)

        I_total = bm.full(self._n_output, self._I_bg) + I_syn + I_rec + I_inh
        self.output_pop.update(x=I_total)

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

        self.rec_synapse.r1.value = bm.zeros(self._n_output)
        self.rec_synapse.r2.value = bm.zeros(self._n_output)
        self.rec_synapse.o1.value = bm.zeros(self._n_output)
        self.rec_synapse.o2.value = bm.zeros(self._n_output)
        self.rec_synapse.g.value  = bm.zeros((self._n_output, self._n_output))

    def post_trial(
        self,
        trial_id: int,
        state_trace: list,
        modulator: float = 1.0,
    ) -> None:
        """Apply homeostatic plasticity at end of trial (ITI)."""
        self.synapse.modulator = modulator
        self.homeo.update(self._trial_spike_counts.copy())

    @property
    def weight_stats(self) -> dict:
        return self.synapse.weight_stats

    @property
    def homeo_stats(self) -> dict:
        return self.homeo.stats
