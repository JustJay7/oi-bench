"""
CAdExNetwork — Reference OI Model implementing OIModel adapter interface.
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

        self.input_pop  = CAdExNeuron(size=n_input, dt=dt)
        self.output_pop = CAdExFractalNeuron(size=n_output, alpha=alpha, dt=dt)
        self.output_pop.V_T = bm.Variable(
            bm.full(n_output, self.output_pop.V_T))

        conn = (np.ones((n_input, n_output), dtype=bool)
                if conn_prob >= 1.0 else {'prob': conn_prob})
        self.synapse = TripletSTDPSynapse(
            pre=self.input_pop, post=self.output_pop,
            conn=conn, w_init=0.3, g_max=3.0, tau_syn=15.0,
            plasticity=plasticity,
        )

        rec_conn = np.random.rand(n_output, n_output) < 0.2
        np.fill_diagonal(rec_conn, False)
        self.rec_synapse = TripletSTDPSynapse(
            pre=self.output_pop, post=self.output_pop,
            conn=rec_conn, w_init=0.1, g_max=1.5, tau_syn=15.0,
            plasticity=plasticity,
        )

        self._inhibition_strength = 2.0
        self.homeo = HomeostaticPlasticity(
            neurons=self.output_pop, synapse=self.synapse,
            r_target=110.0, trial_dur_ms=400.0,
            gamma=0.5, eta_h=0.001, enabled=homeostasis,
        )
        self._trial_spike_counts = np.zeros(n_output, dtype=np.float32)

    def configure_homeostasis(self, r_target, trial_dur_ms, enabled=True):
        self.homeo.r_target     = r_target
        self.homeo.trial_dur_ms = trial_dur_ms
        self.homeo.r_mean[:]    = r_target
        self.homeo.alpha        = 1.0 - np.exp(-1.0 / 5.0)
        self.homeo.enabled      = enabled

    def configure_stdp(
        self,
        A2_plus: float  = 0.006,
        A3_plus: float  = 0.009,
        A2_minus: float = 0.003,
    ) -> None:
        """
        Update STDP amplitudes for the current task.

        Called from run.py before each task. Allows T4 to use reduced
        amplitudes (preventing weight saturation from dopamine reward)
        without affecting other tasks which need full amplitudes.

        Default values restore Pfister & Gerstner (2006) Table 1.
        """
        self.synapse.configure_stdp(A2_plus, A3_plus, A2_minus)
        self.rec_synapse.configure_stdp(A2_plus, A3_plus, A2_minus)

    def configure_eligibility_trace(self, enabled=False, tau_e=1000.0):
        self.synapse.use_eligibility_trace     = enabled
        self.synapse.tau_e                     = tau_e
        self.rec_synapse.use_eligibility_trace = enabled
        self.rec_synapse.tau_e                 = tau_e
        if not enabled:
            self.synapse.reset_eligibility()
            self.rec_synapse.reset_eligibility()

    @property
    def n_input(self):  return self._n_input
    @property
    def n_output(self): return self._n_output
    @property
    def dt(self):       return self._dt

    def reset(self):
        self.input_pop.V.value      = bm.full(self._n_input,  self.input_pop.E_L)
        self.input_pop.w.value      = bm.zeros(self._n_input)
        self.input_pop.Ca.value     = bm.zeros(self._n_input)
        self.input_pop.spike.value  = bm.zeros(self._n_input, dtype=bool)
        self.output_pop.V.value     = bm.full(self._n_output, self.output_pop.E_L)
        self.output_pop.w.value     = bm.zeros(self._n_output)
        self.output_pop.Ca.value    = bm.zeros(self._n_output)
        self.output_pop.spike.value = bm.zeros(self._n_output, dtype=bool)
        self.synapse.r1.value       = bm.zeros(self._n_input)
        self.synapse.r2.value       = bm.zeros(self._n_input)
        self.synapse.o1.value       = bm.zeros(self._n_output)
        self.synapse.o2.value       = bm.zeros(self._n_output)
        self.synapse.g.value        = bm.zeros((self._n_input,  self._n_output))
        self.synapse.reset_eligibility()
        self.rec_synapse.r1.value   = bm.zeros(self._n_output)
        self.rec_synapse.r2.value   = bm.zeros(self._n_output)
        self.rec_synapse.o1.value   = bm.zeros(self._n_output)
        self.rec_synapse.o2.value   = bm.zeros(self._n_output)
        self.rec_synapse.g.value    = bm.zeros((self._n_output, self._n_output))
        self.rec_synapse.reset_eligibility()
        self._trial_spike_counts[:] = 0.0

    def step(self, stimulus: Stimulus) -> ModelState:
        I_input = stimulus.current + stimulus.spike_train * 100.0
        self.input_pop.update(x=bm.array(I_input, dtype=bm.float32))
        S_pre  = self.input_pop.spike.value.astype(bm.float32)
        S_post = self.output_pop.spike.value.astype(bm.float32)
        I_syn  = self.synapse.update(S_pre, S_post, self.output_pop.V.value)
        I_rec  = self.rec_synapse.update(S_post, S_post, self.output_pop.V.value)
        mean_activity = jnp.mean(S_post)
        I_inh   = -self._inhibition_strength * mean_activity * bm.ones(self._n_output)
        I_total = bm.full(self._n_output, self._I_bg) + I_syn + I_rec + I_inh
        self.output_pop.update(x=I_total)
        self._trial_spike_counts += np.array(
            self.output_pop.spike.value.astype(bm.float32))
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
        self._trial_spike_counts[:] = 0.0
        self.synapse.r1.value     = bm.zeros(self._n_input)
        self.synapse.r2.value     = bm.zeros(self._n_input)
        self.synapse.o1.value     = bm.zeros(self._n_output)
        self.synapse.o2.value     = bm.zeros(self._n_output)
        self.synapse.g.value      = bm.zeros((self._n_input,  self._n_output))
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
        e_ff  = None,
        e_rec = None,
    ) -> None:
        if self.synapse.use_eligibility_trace:
            self.synapse.apply_neuromodulation(modulator, e=e_ff)
            self.rec_synapse.apply_neuromodulation(modulator, e=e_rec)
        else:
            self.synapse.modulator = modulator
        self.homeo.update(self._trial_spike_counts.copy())

    @property
    def weight_stats(self):
        return self.synapse.weight_stats

    @property
    def homeo_stats(self):
        return self.homeo.stats
