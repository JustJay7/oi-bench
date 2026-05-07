import os
os.environ["JAX_PLATFORMS"] = "mps"

import numpy as np
import brainpy as bp
import brainpy.math as bm
import jax.numpy as jnp

from oi_bench.core.adapter import OIModel
from oi_bench.core.types import Stimulus, ModelState
from oi_bench.models.cadex.synapse import TripletSTDPSynapse
from oi_bench.models.cadex.homeostatic import HomeostaticPlasticity


class LIFNeuron(bp.dyn.NeuDyn):
    def __init__(self, size, C_m=200., g_L=10., E_L=-70., V_peak=20.,
                 V_r=-65., tau_ref=2.0, dt=0.1, **kwargs):
        super().__init__(size=size, **kwargs)
        self.C_m = C_m; self.g_L = g_L; self.E_L = E_L
        self.V_peak = V_peak; self.V_r = V_r; self.tau_ref = tau_ref; self.dt = dt
        self.V         = bm.Variable(bm.full(size, E_L))
        self.spike     = bm.Variable(bm.zeros(size, dtype=bool))
        self.input     = bm.Variable(bm.zeros(size))
        self.V_T       = bm.Variable(bm.full(size, -50.0))
        self.ref_count = bm.Variable(bm.zeros(size))

    def update(self, x=None):
        I_ext = self.input.value if x is None else x
        ref   = self.ref_count.value
        V     = self.V.value
        dV    = (-self.g_L * (V - self.E_L) + I_ext) / self.C_m
        V_new = V + dV * self.dt
        in_ref    = (ref > 0.0).astype(bm.float32)
        V_thr_eff = self.V_T.value + in_ref * 1000.0
        spike     = V_new >= V_thr_eff
        V_new     = bm.where(spike, self.V_r, V_new)
        ref_new   = bm.where(spike, self.tau_ref, bm.maximum(ref - self.dt, 0.0))
        self.V.value         = V_new
        self.spike.value     = spike
        self.ref_count.value = ref_new
        self.input.value     = bm.zeros(self.num)


class LIFNetwork(OIModel):
    def __init__(self, n_input=100, n_output=50, conn_prob=0.2, dt=0.1,
                 I_background=150.0, plasticity=True, homeostasis=True, seed=0):
        np.random.seed(seed)
        self._n_input  = n_input
        self._n_output = n_output
        self._dt       = dt
        self._I_bg     = I_background

        self.input_pop  = LIFNeuron(size=n_input,  dt=dt)
        self.output_pop = LIFNeuron(size=n_output, dt=dt)

        conn = (np.ones((n_input, n_output), dtype=bool)
                if conn_prob >= 1.0 else {'prob': conn_prob})

        self.synapse = TripletSTDPSynapse(
            pre=self.input_pop, post=self.output_pop,
            conn=conn, w_init=0.5, g_max=0.3, tau_syn=15.0, w_min=0.0, w_max=1.0, soft_bounds=True,
            plasticity=plasticity,
        )

        rec_conn = np.random.rand(n_output, n_output) < 0.2
        np.fill_diagonal(rec_conn, False)
        self.rec_synapse = TripletSTDPSynapse(
            pre=self.output_pop, post=self.output_pop,
            conn=rec_conn, w_init=0.1, g_max=0.1, tau_syn=15.0, w_min=0.0, w_max=1.0, soft_bounds=True,
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

    def configure_stdp(self, A2_plus=0.006, A3_plus=0.009, A2_minus=0.003):
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
        self.input_pop.V.value         = bm.full(self._n_input,  self.input_pop.E_L)
        self.input_pop.spike.value     = bm.zeros(self._n_input, dtype=bool)
        self.input_pop.ref_count.value = bm.zeros(self._n_input)
        self.output_pop.V.value        = bm.full(self._n_output, self.output_pop.E_L)
        self.output_pop.spike.value    = bm.zeros(self._n_output, dtype=bool)
        self.output_pop.ref_count.value = bm.zeros(self._n_output)
        self.synapse.r1.value     = bm.zeros(self._n_input)
        self.synapse.r2.value     = bm.zeros(self._n_input)
        self.synapse.o1.value     = bm.zeros(self._n_output)
        self.synapse.o2.value     = bm.zeros(self._n_output)
        self.synapse.g.value      = bm.zeros((self._n_input,  self._n_output))
        self.synapse.reset_eligibility()
        self.rec_synapse.r1.value = bm.zeros(self._n_output)
        self.rec_synapse.r2.value = bm.zeros(self._n_output)
        self.rec_synapse.o1.value = bm.zeros(self._n_output)
        self.rec_synapse.o2.value = bm.zeros(self._n_output)
        self.rec_synapse.g.value  = bm.zeros((self._n_output, self._n_output))
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
        I_inh  = -self._inhibition_strength * mean_activity * bm.ones(self._n_output)
        I_total = bm.full(self._n_output, self._I_bg) + I_syn + I_rec + I_inh
        self.output_pop.update(x=I_total)
        spikes = self.output_pop.spike.value.astype(bm.float32)
        self._trial_spike_counts += np.array(spikes)
        return ModelState(
            spikes   = spikes,
            membrane = self.output_pop.V.value,
            weights  = self.synapse.W.value,
            extras   = {
                'input_spikes': self.input_pop.spike.value.astype(bm.float32),
                'ltp_total':    float(self.synapse.n_ltp.value),
                'ltd_total':    float(self.synapse.n_ltd.value),
            }
        )

    def pre_trial(self, trial_id: int):
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
        self.input_pop.ref_count.value  = bm.zeros(self._n_input)
        self.output_pop.ref_count.value = bm.zeros(self._n_output)

    def post_trial(self, trial_id, state_trace, modulator=1.0, e_ff=None, e_rec=None):
        if self.synapse.use_eligibility_trace:
            self.synapse.apply_neuromodulation(modulator, e=e_ff)
            self.rec_synapse.apply_neuromodulation(modulator, e=e_rec)
        else:
            self.synapse.modulator = modulator
        self.homeo.update(self._trial_spike_counts.copy())

    @property
    def weight_stats(self): return self.synapse.weight_stats
    @property
    def homeo_stats(self):  return self.homeo.stats
