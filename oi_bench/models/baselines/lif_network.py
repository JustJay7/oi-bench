"""
LIF Baseline Network — Standard Leaky Integrate-and-Fire

Implements a feedforward LIF network as the ablation baseline for CAdEx.

Purpose in OI-Bench:
  Isolates the contribution of CAdEx-specific features by comparing against
  the simplest biologically-plausible neuron model. Differences between
  LIF and CAdEx results quantify the contribution of:
    - Calcium dynamics
    - Subthreshold adaptation
    - Fractional membrane capacitance

Uses the same TripletSTDPSynapse and HomeostaticPlasticity as CAdEx,
so differences in benchmark performance are attributable to the neuron
model alone, not the learning rules.

Architecture:
  Input population  : LIFNeuron (n_input neurons)
  Output population : LIFNeuron (n_output neurons)
  Synapse           : TripletSTDPSynapse (same as CAdEx)
  Homeostasis       : HomeostaticPlasticity (same as CAdEx)

LIF membrane equation:
  C_m dV/dt = -g_L(V - E_L) + I_ext
  Spike when V >= V_peak, reset to V_r

No adaptation variable w, no calcium dynamics, no fractional memory.
"""

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
    """
    Standard Leaky Integrate-and-Fire neuron.

    Parameters (matched to CAdEx resting state for fair comparison):
      C_m    = 200 pF
      g_L    = 10 nS
      E_L    = -70 mV
      V_peak = 20 mV
      V_r    = -65 mV
    """

    def __init__(
        self,
        size: int,
        C_m: float    = 200.0,
        g_L: float    = 10.0,
        E_L: float    = -70.0,
        V_peak: float = 20.0,
        V_r: float    = -65.0,
        dt: float     = 0.1,
        **kwargs,
    ):
        super().__init__(size=size, **kwargs)
        self.C_m    = C_m
        self.g_L    = g_L
        self.E_L    = E_L
        self.V_peak = V_peak
        self.V_r    = V_r
        self.dt     = dt

        self.V     = bm.Variable(bm.full(size, E_L))
        self.spike = bm.Variable(bm.zeros(size, dtype=bool))
        self.input = bm.Variable(bm.zeros(size))

        # Expose V_T as Variable for homeostasis compatibility
        self.V_T   = bm.Variable(bm.full(size, -50.0))

    def update(self, x=None):
        I_ext = self.input.value if x is None else x
        V     = self.V.value

        dV    = (-self.g_L * (V - self.E_L) + I_ext) / self.C_m
        V_new = V + dV * self.dt
        spike = V_new >= self.V_peak
        V_new = bm.where(spike, self.V_r, V_new)

        self.V.value     = V_new
        self.spike.value = spike
        self.input.value = bm.zeros(self.num)


class LIFNetwork(OIModel):
    """
    LIF baseline network implementing OIModel.

    Identical architecture to CAdExNetwork except:
      - LIFNeuron replaces CAdExNeuron/CAdExFractalNeuron
      - No adaptation, no calcium, no fractional memory

    Parameters match CAdExNetwork defaults for fair comparison.
    """

    def __init__(
        self,
        n_input: int        = 100,
        n_output: int       = 50,
        conn_prob: float    = 1.0,
        dt: float           = 0.1,
        I_background: float = 150.0,
        plasticity: bool    = True,
        homeostasis: bool   = True,
        seed: int           = 0,
    ):
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
            pre        = self.input_pop,
            post       = self.output_pop,
            conn       = conn,
            w_init     = 0.3,
            g_max      = 3.0,
            tau_syn    = 15.0,
            plasticity = plasticity,
        )

        self.homeo = HomeostaticPlasticity(
            neurons      = self.output_pop,
            synapse      = self.synapse,
            r_target     = 5.0,
            trial_dur_ms = 100.0,
            gamma        = 0.5,
            eta_h        = 0.001,
            enabled      = homeostasis,
        )

        self._trial_spike_counts = np.zeros(n_output, dtype=np.float32)

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
        self.input_pop.V.value     = bm.full(self._n_input,  self.input_pop.E_L)
        self.input_pop.spike.value = bm.zeros(self._n_input, dtype=bool)
        self.output_pop.V.value    = bm.full(self._n_output, self.output_pop.E_L)
        self.output_pop.spike.value = bm.zeros(self._n_output, dtype=bool)
        self.synapse.r1.value = bm.zeros(self._n_input)
        self.synapse.r2.value = bm.zeros(self._n_input)
        self.synapse.o1.value = bm.zeros(self._n_output)
        self.synapse.o2.value = bm.zeros(self._n_output)
        self.synapse.g.value  = bm.zeros((self._n_input, self._n_output))
        self._trial_spike_counts[:] = 0.0

    def step(self, stimulus: Stimulus) -> ModelState:
        I_input = stimulus.current + stimulus.spike_train * 100.0
        self.input_pop.update(x=bm.array(I_input, dtype=bm.float32))

        S_pre  = self.input_pop.spike.value.astype(bm.float32)
        S_post = self.output_pop.spike.value.astype(bm.float32)
        I_syn  = self.synapse.update(S_pre, S_post, self.output_pop.V.value)

        self.output_pop.update(x=bm.full(self._n_output, self._I_bg) + I_syn)

        self._trial_spike_counts += np.array(
            self.output_pop.spike.value.astype(bm.float32)
        )

        return ModelState(
            spikes   = self.output_pop.spike.value.astype(bm.float32),
            membrane = self.output_pop.V.value,
            weights  = self.synapse.W.value,
            extras   = {
                'input_spikes': self.input_pop.spike.value.astype(bm.float32),
                'ltp_total':    float(self.synapse.n_ltp.value),
                'ltd_total':    float(self.synapse.n_ltd.value),
            }
        )

    def pre_trial(self, trial_id: int) -> None:
        self._trial_spike_counts[:] = 0.0
        self.synapse.r1.value = bm.zeros(self._n_input)
        self.synapse.r2.value = bm.zeros(self._n_input)
        self.synapse.o1.value = bm.zeros(self._n_output)
        self.synapse.o2.value = bm.zeros(self._n_output)
        self.synapse.g.value  = bm.zeros((self._n_input, self._n_output))

    def post_trial(
        self,
        trial_id: int,
        state_trace: list,
        modulator: float = 1.0,
    ) -> None:
        self.synapse.modulator = modulator
        self.homeo.update(self._trial_spike_counts.copy())

    @property
    def weight_stats(self) -> dict:
        return self.synapse.weight_stats
