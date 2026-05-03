"""
Liquid State Machine (LSM) Baseline — Spiking Reservoir Network

Fixed random reservoir, no STDP. Output = fixed linear projection.

W_out initialization: Xavier scale 1/sqrt(n_reservoir).
Previous: 0.01 (arbitrary). With 0.01, outputs S_res @ W_out were so
small they rarely crossed the 0.0 threshold, giving near-zero output
spikes. Xavier scale gives correct expected output magnitude.

References:
  Maass et al. (2002) Science 296:2044-2046
  Jaeger (2001) GMD Report 148
  Glorot & Bengio (2010) AISTATS — Xavier initialization
"""

import os
os.environ["JAX_PLATFORMS"] = "mps"

import numpy as np
import brainpy as bp
import brainpy.math as bm

from oi_bench.core.adapter import OIModel
from oi_bench.core.types import Stimulus, ModelState
from oi_bench.models.baselines.lif_network import LIFNeuron


class LiquidStateMachine(OIModel):

    def __init__(
        self,
        n_input: int           = 100,
        n_reservoir: int       = 200,
        n_output: int          = 50,
        conn_prob_in: float    = 0.1,
        conn_prob_rec: float   = 0.1,
        spectral_radius: float = 0.9,
        dt: float              = 0.1,
        I_background: float    = 150.0,
        seed: int              = 0,
    ):
        rng = np.random.RandomState(seed)

        self._n_input     = n_input
        self._n_reservoir = n_reservoir
        self._n_output    = n_output
        self._dt          = dt
        self._I_bg        = I_background

        self.input_pop = LIFNeuron(size=n_input, dt=dt)
        self.reservoir = LIFNeuron(size=n_reservoir, dt=dt)

        # Fixed input → reservoir weights
        W_in_mask  = rng.rand(n_input, n_reservoir) < conn_prob_in
        W_in       = rng.uniform(-1, 1, (n_input, n_reservoir)) * W_in_mask
        self._W_in = bm.array(W_in.astype(np.float32))

        # Fixed recurrent weights scaled to target spectral radius
        W_rec_mask = rng.rand(n_reservoir, n_reservoir) < conn_prob_rec
        np.fill_diagonal(W_rec_mask, False)
        W_rec = rng.randn(n_reservoir, n_reservoir) * W_rec_mask
        if np.any(W_rec != 0):
            eigvals    = np.linalg.eigvals(W_rec)
            current_sr = np.max(np.abs(eigvals))
            if current_sr > 1e-6:
                W_rec = W_rec * (spectral_radius / current_sr)
        self._W_rec = bm.array(W_rec.astype(np.float32))

        # Fixed output projection: Xavier scale = 1/sqrt(n_reservoir)
        # Ensures output magnitudes are correctly scaled regardless of reservoir size.
        # Previous 0.01 was too small — outputs rarely crossed the 0.0 spike threshold.
        xavier_scale = 1.0 / np.sqrt(n_reservoir)
        self._W_out  = (rng.randn(n_reservoir, n_output) * xavier_scale
                        ).astype(np.float32)

        self._g_in    = np.zeros((n_input,     n_reservoir), dtype=np.float32)
        self._g_rec   = np.zeros((n_reservoir, n_reservoir), dtype=np.float32)
        self._tau_syn = 15.0
        self._g_max   = 3.0

    @property
    def n_input(self):  return self._n_input
    @property
    def n_output(self): return self._n_output
    @property
    def dt(self):       return self._dt

    def reset(self):
        self.input_pop.V.value     = bm.full(self._n_input,     self.input_pop.E_L)
        self.input_pop.spike.value = bm.zeros(self._n_input,    dtype=bool)
        self.reservoir.V.value     = bm.full(self._n_reservoir, self.reservoir.E_L)
        self.reservoir.spike.value = bm.zeros(self._n_reservoir, dtype=bool)
        self._g_in[:]  = 0.0
        self._g_rec[:] = 0.0

    def step(self, stimulus: Stimulus) -> ModelState:
        I_input = stimulus.current + stimulus.spike_train * 100.0
        self.input_pop.update(x=bm.array(I_input, dtype=bm.float32))
        S_in  = np.array(self.input_pop.spike.value.astype(bm.float32))
        S_rec = np.array(self.reservoir.spike.value.astype(bm.float32))

        dt = self._dt
        self._g_in  *= (1.0 - dt / self._tau_syn)
        self._g_rec *= (1.0 - dt / self._tau_syn)
        self._g_in  += (np.outer(S_in,  np.ones(self._n_reservoir))
                        * np.array(self._W_in))
        self._g_rec += (np.outer(S_rec, np.ones(self._n_reservoir))
                        * np.array(self._W_rec))

        V_res     = np.array(self.reservoir.V.value)
        I_syn_in  = -self._g_max * self._g_in.sum(axis=0)  * (V_res - 0.0)
        I_syn_rec = -self._g_max * self._g_rec.sum(axis=0) * (V_res - 0.0)
        I_total   = self._I_bg + I_syn_in + I_syn_rec

        self.reservoir.update(x=bm.array(I_total, dtype=bm.float32))
        S_res = np.array(self.reservoir.spike.value.astype(bm.float32))

        output        = S_res @ self._W_out
        output_spikes = (output > 0.0).astype(np.float32)

        return ModelState(
            spikes   = bm.array(output_spikes),
            membrane = bm.zeros(self._n_output),
            weights  = None,
            extras   = {'reservoir_spikes': S_res, 'input_spikes': S_in}
        )

    def pre_trial(self, trial_id: int) -> None:
        self._g_in[:]  = 0.0
        self._g_rec[:] = 0.0

    def post_trial(self, trial_id, state_trace, modulator=1.0):
        pass
