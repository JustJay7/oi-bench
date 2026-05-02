"""
Liquid State Machine (LSM) Baseline — Spiking Reservoir Network

Implements a Liquid State Machine as the no-plasticity baseline for OI-Bench.

Purpose in OI-Bench:
  Isolates the contribution of STDP learning by comparing against a network
  with identical input/output population sizes but FIXED synaptic weights.
  Differences between LSM and CAdEx results quantify what STDP contributes
  beyond fixed reservoir dynamics.

Key distinction from CAdEx:
  - Fixed random recurrent reservoir (weights never change)
  - No STDP, no homeostasis
  - Output = linear projection of reservoir population rate via fixed W_out
  - W_out is NOT trained during the benchmark — it is fixed at initialisation

Why no readout training:
  The benchmark evaluates learning that emerges from the model's internal
  dynamics, not from a separately trained readout. Training W_out would
  conflate reservoir dynamics with supervised learning, making the LSM
  result uninterpretable as a baseline. The LSM output is intentionally a
  fixed random projection — its benchmark score reflects what an unlearned,
  fixed-weight system achieves.

Architecture (Maass et al. 2002, Science 296:2044-2046):
  Input population   : LIFNeuron (n_input)
  Reservoir          : LIFNeuron (n_reservoir, recurrently connected)
  Input→Reservoir    : Random sparse weights (fixed, conn_prob_in=0.1)
  Reservoir→Reservoir: Random sparse weights (fixed, spectral_radius=0.9)
  Output             : Fixed linear projection reservoir → n_output

References:
  Maass et al. (2002) Science 296:2044-2046
  Jaeger (2001) GMD Report 148
"""

import os
os.environ["JAX_PLATFORMS"] = "mps"

import numpy as np
import brainpy as bp
import brainpy.math as bm
import jax.numpy as jnp

from oi_bench.core.adapter import OIModel
from oi_bench.core.types import Stimulus, ModelState
from oi_bench.models.baselines.lif_network import LIFNeuron


class LiquidStateMachine(OIModel):
    """
    LSM baseline implementing OIModel.

    All weights are fixed at initialisation and never updated.
    The output is a fixed random linear projection of reservoir activity.
    """

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

        # Input population
        self.input_pop = LIFNeuron(size=n_input, dt=dt)

        # Reservoir population
        self.reservoir = LIFNeuron(size=n_reservoir, dt=dt)

        # Fixed input → reservoir weights
        W_in_mask   = rng.rand(n_input, n_reservoir) < conn_prob_in
        W_in        = rng.uniform(-1, 1, (n_input, n_reservoir)) * W_in_mask
        self._W_in  = bm.array(W_in.astype(np.float32))

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

        # Fixed output projection: reservoir → n_output
        # Small random weights — NOT trained. The LSM score reflects what
        # a random fixed projection of reservoir activity achieves.
        self._W_out = rng.randn(n_reservoir, n_output).astype(np.float32) * 0.01

        # Synaptic conductance states
        self._g_in    = np.zeros((n_input,     n_reservoir), dtype=np.float32)
        self._g_rec   = np.zeros((n_reservoir, n_reservoir), dtype=np.float32)
        self._tau_syn = 15.0   # ms — same as CAdEx for fair comparison
        self._g_max   = 3.0    # nS

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
        self.input_pop.V.value      = bm.full(self._n_input,     self.input_pop.E_L)
        self.input_pop.spike.value  = bm.zeros(self._n_input,    dtype=bool)
        self.reservoir.V.value      = bm.full(self._n_reservoir, self.reservoir.E_L)
        self.reservoir.spike.value  = bm.zeros(self._n_reservoir, dtype=bool)
        self._g_in[:]  = 0.0
        self._g_rec[:] = 0.0

    def step(self, stimulus: Stimulus) -> ModelState:
        # Step input population
        I_input = stimulus.current + stimulus.spike_train * 100.0
        self.input_pop.update(x=bm.array(I_input, dtype=bm.float32))
        S_in  = np.array(self.input_pop.spike.value.astype(bm.float32))
        S_rec = np.array(self.reservoir.spike.value.astype(bm.float32))

        # Update conductances
        dt = self._dt
        self._g_in  *= (1.0 - dt / self._tau_syn)
        self._g_rec *= (1.0 - dt / self._tau_syn)
        self._g_in  += np.outer(S_in,  np.ones(self._n_reservoir)) \
                       * np.array(self._W_in)
        self._g_rec += np.outer(S_rec, np.ones(self._n_reservoir)) \
                       * np.array(self._W_rec)

        # Reservoir input current
        V_res     = np.array(self.reservoir.V.value)
        I_syn_in  = -self._g_max * self._g_in.sum(axis=0)  * (V_res - 0.0)
        I_syn_rec = -self._g_max * self._g_rec.sum(axis=0) * (V_res - 0.0)
        I_total   = self._I_bg + I_syn_in + I_syn_rec

        self.reservoir.update(x=bm.array(I_total, dtype=bm.float32))
        S_res = np.array(self.reservoir.spike.value.astype(bm.float32))

        # Fixed linear projection of reservoir spikes → output
        output        = S_res @ self._W_out          # (n_reservoir,) @ (n_reservoir, n_output)
        output_spikes = (output > 0.0).astype(np.float32)

        return ModelState(
            spikes   = bm.array(output_spikes),
            membrane = bm.zeros(self._n_output),
            weights  = None,   # LSM does not expose internal weights
            extras   = {
                'reservoir_spikes': S_res,
                'input_spikes':     S_in,
            }
        )

    def pre_trial(self, trial_id: int) -> None:
        # Reset conductances between trials — not weights (they're fixed)
        self._g_in[:]  = 0.0
        self._g_rec[:] = 0.0

    def post_trial(
        self,
        trial_id: int,
        state_trace: list,
        modulator: float = 1.0,
    ) -> None:
        # LSM has no learning — post_trial is a no-op
        pass
