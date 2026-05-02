"""
Liquid State Machine (LSM) Baseline — Spiking Reservoir Network

Implements a Liquid State Machine as the no-plasticity baseline for OI-Bench.

Purpose in OI-Bench:
  Isolates the contribution of STDP learning by comparing against a network
  with identical neuron dynamics but FIXED synaptic weights. Differences
  between LSM and CAdEx results quantify what STDP learning contributes
  beyond fixed reservoir dynamics.

Key distinction from CAdEx:
  - Recurrent reservoir with random fixed weights (not trained)
  - Linear readout trained via ridge regression on output population activity
  - No STDP, no homeostasis
  - Weights never change during the benchmark

Architecture (Maass et al. 2002, Science 296:2044-2046):
  Input population   : LIFNeuron (n_input)
  Reservoir          : LIFNeuron (n_reservoir, recurrently connected)
  Readout            : Linear regression on reservoir population rate
  Input→Reservoir    : Random sparse weights (fixed)
  Reservoir→Reservoir: Random sparse recurrent weights (fixed, spectral radius < 1)

Spectral radius < 1 ensures echo state property — reservoir activity
decays without input, preventing runaway dynamics.

References:
  Maass et al. (2002) Science 296:2044-2046 — LSM original
  Jaeger (2001) GMD Report 148 — Echo State Networks
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

    Parameters
    ----------
    n_input : int
        Input population size. Default 100.
    n_reservoir : int
        Reservoir size. Default 200.
    n_output : int
        Readout population size (virtual — LSM has no output neurons,
        the readout is a linear projection from reservoir activity).
        Default 50.
    conn_prob_in : float
        Input→reservoir connection probability. Default 0.1.
    conn_prob_rec : float
        Reservoir→reservoir connection probability. Default 0.1.
    spectral_radius : float
        Target spectral radius of recurrent weight matrix. Default 0.9.
        Must be < 1 for echo state property.
    dt : float
        Timestep in ms. Default 0.1.
    I_background : float
        Background current to reservoir (pA). Default 150.0.
    seed : int
        Random seed. Default 0.
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

        # --- Fixed input → reservoir weights ---
        W_in_mask = rng.rand(n_input, n_reservoir) < conn_prob_in
        W_in      = rng.uniform(-1, 1, (n_input, n_reservoir)) * W_in_mask
        self._W_in = bm.array(W_in.astype(np.float32))   # fixed

        # --- Fixed recurrent weights with target spectral radius ---
        W_rec_mask = rng.rand(n_reservoir, n_reservoir) < conn_prob_rec
        np.fill_diagonal(W_rec_mask, False)   # no self-connections
        W_rec = rng.randn(n_reservoir, n_reservoir) * W_rec_mask

        # Scale to target spectral radius
        if np.any(W_rec != 0):
            eigvals = np.linalg.eigvals(W_rec)
            current_sr = np.max(np.abs(eigvals))
            if current_sr > 1e-6:
                W_rec = W_rec * (spectral_radius / current_sr)

        self._W_rec = bm.array(W_rec.astype(np.float32))  # fixed

        # --- Readout weights: linear projection reservoir → n_output ---
        # Initialised randomly, updated via ridge regression in post_trial()
        self._W_out = rng.randn(n_reservoir, n_output).astype(np.float32) * 0.01

        # Reservoir activity buffer for readout training
        self._reservoir_trace = []   # list of (n_reservoir,) arrays per step
        self._trial_targets   = []   # list of target outputs per trial

        # Conductance states for input→reservoir and rec→reservoir
        self._g_in  = np.zeros((n_input,     n_reservoir), dtype=np.float32)
        self._g_rec = np.zeros((n_reservoir, n_reservoir), dtype=np.float32)
        self._tau_syn = 15.0   # ms, NMDA-like (same as CAdEx for fair comparison)
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
        self.input_pop.V.value     = bm.full(self._n_input,     self.input_pop.E_L)
        self.input_pop.spike.value = bm.zeros(self._n_input,    dtype=bool)
        self.reservoir.V.value     = bm.full(self._n_reservoir, self.reservoir.E_L)
        self.reservoir.spike.value = bm.zeros(self._n_reservoir, dtype=bool)
        self._g_in[:]  = 0.0
        self._g_rec[:] = 0.0
        self._reservoir_trace = []

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
        V_res   = np.array(self.reservoir.V.value)
        I_syn_in  = -self._g_max * self._g_in.sum(axis=0)  * (V_res - 0.0)
        I_syn_rec = -self._g_max * self._g_rec.sum(axis=0) * (V_res - 0.0)
        I_total   = self._I_bg + I_syn_in + I_syn_rec

        self.reservoir.update(x=bm.array(I_total, dtype=bm.float32))
        S_res = np.array(self.reservoir.spike.value.astype(bm.float32))

        # Buffer reservoir activity for readout
        self._reservoir_trace.append(S_res.copy())

        # Readout: linear projection of reservoir spikes → output
        output = S_res @ self._W_out   # (n_reservoir,) @ (n_reservoir, n_output)
        output_spikes = (output > 0.5).astype(np.float32)

        return ModelState(
            spikes   = bm.array(output_spikes),
            membrane = bm.zeros(self._n_output),   # LSM has no output membrane
            weights  = None,                        # LSM hides internal weights
            extras   = {
                'reservoir_spikes': S_res,
                'input_spikes':     S_in,
                'readout_output':   output,
            }
        )

    def pre_trial(self, trial_id: int) -> None:
        self._reservoir_trace = []

    def post_trial(
        self,
        trial_id: int,
        state_trace: list,
        modulator: float = 1.0,
    ) -> None:
        """
        Update readout weights via ridge regression on reservoir activity.

        This is the ONLY learning that happens in the LSM — the readout.
        The reservoir itself never changes.
        """
        # Only update readout if we have enough data
        if len(self._reservoir_trace) < 10:
            return

        # Build activity matrix X: (n_steps, n_reservoir)
        X = np.array(self._reservoir_trace, dtype=np.float32)

        # Target: desired output (ones for all output neurons — simple baseline)
        # In Task T1, the runner will provide trial labels via modulator
        # For now, use mean reservoir activity as self-supervised target
        target = np.tile(X.mean(axis=0)[:self._n_output],
                         (len(X), 1))

        # Ridge regression: W_out = (X^T X + λI)^{-1} X^T Y
        lambda_reg = 1e-4
        XtX = X.T @ X + lambda_reg * np.eye(self._n_reservoir)
        XtY = X.T @ target
        try:
            self._W_out = np.linalg.solve(XtX, XtY).astype(np.float32)
        except np.linalg.LinAlgError:
            pass   # keep existing weights if solve fails
