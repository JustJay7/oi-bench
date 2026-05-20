"""
Liquid State Machine (LSM) Baseline — Spiking Reservoir Network

Fixed random reservoir, no STDP. Output = fixed linear projection.

JIT EXECUTION
-------------
Previous version used numpy arrays for conductances (_g_in, _g_rec)
and numpy ops in step(), forcing Python loop fallback at ~40s/trial.

This version stores all mutable state as bm.Variable so bm.for_loop
can JIT-compile the scan, giving ~1-2s/trial identical to LIF/CAdEx.

The runner's _build_step_fn path requires:
  - model.input_pop  : BrainPy NeuDyn with .update(x=...)
  - model.output_pop : BrainPy NeuDyn with .update(x=...)
  - model.synapse    : has .update(S_pre, S_post, V_post, plasticity_scale)
  - model._inhibition_strength : float

LSM satisfies this by implementing a synthetic synapse wrapper
(LSMSynapse) that handles the reservoir dynamics internally and
exposes the same .update() interface as TripletSTDPSynapse.

W_out: Xavier scale 1/sqrt(n_reservoir). Fixed, never trained.

References:
  Maass et al. (2002) Science 296:2044-2046
  Jaeger (2001) GMD Report 148
  Glorot & Bengio (2010) AISTATS
"""


import numpy as np
import brainpy as bp
import brainpy.math as bm
import jax.numpy as jnp

from oi_bench.core.adapter import OIModel
from oi_bench.core.types import Stimulus, ModelState
from oi_bench.models.baselines.lif_network import LIFNeuron


class LSMSynapse(bp.Projection):
    """
    Synthetic synapse wrapper for LSM that satisfies the runner's
    _build_step_fn interface. Internally implements:
      - Input → reservoir conductance (g_in)
      - Reservoir → reservoir conductance (g_rec)
      - Fixed linear readout W_out: reservoir → output

    All state is bm.Variable so bm.for_loop JIT compiles correctly.
    plasticity_scale argument accepted but ignored (LSM never updates weights).
    """

    def __init__(self, pre, post, W_in, W_rec, W_out,
                 tau_syn=15.0, g_max=3.0, dt=0.1):
        super().__init__()
        self.pre     = pre
        self.post    = post
        self.tau_syn = tau_syn
        self.g_max   = g_max
        self.dt      = dt

        n_in  = pre.num
        n_res = W_in.shape[1]
        n_out = post.num

        self.W_in  = bm.array(W_in,  dtype=bm.float32)
        self.W_rec = bm.array(W_rec, dtype=bm.float32)
        self.W_out = bm.array(W_out, dtype=bm.float32)

        # All conductance state as bm.Variable for JIT compatibility
        self.g_in  = bm.Variable(bm.zeros((n_in,  n_res), dtype=bm.float32))
        self.g_rec = bm.Variable(bm.zeros((n_res, n_res), dtype=bm.float32))

        # Expose W as a bm.Variable so weight_stats works
        self.W    = bm.Variable(bm.zeros((n_in, n_out), dtype=bm.float32))
        self.mask = bm.array(np.ones((n_in, n_out), dtype=bool))

        # Reservoir neuron state
        self.reservoir = LIFNeuron(size=n_res, dt=dt)

        # Required by runner interface
        self.plasticity = False
        self.use_eligibility_trace = False
        self.tau_syn = tau_syn

    def update(self, S_pre, S_post, V_post, plasticity_scale=None):
        """
        One timestep of reservoir dynamics.
        S_pre: input spikes (n_input,)
        S_post: output spikes from previous step (n_output,) — unused
        V_post: output membrane voltage — unused
        Returns: I_out (n_output,) — output current from linear readout
        """
        dt = self.dt
        S_res = self.reservoir.spike.value.astype(bm.float32)

        # Decay conductances
        g_in_new  = self.g_in.value  * (1.0 - dt / self.tau_syn)
        g_rec_new = self.g_rec.value * (1.0 - dt / self.tau_syn)

        # Add new spikes
        g_in_new  = g_in_new  + jnp.outer(S_pre, jnp.ones(self.reservoir.num)) \
                    * self.W_in
        g_rec_new = g_rec_new + jnp.outer(S_res, jnp.ones(self.reservoir.num)) \
                    * self.W_rec

        self.g_in.value  = g_in_new
        self.g_rec.value = g_rec_new

        # Reservoir input current
        V_res     = self.reservoir.V.value
        I_syn_in  = -self.g_max * jnp.sum(g_in_new,  axis=0) * (V_res - 0.0)
        I_syn_rec = -self.g_max * jnp.sum(g_rec_new, axis=0) * (V_res - 0.0)
        I_total   = bm.full(self.reservoir.num, 150.0) + I_syn_in + I_syn_rec

        self.reservoir.update(x=I_total)
        S_res_new = self.reservoir.spike.value.astype(bm.float32)

        # Linear readout: reservoir → output (no threshold, just current)
        I_out = jnp.dot(S_res_new, self.W_out)
        return I_out

    def reset(self):
        n_res = self.reservoir.num
        self.g_in.value  = bm.zeros_like(self.g_in.value)
        self.g_rec.value = bm.zeros_like(self.g_rec.value)
        self.reservoir.V.value     = bm.full(n_res, self.reservoir.E_L)
        self.reservoir.spike.value = bm.zeros(n_res, dtype=bool)

    @property
    def weight_stats(self):
        return {'mean': 0.0, 'std': 0.0,
                'frac_silent': 0.0, 'frac_strong': 0.0}


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

        # Input and output populations (runner expects these)
        self.input_pop  = LIFNeuron(size=n_input,  dt=dt)
        self.output_pop = LIFNeuron(size=n_output, dt=dt)

        # Fixed input → reservoir weights
        W_in_mask = rng.rand(n_input, n_reservoir) < conn_prob_in
        W_in      = (rng.uniform(-1, 1, (n_input, n_reservoir))
                     * W_in_mask).astype(np.float32)

        # Fixed recurrent weights scaled to target spectral radius
        W_rec_mask = rng.rand(n_reservoir, n_reservoir) < conn_prob_rec
        np.fill_diagonal(W_rec_mask, False)
        W_rec = (rng.randn(n_reservoir, n_reservoir)
                 * W_rec_mask).astype(np.float32)
        if np.any(W_rec != 0):
            eigvals    = np.linalg.eigvals(W_rec)
            current_sr = np.max(np.abs(eigvals))
            if current_sr > 1e-6:
                W_rec = (W_rec * (spectral_radius / current_sr)).astype(np.float32)

        # Xavier-scaled output projection
        xavier_scale = 1.0 / np.sqrt(n_reservoir)
        W_out = (rng.randn(n_reservoir, n_output) * xavier_scale).astype(np.float32)

        # Single synapse object that encapsulates all reservoir dynamics
        self.synapse = LSMSynapse(
            pre=self.input_pop, post=self.output_pop,
            W_in=W_in, W_rec=W_rec, W_out=W_out,
            tau_syn=15.0, g_max=3.0, dt=dt,
        )

        # Runner expects rec_synapse; give it a dummy no-op
        self.rec_synapse = _NullSynapse(self.output_pop, self.output_pop, n_output)

        # Runner expects _inhibition_strength
        self._inhibition_strength = 0.0  # LSM has no global inhibition
        self._trial_spike_counts  = np.zeros(n_output, dtype=np.float32)

    def configure_homeostasis(self, **kwargs):
        pass  # LSM has no homeostasis

    def configure_stdp(self, **kwargs):
        pass  # LSM has no STDP

    def configure_eligibility_trace(self, **kwargs):
        pass  # LSM has no eligibility trace

    @property
    def n_input(self):  return self._n_input
    @property
    def n_output(self): return self._n_output
    @property
    def dt(self):       return self._dt

    def reset(self):
        self.input_pop.V.value     = bm.full(self._n_input,  self.input_pop.E_L)
        self.input_pop.spike.value = bm.zeros(self._n_input, dtype=bool)
        self.output_pop.V.value    = bm.full(self._n_output, self.output_pop.E_L)
        self.output_pop.spike.value = bm.zeros(self._n_output, dtype=bool)
        self.synapse.reset()
        self._trial_spike_counts[:] = 0.0

    def step(self, stimulus: Stimulus) -> ModelState:
        """Python loop fallback — only used if JIT fails."""
        I_input = stimulus.current
        self.input_pop.update(x=bm.array(I_input, dtype=bm.float32))
        S_pre   = self.input_pop.spike.value.astype(bm.float32)
        S_post  = self.output_pop.spike.value.astype(bm.float32)
        I_out   = self.synapse.update(S_pre, S_post, self.output_pop.V.value)
        # Drive output population with readout current
        self.output_pop.update(x=bm.array(I_out, dtype=bm.float32))
        spikes = self.output_pop.spike.value.astype(bm.float32)
        self._trial_spike_counts += np.array(spikes)
        return ModelState(
            spikes=spikes, membrane=self.output_pop.V.value,
            weights=None, extras={})

    def pre_trial(self, trial_id: int) -> None:
        self._trial_spike_counts[:] = 0.0
        self.synapse.reset()

    def post_trial(self, trial_id, state_trace, modulator=1.0,
                   e_ff=None, e_rec=None):
        pass  # No learning

    @property
    def weight_stats(self):
        return self.synapse.weight_stats


class _NullSynapse:
    """No-op recurrent synapse placeholder for LSM."""
    def __init__(self, pre, post, n):
        self.pre  = pre
        self.post = post
        self.W    = bm.Variable(bm.zeros((n, n), dtype=bm.float32))
        self.mask = bm.array(np.zeros((n, n), dtype=bool))
        self.plasticity            = False
        self.use_eligibility_trace = False
        self.tau_syn               = 15.0

    def update(self, S_pre, S_post, V_post, plasticity_scale=None):
        return bm.zeros(self.post.num)

    def reset_eligibility(self): pass
    def apply_neuromodulation(self, *a, **kw): pass

    @property
    def weight_stats(self):
        return {'mean': 0.0, 'std': 0.0,
                'frac_silent': 0.0, 'frac_strong': 0.0}
