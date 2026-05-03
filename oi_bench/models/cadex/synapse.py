"""
Triplet STDP Synapse with Three-Factor Eligibility Trace

Implements Pfister & Gerstner (2006) triplet STDP extended with
Izhikevich (2007) three-factor eligibility trace.

TWO EXECUTION MODES
--------------------
1. Standard mode (use_eligibility_trace=False):
   update() applies dW directly to W each timestep.
   Compatible with bm.for_loop JIT scan.

2. Eligibility trace mode (use_eligibility_trace=True):
   stdp_step_functional() computes dW and accumulates into eligibility
   trace e_ij as pure JAX array operations with no Variable side effects.
   Used with jax.lax.scan in runner.py where carry threads e explicitly.
   apply_neuromodulation(da_level, e) gates the final trace into W.

The functional path exists because JAX's XLA compilation treats BrainPy
Variable.value updates as static — side effects inside a scan are silently
dropped. Pure functional state (carry arrays) threads correctly through
jax.lax.scan.

References:
  Pfister & Gerstner (2006) J. Neurosci. 26(38):9673-9682
  Izhikevich (2007) Cereb. Cortex 17:2443-2452
"""

import os
os.environ["JAX_PLATFORMS"] = "mps"

import numpy as np
import brainpy as bp
import brainpy.math as bm
import jax.numpy as jnp


class TripletSTDPSynapse(bp.Projection):
    """
    Triplet STDP synapse with optional three-factor eligibility trace.

    Parameters
    ----------
    pre, post : bp.dyn.NeuDyn
    conn : dict {'prob': p} or np.ndarray (n_pre, n_post)
    use_eligibility_trace : bool
        If True, stdp_step_functional() is used via jax.lax.scan carry.
        If False, update() applies dW directly each step.
    tau_e : float
        Eligibility trace decay (ms). Default 1000ms.
    """

    def __init__(
        self,
        pre, post, conn,
        use_eligibility_trace: bool = False,
        tau_e: float             = 1000.0,
        A2_plus: float           = 0.006,
        A3_plus: float           = 0.009,
        A2_minus: float          = 0.003,
        tau_plus: float          = 16.8,
        tau_minus: float         = 33.7,
        tau_x: float             = 101.0,
        tau_y: float             = 125.0,
        tau_syn: float           = 15.0,
        E_syn: float             = 0.0,
        g_max: float             = 3.0,
        w_min: float             = 0.0,
        w_max: float             = 1.0,
        w_init                   = 'random',
        plasticity: bool         = True,
        modulator: float         = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.pre  = pre
        self.post = post
        self.use_eligibility_trace = use_eligibility_trace
        self.tau_e     = tau_e
        self.A2_plus   = A2_plus
        self.A3_plus   = A3_plus
        self.A2_minus  = A2_minus
        self.tau_plus  = tau_plus
        self.tau_minus = tau_minus
        self.tau_x     = tau_x
        self.tau_y     = tau_y
        self.tau_syn   = tau_syn
        self.E_syn     = E_syn
        self.g_max     = g_max
        self.w_min     = w_min
        self.w_max     = w_max
        self.plasticity  = plasticity
        self.modulator   = modulator
        self.dt = pre.dt

        n_pre  = pre.num
        n_post = post.num

        # Connectivity mask
        if isinstance(conn, dict) and 'prob' in conn:
            mask = np.random.rand(n_pre, n_post) < conn['prob']
        elif isinstance(conn, np.ndarray):
            mask = conn.astype(bool)
        else:
            raise ValueError("conn must be {'prob': float} or np.ndarray")
        self.mask    = bm.array(mask, dtype=bool)
        # JAX array version of mask for functional path
        self.mask_jax = jnp.array(mask, dtype=jnp.float32)

        # Synaptic weights
        if w_init == 'random':
            W_init = np.random.uniform(w_min, w_max, (n_pre, n_post))
        else:
            W_init = np.full((n_pre, n_post), float(w_init))
        W_init *= mask.astype(float)
        self.W = bm.Variable(bm.array(W_init, dtype=bm.float32))

        # Synaptic conductance (Variable — used in standard mode)
        self.g = bm.Variable(bm.zeros((n_pre, n_post), dtype=bm.float32))

        # STDP traces (Variables — used in standard mode)
        self.r1 = bm.Variable(bm.zeros(n_pre))
        self.r2 = bm.Variable(bm.zeros(n_pre))
        self.o1 = bm.Variable(bm.zeros(n_post))
        self.o2 = bm.Variable(bm.zeros(n_post))

        # Eligibility trace (Variable — only used for reset/init,
        # actual accumulation in eligibility mode uses functional path)
        self.e = bm.Variable(bm.zeros((n_pre, n_post), dtype=bm.float32))

        # Cumulative statistics
        self.n_ltp = bm.Variable(bm.array(0.0))
        self.n_ltd = bm.Variable(bm.array(0.0))

    # ------------------------------------------------------------------
    # Standard mode: Variable-based update (compatible with bm.for_loop)
    # ------------------------------------------------------------------

    def update(self, S_pre, S_post, V_post):
        """
        Standard STDP step — updates Variables directly.
        Use with bm.for_loop for all tasks except T4.
        """
        S_pre  = S_pre.astype(bm.float32)
        S_post = S_post.astype(bm.float32)
        dt = self.dt

        r1 = self.r1.value
        r2 = self.r2.value
        o1 = self.o1.value
        o2 = self.o2.value

        # Synaptic conductance
        g_new = self.g.value * (1.0 - dt / self.tau_syn)
        g_new = g_new + jnp.outer(S_pre, jnp.ones(self.post.num)) \
                * self.W.value * self.mask
        self.g.value = g_new
        I_syn = -self.g_max * jnp.sum(g_new, axis=0) * (V_post - self.E_syn)

        # Triplet STDP
        if self.plasticity:
            ltp_factor = self.A2_plus + self.A3_plus * o2
            dW_ltp     = jnp.outer(r1, S_post * ltp_factor)
            dW_ltd     = self.A2_minus * jnp.outer(S_pre, o1)
            dW         = (dW_ltp - dW_ltd) * self.mask * self.modulator
            self.W.value = jnp.clip(
                self.W.value + dW, self.w_min, self.w_max)
            self.n_ltp.value += jnp.sum(dW_ltp * self.mask)
            self.n_ltd.value += jnp.sum(dW_ltd * self.mask)

        self.r1.value = r1 * (1.0 - dt / self.tau_plus)  + S_pre
        self.r2.value = r2 * (1.0 - dt / self.tau_x)     + S_pre
        self.o1.value = o1 * (1.0 - dt / self.tau_minus)  + S_post
        self.o2.value = o2 * (1.0 - dt / self.tau_y)      + S_post

        return I_syn

    # ------------------------------------------------------------------
    # Eligibility trace mode: pure functional step for jax.lax.scan
    # ------------------------------------------------------------------

    def stdp_step_functional(
        self,
        r1, r2, o1, o2, e, g,
        S_pre, S_post, V_post, W,
    ):
        """
        Pure functional STDP step — no Variable side effects.

        All state passed in as JAX arrays, all state returned as JAX arrays.
        Compatible with jax.lax.scan carry threading.

        Parameters (all JAX arrays):
          r1, r2 : (n_pre,)   — pre-synaptic STDP traces
          o1, o2 : (n_post,)  — post-synaptic STDP traces
          e      : (n_pre, n_post) — eligibility trace
          g      : (n_pre, n_post) — synaptic conductance
          S_pre  : (n_pre,)   — pre-synaptic spikes this step
          S_post : (n_post,)  — post-synaptic spikes this step
          V_post : (n_post,)  — post-synaptic membrane voltage
          W      : (n_pre, n_post) — current weight matrix (read-only here)

        Returns:
          (r1_new, r2_new, o1_new, o2_new, e_new, g_new, I_syn)
        """
        dt = self.dt

        # Synaptic conductance
        g_new = g * (1.0 - dt / self.tau_syn)
        g_new = g_new + jnp.outer(S_pre, jnp.ones(self.post.num)) \
                * W * self.mask_jax
        I_syn = -self.g_max * jnp.sum(g_new, axis=0) * (V_post - self.E_syn)

        # STDP — accumulate into eligibility trace, NOT into W
        ltp_factor = self.A2_plus + self.A3_plus * o2
        dW_ltp     = jnp.outer(r1, S_post * ltp_factor)
        dW_ltd     = self.A2_minus * jnp.outer(S_pre, o1)
        dW         = (dW_ltp - dW_ltd) * self.mask_jax

        # Eligibility trace: leaky accumulator of STDP events
        e_new = e * (1.0 - dt / self.tau_e) + dW

        # Update STDP traces
        r1_new = r1 * (1.0 - dt / self.tau_plus)  + S_pre
        r2_new = r2 * (1.0 - dt / self.tau_x)     + S_pre
        o1_new = o1 * (1.0 - dt / self.tau_minus)  + S_post
        o2_new = o2 * (1.0 - dt / self.tau_y)      + S_post

        return r1_new, r2_new, o1_new, o2_new, e_new, g_new, I_syn

    def apply_neuromodulation(self, da_level: float, e: jnp.ndarray = None) -> None:
        """
        Gate eligibility trace into weight change via dopamine signal.

        In functional mode: pass e explicitly (from scan carry final state).
        In Variable mode: reads self.e.value.

        Parameters
        ----------
        da_level : float
            Dopamine signal. Positive = reward.
        e : jnp.ndarray or None
            Eligibility trace from scan carry. If None, uses self.e.value.
        """
        if not self.plasticity:
            return
        if abs(da_level) < 1e-8:
            return

        e_use = e if e is not None else self.e.value
        dW    = da_level * e_use * self.mask_jax
        self.W.value = jnp.clip(
            self.W.value + dW, self.w_min, self.w_max)

    def reset_eligibility(self) -> None:
        """Reset eligibility trace Variable."""
        self.e.value = bm.zeros_like(self.e.value)

    def get_initial_functional_state(self):
        """
        Return initial carry state for jax.lax.scan functional path.
        Returns (r1, r2, o1, o2, e, g) as JAX arrays.
        """
        n_pre  = self.pre.num
        n_post = self.post.num
        return (
            jnp.zeros(n_pre),
            jnp.zeros(n_pre),
            jnp.zeros(n_post),
            jnp.zeros(n_post),
            jnp.zeros((n_pre, n_post)),
            jnp.zeros((n_pre, n_post)),
        )

    @property
    def weight_stats(self) -> dict:
        W_conn = np.array(self.W.value)[np.array(self.mask)]
        if len(W_conn) == 0:
            return {}
        return {
            'mean':          float(np.mean(W_conn)),
            'std':           float(np.std(W_conn)),
            'frac_silent':   float(np.mean(W_conn < 0.05)),
            'frac_strong':   float(np.mean(W_conn > 0.8)),
            'ltp_ltd_ratio': float(self.n_ltp.value / (self.n_ltd.value + 1e-10)),
        }
