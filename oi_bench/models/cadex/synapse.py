"""
Triplet STDP Synapse with Three-Factor Eligibility Trace

Implements Pfister & Gerstner (2006) triplet STDP extended with
Izhikevich (2007) three-factor eligibility trace.

PLASTICITY SCALING
------------------
synapse.update() accepts a plasticity_scale argument (JAX float32).
  plasticity_scale=1.0: normal STDP (learning trials)
  plasticity_scale=0.0: weights frozen (test trials)

This is a runtime JAX value, not a Python bool. This matters because
bm.for_loop JIT-compiles the step function and caches it — a Python
bool captured at trace time cannot be changed between trials without
invalidating the cache. A JAX scalar multiplier on dW is evaluated at
runtime and correctly freezes weights without recompilation.

The runner passes plasticity_scale via the I_us channel (extra dim) or
directly constructs I_bg with a scale factor. Actually, the simplest
correct approach: runner rebuilds _build_step_fn each trial with
plasticity_scale as a captured JAX constant. Since the constant value
differs between learning and test, JAX traces a new function for each
value — but bm.for_loop caches by shape only, not by constant value.
So we pass plasticity_scale as a dynamic input in the xs array instead.

References:
  Pfister & Gerstner (2006) J. Neurosci. 26(38):9673-9682
  Izhikevich (2007) Cereb. Cortex 17:2443-2452
"""


import numpy as np
import brainpy as bp
import brainpy.math as bm
import jax.numpy as jnp


class TripletSTDPSynapse(bp.Projection):
    """
    Triplet STDP synapse with optional three-factor eligibility trace.
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
        soft_bounds: bool        = False,
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
        self.soft_bounds = soft_bounds
        self.dt = pre.dt

        n_pre  = pre.num
        n_post = post.num

        if isinstance(conn, dict) and 'prob' in conn:
            mask = np.random.rand(n_pre, n_post) < conn['prob']
        elif isinstance(conn, np.ndarray):
            mask = conn.astype(bool)
        else:
            raise ValueError("conn must be {'prob': float} or np.ndarray")
        self.mask     = bm.array(mask, dtype=bool)
        self.mask_jax = jnp.array(mask, dtype=jnp.float32)

        if w_init == 'random':
            W_init = np.random.uniform(w_min, w_max, (n_pre, n_post))
        else:
            W_init = np.full((n_pre, n_post), float(w_init))
        W_init *= mask.astype(float)
        self.W = bm.Variable(bm.array(W_init, dtype=bm.float32))

        self.g  = bm.Variable(bm.zeros((n_pre, n_post), dtype=bm.float32))
        self.r1 = bm.Variable(bm.zeros(n_pre))
        self.r2 = bm.Variable(bm.zeros(n_pre))
        self.o1 = bm.Variable(bm.zeros(n_post))
        self.o2 = bm.Variable(bm.zeros(n_post))
        self.e  = bm.Variable(bm.zeros((n_pre, n_post), dtype=bm.float32))

        self.n_ltp = bm.Variable(bm.array(0.0))
        self.n_ltd = bm.Variable(bm.array(0.0))

    def configure_stdp(self, A2_plus: float, A3_plus: float,
                       A2_minus: float) -> None:
        """Update STDP amplitudes for the current task."""
        self.A2_plus  = A2_plus
        self.A3_plus  = A3_plus
        self.A2_minus = A2_minus

    # ------------------------------------------------------------------
    # Standard mode — plasticity_scale is a JAX runtime scalar
    # ------------------------------------------------------------------

    def update(self, S_pre, S_post, V_post, plasticity_scale=None):
        """
        Standard STDP step.

        plasticity_scale: JAX float32 scalar.
          1.0 = normal learning, 0.0 = weights frozen.
          Passed as a runtime value so test-phase freezing works
          correctly under JIT compilation.
          If None, falls back to self.plasticity Python bool (for
          compatibility with code that calls update() without the arg).
        """
        S_pre  = S_pre.astype(bm.float32)
        S_post = S_post.astype(bm.float32)
        dt = self.dt

        r1 = self.r1.value
        r2 = self.r2.value
        o1 = self.o1.value
        o2 = self.o2.value

        g_new = self.g.value * (1.0 - dt / self.tau_syn)
        g_new = g_new + jnp.outer(S_pre, jnp.ones(self.post.num)) \
                * self.W.value * self.mask
        self.g.value = g_new
        I_syn = -self.g_max * jnp.sum(g_new, axis=0) * (V_post - self.E_syn)

        # Determine effective plasticity scale
        if plasticity_scale is not None:
            p_scale = plasticity_scale
        else:
            p_scale = 1.0 if self.plasticity else 0.0

        ltp_factor = self.A2_plus + self.A3_plus * o2
        dW_ltp     = jnp.outer(r1, S_post * ltp_factor)
        dW_ltd     = self.A2_minus * jnp.outer(S_pre, o1)
        # Soft weight bounds (van Rossum et al. 2000; Gütig et al. 2003):
        # LTP scales with (w_max - W), LTD scales with (W - w_min).
        if self.soft_bounds:
            W_cur  = self.W.value
            dW_ltp = dW_ltp * (self.w_max - W_cur)
            dW_ltd = dW_ltd * (W_cur - self.w_min)
        dW         = (dW_ltp - dW_ltd) * self.mask * self.modulator * p_scale
        self.W.value = jnp.clip(
            self.W.value + dW, self.w_min, self.w_max)
        self.n_ltp.value += jnp.sum(dW_ltp * self.mask) * p_scale
        self.n_ltd.value += jnp.sum(dW_ltd * self.mask) * p_scale

        self.r1.value = r1 * (1.0 - dt / self.tau_plus)  + S_pre
        self.r2.value = r2 * (1.0 - dt / self.tau_x)     + S_pre
        self.o1.value = o1 * (1.0 - dt / self.tau_minus)  + S_post
        self.o2.value = o2 * (1.0 - dt / self.tau_y)      + S_post

        return I_syn

    # ------------------------------------------------------------------
    # Eligibility trace mode
    # ------------------------------------------------------------------

    def stdp_step_functional(
        self, r1, r2, o1, o2, e, g,
        S_pre, S_post, V_post, W,
    ):
        dt = self.dt

        g_new = g * (1.0 - dt / self.tau_syn)
        g_new = g_new + jnp.outer(S_pre, jnp.ones(self.post.num)) \
                * W * self.mask_jax
        I_syn = -self.g_max * jnp.sum(g_new, axis=0) * (V_post - self.E_syn)

        ltp_factor = self.A2_plus + self.A3_plus * o2
        dW_ltp     = jnp.outer(r1, S_post * ltp_factor)
        dW_ltd     = self.A2_minus * jnp.outer(S_pre, o1)
        dW_stdp    = (dW_ltp - dW_ltd) * self.mask_jax

        # C_eff > C_m for CAdEx (α<1) means fewer spikes → weaker eligibility trace.
        # This scale normalises STDP amplitude so T4 eligibility accumulates at a
        # comparable rate regardless of membrane capacitance. For LIF (no C_eff),
        # getattr fallback gives scale=1.0 (no effect).
        cadex_scale = getattr(self.post, 'C_m', 200.0) / getattr(self.post, 'C_eff', 200.0)
        dW_stdp = dW_stdp * cadex_scale

        e_new  = e * (1.0 - dt / self.tau_e) + dW_stdp
        r1_new = r1 * (1.0 - dt / self.tau_plus)  + S_pre
        r2_new = r2 * (1.0 - dt / self.tau_x)     + S_pre
        o1_new = o1 * (1.0 - dt / self.tau_minus)  + S_post
        o2_new = o2 * (1.0 - dt / self.tau_y)      + S_post

        return r1_new, r2_new, o1_new, o2_new, e_new, g_new, I_syn

    def apply_neuromodulation(
        self, da_level: float, e: jnp.ndarray = None,
        lr: float = 0.05,
    ) -> None:
        if not self.plasticity:
            return
        if abs(da_level) < 1e-8:
            return
        e_use = e if e is not None else self.e.value
        dW    = da_level * lr * e_use * self.mask_jax
        self.W.value = jnp.clip(
            self.W.value + dW, self.w_min, self.w_max)
                        
    def reset_eligibility(self) -> None:
        self.e.value = bm.zeros_like(self.e.value)

    def get_initial_functional_state(self):
        n_pre  = self.pre.num
        n_post = self.post.num
        return (
            jnp.zeros(n_pre), jnp.zeros(n_pre),
            jnp.zeros(n_post), jnp.zeros(n_post),
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
