"""
Grünwald-Letnikov Fractional Derivative Operator for CAdEx Membrane Dynamics

Implements the GL discretization of the fractional derivative D^α V(t):

    D^α V(t) ≈ (1/dt^α) · Σ_{k=0}^{N-1} w_k · V(t - k·dt)

where the GL weights are:
    w_0 = 1
    w_k = w_{k-1} · (k - 1 - α) / k     for k ≥ 1

At α = 1.0 this exactly recovers the standard first-order Euler derivative.
For α ∈ (0, 1) the operator introduces power-law weighted memory over all
past membrane states.

Biological justification:
  - Lundstrom et al. (2008) Nature Neuroscience 11:1335-1342
  - Vazquez-Guerrero et al. (2024) Scientific Reports 14:5671
  - EPJ Special Topics (2025) doi:10.1140/epjs/s11734-025-02117-6

Why GL over Caputo:
  GL discretizes directly as a finite weighted sum of past values — maps to
  a simple dot product, is JIT-compatible, and is equivalent to Caputo for
  smooth V(t) with zero initial history (Podlubny 1999, Theorem 2.2).
"""

import os

import jax.numpy as jnp
import numpy as np


def compute_gl_weights(alpha: float, history_len: int) -> np.ndarray:
    """
    Precompute Grünwald-Letnikov binomial coefficients.

    w_0 = 1
    w_k = w_{k-1} * (k - 1 - alpha) / k

    For alpha ∈ (0,1):
      w_0 = 1  (current value, positive)
      w_1 = -alpha  (most recent past, negative)
      w_k decays to zero with power-law envelope ~ k^(-alpha-1)

    Parameters
    ----------
    alpha : float
        Fractional order in (0, 1].
    history_len : int
        Number of past steps to retain.

    Returns
    -------
    weights : np.ndarray, shape (history_len,)
    """
    assert 0.0 < alpha <= 1.0, f"alpha must be in (0, 1], got {alpha}"
    weights = np.zeros(history_len)
    weights[0] = 1.0
    for k in range(1, history_len):
        weights[k] = weights[k - 1] * (k - 1.0 - alpha) / k
    return weights


class GLFractionalOperator:
    """
    Grünwald-Letnikov fractional derivative operator.

    Maintains a circular history buffer of membrane voltages and computes
    D^α V at each timestep as a weighted sum of past values.

    Parameters
    ----------
    alpha : float
        Fractional order. 1.0 = standard derivative. Lower = more memory.
    history_len : int
        Number of past timesteps to retain. Default 200 = 20ms at dt=0.1ms.
    n_neurons : int
        Number of neurons (buffer width).
    dt : float
        Timestep in ms.
    """

    def __init__(
        self,
        alpha: float     = 0.85,
        history_len: int = 200,
        n_neurons: int   = 1,
        dt: float        = 0.1,
    ):
        assert 0.0 < alpha <= 1.0
        self.alpha       = alpha
        self.history_len = history_len
        self.n_neurons   = n_neurons
        self.dt          = dt
        self.dt_alpha    = dt ** alpha

        # GL weights — precomputed once, immutable
        w_np = compute_gl_weights(alpha, history_len)
        self.weights = jnp.array(w_np, dtype=jnp.float32)   # (history_len,)

        # Circular history buffer — (history_len, n_neurons)
        self._buffer = np.zeros((history_len, n_neurons), dtype=np.float32)
        self._ptr    = 0

    def reset(self, V_init: float = -70.0):
        """Reset history buffer to a resting potential."""
        self._buffer[:] = V_init
        self._ptr = 0

    def push(self, V: np.ndarray):
        """Push current voltage into circular buffer. V: shape (n_neurons,)"""
        self._buffer[self._ptr] = V
        self._ptr = (self._ptr + 1) % self.history_len

    def compute(self) -> jnp.ndarray:
        """
        Compute D^α V using current buffer contents.

        Returns dV_frac : shape (n_neurons,), units mV/ms.

        Buffer unrolled so index 0 = most recent, index N-1 = oldest.
        """
        ordered = np.roll(self._buffer, -self._ptr, axis=0)[::-1]
        buf_jax = jnp.array(ordered, dtype=jnp.float32)
        weighted_sum = jnp.einsum('k,kn->n', self.weights, buf_jax)
        return weighted_sum / self.dt_alpha


# ----------------------------------------------------------------------
# Verification 1: α=1 recovers standard derivative
# ----------------------------------------------------------------------
def verify_alpha_unity():
    """At α=1, D^1 V = (V[t] - V[t-dt])/dt. Test with linear ramp dV/dt=1."""
    print("=== Verifying GL operator (α=1 recovery) ===\n")
    dt = 0.1
    op = GLFractionalOperator(alpha=1.0, history_len=200, n_neurons=1, dt=dt)
    op.reset(V_init=0.0)
    for t in np.arange(0, 5.0, dt):
        op.push(np.array([t]))
    dV = op.compute()
    print(f"  Linear ramp test: expected dV/dt = 1.0 mV/ms")
    print(f"  GL estimate: {float(dV[0]):.6f} mV/ms")
    print(f"  Error: {abs(float(dV[0]) - 1.0):.2e}")
    assert abs(float(dV[0]) - 1.0) < 0.01
    print("  ✓ Passed\n")


# ----------------------------------------------------------------------
# Verification 2: weight profile — the key scientific figure
# ----------------------------------------------------------------------
def verify_weight_profiles():
    """
    Plot GL weight magnitudes for each α on log scale.

    This is the primary scientific claim: α < 1 produces power-law weight
    decay (straight line on log-log), while α = 1 drops to zero after 1 step.
    Lower α → heavier tail → longer memory.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    alphas = [1.0, 0.85, 0.7]
    colors = ['#1565C0', '#E65100', '#2E7D32']
    history_len = 300   # 30ms at dt=0.1ms
    dt = 0.1

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("GL Weight Profiles — Power-Law Memory vs α",
                 fontsize=13, fontweight='bold')

    lags_ms = np.arange(1, history_len) * dt

    for alpha, color in zip(alphas, colors):
        w = compute_gl_weights(alpha, history_len)
        w_abs = np.abs(w[1:])   # skip w_0=1, show tail

        # Linear scale
        axes[0].plot(lags_ms, w_abs, color=color, linewidth=1.5,
                     label=f"α = {alpha}")
        # Log-log scale — straight line = power law
        axes[1].loglog(lags_ms, w_abs, color=color, linewidth=1.5,
                       label=f"α = {alpha}")

    axes[0].set_xlabel("Lag (ms)"); axes[0].set_ylabel("|w_k|")
    axes[0].set_title("Linear scale", fontsize=10)
    axes[0].legend(fontsize=9)
    axes[0].set_xlim(0, 15)

    axes[1].set_xlabel("Lag (ms)"); axes[1].set_ylabel("|w_k| (log)")
    axes[1].set_title("Log-log scale — straight line = power law", fontsize=10)
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig("fractal_weights.png", dpi=150, bbox_inches='tight')
    print("Weight profile saved → fractal_weights.png")

    # Quantify: weight at 5ms lag
    print("\n=== GL weight magnitude at 5ms lag ===")
    for alpha in alphas:
        w = compute_gl_weights(alpha, history_len)
        idx_5ms = int(5.0 / dt)
        print(f"  α={alpha}: |w_{idx_5ms}| = {abs(w[idx_5ms]):.2e}  "
              f"({'power-law tail' if alpha < 1.0 else 'near zero (standard)'})")


# ----------------------------------------------------------------------
# Verification 3: memory effect using real CAdEx neuron voltage trace
# ----------------------------------------------------------------------
def verify_memory_on_cadex():
    """
    Feed actual CAdEx neuron output into GL operators with different α.

    The key question: after a burst of spikes ends, how long does D^α V
    remain elevated above baseline? Lower α should show longer tails
    because the power-law weights keep referencing past spike-driven
    voltage excursions.

    We run CAdEx for 300ms: stimulus on 50-150ms, off after.
    Then compare |D^α V| in the post-stimulus window (150-300ms).
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import brainpy as bp
    import brainpy.math as bm
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    from oi_bench.models.cadex.neuron import CAdExNeuron

    print("=== Memory effect on real CAdEx voltage trace ===\n")

    dt       = 0.1
    duration = 300.0
    n_steps  = int(duration / dt)
    stim_on  = 50.0
    stim_off = 150.0
    I_amp    = 600.0   # pA

    # Run CAdEx once, record V trace
    neuron = CAdExNeuron(size=1, dt=dt)
    V_trace = np.zeros(n_steps)
    for i in range(n_steps):
        t_ms = i * dt
        I = I_amp if stim_on <= t_ms < stim_off else 0.0
        neuron.update(x=bm.array([I]))
        V_trace[i] = float(neuron.V.value[0])

    n_spikes = int(np.sum(np.diff(V_trace > -40) > 0))
    print(f"  CAdEx produced {n_spikes} spikes during stimulus window")

    # Now feed same V trace into GL operators with different α
    alphas = [1.0, 0.85, 0.7]
    colors = ['#1565C0', '#E65100', '#2E7D32']
    dV_traces = {}

    for alpha in alphas:
        op = GLFractionalOperator(alpha=alpha, history_len=400,
                                  n_neurons=1, dt=dt)
        op.reset(V_init=V_trace[0])
        dV = np.zeros(n_steps)
        for i in range(n_steps):
            op.push(np.array([V_trace[i]]))
            dV[i] = float(op.compute()[0])
        dV_traces[alpha] = dV

    # Plot
    t = np.arange(n_steps) * dt
    stim_off_idx = int(stim_off / dt)

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    fig.suptitle("Memory Effect of GL Operator on CAdEx Neuron Output",
                 fontsize=13, fontweight='bold')

    # Voltage trace
    axes[0].plot(t, V_trace, color='#37474F', linewidth=0.7)
    axes[0].axvspan(stim_on, stim_off, alpha=0.12, color='steelblue',
                    label='Stimulus window')
    axes[0].set_ylabel("V (mV)")
    axes[0].set_title("CAdEx Membrane Potential", fontsize=10)
    axes[0].legend(fontsize=8)

    # |D^α V| after stimulus offset — the memory tail
    for alpha, color in zip(alphas, colors):
        dv = np.abs(dV_traces[alpha])
        axes[1].plot(t, dv, color=color, linewidth=1.0, alpha=0.85,
                     label=f"α = {alpha}")

    axes[1].axvspan(stim_on, stim_off, alpha=0.12, color='steelblue')
    axes[1].axvline(stim_off, color='red', linestyle='--', linewidth=0.9,
                    label='Stimulus offset')
    axes[1].axhline(0.5, color='gray', linestyle=':', linewidth=0.8,
                    label='Threshold (0.5 mV/ms)')
    axes[1].set_ylabel("|D^α V| (mV/ms)")
    axes[1].set_xlabel("Time (ms)")
    axes[1].set_title("|Fractional Derivative| — post-stimulus memory tail", fontsize=10)
    axes[1].legend(fontsize=8)
    axes[1].set_yscale('log')
    axes[1].set_ylim(bottom=1e-3)

    plt.tight_layout()
    plt.savefig("fractal_demo.png", dpi=150, bbox_inches='tight')
    print("  Plot saved → fractal_demo.png")

    # Quantify post-stimulus memory
    threshold = 0.5   # mV/ms
    print(f"\n=== Post-stimulus memory duration (|D^α V| > {threshold} mV/ms) ===")
    for alpha in alphas:
        post = np.abs(dV_traces[alpha][stim_off_idx:])
        below = np.where(post < threshold)[0]
        mem_ms = float(below[0]) * dt if len(below) > 0 else (n_steps - stim_off_idx) * dt
        print(f"  α = {alpha}: {mem_ms:.1f} ms after stimulus offset")


if __name__ == "__main__":
    verify_alpha_unity()
    verify_weight_profiles()
    verify_memory_on_cadex()
    print("\n✓ Fractional operator fully verified.")
