"""
Triplet STDP Synapse

Implements the triplet STDP rule from Pfister & Gerstner (2006),
J. Neurosci. 26(38):9673-9682.

TRIPLET STDP EQUATIONS (Eq. 3 & 4)
------------------------------------
Four traces per synapse: r1, r2 (pre-synaptic), o1, o2 (post-synaptic)

Trace dynamics (exponential decay + spike increment):
  dr1/dt = -r1/τ+   | r1 += 1 at each pre spike
  dr2/dt = -r2/τx   | r2 += 1 at each pre spike
  do1/dt = -o1/τ-   | o1 += 1 at each post spike
  do2/dt = -o2/τy   | o2 += 1 at each post spike

Weight updates (nearest-neighbour, traces read BEFORE increment):
  At each PRE spike:
    LTD: ΔW -= A2- · o1(t_pre)

  At each POST spike:
    LTP: ΔW += r1 · (A2+ + A3+ · o2(t_post-ε))

A2+ = pairwise LTP term (active even on first spike pair)
A3+ · o2 = triplet LTP term (amplified by recent post history)

Time constants — visual cortex fit, Pfister & Gerstner (2006) Table 1:
  τ+ = 16.8ms, τ- = 33.7ms, τx = 101ms, τy = 125ms

SYNAPTIC CONDUCTANCE
---------------------
Exponential conductance (not delta pulse) per connection:
  dg_ij/dt = -g_ij/τ_syn + W_ij · S_pre_i
  I_syn_j  = g_max · Σ_i g_ij · (V_post_j - E_syn)

τ_syn = 15ms (NMDA-like). Without sustained conductance, sparse pre spikes
(dt=0.1ms pulses) cannot drive post above threshold.

THREE-FACTOR HOOK
-----------------
self.modulator (default 1.0) scales all weight updates.
Pass neuromodulatory signal for neoHebbian learning (Stage 4, future work).

RESOURCE-DEPENDENT HETEROSYNAPTIC STABILITY
--------------------------------------------
Deferred to Stage 4. Requires recurrent network architecture to function
correctly. Will implement Humble (2025) in full with:
  - Recurrent connectivity (25% probability)
  - Poisson input spike trains
  - Axonal delays (1-5ms uniform)
  - Lognormal resource pool initialization
  - Weight-dependent depression offset (0.18)
"""

import os
os.environ["JAX_PLATFORMS"] = "mps"

import numpy as np
import brainpy as bp
import brainpy.math as bm
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class TripletSTDPSynapse(bp.Projection):
    """
    Triplet STDP synapse with exponential synaptic conductance.

    Parameters
    ----------
    pre, post : bp.dyn.NeuDyn
    conn : dict {'prob': p} or np.ndarray (n_pre, n_post)
    A2_plus : float   Pairwise LTP amplitude. Default 0.006.
    A3_plus : float   Triplet LTP amplitude.  Default 0.009.
    A2_minus : float  Pairwise LTD amplitude. Default 0.003.
    tau_plus : float  Fast pre trace τ+ (ms).  Default 16.8.
    tau_minus : float Fast post trace τ- (ms). Default 33.7.
    tau_x : float     Slow pre trace τx (ms).  Default 101.0.
    tau_y : float     Slow post trace τy (ms). Default 125.0.
    tau_syn : float   Conductance decay (ms).  Default 15.0.
    E_syn : float     Reversal potential (mV). Default 0.0 (excitatory).
    g_max : float     Peak conductance (nS).   Default 3.0.
    w_min, w_max : float  Weight bounds [0, 1].
    w_init : float or 'random'.
    plasticity : bool Default True.
    modulator : float Three-factor hook. Default 1.0.
    """

    def __init__(
        self,
        pre, post, conn,
        A2_plus: float   = 0.006,
        A3_plus: float   = 0.009,
        A2_minus: float  = 0.003,
        tau_plus: float  = 16.8,
        tau_minus: float = 33.7,
        tau_x: float     = 101.0,
        tau_y: float     = 125.0,
        tau_syn: float   = 15.0,
        E_syn: float     = 0.0,
        g_max: float     = 3.0,
        w_min: float     = 0.0,
        w_max: float     = 1.0,
        w_init           = 'random',
        plasticity: bool = True,
        modulator: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.pre = pre;  self.post = post
        self.A2_plus  = A2_plus;   self.A3_plus  = A3_plus
        self.A2_minus = A2_minus
        self.tau_plus = tau_plus;  self.tau_minus = tau_minus
        self.tau_x    = tau_x;     self.tau_y     = tau_y
        self.tau_syn  = tau_syn;   self.E_syn     = E_syn
        self.g_max    = g_max
        self.w_min    = w_min;     self.w_max     = w_max
        self.plasticity = plasticity
        self.modulator  = modulator
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
        self.mask = bm.array(mask, dtype=bool)

        # Synaptic weights
        if w_init == 'random':
            W_init = np.random.uniform(w_min, w_max, (n_pre, n_post))
        else:
            W_init = np.full((n_pre, n_post), float(w_init))
        W_init *= mask.astype(float)
        self.W = bm.Variable(bm.array(W_init, dtype=bm.float32))

        # Synaptic conductance state — (n_pre, n_post)
        self.g = bm.Variable(bm.zeros((n_pre, n_post), dtype=bm.float32))

        # Four STDP traces
        self.r1 = bm.Variable(bm.zeros(n_pre))   # fast pre
        self.r2 = bm.Variable(bm.zeros(n_pre))   # slow pre
        self.o1 = bm.Variable(bm.zeros(n_post))  # fast post
        self.o2 = bm.Variable(bm.zeros(n_post))  # slow post

        # Cumulative statistics
        self.n_ltp = bm.Variable(bm.array(0.0))
        self.n_ltd = bm.Variable(bm.array(0.0))

    def update(self, S_pre, S_post, V_post):
        """
        One STDP timestep.

        Parameters
        ----------
        S_pre  : float array (n_pre,)  — pre spikes this step
        S_post : float array (n_post,) — post spikes this step
        V_post : float array (n_post,) — post membrane voltage

        Returns
        -------
        I_syn : float array (n_post,) — pA, positive = depolarising
        """
        S_pre  = S_pre.astype(bm.float32)
        S_post = S_post.astype(bm.float32)
        dt = self.dt

        r1 = self.r1.value;  r2 = self.r2.value
        o1 = self.o1.value;  o2 = self.o2.value

        # --- Exponential synaptic conductance ---
        g_new = self.g.value * (1.0 - dt / self.tau_syn)
        g_new = g_new + jnp.outer(S_pre, jnp.ones(self.post.num)) \
                * self.W.value * self.mask
        self.g.value = g_new
        I_syn = -self.g_max * jnp.sum(g_new, axis=0) * (V_post - self.E_syn)

        # --- Triplet STDP ---
        if self.plasticity:
            # LTP at post spike — read o2 BEFORE increment (t_post-ε)
            ltp_factor = self.A2_plus + self.A3_plus * o2
            dW_ltp     = jnp.outer(r1, S_post * ltp_factor)

            # LTD at pre spike — read o1 BEFORE increment
            dW_ltd     = self.A2_minus * jnp.outer(S_pre, o1)

            dW = (dW_ltp - dW_ltd) * self.mask * self.modulator
            self.W.value = jnp.clip(self.W.value + dW, self.w_min, self.w_max)

            self.n_ltp.value = self.n_ltp.value + jnp.sum(dW_ltp * self.mask)
            self.n_ltd.value = self.n_ltd.value + jnp.sum(dW_ltd * self.mask)

        # --- Update traces AFTER weight change (t_post-ε convention) ---
        self.r1.value = r1 * (1.0 - dt / self.tau_plus)  + S_pre
        self.r2.value = r2 * (1.0 - dt / self.tau_x)     + S_pre
        self.o1.value = o1 * (1.0 - dt / self.tau_minus)  + S_post
        self.o2.value = o2 * (1.0 - dt / self.tau_y)      + S_post

        return I_syn

    @property
    def weight_stats(self) -> dict:
        W_conn = np.array(self.W.value)[np.array(self.mask)]
        if len(W_conn) == 0:
            return {}
        # Compute per-trial LTP/LTD from cumulative counters
        return {
            'mean':          float(np.mean(W_conn)),
            'std':           float(np.std(W_conn)),
            'frac_silent':   float(np.mean(W_conn < 0.05)),
            'frac_strong':   float(np.mean(W_conn > 0.8)),
            'ltp_ltd_ratio': float(self.n_ltp.value / (self.n_ltd.value + 1e-10)),
        }


# ----------------------------------------------------------------------
# Integration test: 10 pre → 5 post, triplet STDP
# ----------------------------------------------------------------------
def run_integration_test():
    """
    10 CAdEx pre-neurons → 5 CAdEx post-neurons via TripletSTDPSynapse.

    Starting weight w=0.3. Population drive fires post reliably.
    Triplet STDP (Pfister & Gerstner 2006) drives weight evolution.

    Expected behavior:
    - Post fires 10-15 spikes per trial from population synaptic drive
    - Weights evolve based on LTP/LTD balance
    - At these firing rates (~14Hz post, ~110Hz pre total) and with
      A2+=0.006, A3+=0.009, A2-=0.003, LTP slightly dominates per trial
    - Over 150 trials weights should slowly climb toward w_max

    Note on runtime: ~60s per trial on M1 (pure Python loop).
    150 trials ≈ 2.5 hours. This is a one-time validation run.
    """
    from oi_bench.models.cadex.neuron import CAdExNeuron

    print("=== Triplet STDP Integration Test: 10 pre → 5 post ===")
    print("  Pfister & Gerstner (2006)\n")

    dt        = 0.1
    trial_dur = 100.0
    n_trials  = 150
    n_steps   = int(trial_dur / dt)
    n_pre     = 10
    n_post    = 5

    np.random.seed(42)
    pre  = CAdExNeuron(size=n_pre,  dt=dt)
    post = CAdExNeuron(size=n_post, dt=dt)
    syn  = TripletSTDPSynapse(
        pre, post,
        conn     = np.ones((n_pre, n_post), dtype=bool),
        w_init   = 0.3,
        g_max    = 3.0,
        tau_syn  = 15.0,
    )

    W_history             = [np.array(syn.W.value).copy()]
    ltp_per_trial         = []
    ltd_per_trial         = []
    post_spikes_per_trial = []

    print(f"  Running {n_trials} trials × {trial_dur}ms...")
    print(f"  ~60s per trial on M1. Total ≈ 2.5 hours.\n")

    ltp_prev = 0.0
    ltd_prev = 0.0

    for trial in range(n_trials):
        # Reset neuron state — preserve weights
        pre.V.value      = bm.full(n_pre,  pre.E_L)
        pre.w.value      = bm.zeros(n_pre)
        pre.Ca.value     = bm.zeros(n_pre)
        pre.spike.value  = bm.zeros(n_pre,  dtype=bool)
        post.V.value     = bm.full(n_post, post.E_L)
        post.w.value     = bm.zeros(n_post)
        post.Ca.value    = bm.zeros(n_post)
        post.spike.value = bm.zeros(n_post, dtype=bool)
        syn.r1.value     = bm.zeros(n_pre)
        syn.r2.value     = bm.zeros(n_pre)
        syn.o1.value     = bm.zeros(n_post)
        syn.o2.value     = bm.zeros(n_post)
        syn.g.value      = bm.zeros((n_pre, n_post))

        post_s = np.zeros(n_post)

        for step in range(n_steps):
            pre.update(x=bm.full(n_pre, 800.0))
            S_pre  = pre.spike.value.astype(bm.float32)
            S_post = post.spike.value.astype(bm.float32)
            I_syn  = syn.update(S_pre, S_post, post.V.value)
            post.update(x=bm.full(n_post, 150.0) + I_syn)
            post_s += np.array(post.spike.value.astype(bm.float32))

        W_history.append(np.array(syn.W.value).copy())

        # Per-trial LTP/LTD (difference from previous cumulative)
        ltp_now = float(syn.n_ltp.value)
        ltd_now = float(syn.n_ltd.value)
        ltp_per_trial.append(ltp_now - ltp_prev)
        ltd_per_trial.append(ltd_now - ltd_prev)
        ltp_prev = ltp_now
        ltd_prev = ltd_now
        post_spikes_per_trial.append(post_s.copy())

        if (trial + 1) % 5 == 0:
            s = syn.weight_stats
            per_trial_ratio = ltp_per_trial[-1] / (ltd_per_trial[-1] + 1e-10)
            print(f"  Trial {trial+1:3d}: mean_w={s['mean']:.4f} | "
                  f"post={post_s.astype(int)} | "
                  f"LTP/LTD(trial)={per_trial_ratio:.2f} | "
                  f"strong={s['frac_strong']:.2f}")

    s = syn.weight_stats
    print(f"\n  Final weight stats:")
    print(f"    Mean            : {s['mean']:.4f}")
    print(f"    Std             : {s['std']:.4f}")
    print(f"    Frac strong     : {s['frac_strong']:.2f}")
    print(f"    Frac silent     : {s['frac_silent']:.2f}")
    print(f"    LTP/LTD (total) : {s['ltp_ltd_ratio']:.2f}")

    # 4-panel figure
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(
        "Triplet STDP: 10 pre → 5 post\n"
        "Pfister & Gerstner (2006) | w_init=0.3 | CAdEx neurons",
        fontsize=11, fontweight='bold'
    )

    colors = ['#1565C0', '#E65100', '#2E7D32', '#6A1B9A', '#B71C1C']
    W_arr  = np.array(W_history)        # (n_trials+1, n_pre, n_post)
    t_arr  = np.arange(len(W_history))

    # Panel 1: Mean weight per post-neuron
    ax = axes[0, 0]
    for j in range(n_post):
        ax.plot(t_arr, W_arr[:, :, j].mean(axis=1),
                color=colors[j], linewidth=1.2, label=f"Post {j+1}")
    ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.7)
    ax.axhline(0.3, color='orange', linestyle=':', linewidth=0.8, label='w_init')
    ax.set_ylabel("Mean Incoming Weight"); ax.set_xlabel("Trial")
    ax.set_title("Per-Post Weight Trajectory", fontsize=9)
    ax.set_ylim(-0.05, 1.05); ax.legend(fontsize=7)

    # Panel 2: Weight distribution initial vs final
    ax = axes[0, 1]
    ax.hist(W_arr[0].flatten(),  bins=20, alpha=0.6,
            color='gray',    label='Initial', density=True)
    ax.hist(W_arr[-1].flatten(), bins=20, alpha=0.7,
            color='#1565C0', label='Final',   density=True)
    ax.set_xlabel("Synaptic Weight"); ax.set_ylabel("Density")
    ax.set_title("Weight Distribution: Initial → Final", fontsize=9)
    ax.legend(fontsize=8)

    # Panel 3: Post spikes per trial
    ax = axes[1, 0]
    post_arr = np.array(post_spikes_per_trial)
    for j in range(n_post):
        ax.plot(post_arr[:, j], color=colors[j],
                linewidth=1.0, alpha=0.8, label=f"Post {j+1}")
    ax.set_ylabel("Spikes per Trial"); ax.set_xlabel("Trial")
    ax.set_title("Post-Neuron Firing per Trial", fontsize=9)
    ax.legend(fontsize=7)

    # Panel 4: Per-trial LTP vs LTD
    ax = axes[1, 1]
    t_trials = np.arange(n_trials)
    ax.bar(t_trials, ltp_per_trial,
           color='#43A047', alpha=0.7, label='LTP', width=0.8)
    ax.bar(t_trials, [-x for x in ltd_per_trial],
           color='#E53935', alpha=0.7, label='LTD (inverted)', width=0.8)
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.set_ylabel("Plasticity Events"); ax.set_xlabel("Trial")
    ax.set_title("Per-Trial LTP vs LTD Balance", fontsize=9)
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("stdp_demo.png", dpi=150, bbox_inches='tight')
    print("\nPlot saved → stdp_demo.png")


if __name__ == "__main__":
    run_integration_test()
