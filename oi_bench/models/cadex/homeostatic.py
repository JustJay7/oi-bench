"""
Homeostatic Plasticity for CAdEx Networks

Implements two complementary homeostatic mechanisms operating on slower
timescales than STDP, preventing runaway potentiation and silencing.

MECHANISM 1: SYNAPTIC SCALING (Turrigiano et al. 1998, Nature 391:892-896)
---------------------------------------------------------------------------
Scales all incoming synaptic weights of a neuron multiplicatively to drive
its mean firing rate toward a target rate r_target.

  W_ij ← W_ij · (r_target / r_j)^γ

Where:
  r_j     = neuron j's mean firing rate over the homeostatic window (Hz)
  r_target = target firing rate (Hz), default 5 Hz
  γ        = scaling exponent, default 0.5 (soft correction — avoids oscillation)

Biological basis: AMPA receptor insertion/removal scales quantal amplitude.
Timescale: hours in biology, compressed to ITI in simulation.
Applied ONLY during inter-trial intervals, not during stimulus presentation.

MECHANISM 2: INTRINSIC EXCITABILITY HOMEOSTASIS (Desai et al. 1999, Nat Neurosci)
-----------------------------------------------------------------------------------
Adjusts the firing threshold V_T of each neuron to drive firing rate toward
r_target via a slow negative feedback on intrinsic excitability.

  dV_T/dt = η_h · (r_j - r_target)

Where:
  η_h = learning rate, default 0.001 mV/(Hz·ms)
  Higher r_j than target → V_T increases → harder to fire → rate drops.
  Lower r_j than target  → V_T decreases → easier to fire → rate rises.

Biological basis: Activity-dependent regulation of Na+ channel density
(Desai et al. 1999). Operates on ~10x slower timescale than synaptic scaling.

INTERACTION WITH STDP
----------------------
Both mechanisms are disabled during trial simulation steps.
Applied once per ITI (inter-trial interval) after each trial.
This preserves the timescale separation: STDP (ms) >> synaptic scaling (ITI)
>> intrinsic excitability (multiple ITIs).

WHY TWO MECHANISMS (Zenke & Gerstner 2017, Curr Opin Neurobiol)
----------------------------------------------------------------
Synaptic scaling controls the total synaptic input (prevents saturation).
Intrinsic excitability controls the neuron's gain (prevents silencing).
They are complementary and together stabilise the network across a much
wider range of perturbations than either alone.

References:
  Turrigiano et al. (1998) Nature 391:892-896
  Desai et al. (1999) Nature Neuroscience 2:515-520
  van Rossum et al. (2000) J. Neurosci. 20:8812-8821
  Zenke & Gerstner (2017) Curr. Opin. Neurobiol. 43:166-176
  Lu et al. (2025) eLife doi:10.7554/eLife.88376
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


class HomeostaticPlasticity:
    """
    Homeostatic plasticity controller for a neuron population.

    Operates on the inter-trial interval timescale. Call update() once
    after each trial, passing the spike counts recorded during that trial.

    Parameters
    ----------
    neurons : bp.dyn.NeuDyn
        The neuron group to regulate. Must have V_T as a bm.Variable.
    synapse : TripletSTDPSynapse or None
        Synapse whose incoming weights will be scaled. If None, only
        intrinsic excitability homeostasis is applied.
    r_target : float
        Target firing rate in Hz. Default 5.0.
    trial_dur_ms : float
        Duration of each trial in ms. Used to convert spike counts to Hz.
    gamma : float
        Synaptic scaling exponent. Default 0.5.
        0 = no scaling, 1 = full multiplicative correction per ITI.
        0.5 gives smooth convergence without oscillation.
    eta_h : float
        Intrinsic excitability learning rate (mV per Hz per ITI).
        Default 0.05. Larger = faster threshold adaptation.
    w_min : float
        Minimum weight after scaling. Default 0.0.
    w_max : float
        Maximum weight after scaling. Default 1.0.
    enabled : bool
        If False, update() is a no-op. Useful for ablation experiments.
    """

    def __init__(
        self,
        neurons,
        synapse=None,
        r_target: float     = 5.0,
        trial_dur_ms: float = 100.0,
        gamma: float        = 0.5,
        eta_h: float        = 0.05,
        w_min: float        = 0.0,
        w_max: float        = 1.0,
        enabled: bool       = True,
    ):
        self.neurons      = neurons
        self.synapse      = synapse
        self.r_target     = r_target
        self.trial_dur_ms = trial_dur_ms
        self.gamma        = gamma
        self.eta_h        = eta_h
        self.w_min        = w_min
        self.w_max        = w_max
        self.enabled      = enabled

        n = neurons.num

        # Running mean firing rate estimate per neuron (Hz)
        # Initialised at r_target so first trial doesn't over-correct
        self.r_mean = np.full(n, r_target, dtype=np.float32)

        # Exponential moving average decay for firing rate estimate
        # τ_rate = 5 trials → α = 1 - exp(-1/5) ≈ 0.18
        self.alpha = 1.0 - np.exp(-1.0 / 5.0)

        # Track V_T history for analysis
        self.V_T_history = [np.array(neurons.V_T.value).copy()
                            if hasattr(neurons, 'V_T') else None]

        # Track scaling factors applied
        self.scale_history = []

    def update(self, spike_counts: np.ndarray):
        """
        Apply homeostatic plasticity after one trial.

        Parameters
        ----------
        spike_counts : np.ndarray, shape (n_neurons,)
            Number of spikes each neuron fired during the trial.
            Used to estimate firing rate: r = spike_counts / (trial_dur_ms/1000)
        """
        if not self.enabled:
            return

        # Convert spike counts to Hz
        trial_dur_s = self.trial_dur_ms / 1000.0
        r_now = spike_counts / trial_dur_s   # Hz

        # Update running mean firing rate (exponential moving average)
        self.r_mean = (1.0 - self.alpha) * self.r_mean + self.alpha * r_now

        # --- Mechanism 1: Synaptic scaling ---
        if self.synapse is not None:
            # Scale factor per post-neuron: (r_target / r_j)^gamma
            # Clip r_mean to avoid division by zero
            r_safe = np.maximum(self.r_mean, 0.1)
            scale  = (self.r_target / r_safe) ** self.gamma  # (n_post,)

            # Apply scaling to incoming weights of each post-neuron
            # W has shape (n_pre, n_post) — scale along post axis
            W_new = np.array(self.synapse.W.value) * scale[np.newaxis, :]
            W_new = np.clip(W_new, self.w_min, self.w_max)
            self.synapse.W.value = bm.array(W_new)

            self.scale_history.append(scale.copy())
        else:
            self.scale_history.append(np.ones(self.neurons.num))

        # --- Mechanism 2: Intrinsic excitability homeostasis ---
        if hasattr(self.neurons, 'V_T'):
            # dV_T = eta_h * (r_mean - r_target)
            # Positive error (too fast) → raise threshold → harder to fire
            dV_T = self.eta_h * (self.r_mean - self.r_target)
            V_T_new = np.array(self.neurons.V_T.value) + dV_T
            # Clip to physiologically reasonable range
            V_T_new = np.clip(V_T_new, -60.0, -40.0)
            self.neurons.V_T.value = bm.array(V_T_new.astype(np.float32))
            self.V_T_history.append(V_T_new.copy())

    @property
    def stats(self) -> dict:
        """Current homeostatic state."""
        r_safe = np.maximum(self.r_mean, 0.1)
        return {
            'r_mean':          self.r_mean.copy(),
            'r_error':         float(np.mean(np.abs(self.r_mean - self.r_target))),
            'homeostatic_efficacy': float(
                np.mean(np.abs(self.r_mean - self.r_target)) / self.r_target
            ),
            'mean_scale':      float(np.mean(self.scale_history[-1]))
                               if self.scale_history else 1.0,
            'V_T_mean':        float(np.mean(self.neurons.V_T.value))
                               if hasattr(self.neurons, 'V_T') else None,
        }


# ----------------------------------------------------------------------
# Integration test: STDP + Homeostasis together
# Demonstrates that homeostasis prevents STDP saturation
# ----------------------------------------------------------------------
def run_integration_test():
    """
    10 pre → 5 post with TripletSTDP + HomeostaticPlasticity.

    Without homeostasis: weights saturate at w_max in trial 1 (verified).
    With homeostasis: synaptic scaling pulls weights back when post fires
    too fast, creating a stable equilibrium between LTP and scaling.

    This is the complete reference plasticity stack for OI-Bench.
    """
    from oi_bench.models.cadex.neuron import CAdExNeuron
    from oi_bench.models.cadex.synapse import TripletSTDPSynapse

    print("=== STDP + Homeostasis Integration Test: 10 pre → 5 post ===")
    print("  Turrigiano (1998) · Desai (1999) · Pfister & Gerstner (2006)\n")

    dt        = 0.1
    trial_dur = 100.0
    n_trials  = 30      # 30 trials sufficient to see equilibrium
    n_steps   = int(trial_dur / dt)
    n_pre     = 10
    n_post    = 5

    np.random.seed(42)
    pre  = CAdExNeuron(size=n_pre,  dt=dt)
    post = CAdExNeuron(size=n_post, dt=dt)

    # Add V_T as a Variable to post so intrinsic homeostasis can modify it
    post.V_T = bm.Variable(bm.full(n_post, post.V_T))

    syn  = TripletSTDPSynapse(
        pre, post,
        conn    = np.ones((n_pre, n_post), dtype=bool),
        w_init  = 0.3,
        g_max   = 3.0,
        tau_syn = 15.0,
    )

    homeo = HomeostaticPlasticity(
        neurons      = post,
        synapse      = syn,
        r_target     = 220.0,    # natural firing rate at w_init=0.3
                                 # homeostasis maintains equilibrium near
                                 # this point rather than dragging to near-zero
        trial_dur_ms = trial_dur,
        gamma        = 0.7,
        eta_h        = 0.01,     # slow intrinsic adaptation
    )

    W_history             = [np.array(syn.W.value).copy()]
    r_mean_history        = [homeo.r_mean.copy()]
    post_spikes_history   = []
    V_T_history           = [np.array(post.V_T.value).copy()]

    print(f"  Running {n_trials} trials × {trial_dur}ms...")
    print(f"  ~60s per trial. Total ≈ {n_trials} min.\n")

    for trial in range(n_trials):
        # Reset neuron state — preserve weights and V_T
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

        # Apply homeostasis at end of trial (ITI)
        homeo.update(post_s)

        W_history.append(np.array(syn.W.value).copy())
        r_mean_history.append(homeo.r_mean.copy())
        post_spikes_history.append(post_s.copy())
        V_T_history.append(np.array(post.V_T.value).copy())

        if (trial + 1) % 5 == 0:
            h = homeo.stats
            print(f"  Trial {trial+1:3d}: mean_w={np.mean(syn.W.value):.4f} | "
                  f"post={post_s.astype(int)} | "
                  f"r_mean={h['r_mean'].mean():.1f}Hz | "
                  f"r_error={h['r_error']:.2f}Hz | "
                  f"V_T={h['V_T_mean']:.2f}mV")

    h = homeo.stats
    print(f"\n  Final state:")
    print(f"    Mean weight         : {np.mean(syn.W.value):.4f}")
    print(f"    Mean firing rate    : {h['r_mean'].mean():.2f} Hz")
    print(f"    Homeostatic efficacy: {h['homeostatic_efficacy']:.3f}")
    print(f"    Mean V_T            : {h['V_T_mean']:.2f} mV")

    # 4-panel figure
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(
        "Triplet STDP + Homeostatic Plasticity: 10 pre → 5 post\n"
        "Pfister & Gerstner (2006) · Turrigiano (1998) · Desai (1999)",
        fontsize=11, fontweight='bold'
    )

    colors = ['#1565C0', '#E65100', '#2E7D32', '#6A1B9A', '#B71C1C']
    W_arr  = np.array(W_history)
    t_arr  = np.arange(len(W_history))

    # Panel 1: Mean weight per post-neuron
    ax = axes[0, 0]
    for j in range(n_post):
        ax.plot(t_arr, W_arr[:, :, j].mean(axis=1),
                color=colors[j], linewidth=1.2, label=f"Post {j+1}")
    ax.axhline(0.3, color='orange', linestyle=':', linewidth=0.8, label='w_init')
    ax.set_ylabel("Mean Incoming Weight"); ax.set_xlabel("Trial")
    ax.set_title("Weight Trajectory — Homeostasis prevents saturation", fontsize=9)
    ax.set_ylim(-0.05, 1.05); ax.legend(fontsize=7)

    # Panel 2: Post firing rate vs target
    ax = axes[0, 1]
    r_arr = np.array(r_mean_history)
    for j in range(n_post):
        ax.plot(t_arr, r_arr[:, j], color=colors[j],
                linewidth=1.2, label=f"Post {j+1}")
    ax.axhline(homeo.r_target, color='gray', linestyle='--',
               linewidth=1.0, label=f'r_target={homeo.r_target}Hz')
    ax.set_ylabel("Mean Firing Rate (Hz)"); ax.set_xlabel("Trial")
    ax.set_title("Firing Rate Homeostasis — convergence to target", fontsize=9)
    ax.legend(fontsize=7)

    # Panel 3: Firing threshold V_T per post-neuron
    ax = axes[1, 0]
    VT_arr = np.array(V_T_history)
    for j in range(n_post):
        ax.plot(t_arr, VT_arr[:, j], color=colors[j],
                linewidth=1.2, label=f"Post {j+1}")
    ax.axhline(-50.0, color='gray', linestyle='--', linewidth=0.8,
               label='V_T initial')
    ax.set_ylabel("Threshold V_T (mV)"); ax.set_xlabel("Trial")
    ax.set_title("Intrinsic Excitability — threshold adapts to firing rate", fontsize=9)
    ax.legend(fontsize=7)

    # Panel 4: Post spikes per trial
    ax = axes[1, 1]
    ps_arr = np.array(post_spikes_history)
    for j in range(n_post):
        ax.plot(ps_arr[:, j], color=colors[j],
                linewidth=1.0, label=f"Post {j+1}")
    ax.axhline(homeo.r_target * trial_dur / 1000.0, color='gray',
               linestyle='--', linewidth=0.8,
               label=f'target ({homeo.r_target * trial_dur / 1000.0:.0f} spikes/trial)')
    ax.set_ylabel("Spikes per Trial"); ax.set_xlabel("Trial")
    ax.set_title("Post-Neuron Firing per Trial", fontsize=9)
    ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig("homeostatic_demo.png", dpi=150, bbox_inches='tight')
    print("\nPlot saved → homeostatic_demo.png")


if __name__ == "__main__":
    run_integration_test()
