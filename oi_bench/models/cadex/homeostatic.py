"""
Homeostatic Plasticity for CAdEx Networks

Implements two complementary homeostatic mechanisms as specified in
OI-Bench architecture spec Section 4.4.

MECHANISM 1: SYNAPTIC SCALING (Turrigiano et al. 1998, Nature 391:892-896)
---------------------------------------------------------------------------
Multiplicative scaling of all incoming synaptic weights to drive each
neuron's mean firing rate toward r_target.

  W_ij ← W_ij · (r_target / r_j)^γ

Parameters (spec defaults):
  r_target = 5.0 Hz   — target firing rate
  γ        = 0.5      — scaling exponent (soft correction)
  τ_homeo  = 1000 ms  — homeostatic window over which r_j is measured

Biological basis: AMPA receptor insertion/removal scales quantal amplitude.
Reference: Turrigiano et al. (1998), van Rossum et al. (2000) J. Neurosci.

MECHANISM 2: INTRINSIC EXCITABILITY HOMEOSTASIS (Desai et al. 1999)
---------------------------------------------------------------------
Slow adjustment of firing threshold V_T toward a target firing rate.

  dV_T/dt = η_h · (r_j - r_target)

Parameters (spec defaults):
  η_h = 0.001 mV/(Hz·ms)

Higher r_j → V_T increases → harder to fire → rate drops.
Lower r_j  → V_T decreases → easier to fire → rate rises.

Biological basis: Activity-dependent regulation of Na+ channel density.
Reference: Desai et al. (1999) Nature Neuroscience 2:515-520.

TIMESCALE SEPARATION
---------------------
Both mechanisms are DISABLED during trial stimulus windows.
Applied once per ITI (inter-trial interval) after each trial ends.

This preserves the biological timescale hierarchy:
  STDP (ms) << synaptic scaling (ITI) << intrinsic excitability (multiple ITIs)

Correct validation requires sparse task stimuli (Task T1 onwards),
not continuous high-current drive. See benchmark spec Section 4.4.

References:
  Turrigiano et al. (1998) Nature 391:892-896
  Desai et al. (1999) Nature Neuroscience 2:515-520
  van Rossum et al. (2000) J. Neurosci. 20:8812-8821
  Zenke & Gerstner (2017) Curr. Opin. Neurobiol. 43:166-176
"""

import os
os.environ["JAX_PLATFORMS"] = "mps"

import numpy as np
import brainpy.math as bm
import jax.numpy as jnp


class HomeostaticPlasticity:
    """
    Homeostatic plasticity controller for a CAdEx neuron population.

    Applied once per ITI after each trial. Both mechanisms operate on
    the measured firing rate from the completed trial.

    Parameters
    ----------
    neurons : bp.dyn.NeuDyn
        Neuron group to regulate. Must expose V_T as bm.Variable for
        intrinsic excitability homeostasis.
    synapse : TripletSTDPSynapse or None
        Incoming synapse whose weights will be scaled. If None, only
        intrinsic excitability homeostasis is applied.
    r_target : float
        Target firing rate (Hz). Default 5.0 Hz per spec Section 4.4.
    trial_dur_ms : float
        Trial duration (ms). Used to convert spike counts to Hz.
    gamma : float
        Synaptic scaling exponent. Default 0.5 per spec.
    eta_h : float
        Intrinsic excitability learning rate mV/(Hz·ms). Default 0.001.
    w_min : float
        Minimum weight after scaling. Default 0.0.
    w_max : float
        Maximum weight after scaling. Default 1.0.
    enabled : bool
        If False, update() is a no-op. Use for ablation experiments.
    """

    def __init__(
        self,
        neurons,
        synapse          = None,
        r_target: float  = 5.0,
        trial_dur_ms: float = 100.0,
        gamma: float     = 0.5,
        eta_h: float     = 0.001,
        w_min: float     = 0.0,
        w_max: float     = 1.0,
        enabled: bool    = True,
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

        # Running mean firing rate per neuron — initialised at r_target
        self.r_mean = np.full(n, r_target, dtype=np.float32)

        # Exponential moving average decay — τ = 5 trials
        self.alpha = 1.0 - np.exp(-1.0 / 5.0)

        # History for analysis and paper figures
        self.r_mean_history  = [self.r_mean.copy()]
        self.scale_history   = []
        self.V_T_history     = [
            np.array(neurons.V_T.value).copy()
            if hasattr(neurons, 'V_T') else None
        ]

    def update(self, spike_counts: np.ndarray):
        """
        Apply homeostatic plasticity after one trial.

        Called once per ITI. Both mechanisms use the firing rate
        measured during the completed trial.

        Parameters
        ----------
        spike_counts : np.ndarray, shape (n_neurons,)
            Spikes fired by each neuron during the trial.
        """
        if not self.enabled:
            return

        # Firing rate this trial (Hz)
        r_now = spike_counts / (self.trial_dur_ms / 1000.0)

        # Update EMA firing rate estimate
        self.r_mean = (1.0 - self.alpha) * self.r_mean + self.alpha * r_now

        # --- Mechanism 1: Synaptic scaling ---
        if self.synapse is not None:
            # Avoid division by zero
            r_safe = np.maximum(self.r_mean, 0.01)
            scale  = (self.r_target / r_safe) ** self.gamma  # (n_post,)

            W_new = np.array(self.synapse.W.value) * scale[np.newaxis, :]
            W_new = np.clip(W_new, self.w_min, self.w_max)
            self.synapse.W.value = bm.array(W_new)
            self.scale_history.append(scale.copy())

        # --- Mechanism 2: Intrinsic excitability homeostasis ---
        if hasattr(self.neurons, 'V_T'):
            dV_T    = self.eta_h * (self.r_mean - self.r_target)
            V_T_new = np.array(self.neurons.V_T.value) + dV_T
            # Physiological bounds: -60mV to -40mV
            V_T_new = np.clip(V_T_new, -60.0, -40.0)
            self.neurons.V_T.value = bm.array(V_T_new.astype(np.float32))

        # Record history
        self.r_mean_history.append(self.r_mean.copy())
        if hasattr(self.neurons, 'V_T'):
            self.V_T_history.append(np.array(self.neurons.V_T.value).copy())

    @property
    def stats(self) -> dict:
        """Current homeostatic state summary."""
        return {
            'r_mean':              self.r_mean.copy(),
            'r_error_hz':          float(np.mean(np.abs(self.r_mean - self.r_target))),
            'homeostatic_efficacy': float(
                1.0 - np.mean(np.abs(self.r_mean - self.r_target)) / self.r_target
            ),
            'mean_scale': float(np.mean(self.scale_history[-1]))
                          if self.scale_history else 1.0,
            'V_T_mean':   float(np.mean(self.neurons.V_T.value))
                          if hasattr(self.neurons, 'V_T') else None,
        }
