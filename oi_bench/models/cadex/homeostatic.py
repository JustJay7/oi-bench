"""
Homeostatic Plasticity for CAdEx Networks

Implements two complementary homeostatic mechanisms as specified in
OI-Bench architecture spec Section 4.4.

MECHANISM 1: SYNAPTIC SCALING (Turrigiano et al. 1998, Nature 391:892-896)
  W_ij ← W_ij · (r_target / r_j)^γ

MECHANISM 2: INTRINSIC EXCITABILITY HOMEOSTASIS (Desai et al. 1999)
  dV_T/dt = η_h · (r_j - r_target)

V_T BOUNDS — derived from neuron parameters, not hardcoded:
  lower = E_L + 10mV
    Floor 10mV above resting potential — always subthreshold.
    For CAdEx defaults (E_L=-70): lower = -60mV.

  upper = V_T_init + 20mV
    Ceiling 20mV above initial threshold. Desai et al. (1999) measured
    threshold shifts up to 15-20mV above resting V_T in cortical neurons
    under sustained hyperactivity. 20mV headroom is biologically correct.
    For CAdEx defaults (V_T_init=-50): upper = -30mV.

    Previous hardcoded upper=-40mV only allowed 10mV headroom, preventing
    homeostasis from fully regulating high-firing networks. This caused
    persistent over-firing in tasks with dense stimuli (T2, T3).

Both bounds are computed once in __init__ from the neuron's actual
E_L and V_T values — if you change the neuron model, bounds update
automatically.

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


class HomeostaticPlasticity:
    """
    Homeostatic plasticity controller for a CAdEx neuron population.

    Parameters
    ----------
    neurons : bp.dyn.NeuDyn
        Neuron group to regulate. Must expose V_T as bm.Variable.
    synapse : TripletSTDPSynapse or None
        Incoming synapse for synaptic scaling.
    r_target : float
        Target firing rate (Hz). Default 5.0 Hz per spec Section 4.4.
    trial_dur_ms : float
        Trial duration (ms).
    gamma : float
        Synaptic scaling exponent. Default 0.5 per spec.
    eta_h : float
        Intrinsic excitability learning rate mV/(Hz·ms). Default 0.001.
    w_min, w_max : float
        Weight bounds after scaling.
    enabled : bool
        If False, update() is a no-op.
    """

    def __init__(
        self,
        neurons,
        synapse             = None,
        r_target: float     = 5.0,
        trial_dur_ms: float = 100.0,
        gamma: float        = 0.5,
        eta_h: float        = 0.001,
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

        n           = neurons.num
        self.r_mean = np.full(n, r_target, dtype=np.float32)
        # EMA decay — τ = 5 trials (spec Section 4.4)
        self.alpha  = 1.0 - np.exp(-1.0 / 5.0)

        # V_T bounds derived from neuron parameters
        E_L        = getattr(neurons, 'E_L',    -70.0)
        V_T_init   = float(np.mean(np.array(neurons.V_T.value))) \
                     if hasattr(neurons, 'V_T') else -50.0
        self._V_T_lo = E_L      + 10.0   # floor: 10mV above resting
        self._V_T_hi = V_T_init + 20.0   # ceiling: 20mV above initial threshold

        self.r_mean_history = [self.r_mean.copy()]
        self.scale_history  = []
        self.V_T_history    = [
            np.array(neurons.V_T.value).copy()
            if hasattr(neurons, 'V_T') else None
        ]

    def update(self, spike_counts: np.ndarray) -> None:
        """Apply homeostatic plasticity after one trial (ITI)."""
        if not self.enabled:
            return

        r_now       = spike_counts / (self.trial_dur_ms / 1000.0)
        self.r_mean = (1.0 - self.alpha) * self.r_mean + self.alpha * r_now

        # Mechanism 1: Synaptic scaling
        if self.synapse is not None:
            r_safe = np.maximum(self.r_mean, 0.01)
            scale  = (self.r_target / r_safe) ** self.gamma
            W_new  = np.array(self.synapse.W.value) * scale[np.newaxis, :]
            W_new  = np.clip(W_new, self.w_min, self.w_max)
            self.synapse.W.value = bm.array(W_new)
            self.scale_history.append(scale.copy())

        # Mechanism 2: Intrinsic excitability homeostasis
        if hasattr(self.neurons, 'V_T'):
            dV_T    = self.eta_h * (self.r_mean - self.r_target)
            V_T_new = np.array(self.neurons.V_T.value) + dV_T
            # Bounds derived from neuron params in __init__
            V_T_new = np.clip(V_T_new, self._V_T_lo, self._V_T_hi)
            self.neurons.V_T.value = bm.array(V_T_new.astype(np.float32))

        self.r_mean_history.append(self.r_mean.copy())
        if hasattr(self.neurons, 'V_T'):
            self.V_T_history.append(np.array(self.neurons.V_T.value).copy())

    @property
    def stats(self) -> dict:
        return {
            'r_mean':               self.r_mean.copy(),
            'r_error_hz':           float(np.mean(np.abs(self.r_mean - self.r_target))),
            'homeostatic_efficacy': float(
                1.0 - np.mean(np.abs(self.r_mean - self.r_target))
                / max(self.r_target, 1e-6)
            ),
            'mean_scale': float(np.mean(self.scale_history[-1]))
                          if self.scale_history else 1.0,
            'V_T_mean':   float(np.mean(self.neurons.V_T.value))
                          if hasattr(self.neurons, 'V_T') else None,
        }
