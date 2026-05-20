"""
CAdEx Neuron — Calcium-Augmented Adaptive Exponential Integrate-and-Fire

Base AdEx formulation: Brette & Gerstner (2005), J. Neurophysiol. 94:3637-3642
Calcium current: Reuveni et al. (1993), J. Neurosci. 13:4609-4621
Burst/adaptation in organoids: Trujillo et al. (2019), Cell Stem Cell 25:558-569

All quantities in consistent units:
  Voltages    : mV
  Currents    : pA
  Time        : ms
  Capacitance : pF
  Conductance : nS   (nS * mV = pA ✓)

Membrane equation:
  C_m dV/dt = -g_L(V-E_L) + g_L*dT*exp((V-V_T)/dT) - I_Ca(V) + I_ext - w

  w is kept in pA throughout.

Adaptation:
  tau_w dw/dt = a*(V-E_L) - w     [a in nS → a*(V-E_L) in pA ✓]
  At each spike: w += b            [b in pA]

Calcium:
  d[Ca]/dt = -[Ca]/tau_Ca + rho * spike
  I_Ca = g_Ca * m_inf(V)^2 * (V - E_Ca)
  Since V < E_Ca always, I_Ca < 0 (inward/depolarising).
  In membrane eq: -I_Ca → subtracting a negative = depolarising. ✓
"""


import brainpy as bp
import brainpy.math as bm
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


class CAdExNeuron(bp.dyn.NeuDyn):
    """
    Single-compartment CAdEx neuron group.

    All internal currents in pA, conductances in nS, voltages in mV, time in ms.
    """

    def __init__(
        self,
        size: int,
        C_m: float     = 200.0,  # pF
        g_L: float     = 10.0,   # nS
        E_L: float     = -70.0,  # mV
        delta_T: float = 2.0,    # mV
        V_T: float     = -50.0,  # mV
        V_peak: float  = 20.0,   # mV  hard spike cutoff
        V_r: float     = -65.0,  # mV  reset
        tau_w: float   = 100.0,  # ms
        a: float       = 2.0,    # nS
        b: float       = 20.0,   # pA  spike-triggered adaptation
        tau_Ca: float  = 80.0,   # ms
        rho: float     = 0.1,    # μM/spike
        g_Ca: float    = 1.5,    # nS
        E_Ca: float    = 120.0,  # mV
        dt: float      = 0.1,    # ms
        **kwargs,
    ):
        super().__init__(size=size, **kwargs)

        self.C_m = C_m; self.g_L = g_L; self.E_L = E_L
        self.delta_T = delta_T; self.V_T = V_T
        self.V_peak = V_peak; self.V_r = V_r
        self.tau_w = tau_w; self.a = a; self.b = b
        self.tau_Ca = tau_Ca; self.rho = rho
        self.g_Ca = g_Ca; self.E_Ca = E_Ca
        self.dt = dt

        self.V     = bm.Variable(bm.full(size, E_L))
        self.w     = bm.Variable(bm.zeros(size))
        self.Ca    = bm.Variable(bm.zeros(size))
        self.spike = bm.Variable(bm.zeros(size, dtype=bool))
        self.input = bm.Variable(bm.zeros(size))

    def m_inf(self, V):
        """Calcium activation gate — half at -30 mV, slope 6 mV."""
        return 1.0 / (1.0 + jnp.exp(-(V + 30.0) / 6.0))

    def I_Ca(self, V):
        """Calcium current (pA). Negative = inward = depolarising."""
        return self.g_Ca * self.m_inf(V) ** 2 * (V - self.E_Ca)

    def dV_dt(self, V, w, I_ext):
        I_leak = self.g_L * (V - self.E_L)
        I_exp  = self.g_L * self.delta_T * jnp.exp((V - self.V_T) / self.delta_T)
        I_ca   = self.I_Ca(V)          # negative (inward)
        return (-I_leak + I_exp - I_ca + I_ext - w) / self.C_m

    def dw_dt(self, w, V):
        return (self.a * (V - self.E_L) - w) / self.tau_w

    def dCa_dt(self, Ca):
        return -Ca / self.tau_Ca

    def update(self, x=None):
        I_ext = self.input.value if x is None else x

        V  = self.V.value
        w  = self.w.value
        Ca = self.Ca.value

        V_new  = V  + self.dV_dt(V, w, I_ext) * self.dt
        w_new  = w  + self.dw_dt(w, V)        * self.dt
        Ca_new = Ca + self.dCa_dt(Ca)         * self.dt

        spike = V_new >= self.V_peak

        V_new  = bm.where(spike, self.V_r,          V_new)
        w_new  = bm.where(spike, w_new + self.b,    w_new)
        Ca_new = bm.where(spike, Ca_new + self.rho, Ca_new)

        self.V.value     = V_new
        self.w.value     = w_new
        self.Ca.value    = Ca_new
        self.spike.value = spike
        self.input.value = bm.zeros(self.num)


def run_demo():
    dt       = 0.1
    duration = 500.0
    n_steps  = int(duration / dt)

    neuron = CAdExNeuron(size=1, dt=dt)

    V_trace     = np.zeros(n_steps)
    w_trace     = np.zeros(n_steps)
    Ca_trace    = np.zeros(n_steps)
    spike_trace = np.zeros(n_steps, dtype=bool)

    I_amp = 600.0  # pA

    for i in range(n_steps):
        t_ms = i * dt
        I = I_amp if t_ms >= 50.0 else 0.0
        neuron.update(x=bm.array([I]))
        V_trace[i]     = float(neuron.V.value[0])
        w_trace[i]     = float(neuron.w.value[0])
        Ca_trace[i]    = float(neuron.Ca.value[0])
        spike_trace[i] = bool(neuron.spike.value[0])

    spike_times = np.where(spike_trace)[0] * dt
    n_spikes = len(spike_times)
    print(f"Simulation complete: {n_spikes} spikes in {duration} ms")

    if n_spikes > 1:
        isis = np.diff(spike_times)
        print(f"  First ISI : {isis[0]:.1f} ms  ({1000/isis[0]:.1f} Hz)")
        print(f"  Last ISI  : {isis[-1]:.1f} ms  ({1000/isis[-1]:.1f} Hz)")
        print(f"  Adaptation ratio (last/first ISI): {isis[-1]/isis[0]:.2f}x")
        strong = isis[-1]/isis[0] > 1.5
        print(f"  → {'Strong' if strong else 'Mild'} spike-frequency adaptation")
    elif n_spikes == 1:
        print("  Only 1 spike — neuron is near rheobase")
    else:
        print("  No spikes detected — check I_amp")

    t = np.arange(n_steps) * dt
    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    fig.suptitle("CAdEx Neuron — Step Current Response (600 pA)", fontsize=13, fontweight='bold')

    axes[0].plot(t, V_trace, color='#1565C0', linewidth=0.7)
    axes[0].axvline(50, color='gray', linestyle='--', linewidth=0.8, label='Stimulus onset')
    axes[0].set_ylabel("V (mV)"); axes[0].set_ylim(-80, 30)
    axes[0].legend(fontsize=8); axes[0].set_title("Membrane Potential", fontsize=9)

    if n_spikes > 1:
        axes[1].plot(spike_times[:-1], np.diff(spike_times), 'o-',
                     color='#6A1B9A', markersize=3, linewidth=1.2)
        axes[1].set_ylabel("ISI (ms)"); axes[1].set_title("Inter-Spike Interval", fontsize=9)
    else:
        axes[1].text(0.5, 0.5, f'{n_spikes} spike(s) detected',
                     transform=axes[1].transAxes, ha='center', va='center')

    axes[2].plot(t, w_trace, color='#C62828', linewidth=0.8)
    axes[2].set_ylabel("w (pA)"); axes[2].set_title("Adaptation Current", fontsize=9)

    axes[3].plot(t, Ca_trace, color='#2E7D32', linewidth=0.8)
    axes[3].set_ylabel("[Ca²⁺] (μM)"); axes[3].set_xlabel("Time (ms)")
    axes[3].set_title("Intracellular Calcium", fontsize=9)

    plt.tight_layout()
    plt.savefig("cadex_demo.png", dpi=150, bbox_inches='tight')
    print("Plot saved → cadex_demo.png")


if __name__ == "__main__":
    run_demo()
