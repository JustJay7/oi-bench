"""Diagnostic: does CAdEx show SSA (reduced response on repeated stimulus)?

Presents stimulus A (neurons 0-29 @ 400pA) for 500ms, then again for 500ms
with no reset between. CAdEx should fire less on the second presentation
because the adaptation current w builds up. LIF has no adaptation — flat.
"""

import numpy as np
from oi_bench.core.types import Stimulus
from oi_bench.models.cadex.network import CAdExNetwork
from oi_bench.models.baselines.lif_network import LIFNetwork

DT       = 0.1
N_INPUT  = 100
N_OUTPUT = 50
STIM_I   = 400.0   # pA, same as oddball task
DUR_MS   = 500.0
N_STEPS  = int(DUR_MS / DT)


def run_window(model, offset_ms, drive_neurons):
    """Run one 500ms window, return mean output rate in Hz."""
    current_template = np.zeros(N_INPUT, dtype=np.float32)
    current_template[drive_neurons] = STIM_I
    spikes = 0.0
    for i in range(N_STEPS):
        stim = Stimulus(
            current=current_template.copy(),
            spike_train=np.zeros(N_INPUT, dtype=np.float32),
            t=offset_ms + i * DT,
            label=0,
        )
        state = model.step(stim)
        spikes += float(np.sum(np.array(state.spikes)))
    return spikes / (N_OUTPUT * (DUR_MS / 1000.0))


def probe_adaptation(model):
    """Return mean adaptation current w after all stimulus steps."""
    if hasattr(model, 'output_pop') and hasattr(model.output_pop, 'w'):
        return float(np.mean(np.array(model.output_pop.w.value)))
    return float('nan')


def diagnose(name, model):
    model.reset()
    stim_neurons = np.arange(0, 30)

    r1  = run_window(model, offset_ms=0.0,    drive_neurons=stim_neurons)
    w1  = probe_adaptation(model)
    r2  = run_window(model, offset_ms=DUR_MS, drive_neurons=stim_neurons)
    w2  = probe_adaptation(model)

    ratio = r2 / r1 if r1 > 0 else float('nan')
    verdict = "SUPPRESSED (SSA)" if ratio < 0.85 else "flat — no SSA"

    print(f"{name}:")
    print(f"  P1 rate: {r1:6.1f} Hz  |  adapt-w after P1: {w1:.2f} pA")
    print(f"  P2 rate: {r2:6.1f} Hz  |  adapt-w after P2: {w2:.2f} pA")
    print(f"  P2/P1 = {ratio:.3f}  →  {verdict}")
    print()


cadex = CAdExNetwork(n_input=N_INPUT, n_output=N_OUTPUT,
                     plasticity=False, homeostasis=False, seed=42)
lif   = LIFNetwork(  n_input=N_INPUT, n_output=N_OUTPUT,
                     plasticity=False, homeostasis=False, seed=42)

print("=== SSA Diagnostic: two back-to-back 500ms presentations of stimulus A ===")
print(f"Stimulus: neurons 0-29 @ {STIM_I:.0f}pA | {DUR_MS:.0f}ms each\n")
diagnose("CAdEx", cadex)
diagnose("LIF",   lif)
