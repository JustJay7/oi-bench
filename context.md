# OI-Bench: T4 Interval Timing — Full Context

## Project Overview

OI-Bench is a substrate-agnostic Python benchmark suite for evaluating learning in organoid intelligence (OI) simulations. Stack: BrainPy + JAX, CAdEx neurons with Grünwald-Letnikov fractal membrane dynamics, triplet STDP + homeostatic plasticity. GitHub: JustJay7/oi-bench. M1 Mac, Metal JAX backend.

3 models × 6 tasks:
- Models: CAdEx (reference), LIF (ablation), LSM (no-plasticity baseline)
- Tasks: T1 Classical Conditioning, T2 Pattern Completion, T3 Sequence Prediction, T4 Interval Timing, T5 Delay Match-to-Sample, T6 N-Back

**T1, T2, T3, T5, T6 are working. T4 is broken.**

---

## What T4 Is Supposed To Do

Model learns to produce an output burst at T_target (200ms, 500ms, or 1000ms) after a cue stimulus. Mechanism: three-factor STDP + dopamine modulation (Izhikevich 2007, Fremaux & Gerstner 2016). Eligibility trace accumulates during trial, dopamine gates weight update based on whether burst occurred near T_target. Weber's law scaling expected across intervals.

---

## File Structure

```
oi_bench/
├── core/
│   ├── adapter.py          # OIModel ABC
│   ├── protocol.py         # BenchmarkTask ABC
│   └── types.py            # Stimulus, ModelState, TrialResult
├── models/
│   ├── cadex/
│   │   ├── neuron.py       # CAdExNeuron
│   │   ├── fractal_neuron.py
│   │   ├── fractal.py      # Grünwald-Letnikov operator
│   │   ├── synapse.py      # TripletSTDPSynapse + eligibility trace
│   │   ├── homeostatic.py
│   │   └── network.py      # CAdExNetwork
│   └── baselines/
│       ├── lif_network.py
│       └── reservoir.py
├── tasks/
│   ├── associative/
│   │   ├── classical_conditioning.py  # T1 ✅
│   │   └── pattern_completion.py      # T2 ✅
│   ├── temporal/
│   │   ├── sequence_prediction.py     # T3 ✅
│   │   └── interval_timing.py         # T4 ❌
│   └── working_memory/
│       ├── delay_match.py             # T5 ✅
│       └── n_back.py                  # T6 ✅
├── harness/
│   ├── runner.py
│   └── logger.py
└── metrics/
    ├── learning_curve.py
    ├── plasticity.py
    ├── information.py
    └── stability.py
run.py
```

---

## Current T4 Implementation (interval_timing.py)

- Input population (100 neurons) divided into 10 groups × 10 neurons
- Each group fires for 30ms at a fixed delay post-cue
- delta_t = 100ms per group, covering 0–1000ms
- group_current = 3000pA
- tonic_output_current = 75pA to output via us_current_for_step
- exclusion_ms = 65ms (cue_dur + 3 × tau_syn)
- burst_threshold = 1 (calibrated from silent baseline)

---

## Current run.py T4 Configuration

- T4 model: CAdExNetwork with I_background=0.0, conn_prob=1.0, homeostasis=False
- ELIGIBILITY_CONFIG: T4 enabled=True, tau_e=100ms
- STDP: A2_plus=0.002, A3_plus=0.003, A2_minus=0.002
- Dopamine modulator: weber_error curriculum, tolerance shrinks 1.0→0.15 over 150 trials

---

## runner.py Execution Paths

1. **ELIG path** (use_eligibility_trace=True): jax.lax.scan accumulates e_ff over all timesteps. post_trial calls apply_neuromodulation(da, e_ff). W_ff captured once at trial start, never updated during scan.

2. **JIT path** (use_eligibility_trace=False): bm.for_loop JIT-compiles step function. STDP fires online every step via synapse.update(). modulator applied in post_trial but has no effect on online STDP.

3. **Python loop**: fallback, slow (12,200 steps × model.step() calls).

---

## synapse.py apply_neuromodulation (current)

```python
def apply_neuromodulation(self, da_level, e=None):
    if not self.plasticity:
        return
    if abs(da_level) < 1e-8:
        return
    e_use = e if e is not None else self.e.value
    dW    = da_level * e_use * self.mask_jax
    self.W.value = jnp.clip(self.W.value + dW, self.w_min, self.w_max)
```

---

## Confirmed Failure Modes (from actual runs)

### ELIG path
- w moves 0.3000→0.3022 then freezes
- acc locked at ~0.225 (200ms block), ~0.090 (500ms block), ~0.046 (1000ms block)
- reproduced_ms always ~45ms regardless of target interval or trial number
- e_ff mean abs = 0.000213, max = 0.026, nonzero = 463/5000
- With lr added (dW = da * 0.002 * e_ff): dW max = 0.000052 per trial, weights barely move

### JIT path  
- w saturates to 1.0 within 10–20 trials
- STDP fires unconditionally every step × 12,200 steps per trial
- Even A2_plus=0.0001 saturates by trial 40

### Both paths
- burst position (reproduced_ms) never changes across 150 trials
- acc drops monotonically as curriculum tolerance tightens (dopamine→0)
- The output fires at the first group activation after exclusion window every trial

---

## Confirmed Diagnostics

### CAdEx threshold from rest (I_bg=0, I_us=75pA)
- Fires at I ≥ 600pA
- Below 600pA: zero spikes regardless of duration

### I_syn from group activation (10 neurons, all-to-all, w=0.3)
- group_current=3000pA → I_syn peak=625pA, mean=302pA
- Output fires during group activation (I_syn peak > 600pA threshold)
- But: output fires unconditionally on EVERY group regardless of weights
- Burst always detected at first group after exclusion window (~71ms)

### Pure background (I_bg=0, I_us=75pA, no input)
- Zero spontaneous output spikes over 2000 steps

### w_init vs group_current SNR test (I_bg=0, I_us=75pA, 10 neurons active)
| w_init | group_I | spikes_during | spikes_outside | SNR  |
|--------|---------|---------------|----------------|------|
| 0.8    | 500pA   | 0.17          | 0.14           | 1.2x |
| 0.8    | 800pA   | 0.50          | 0.14           | 3.5x |
| 0.8    | 1000pA  | 0.67          | 0.36           | 1.9x |
| 0.5    | 1000pA  | 0.50          | 0.21           | 2.3x |
| 0.5    | 1500pA  | 0.67          | 0.29           | 2.3x |

---

## Constraints

- Must not break T1, T2, T3, T5, T6
- Must work with existing runner.py (modifying runner.py is allowed if needed)
- Must be biologically correct: three-factor STDP + dopamine is the right mechanism
- M1 Mac, Metal JAX backend, BrainPy
- jax.lax.scan cannot call Python methods — pure JAX only inside scan
- dt=0.1ms, n_input=100, n_output=50, trial_dur=1220ms, 150 trials (50 per interval)
- Target intervals: 200ms, 500ms, 1000ms
