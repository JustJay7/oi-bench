"""
OI-Bench Full Benchmark Run

Executes the complete 3 models × 6 tasks evaluation matrix.
Results logged to results/{run_id}/ as HDF5 + JSON + CSV.

Usage:
    python run.py                          # full run, all models and tasks
    python run.py --models cadex           # single model
    python run.py --tasks T1 T2            # subset of tasks
    python run.py --smoke                  # minimal trial counts (fast)
    python run.py --model_seed 42          # reproducibility seed

T4 DISCRETE POPULATION PACKET ARCHITECTURE
-------------------------------------------
Input population divided into 10 sequential groups × 10 neurons.
Each group fires for 30ms at a fixed delay post-cue, covering
[0, 1000ms] uniformly (delta_t=100ms per group).

T4 model uses I_background=0 (vs 150pA for other tasks):
  - CAdEx fires at ~600pA from rest
  - With I_bg=150pA, spontaneous output firing drowns group signal
  - With I_bg=0 and I_us=75pA, output is silent between groups
  - Group activation (3000pA → I_syn ~625pA) reliably crosses threshold
  - Result: clean temporal code with zero false positives

Three-factor STDP + dopamine (Izhikevich 2007):
  - Eligibility trace accumulates during group → output co-activation
  - Dopamine gates weight update only for rewarded bursts
  - Over trials, T_target group synapses are selectively potentiated

MODEL-SPECIFIC STDP AMPLITUDES
-------------------------------
CAdEx: A2_plus=0.006 (Pfister & Gerstner 2006 Table 1)
LIF:   A2_plus=0.001 (6× smaller — LIF fires more readily)
LSM:   no STDP
"""

import os
os.environ["JAX_PLATFORMS"] = "mps"

import argparse
import time
import jax
import numpy as np
from datetime import datetime

from oi_bench.models.cadex.network import CAdExNetwork
from oi_bench.models.baselines.lif_network import LIFNetwork
from oi_bench.models.baselines.reservoir import LiquidStateMachine

from oi_bench.tasks.associative.classical_conditioning import ClassicalConditioningTask
from oi_bench.tasks.associative.pattern_completion import PatternCompletionTask
from oi_bench.tasks.temporal.sequence_prediction import SequencePredictionTask
from oi_bench.tasks.temporal.interval_timing import IntervalTimingTask
from oi_bench.tasks.working_memory.delay_match import DelayMatchTask
from oi_bench.tasks.working_memory.n_back import NBackTask

from oi_bench.harness.runner import BenchmarkRunner
from oi_bench.harness.logger import BenchmarkLogger


# ------------------------------------------------------------------
# Per-task homeostasis calibration
# ------------------------------------------------------------------
HOMEO_CONFIG = {
    'T1': {'r_target': 110.0, 'trial_dur_ms':  400.0, 'enabled': True},
    'T2': {'r_target': 200.0, 'trial_dur_ms':  200.0, 'enabled': True},
    'T3': {'r_target':  25.0, 'trial_dur_ms':  250.0, 'enabled': False},
    'T4': {'r_target':   5.0, 'trial_dur_ms': 1220.0, 'enabled': False},
    'T5': {'r_target':  50.0, 'trial_dur_ms':  800.0, 'enabled': True},
    'T6': {'r_target':  50.0, 'trial_dur_ms':  500.0, 'enabled': True},
}

# ------------------------------------------------------------------
# Per-model, per-task STDP amplitude configuration
# ------------------------------------------------------------------
STDP_CONFIG = {
    'cadex': {
        'T1': {'A2_plus': 0.006, 'A3_plus': 0.009, 'A2_minus': 0.003},
        # T2: CA3 has stronger NMDA-driven LTP than cortex (Rolls 2007).
        # Higher amplitude needed to differentiate 5 patterns within 20 reps.
        'T2': {'A2_plus': 0.015, 'A3_plus': 0.020, 'A2_minus': 0.006},
        # A3_plus=0: LIF fires during all 5 elements (no adaptation), so o2
        # accumulates to ~3.9 within a trial. With A3_plus=0.002, ltp_factor=0.0088
        # (8.8× A2_plus) → FF weights hit 0.8289 in 10 trials. A3_plus=0 uses
        # pure pair STDP on FF; C→D selectivity comes from rec, not FF.
        'T3': {'A2_plus': 0.001, 'A3_plus': 0.000, 'A2_minus': 0.001},
        'T4': {'A2_plus': 0.01,  'A3_plus': 0.01,   'A2_minus': 0.001},
        'T5': {'A2_plus': 0.006, 'A3_plus': 0.009, 'A2_minus': 0.003},
        'T6': {'A2_plus': 0.006, 'A3_plus': 0.009, 'A2_minus': 0.003},
    },
    'lif': {
        'T1': {'A2_plus': 0.001, 'A3_plus': 0.002, 'A2_minus': 0.001},
        'T2': {'A2_plus': 0.004, 'A3_plus': 0.006, 'A2_minus': 0.002},
        # A3_plus=0: same rationale as cadex T3 above.
        'T3': {'A2_plus': 0.001, 'A3_plus': 0.000, 'A2_minus': 0.001},
        'T4': {'A2_plus': 0.003, 'A3_plus': 0.003,  'A2_minus': 0.005},
        'T5': {'A2_plus': 0.001, 'A3_plus': 0.002, 'A2_minus': 0.001},
        'T6': {'A2_plus': 0.001, 'A3_plus': 0.002, 'A2_minus': 0.001},
    },
    'lsm': {t: {} for t in ['T1', 'T2', 'T3', 'T4', 'T5', 'T6']},
}

# ------------------------------------------------------------------
# T4 eligibility trace: DA at burst time (Izhikevich 2007).
# tau_e=100ms = one group spacing (delta_t=100ms).  The eligibility
# scan snapshots e_ff at each group's burst-off step; the runner
# applies DA using e(T_burst) not e(T_trial).  At tau_e=100ms only
# the group that fired within the last ~100ms carries significant
# trace at T_burst, giving near-perfect credit assignment.
# ------------------------------------------------------------------
ELIGIBILITY_CONFIG = {
    'T4': {'enabled': True, 'tau_e': 100.0},
}


# ------------------------------------------------------------------
# T4 dopamine modulator — weber_error curriculum
# ------------------------------------------------------------------
def make_t4_modulator_fn(n_trials: int = 150):
    """
    Dopamine modulator for T4 with positive reward + negative correction.

    da > 0 when burst near T_target → LTP for target-group synapses.
    da < 0 when burst far from T_target → slow LTD drives wrong groups
    below threshold so the target group can emerge.
    """
    def t4_modulator_fn(trial_id: int, scores: dict) -> float:
        weber_error = scores.get('weber_error', 1.0)
        # Positive reward only within ~1.5 group-widths of target (weber<0.12).
        # Neutral zone 0.12–0.18 covers the adjacent group (doesn't block cascade).
        # Beyond 0.18: negative reward — ensures wrong-group attractors are unstable.
        if weber_error < 0.12:
            return 1.0
        elif weber_error < 0.18:
            return 0.0
        else:
            return -0.3
    return t4_modulator_fn


# ------------------------------------------------------------------
# Standard model registry (T1-T3, T5-T6)
# ------------------------------------------------------------------
def build_models(seed: int = 0) -> dict:
    return {
        'cadex': CAdExNetwork(
            n_input=100, n_output=50,
            alpha=0.85, dt=0.1,
            plasticity=True, homeostasis=True,
            seed=seed,
        ),
        'lif': LIFNetwork(
            n_input=100, n_output=50,
            dt=0.1,
            I_background=150.0,
            plasticity=True, homeostasis=True,
            seed=seed,
        ),
        'lsm': LiquidStateMachine(
            n_input=100, n_reservoir=200, n_output=50,
            dt=0.1, seed=seed,
        ),
    }


# ------------------------------------------------------------------
# T2/T3-specific model registry — sparse FF connectivity
#
# conn_prob=0.5: each output neuron receives ~50 of 100 input
# connections (SD≈5), creating element- and pattern-selective
# populations:
#   T3: C-selective ≠ D-selective neurons → gap burst (t=120–150ms)
#       hits C-selective neurons; D-selective neurons escape adaptation
#       depletion and fire in the D window driven by recurrent NMDA.
#   T2: partial pattern coverage → heterogeneous output representations
#       → cosine similarity can discriminate stored patterns.
# With conn_prob=1.0 (all-to-all), every output neuron responds
# identically to every element — no selectivity, no sequence coding.
# Biological basis: CA3 and cortex have 10–50% connectivity (Rolls 2007).
# ------------------------------------------------------------------
def build_sparse_models(seed: int = 0) -> dict:
    return {
        'cadex': CAdExNetwork(
            n_input=100, n_output=50,
            alpha=0.85, conn_prob=0.5, dt=0.1,
            plasticity=True, homeostasis=True,
            seed=seed,
        ),
        'lif': LIFNetwork(
            n_input=100, n_output=50,
            conn_prob=0.5, dt=0.1,
            I_background=150.0,
            plasticity=True, homeostasis=True,
            seed=seed,
        ),
        'lsm': LiquidStateMachine(
            n_input=100, n_reservoir=200, n_output=50,
            dt=0.1, seed=seed,
        ),
    }


# ------------------------------------------------------------------
# T3-specific model registry — sparse FF connectivity
#
# conn_prob=0.5: each output receives ~50 FF connections from 100 inputs,
# creating element-selective populations (C-selective ≠ D-selective neurons)
# while keeping enough coverage for reliable firing. With conn_prob=1.0,
# every output fires identically during every element (no selectivity).
# With conn_prob=0.1, output neurons are too silent (~1.5 FF inputs per
# element) → insufficient drive for LIF at 400pA.
#
# LIF I_background=150pA (matches CAdEx default):
# LIF spike threshold is V_T=-50mV (bm.Variable, used in LIFNeuron.update).
# Rheobase = g_L×(V_T-E_L) = 10×20 = 200pA. V_ss from I_bg alone:
#   V_ss = (g_L*E_L + I_bg) / g_L = (-700+150)/10 = -55mV < V_T=-50mV
# → output is SILENT from background alone (no 224Hz noise).
# During each element (15 active input neurons fire, ~7 connect per output):
#   G_syn = g_max×W×n_conn = 3.0×0.3×7 = 6.3nS
#   V_ss(element) = (-700+150)/(10+6.3) = -33.7mV > -50mV → output fires ✓
# This gives element-selective output firing → STDP builds C→D→E chain.
# With I_bg=800pA (default): V_ss=10mV > V_T=-50mV → ALL neurons fire at
# 224Hz continuously → no selectivity → STDP learning cannot converge.
#
# Biological basis: CA3 connectivity ~10-50% (Rolls 2007).
# ------------------------------------------------------------------
def build_t3_models(seed: int = 0) -> dict:
    return {
        'cadex': CAdExNetwork(
            n_input=100, n_output=50,
            alpha=0.85, conn_prob=0.5, dt=0.1,
            plasticity=True, homeostasis=True,
            seed=seed,
        ),
        'lif': LIFNetwork(
            n_input=100, n_output=50,
            conn_prob=0.5, dt=0.1,
            I_background=150.0,
            plasticity=True, homeostasis=True,
            seed=seed,
        ),
        'lsm': LiquidStateMachine(
            n_input=100, n_reservoir=200, n_output=50,
            dt=0.1, seed=seed,
        ),
    }


# ------------------------------------------------------------------
# T4-specific model registry
# I_background=0: eliminates spontaneous output firing so group
# activations produce clean detectable bursts above silent baseline.
# conn_prob=1.0: all-to-all ensures every output neuron receives
# the full group signal reliably.
# homeostasis=False: homeostasis would fight the sparse firing pattern.
# ------------------------------------------------------------------
def build_t4_models(seed: int = 0) -> dict:
    return {
        'cadex': CAdExNetwork(
            n_input=100, n_output=50,
            alpha=0.85, conn_prob=1.0, dt=0.1,
            I_background=0.0,
            plasticity=True, homeostasis=False,
            w_init=0.4,
            seed=seed,
        ),
        'lif': LIFNetwork(
            n_input=100, n_output=50,
            conn_prob=1.0, dt=0.1,
            I_background=0.0,
            plasticity=True, homeostasis=False,
            w_init=0.05,
            seed=seed,
        ),
        'lsm': LiquidStateMachine(
            n_input=100, n_reservoir=200, n_output=50,
            dt=0.1, seed=seed,
        ),
    }


# ------------------------------------------------------------------
# Task registry
# ------------------------------------------------------------------
def build_tasks(smoke: bool = False) -> dict:
    if smoke:
        return {
            'T1': ClassicalConditioningTask(
                n_trials=10, n_conditioning_trials=8),
            'T2': PatternCompletionTask(
                n_patterns=3, n_learning_reps=5),
            'T3': SequencePredictionTask(
                n_learning_trials=10, n_test_trials=5),
            'T4': IntervalTimingTask(n_trials_per_interval=5),
            'T5': DelayMatchTask(n_trials=10),
            'T6': NBackTask(trials_per_block=10),
        }
    else:
        return {
            'T1': ClassicalConditioningTask(),
            'T2': PatternCompletionTask(n_patterns=5),
            'T3': SequencePredictionTask(),
            'T4': IntervalTimingTask(),
            'T5': DelayMatchTask(),
            'T6': NBackTask(),
        }


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="OI-Bench full benchmark run")
    parser.add_argument('--models', nargs='+', default=None,
                        choices=['cadex', 'lif', 'lsm'])
    parser.add_argument('--tasks', nargs='+', default=None,
                        choices=['T1', 'T2', 'T3', 'T4', 'T5', 'T6'])
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--model_seed', type=int, default=0)
    parser.add_argument('--run_id', type=str, default=None)
    parser.add_argument('--results_dir', type=str, default='results')
    parser.add_argument('--record_traces', action='store_true')
    args = parser.parse_args()

    all_models    = build_models(seed=args.model_seed)
    all_t2_models = build_sparse_models(seed=args.model_seed)
    all_t3_models = build_t3_models(seed=args.model_seed)
    all_t4_models = build_t4_models(seed=args.model_seed)
    all_tasks     = build_tasks(smoke=args.smoke)

    models_to_run = {k: v for k, v in all_models.items()
                     if args.models is None or k in args.models}
    tasks_to_run  = {k: v for k, v in all_tasks.items()
                     if args.tasks is None or k in args.tasks}

    n_runs = len(models_to_run) * len(tasks_to_run)

    runner = BenchmarkRunner(
        record_traces = args.record_traces,
        verbose       = True,
        report_every  = 5 if args.smoke else 10,
        global_seed   = args.model_seed,
    )
    logger = BenchmarkLogger(
        results_dir = args.results_dir,
        run_id      = args.run_id,
    )

    print(f"\n{'='*70}")
    print(f"  OI-Bench Full Benchmark Run")
    print(f"  Models : {list(models_to_run.keys())}")
    print(f"  Tasks  : {list(tasks_to_run.keys())}")
    print(f"  Total  : {n_runs} runs")
    print(f"  Smoke  : {args.smoke}")
    print(f"  Run ID : {logger.run_id}")
    print(f"{'='*70}\n")

    run_count = 0
    t_start   = time.time()

    for model_name in models_to_run:
        for task_name, task in tasks_to_run.items():
            run_count += 1
            print(f"\n[{run_count}/{n_runs}] {model_name} × {task_name}")
            print(f"  Started: {datetime.now().strftime('%H:%M:%S')}")

            # T4: dedicated model with I_background=0 and conn_prob=1.0
            # T2/T3: separate sparse instances (conn_prob=0.5) so learned
            #        weights don't contaminate across tasks
            if task_name == 'T4':
                model = all_t4_models[model_name]
            elif task_name == 'T2':
                model = all_t2_models[model_name]
            elif task_name == 'T3':
                model = all_t3_models[model_name]
            else:
                model = all_models[model_name]

            if hasattr(model, 'configure_homeostasis') and task_name in HOMEO_CONFIG:
                cfg = HOMEO_CONFIG[task_name]
                model.configure_homeostasis(
                    r_target     = cfg['r_target'],
                    trial_dur_ms = cfg['trial_dur_ms'],
                    enabled      = cfg.get('enabled', True),
                )

            if hasattr(model, 'configure_stdp'):
                stdp_cfg = STDP_CONFIG.get(model_name, {}).get(task_name, {})
                if stdp_cfg:
                    model.configure_stdp(**stdp_cfg)

            # T3: rec STDP — symmetric amplitudes, timing does the work.
            #
            # A2_plus=A2_minus=0.003 (ratio=1). For RANDOM rec connections:
            #   LTP rate ∝ A2_plus × τ_plus = 0.003 × 16.8 = 0.050
            #   LTD rate ∝ A2_minus × τ_minus = 0.003 × 33.7 = 0.101
            #   → LTD > LTP → random weights decrease toward 0 over 200 trials.
            #
            # For CAUSAL C→D pairs (C fires ~50ms before D):
            #   LTP: r1_C at D-spike = exp(-50/16.8) = 0.051 (per C-spike)
            #   With ~2 C-spikes per trial: r1_C ≈ 0.16 at D-window → net LTP > LTD.
            #   → C→D weights selectively grow from 0.1 to ~0.24 over 200 trials.
            #
            # During test (A→B→C only):
            #   G_syn_rec from potentiated C→D (W≈0.24, decayed 37% at 50ms) ≈ 1.4nS
            #   I_rec ≈ 1.4 × 55mV ≈ 77pA; I_total = I_bg(150) + 77 = 227pA > rheobase(200pA)
            #   → D fires. D-fire cascades to E via D→E rec (same timing/STDP) → acc=1.0.
            #
            # Previous setting (A2_plus=0.015, A2_minus=0.001, ratio=15:1): drove
            # rec weights to w_max=1.0 within 5 trials → all 50 outputs fire maximally
            # → o2 saturated → FF A3_plus term inflated → FF weights also grew to 0.8289.
            # acc=1.000 on every trial was an artifact of saturation, not sequence coding.
            if (task_name == 'T3'
                    and hasattr(model, 'rec_synapse')
                    and hasattr(model.rec_synapse, 'configure_stdp')):
                model.rec_synapse.configure_stdp(
                    A2_plus  = 0.003,
                    A3_plus  = 0.000,
                    A2_minus = 0.003,
                )

            if hasattr(model, 'configure_eligibility_trace'):
                if task_name in ELIGIBILITY_CONFIG:
                    ecfg = ELIGIBILITY_CONFIG[task_name]
                    model.configure_eligibility_trace(
                        enabled = ecfg['enabled'],
                        tau_e   = ecfg['tau_e'],
                    )
                else:
                    model.configure_eligibility_trace(enabled=False)

            # T3: two fixes for sequence prediction.
            #
            # Fix 1 — homeostasis disabled (HOMEO_CONFIG T3 enabled=False):
            #   Synaptic scaling (Turrigiano 1998) oscillates and collapses FF
            #   weights to ~0 within ~10 trials when r_target=25Hz is set too
            #   low relative to the initial FF drive. With W_ff≈0, output neurons
            #   never fire during elements → STDP has no pre/post pairs on
            #   rec_synapse → C→D recurrent associations never form → test
            #   acc=0.000 on every trial. Disabling homeostasis lets STDP run
            #   consistently. CAdEx intrinsic adaptation (b=20pA, tau_w=100ms)
            #   provides natural rate regulation without synaptic scaling.
            #   Biological basis: CA3 sequence encoding is STDP-driven on
            #   250ms/trial timescales; homeostasis operates on hours-days.
            #
            # Fix 2 — rec_synapse tau_syn=100ms (NMDA-like, recurrent only):
            #   With AMPA tau_syn=15ms, recurrent residual from C at D onset
            #   decays to exp(-30/15)=13.5% — insufficient to drive firing.
            #   tau=100ms was wrong: 74% of C remains at D onset AND 45% of B,
            #   27% of A — even random-weight networks fire in D/E windows from
            #   pure temporal bleedthrough. Not prediction, just decay spillover.
            #   tau=30ms: C at D onset = exp(-30/30)=0.37 — present but requires
            #   potentiated C→D weights to cross threshold. Bleedthrough check:
            #   with W_rec=0.1 (random) and 500Hz output during C, adaptation
            #   from A+B+C exactly cancels generic recurrent bleed →
            #   I_net<threshold. Only STDP-potentiated C→D connections drive
            #   D-window firing. D then cascades to E via D→E rec (same 30ms gap).
            #   FF synapse stays at AMPA 15ms.
            if hasattr(model, 'rec_synapse') and hasattr(model.rec_synapse, 'tau_syn'):
                # T3 rec tau_syn=50ms (NMDA-like): D at E-window onset retains
                # exp(-30/50)=0.549 (vs 0.368 at 30ms), providing the extra ~80pA
                # needed for E-selective to clear its adaptation-elevated threshold.
                # LIF W_DE→0 (anti-causal LTD from D firing during E-window), so
                # longer tau doesn't help it.
                model.rec_synapse.tau_syn = 50.0 if task_name == 'T3' else 15.0
            if hasattr(model, 'rec_synapse') and hasattr(model.rec_synapse, 'g_max'):
                model.rec_synapse.g_max = 3.0 if task_name == 'T3' else 1.5

            # T3: reset neuron state at start of each trial to model the
            # inter-trial interval (ITI). Without this, CAdEx adaptation (w)
            # from D/E firing in trial N suppresses D/E in trial N+1, causing
            # a period-2 alternation (acc=0,1,0,1...) that averages to 0.5.
            # Biological basis: ITI in sequence replay experiments is typically
            # 1-5s; tau_w=100ms means adaptation decays to <1% in ~500ms. The
            # reset models this return to resting state.
            if task_name == 'T3':
                model._reset_state_per_trial = True


            modulator_fn = None
            if task_name == 'T4':
                n_t4_trials  = len(task._intervals) * task._n_per_interval
                modulator_fn = make_t4_modulator_fn(n_trials=n_t4_trials)

            model.reset()

            try:
                result = runner.run(
                    model        = model,
                    task         = task,
                    model_name   = model_name,
                    modulator_fn = modulator_fn,
                )
                logger.log(result)
            except Exception as e:
                print(f"  ERROR: {model_name} × {task_name} failed: {e}")
                import traceback
                traceback.print_exc()
                continue

            jax.clear_caches()

            elapsed   = (time.time() - t_start) / 60.0
            remaining = elapsed / run_count * (n_runs - run_count) if run_count else 0
            print(f"  Completed {run_count}/{n_runs} | "
                  f"Elapsed: {elapsed:.1f}min | "
                  f"ETA: {remaining:.1f}min")

    logger.print_summary()
    total_min = (time.time() - t_start) / 60.0
    print(f"  Total runtime: {total_min:.1f} minutes")
    print(f"  Results saved to: {logger.run_dir}")


if __name__ == "__main__":
    main()
