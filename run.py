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

T2 scoring:
    PatternCompletionTask now uses closest-centroid classification.
    During learning, output response vectors are averaged per pattern.
    During test, the output is compared against stored centroids via
    cosine similarity. This correctly measures attractor basin formation
    rather than bit-level reproduction of the input pattern.

T4 three-factor eligibility trace (Izhikevich 2007):
    Models use dopamine-gated STDP for T4. STDP events accumulate in
    eligibility traces e_ij (τ_e=1000ms). After each trial,
    t4_modulator_fn computes a dopamine reward signal.

    Curriculum tolerance: starts at ±80% of T_target and linearly
    narrows to ±15% by trial 100. This solves the cold-start problem —
    early trials get reward for rough timing, later trials require
    precision. The reward magnitude still scales with proximity so
    there is always a gradient toward exact timing.

    Reference: Izhikevich (2007) Cereb. Cortex 17:2443-2452
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
    'T3': {'r_target':  25.0, 'trial_dur_ms':  250.0, 'enabled': True},
    'T4': {'r_target':   5.0, 'trial_dur_ms': 1220.0, 'enabled': False},
    'T5': {'r_target':  50.0, 'trial_dur_ms':  800.0, 'enabled': True},
    'T6': {'r_target':  50.0, 'trial_dur_ms':  500.0, 'enabled': True},
}

# ------------------------------------------------------------------
# T4 eligibility trace configuration
# ------------------------------------------------------------------
ELIGIBILITY_CONFIG = {
    'T4': {'enabled': True, 'tau_e': 1000.0},
}


# ------------------------------------------------------------------
# T4 dopamine modulator function with curriculum tolerance
#
# Curriculum: tolerance starts wide (80%) and narrows to 15% by
# trial 100. This gives the network a reward signal even from rough
# early timing, then gradually demands precision.
#
# Reward magnitude scales with proximity within the current tolerance
# window: da_level = 1 - (weber_error / tolerance)
#
# This is a closure — we use a mutable container to track trial count
# across calls, since modulator_fn is called once per trial.
# ------------------------------------------------------------------
def make_t4_modulator_fn():
    """
    Factory that returns a stateful T4 modulator function.

    Returns a closure that tracks trial count internally and
    applies curriculum tolerance scheduling.
    """
    state = {'trial_count': 0}

    def t4_modulator_fn(trial_id: int, scores: dict) -> float:
        weber_error = scores.get('weber_error', 1.0)
        t = state['trial_count']
        state['trial_count'] += 1

        # Curriculum: linear decay from 0.80 to 0.15 over 100 trials
        tol_start = 0.80
        tol_end   = 0.15
        tol_trials = 100
        tolerance = max(tol_end, tol_start - (tol_start - tol_end) * t / tol_trials)

        if weber_error <= tolerance:
            da_level = float(1.0 - weber_error / tolerance)
        else:
            da_level = 0.0

        return da_level

    return t4_modulator_fn


# ------------------------------------------------------------------
# Model registry
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
            plasticity=True, homeostasis=True,
            seed=seed,
        ),
        'lsm': LiquidStateMachine(
            n_input=100, n_reservoir=200, n_output=50,
            dt=0.1, seed=seed,
        ),
    }


# ------------------------------------------------------------------
# Task registry
# n_patterns=5: within Hopfield capacity for n_output=50 (~0.14×50=7)
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
            'T4': IntervalTimingTask(
                n_trials_per_interval=5),
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

    all_models = build_models(seed=args.model_seed)
    all_tasks  = build_tasks(smoke=args.smoke)

    models_to_run = {k: v for k, v in all_models.items()
                     if args.models is None or k in args.models}
    tasks_to_run  = {k: v for k, v in all_tasks.items()
                     if args.tasks is None or k in args.tasks}

    n_runs = len(models_to_run) * len(tasks_to_run)

    runner = BenchmarkRunner(
        record_traces = args.record_traces,
        verbose       = True,
        report_every  = 10,
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

    for model_name, model in models_to_run.items():
        for task_name, task in tasks_to_run.items():
            run_count += 1
            print(f"\n[{run_count}/{n_runs}] {model_name} × {task_name}")
            print(f"  Started: {datetime.now().strftime('%H:%M:%S')}")

            # Configure homeostasis for this task
            if hasattr(model, 'configure_homeostasis') and task_name in HOMEO_CONFIG:
                cfg = HOMEO_CONFIG[task_name]
                model.configure_homeostasis(
                    r_target     = cfg['r_target'],
                    trial_dur_ms = cfg['trial_dur_ms'],
                    enabled      = cfg.get('enabled', True),
                )

            # Configure eligibility trace mode
            if hasattr(model, 'configure_eligibility_trace'):
                if task_name in ELIGIBILITY_CONFIG:
                    ecfg = ELIGIBILITY_CONFIG[task_name]
                    model.configure_eligibility_trace(
                        enabled = ecfg['enabled'],
                        tau_e   = ecfg['tau_e'],
                    )
                else:
                    model.configure_eligibility_trace(enabled=False)

            # Build fresh modulator function for T4 (stateful curriculum)
            modulator_fn = make_t4_modulator_fn() if task_name == 'T4' else None

            # Reset fast-timescale state between tasks.
            # Synaptic weights are NOT reset — they persist.
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

            # Release XLA compiled kernels between tasks.
            jax.clear_caches()

            elapsed   = (time.time() - t_start) / 60.0
            remaining = elapsed / run_count * (n_runs - run_count)
            print(f"  Completed {run_count}/{n_runs} | "
                  f"Elapsed: {elapsed:.1f}min | "
                  f"ETA: {remaining:.1f}min")

    logger.print_summary()
    total_min = (time.time() - t_start) / 60.0
    print(f"  Total runtime: {total_min:.1f} minutes")
    print(f"  Results saved to: {logger.run_dir}")


if __name__ == "__main__":
    main()
