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

Runtime estimate (M1, JIT compiled):
    Full run: ~0.9s per trial × ~200 trials × 18 runs ≈ 54 minutes
    Smoke test: ~30 seconds
"""

import os
os.environ["JAX_PLATFORMS"] = "mps"

import argparse
import time
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
# r_target = natural output firing rate for each task
# trial_dur_ms = actual trial duration for each task
# ------------------------------------------------------------------
HOMEO_CONFIG = {
    'T1': {'r_target': 110.0, 'trial_dur_ms': 400.0},
    'T2': {'r_target': 200.0, 'trial_dur_ms': 200.0},
    'T3': {'r_target':  25.0, 'trial_dur_ms': 250.0},
    'T4': {'r_target':  25.0, 'trial_dur_ms': 1200.0},
    'T5': {'r_target':  50.0, 'trial_dur_ms': 800.0},
    'T6': {'r_target':  50.0, 'trial_dur_ms': 500.0},
}


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
# ------------------------------------------------------------------
def build_tasks(smoke: bool = False) -> dict:
    if smoke:
        return {
            'T1': ClassicalConditioningTask(
                n_trials=10, n_conditioning_trials=8),
            'T2': PatternCompletionTask(
                n_patterns=3, n_learning_reps=3),
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
            'T2': PatternCompletionTask(),
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
                )

            # Reset model between tasks
            model.reset()

            try:
                result = runner.run(
                    model      = model,
                    task       = task,
                    model_name = model_name,
                )
                logger.log(result)
            except Exception as e:
                print(f"  ERROR: {model_name} × {task_name} failed: {e}")
                import traceback
                traceback.print_exc()
                continue

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
