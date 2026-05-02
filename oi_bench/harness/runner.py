"""
BenchmarkRunner — Executes model × task evaluation loops.

The runner is the central orchestrator of OI-Bench. It:
  1. Drives the trial loop (generate_trial → step × T → compute_score)
  2. Handles US current injection for tasks that require it (e.g. T1)
  3. Calls pre_trial() and post_trial() hooks at correct times
  4. Records ModelState traces if requested
  5. Computes Learning Index per task via BenchmarkTask.learning_index()
  6. Returns structured results for logging and plotting

Design decisions (from spec Section 7):
  - The runner loop is NOT JIT-compiled. JIT lives inside model.step().
    This keeps the runner debuggable and flexible.
  - Results are accumulated in memory during a run. HDF5 logging
    will be added in the harness/logger.py module.
  - Wall-clock timing is recorded per trial for performance analysis.
  - Seeds are derived from global_seed via jax.random.split for
    full reproducibility.
"""

from __future__ import annotations
import time
import jax
import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from oi_bench.core.adapter import OIModel
from oi_bench.core.protocol import BenchmarkTask
from oi_bench.core.types import ModelState, TrialResult


@dataclass
class RunResult:
    """
    Results from one complete task run (one model × one task).

    Attributes
    ----------
    model_name : str
    task_name : str
    trial_results : list[TrialResult]
    learning_index : float
        LI ∈ [0, 1] computed by task.learning_index().
    wall_time_per_trial : list[float]
        Wall-clock seconds per trial.
    weight_stats_per_trial : list[dict]
        Synapse weight statistics snapshot after each trial.
        Empty if model does not expose weight_stats property.
    """
    model_name:             str
    task_name:              str
    trial_results:          list[TrialResult]
    learning_index:         float
    wall_time_per_trial:    list[float]   = field(default_factory=list)
    weight_stats_per_trial: list[dict]    = field(default_factory=list)


class BenchmarkRunner:
    """
    Executes one model against one task for a full episode.

    Parameters
    ----------
    record_traces : bool
        If True, full per-step ModelState traces are stored in TrialResult.
        Memory-intensive for long trials — disable for large runs.
        Default False.
    verbose : bool
        Print per-trial progress. Default True.
    report_every : int
        Print progress every N trials. Default 10.
    global_seed : int
        Base seed for JAX PRNG key generation. Default 0.
    """

    def __init__(
        self,
        record_traces: bool = False,
        verbose: bool       = True,
        report_every: int   = 10,
        global_seed: int    = 0,
    ):
        self.record_traces = record_traces
        self.verbose       = verbose
        self.report_every  = report_every
        self.base_key      = jax.random.PRNGKey(global_seed)

    def run(
        self,
        model:      OIModel,
        task:       BenchmarkTask,
        model_name: str = "model",
        modulator_fn=None,
    ) -> RunResult:
        """
        Run one complete episode: model × task.

        Parameters
        ----------
        model : OIModel
            The model to evaluate.
        task : BenchmarkTask
            The task to run.
        model_name : str
            Label for logging. Default "model".
        modulator_fn : callable or None
            Optional function (trial_id, trial_result) → float.
            Returns the three-factor modulator signal for post_trial().
            Default None → modulator=1.0 throughout.

        Returns
        -------
        RunResult
        """
        # Setup
        task.setup(model)
        model.reset()

        trial_results          = []
        wall_times             = []
        weight_stats_history   = []

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"  OI-Bench: {model_name} × {task.name}")
            print(f"  {task.n_trials} trials × {task.trial_duration_ms}ms")
            print(f"  dt={model.dt}ms | "
                  f"record_traces={self.record_traces}")
            print(f"{'='*60}\n")

        for trial_id in range(task.n_trials):
            # Per-trial RNG key derived from base key + trial_id
            trial_key = jax.random.fold_in(self.base_key, trial_id)

            # Pre-trial hook
            model.pre_trial(trial_id)

            # Generate stimulus sequence
            stimuli = task.generate_trial(trial_id, trial_key)
            n_steps = len(stimuli)

            # Run trial
            state_trace = [] if self.record_traces else None
            output_spikes_accum = np.zeros(model.n_output, dtype=np.float32)
            t0 = time.time()

            for step_idx, stimulus in enumerate(stimuli):
                t_ms = stimulus.t

                # US current injection (task-specific, e.g. T1 conditioning)
                if hasattr(task, 'us_current_for_step'):
                    us = task.us_current_for_step(t_ms, trial_id)
                    # Deliver US directly to model's output population
                    # by adding to stimulus current via extras mechanism
                    stimulus = type(stimulus)(
                        current     = stimulus.current,
                        spike_train = stimulus.spike_train,
                        t           = stimulus.t,
                        label       = stimulus.label,
                    )
                    # US injected separately through model's output background
                    model_state = model.step(stimulus)
                    # Apply US on top of background to output population
                    if hasattr(model, 'output_pop') and np.any(us > 0):
                        import brainpy.math as bm
                        model.output_pop.update(
                            x=bm.array(us, dtype=bm.float32)
                        )
                        # Re-read state after US
                        model_state = ModelState(
                            spikes   = model.output_pop.spike.value.astype(bm.float32),
                            membrane = model.output_pop.V.value,
                            weights  = model_state.weights,
                            extras   = model_state.extras,
                        )
                else:
                    model_state = model.step(stimulus)

                output_spikes_accum += np.array(model_state.spikes)

                if self.record_traces:
                    state_trace.append(model_state)

            wall_time = time.time() - t0

            # Compute response: mean firing rate over full trial (Hz)
            trial_dur_s = task.trial_duration_ms / 1000.0
            response    = jnp.array(output_spikes_accum / trial_dur_s)

            # Score trial
            scores = task.compute_score(
                response    = response,
                trial_id    = trial_id,
                state_trace = state_trace if self.record_traces else [],
            )

            # Modulator signal for three-factor hook
            modulator = 1.0
            if modulator_fn is not None:
                modulator = float(modulator_fn(trial_id, scores))

            # Post-trial hook (homeostasis, weight consolidation)
            model.post_trial(
                trial_id    = trial_id,
                state_trace = state_trace if self.record_traces else [],
                modulator   = modulator,
            )

            # Assemble TrialResult
            result = TrialResult(
                trial_id    = trial_id,
                correct     = bool(scores.get('accuracy', 0.0) >= 0.5)
                              if trial_id >= getattr(task, '_n_conditioning', 0)
                              else None,
                response    = response,
                scores      = scores,
                state_trace = state_trace if self.record_traces else [],
                metadata    = {
                    'trial_type': 'test' if trial_id >= getattr(
                        task, '_n_conditioning', task.n_trials
                    ) else 'conditioning',
                    'wall_time': wall_time,
                },
            )
            trial_results.append(result)
            wall_times.append(wall_time)

            # Weight stats snapshot
            if hasattr(model, 'weight_stats'):
                weight_stats_history.append(model.weight_stats)

            # Progress reporting
            if self.verbose and (trial_id + 1) % self.report_every == 0:
                acc = scores.get('accuracy', scores.get('performance', 0.0))
                cr  = scores.get('cr_rate', None)
                msg = (f"  Trial {trial_id+1:4d}/{task.n_trials} | "
                       f"acc={acc:.3f} | "
                       f"time={wall_time:.1f}s")
                if cr is not None:
                    msg += f" | cr_rate={cr:.1f}Hz"
                if weight_stats_history:
                    msg += f" | mean_w={weight_stats_history[-1]['mean']:.4f}"
                print(msg)

        # Compute Learning Index
        li = task.learning_index(trial_results)

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"  Learning Index : {li:.4f}")
            print(f"  Total wall time: {sum(wall_times)/60:.1f} min")
            print(f"{'='*60}\n")

        return RunResult(
            model_name             = model_name,
            task_name              = task.name,
            trial_results          = trial_results,
            learning_index         = li,
            wall_time_per_trial    = wall_times,
            weight_stats_per_trial = weight_stats_history,
        )
