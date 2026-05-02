"""
BenchmarkRunner — Executes model × task evaluation loops.

Uses bm.for_loop (BrainPy compiled scan) for the inner step loop,
giving ~80x speedup over pure Python on M1 Metal.

Key design decisions:
  - Stimulus arrays precomputed as JAX arrays before the scan
  - idx converted via jnp.asarray() inside step_fn (required for MPS)
  - US injection baked into I_us array — no Python branching in scan
  - First trial compiles (~2s), subsequent trials use cached XLA kernel (~0.09s/100 steps)
  - Python loop fallback if compilation fails

Spec Section 7.
"""

from __future__ import annotations
import time
import jax
import jax.numpy as jnp
import numpy as np
import brainpy.math as bm
from dataclasses import dataclass, field

from oi_bench.core.adapter import OIModel
from oi_bench.core.protocol import BenchmarkTask
from oi_bench.core.types import ModelState, TrialResult, Stimulus


@dataclass
class RunResult:
    """Results from one complete task run (one model × one task)."""
    model_name:             str
    task_name:              str
    trial_results:          list[TrialResult]
    learning_index:         float
    wall_time_per_trial:    list[float] = field(default_factory=list)
    weight_stats_per_trial: list[dict]  = field(default_factory=list)


class BenchmarkRunner:
    """
    Executes one model against one task for a full episode.

    Parameters
    ----------
    record_traces : bool
        Store per-step ModelState traces. Memory intensive. Default False.
    verbose : bool
        Print per-trial progress. Default True.
    report_every : int
        Print every N trials. Default 10.
    global_seed : int
        Base seed for JAX PRNG. Default 0.
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

    def _build_step_fn(self, model, I_in_jax, I_us_jax, I_bg):
        """
        Build the compiled step function for one trial.

        Closes over the precomputed stimulus arrays.
        idx is converted via jnp.asarray() to satisfy JAX abstractification.
        """
        has_output = hasattr(model, 'output_pop')
        has_synapse = hasattr(model, 'synapse')

        has_rec = hasattr(model, 'rec_synapse')

        def step_fn(idx):
            i       = jnp.asarray(idx, dtype=jnp.int32)
            current = I_in_jax[i]
            us      = I_us_jax[i]

            # Step input population
            model.input_pop.update(x=current)
            S_pre  = model.input_pop.spike.value.astype(bm.float32)
            S_post = model.output_pop.spike.value.astype(bm.float32)

            # Synapse
            I_syn = model.synapse.update(S_pre, S_post,
                                         model.output_pop.V.value)

            # Recurrent synapse + global inhibition (if present)
            if has_rec:
                I_rec  = model.rec_synapse.update(S_post, S_post,
                                                  model.output_pop.V.value)
                mean_act = jnp.mean(S_post)
                I_inh = -model._inhibition_strength * mean_act * bm.ones(model.n_output)
            else:
                I_rec  = bm.zeros(model.n_output)
                I_inh  = bm.zeros(model.n_output)

            # Step output population: background + synaptic + recurrent + inhibition + US
            model.output_pop.update(x=I_bg + I_syn + I_rec + I_inh + us)

            return model.output_pop.spike.value.astype(bm.float32)

        def step_fn_no_output(idx):
            """Fallback for models without output_pop (e.g. LSM)."""
            i       = jnp.asarray(idx, dtype=jnp.int32)
            current = I_in_jax[i]
            stim    = Stimulus(
                current     = np.array(current),
                spike_train = np.zeros(model.n_input, dtype=np.float32),
                t           = 0.0,
            )
            state = model.step(stim)
            return state.spikes.astype(bm.float32)

        return step_fn if has_output and has_synapse else step_fn_no_output

    def run(
        self,
        model:      OIModel,
        task:       BenchmarkTask,
        model_name: str = "model",
        modulator_fn    = None,
    ) -> RunResult:
        """Run one complete episode: model × task."""
        task.setup(model)
        model.reset()

        trial_results        = []
        wall_times           = []
        weight_stats_history = []
        compiled             = False   # track if JIT compiled successfully

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"  OI-Bench: {model_name} × {task.name}")
            print(f"  {task.n_trials} trials × {task.trial_duration_ms}ms")
            print(f"  dt={model.dt}ms")
            print(f"{'='*60}\n")

        # Background current for output population
        I_bg = bm.full(model.n_output, 150.0, dtype=bm.float32) \
               if hasattr(model, 'output_pop') else None

        for trial_id in range(task.n_trials):
            trial_key = jax.random.fold_in(self.base_key, trial_id)
            model.pre_trial(trial_id)

            # Generate and precompute stimulus arrays
            stimuli = task.generate_trial(trial_id, trial_key)
            n_steps = len(stimuli)

            I_in = np.zeros((n_steps, model.n_input),  dtype=np.float32)
            I_us = np.zeros((n_steps, model.n_output), dtype=np.float32)

            for step_idx, stim in enumerate(stimuli):
                I_in[step_idx] = stim.current
                if hasattr(task, 'us_current_for_step'):
                    I_us[step_idx] = task.us_current_for_step(
                        stim.t, trial_id
                    )

            I_in_jax = jnp.array(I_in)
            I_us_jax = jnp.array(I_us)
            indices  = bm.arange(n_steps)

            output_spikes_accum = np.zeros(model.n_output, dtype=np.float32)
            t0 = time.time()

            # --- Attempt compiled scan ---
            if hasattr(model, 'output_pop') and hasattr(model, 'synapse'):
                step_fn = self._build_step_fn(
                    model, I_in_jax, I_us_jax, I_bg
                )
                try:
                    all_spikes = bm.for_loop(step_fn, indices, jit=True)
                    output_spikes_accum = np.array(jnp.sum(all_spikes, axis=0))
                    # Update homeostasis spike counter after scan completes
                    model._trial_spike_counts += output_spikes_accum
                    compiled = True
                except Exception as e:
                    if self.verbose and trial_id == 0:
                        print(f"  bm.for_loop failed ({e}), using Python loop")
                    compiled = False

            # --- Python loop fallback ---
            if not compiled:
                for step_idx in range(n_steps):
                    stim = Stimulus(
                        current     = I_in[step_idx],
                        spike_train = np.zeros(model.n_input, dtype=np.float32),
                        t           = stimuli[step_idx].t,
                    )
                    state = model.step(stim)
                    if hasattr(model, 'output_pop') and np.any(I_us[step_idx] > 0):
                        model.output_pop.update(
                            x=bm.array(I_us[step_idx], dtype=bm.float32)
                        )
                    output_spikes_accum += np.array(state.spikes)

            wall_time = time.time() - t0

            # Response: mean firing rate (Hz) over trial
            trial_dur_s = task.trial_duration_ms / 1000.0
            response    = jnp.array(output_spikes_accum / trial_dur_s)

            # Score
            scores = task.compute_score(
                response    = response,
                trial_id    = trial_id,
                state_trace = [],
            )

            modulator = 1.0
            if modulator_fn is not None:
                modulator = float(modulator_fn(trial_id, scores))

            model.post_trial(
                trial_id    = trial_id,
                state_trace = [],
                modulator   = modulator,
            )

            result = TrialResult(
                trial_id    = trial_id,
                correct     = bool(scores.get('accuracy', 0.0) >= 0.5)
                              if trial_id >= getattr(task, '_n_conditioning', 0)
                              else None,
                response    = response,
                scores      = scores,
                state_trace = [],
                metadata    = {
                    'trial_type': 'test' if trial_id >= getattr(
                        task, '_n_conditioning', task.n_trials
                    ) else 'conditioning',
                    'wall_time':  wall_time,
                    'compiled':   compiled,
                },
            )
            trial_results.append(result)
            wall_times.append(wall_time)

            if hasattr(model, 'weight_stats'):
                weight_stats_history.append(model.weight_stats)

            if self.verbose and (trial_id + 1) % self.report_every == 0:
                acc = scores.get('accuracy', scores.get('performance', 0.0))
                cr  = scores.get('cr_rate', None)
                msg = (f"  Trial {trial_id+1:4d}/{task.n_trials} | "
                       f"acc={acc:.3f} | time={wall_time:.1f}s | "
                       f"{'JIT' if compiled else 'PY'}")
                if cr is not None:
                    msg += f" | cr={cr:.1f}Hz"
                if weight_stats_history:
                    msg += f" | w={weight_stats_history[-1].get('mean',0):.4f}"
                print(msg)

        li = task.learning_index(trial_results)

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"  Learning Index : {li:.4f}")
            print(f"  Compiled       : {compiled}")
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
