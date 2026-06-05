"""
BenchmarkRunner — Executes model × task evaluation loops.

PLASTICITY GATING DURING TEST TRIALS
--------------------------------------
The runner calls task.is_learning_trial(trial_id) before each trial.
For test trials, it passes plasticity_scale=0.0 (JAX float32) into
synapse.update() via the step function.

This is a runtime JAX value, not a Python bool. This is critical because:
- bm.for_loop JIT-compiles the step function by shape/structure
- A Python bool captured at trace time cannot be changed at runtime
- A JAX scalar multiplied into dW IS evaluated at runtime, correctly
  freezing weights even within the compiled scan

The step function is rebuilt each trial via _build_step_fn, capturing
the current plasticity_scale as a Python float (0.0 or 1.0). JAX traces
a fresh function for each distinct captured value — two kernels total.

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
    model_name:             str
    task_name:              str
    trial_results:          list[TrialResult]
    learning_index:         float
    wall_time_per_trial:    list[float] = field(default_factory=list)
    weight_stats_per_trial: list[dict]  = field(default_factory=list)


class BenchmarkRunner:

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

    # ------------------------------------------------------------------
    # Path 1: bm.for_loop JIT scan
    # plasticity_scale captured as Python float (0.0 or 1.0) —
    # JAX traces two distinct kernels, one per value.
    # ------------------------------------------------------------------

    def _build_step_fn(self, model, I_bg, plasticity_scale: float = 1.0):
        """
        Build a step function that receives (step_inputs) from bm.for_loop.

        step_inputs is a 2D slice of xs (shape: (2, max_n)) where:
          step_inputs[0][:n_input] = input current for this step
          step_inputs[1][:n_output] = US current for this step

        I_in and I_us are passed as operands to bm.for_loop (not captured
        in the closure) so the same compiled function object is reused
        across all trials. Creating a new closure each trial — even with
        identical computation — causes a JAX JIT cache miss every trial,
        leading to full XLA recompilation (30–120s on Metal) once the
        cache fills (~trial 130 for T5/T6).
        """
        has_rec     = hasattr(model, 'rec_synapse')
        p_scale_jax = jnp.float32(plasticity_scale)
        n_input     = model.n_input
        n_output    = model.n_output

        def step_fn(step_inputs):
            current = step_inputs[0][:n_input]
            us      = step_inputs[1][:n_output]

            model.input_pop.update(x=current)
            S_pre  = model.input_pop.spike.value.astype(bm.float32)
            S_post = model.output_pop.spike.value.astype(bm.float32)
            I_syn  = model.synapse.update(
                S_pre, S_post, model.output_pop.V.value,
                plasticity_scale=p_scale_jax)

            if has_rec:
                I_rec    = model.rec_synapse.update(
                    S_post, S_post, model.output_pop.V.value,
                    plasticity_scale=p_scale_jax)
                mean_act = jnp.mean(S_post)
                I_inh    = (-model._inhibition_strength
                             * mean_act * bm.ones(n_output))
            else:
                I_rec = bm.zeros(n_output)
                I_inh = bm.zeros(n_output)

            model.output_pop.update(x=I_bg + I_syn + I_rec + I_inh + us)
            return model.output_pop.spike.value.astype(bm.float32)

        return step_fn

    # ------------------------------------------------------------------
    # Path 2: jax.lax.scan with eligibility trace carry (T4)
    # ------------------------------------------------------------------

    def _eligibility_scan(self, model, I_in_jax, I_us_jax, I_bg_val,
                          snap_steps=None):
        n_pre   = model.n_input
        n_out   = model.n_output
        syn     = model.synapse
        rec     = model.rec_synapse
        n_snaps = len(snap_steps) if snap_steps is not None else 0
        snap_steps_jax = (jnp.array(snap_steps, dtype=jnp.int32)
                          if n_snaps > 0 else jnp.zeros(0, dtype=jnp.int32))

        carry0 = (
            jnp.zeros(n_pre), jnp.zeros(n_pre),
            jnp.zeros(n_out), jnp.zeros(n_out),
            jnp.zeros((n_pre, n_out)), jnp.zeros((n_pre, n_out)),
            jnp.zeros(n_out), jnp.zeros(n_out),
            jnp.zeros(n_out), jnp.zeros(n_out),
            jnp.zeros((n_out, n_out)), jnp.zeros((n_out, n_out)),
            jnp.array(model.input_pop.V.value),
            jnp.array(model.input_pop.w.value)
                if hasattr(model.input_pop, 'w') else jnp.zeros(n_pre),
            jnp.array(model.input_pop.Ca.value)
                if hasattr(model.input_pop, 'Ca') else jnp.zeros(n_pre),
            jnp.array(model.output_pop.V.value),
            jnp.array(model.output_pop.w.value)
                if hasattr(model.output_pop, 'w') else jnp.zeros(n_out),
            jnp.array(model.output_pop.Ca.value)
                if hasattr(model.output_pop, 'Ca') else jnp.zeros(n_out),
            jnp.zeros(n_out),
            jnp.int32(0),                                    # step_ctr
            jnp.zeros((n_snaps, n_pre, n_out)),              # e_snaps
        )

        W_ff  = jnp.array(syn.W.value)
        W_rec = jnp.array(rec.W.value)
        V_T   = (jnp.array(model.output_pop.V_T.value)
                 if hasattr(model.output_pop, 'V_T')
                 and hasattr(model.output_pop.V_T, 'value')
                 else jnp.full(n_out, -50.0))

        op = model.output_pop; inp = model.input_pop; dt = model.dt
        has_cadex = hasattr(op, 'delta_T')
        # LIF input uses V_T (-50mV) as spike threshold, not V_peak (20mV).
        # V_ss from group_current=800pA is 10mV < V_peak=20mV → input never spikes
        # → S_pre=0 → eligibility trace never accumulates → weights frozen.
        # Read V_T once before JIT compile; it is static for T4 (homeostasis off).
        V_T_in = (jnp.array(inp.V_T.value)
                  if not has_cadex
                  and hasattr(inp, 'V_T')
                  and hasattr(inp.V_T, 'value')
                  else None)

        def scan_step(carry, x):
            (r1_ff, r2_ff, o1_ff, o2_ff, e_ff, g_ff,
             r1_rec, r2_rec, o1_rec, o2_rec, e_rec, g_rec,
             V_in, w_in, Ca_in, V_out, w_out, Ca_out, S_post_prev,
             step_ctr, e_snaps) = carry

            I_in_step = x[0][:n_pre]
            I_us_step = x[1][:n_out]

            if has_cadex:
                I_ca_in = inp.g_Ca * \
                    (1.0/(1.0+jnp.exp(-(V_in+30.0)/6.0)))**2*(V_in-inp.E_Ca)
                dV_in = (-inp.g_L*(V_in-inp.E_L)
                         + inp.g_L*inp.delta_T*jnp.exp(
                             jnp.clip((V_in-inp.V_T)/inp.delta_T,-20.,20.))
                         - I_ca_in + I_in_step - w_in) / inp.C_m
                dw_in  = (inp.a*(V_in-inp.E_L) - w_in) / inp.tau_w
                dCa_in = -Ca_in / inp.tau_Ca
                V_in_n = V_in+dV_in*dt; w_in_n = w_in+dw_in*dt
                Ca_in_n = Ca_in+dCa_in*dt
                spike_in = V_in_n >= inp.V_peak
                V_in_n = jnp.where(spike_in, inp.V_r, V_in_n)
                w_in_n = jnp.where(spike_in, w_in_n+inp.b, w_in_n)
                Ca_in_n = jnp.where(spike_in, Ca_in_n+inp.rho, Ca_in_n)
            else:
                dV_in = (-inp.g_L*(V_in-inp.E_L)+I_in_step)/inp.C_m
                V_in_n = V_in+dV_in*dt
                thr_in = V_T_in if V_T_in is not None else inp.V_peak
                spike_in = V_in_n >= thr_in
                V_in_n = jnp.where(spike_in, inp.V_r, V_in_n)
                w_in_n = w_in; Ca_in_n = Ca_in

            S_pre = spike_in.astype(jnp.float32)

            r1_ff_n,r2_ff_n,o1_ff_n,o2_ff_n,e_ff_n,g_ff_n,I_syn = \
                syn.stdp_step_functional(
                    r1_ff,r2_ff,o1_ff,o2_ff,e_ff,g_ff,
                    S_pre,S_post_prev,V_out,W_ff)
            r1_rec_n,r2_rec_n,o1_rec_n,o2_rec_n,e_rec_n,g_rec_n,I_rec = \
                rec.stdp_step_functional(
                    r1_rec,r2_rec,o1_rec,o2_rec,e_rec,g_rec,
                    S_post_prev,S_post_prev,V_out,W_rec)

            mean_act = jnp.mean(S_post_prev)
            I_inh   = -model._inhibition_strength*mean_act*jnp.ones(n_out)
            I_total = I_bg_val + I_syn + I_rec + I_inh + I_us_step

            if has_cadex:
                I_ca_out = op.g_Ca * \
                    (1.0/(1.0+jnp.exp(-(V_out+30.0)/6.0)))**2*(V_out-op.E_Ca)
                dV_out = (-op.g_L*(V_out-op.E_L)
                          + op.g_L*op.delta_T*jnp.exp(
                              jnp.clip((V_out-V_T)/op.delta_T,-20.,20.))
                          - I_ca_out+I_total-w_out) / op.C_eff
                dw_out = (op.a*(V_out-op.E_L)-w_out)/op.tau_w
                dCa_out = -Ca_out/op.tau_Ca
                V_out_n = V_out+dV_out*dt; w_out_n = w_out+dw_out*dt
                Ca_out_n = Ca_out+dCa_out*dt
                spike_out = V_out_n >= op.V_peak
                V_out_n = jnp.where(spike_out, op.V_r, V_out_n)
                w_out_n = jnp.where(spike_out, w_out_n+op.b, w_out_n)
                Ca_out_n = jnp.where(spike_out, Ca_out_n+op.rho, Ca_out_n)
            else:
                dV_out = (-op.g_L*(V_out-op.E_L)+I_total)/op.C_m
                V_out_n = V_out+dV_out*dt
                # LIF uses V_T (adaptive threshold) not V_peak.
                # ref_count refractory: elevate threshold by 1000mV when refractory.
                ref   = jnp.array(op.ref_count.value) if hasattr(op, 'ref_count') else jnp.zeros(n_out)
                V_thr = V_T + (ref > 0.0).astype(jnp.float32) * 1000.0
                spike_out = V_out_n >= V_thr
                V_out_n = jnp.where(spike_out, op.V_r, V_out_n)
                w_out_n = w_out; Ca_out_n = Ca_out

            S_out = spike_out.astype(jnp.float32)
            # Snapshot e_ff at each group's burst-off step (Izhikevich 2007:
            # DA at burst time).  Python loop is unrolled at JAX trace time.
            for kg in range(n_snaps):
                e_snaps = jnp.where(
                    step_ctr == snap_steps_jax[kg],
                    e_snaps.at[kg].set(e_ff_n),
                    e_snaps,
                )

            new_carry = (r1_ff_n,r2_ff_n,o1_ff_n,o2_ff_n,e_ff_n,g_ff_n,
                         r1_rec_n,r2_rec_n,o1_rec_n,o2_rec_n,e_rec_n,g_rec_n,
                         V_in_n,w_in_n,Ca_in_n,V_out_n,w_out_n,Ca_out_n,S_out,
                         step_ctr + jnp.int32(1), e_snaps)
            return new_carry, S_out

        max_n    = max(n_pre, n_out)
        I_in_pad = jnp.pad(I_in_jax, ((0,0),(0,max_n-n_pre)))
        I_us_pad = jnp.pad(I_us_jax, ((0,0),(0,max_n-n_out)))
        xs       = jnp.stack([I_in_pad, I_us_pad], axis=1)

        final_carry, all_spikes = jax.lax.scan(scan_step, carry0, xs)
        e_snaps_out = final_carry[20] if n_snaps > 0 else None
        return all_spikes, final_carry[4], final_carry[10], e_snaps_out

    # ------------------------------------------------------------------
    # Path 3: Python loop fallback
    # ------------------------------------------------------------------

    def _python_loop_fallback(self, model, stimuli, I_in, I_us, I_bg,
                               need_spike_ts):
        output_spikes_accum = np.zeros(model.n_output, dtype=np.float32)
        spike_ts_list       = [] if need_spike_ts else None
        for step_idx in range(len(stimuli)):
            stim = Stimulus(
                current     = I_in[step_idx],
                spike_train = np.zeros(model.n_input, dtype=np.float32),
                t           = stimuli[step_idx].t,
            )
            state  = model.step(stim)
            spikes = np.array(state.spikes)
            output_spikes_accum += spikes
            if need_spike_ts:
                spike_ts_list.append(float(spikes.sum()))
        spike_ts = (np.array(spike_ts_list, dtype=np.float32)
                    if need_spike_ts else None)
        return output_spikes_accum, spike_ts

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(
        self,
        model:      OIModel,
        task:       BenchmarkTask,
        model_name: str = "model",
        modulator_fn    = None,
    ) -> RunResult:
        task.setup(model)
        model.reset()

        trial_results        = []
        wall_times           = []
        weight_stats_history = []
        need_spike_ts        = task.requires_spike_times
        use_elig_scan        = (hasattr(model, 'synapse')
                                and getattr(model.synapse,
                                            'use_eligibility_trace', False))

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"  OI-Bench: {model_name} × {task.name}")
            print(f"  {task.n_trials} trials × {task.trial_duration_ms}ms")
            print(f"  dt={model.dt}ms")
            print(f"{'='*60}\n")

        _i_bg    = getattr(model, '_I_bg', 150.0)
        I_bg     = (bm.full(model.n_output, _i_bg, dtype=bm.float32)
                    if hasattr(model, 'output_pop') else None)
        I_bg_val = (jnp.full(model.n_output, _i_bg)
                    if I_bg is not None else None)

        # Cache compiled step functions keyed by plasticity_scale (1.0 or 0.0).
        # Building a new closure each trial causes a JAX JIT cache miss on every
        # trial (new Python object = new function identity). Two closures total.
        _jit_step_fns = {}
        n_input  = model.n_input
        n_output = model.n_output
        max_n    = max(n_input, n_output)

        for trial_id in range(task.n_trials):
            trial_key   = jax.random.fold_in(self.base_key, trial_id)
            is_learning = task.is_learning_trial(trial_id)
            # 1.0 during learning, 0.0 during test — JAX runtime value
            p_scale     = 1.0 if is_learning else 0.0

            model.pre_trial(trial_id)

            stimuli = task.generate_trial(trial_id, trial_key)
            n_steps = len(stimuli)

            I_in = np.zeros((n_steps, model.n_input),  dtype=np.float32)
            I_us = np.zeros((n_steps, model.n_output), dtype=np.float32)
            for step_idx, stim in enumerate(stimuli):
                I_in[step_idx] = stim.current
                if hasattr(task, 'us_current_for_step'):
                    I_us[step_idx] = task.us_current_for_step(stim.t, trial_id)

            I_in_jax = jnp.array(I_in)
            I_us_jax = jnp.array(I_us)
            t0       = time.time()
            compiled = False
            e_ff_final = e_rec_final = None

            if use_elig_scan and hasattr(model, 'output_pop'):
                _snap_steps = (getattr(task, 'group_off_steps', None)
                               if is_learning else None)
                all_spikes, e_ff_final, e_rec_final, e_snaps = \
                    self._eligibility_scan(
                        model, I_in_jax, I_us_jax, I_bg_val,
                        snap_steps=_snap_steps)
                all_spikes_np       = np.array(all_spikes)
                output_spikes_accum = all_spikes_np.sum(axis=0)
                model._trial_spike_counts += output_spikes_accum
                spike_timeseries = (all_spikes_np.sum(axis=1).astype(np.float32)
                                    if need_spike_ts else None)
                compiled = True

            elif hasattr(model, 'output_pop') and hasattr(model, 'synapse'):
                # Reuse cached step_fn — same Python object → JAX cache hit
                if p_scale not in _jit_step_fns:
                    _jit_step_fns[p_scale] = self._build_step_fn(
                        model, I_bg, plasticity_scale=p_scale)
                step_fn = _jit_step_fns[p_scale]
                # Stack I_in and I_us into a single (n_steps, 2, max_n) array
                # so they are passed as operands (dynamic data) not closure
                # captures. bm.for_loop slices axis-0 each step.
                I_in_pad = jnp.pad(I_in_jax, ((0, 0), (0, max_n - n_input)))
                I_us_pad = jnp.pad(I_us_jax, ((0, 0), (0, max_n - n_output)))
                xs = jnp.stack([I_in_pad, I_us_pad], axis=1)
                try:
                    all_spikes    = bm.for_loop(step_fn, xs, jit=True)
                    all_spikes_np = np.array(all_spikes)
                    output_spikes_accum = all_spikes_np.sum(axis=0)
                    model._trial_spike_counts += output_spikes_accum
                    spike_timeseries = (all_spikes_np.sum(axis=1).astype(np.float32)
                                        if need_spike_ts else None)
                    compiled = True
                except Exception as e:
                    if self.verbose and trial_id == 0:
                        print(f"  bm.for_loop failed ({e}), Python fallback")
                    output_spikes_accum, spike_timeseries = \
                        self._python_loop_fallback(
                            model, stimuli, I_in, I_us, I_bg, need_spike_ts)
                    model._trial_spike_counts += output_spikes_accum
            else:
                output_spikes_accum, spike_timeseries = \
                    self._python_loop_fallback(
                        model, stimuli, I_in, I_us, I_bg, need_spike_ts)

            wall_time = time.time() - t0

            trial_dur_s = task.trial_duration_ms / 1000.0
            response    = jnp.array(output_spikes_accum / trial_dur_s)

            scores = task.compute_score(
                response    = response,
                trial_id    = trial_id,
                state_trace = [],
                metadata    = {'spike_counts_timeseries': spike_timeseries},
            )

            modulator = 1.0
            if modulator_fn is not None:
                modulator = float(modulator_fn(trial_id, scores))

            # Only apply plasticity updates on learning trials
            if is_learning:
                if e_ff_final is not None:
                    tau_e = getattr(
                        getattr(model, 'synapse', None), 'tau_e', 1200.0)

                    # Select burst-time trace if snapshots are available
                    # (Izhikevich 2007: DA at burst time, short tau_e).
                    # e_snaps[k] = e_ff captured at group k's burst-off step,
                    # giving precise credit to the causal input group.
                    # Fall back to e_ff_final for tasks without group structure.
                    if (e_snaps is not None
                            and hasattr(task, '_group_times_ms')):
                        reprod_ms = float(scores.get(
                            'reproduced_ms', task.trial_duration_ms))
                        k_winning = int(np.argmin(
                            np.abs(task._group_times_ms - reprod_ms)))
                        e_ff_for_da = e_snaps[k_winning]
                    else:
                        e_ff_for_da = e_ff_final

                    model.post_trial(trial_id, [], modulator=modulator,
                                     e_ff=e_ff_for_da, e_rec=e_rec_final)

                else:
                    model.post_trial(trial_id, [], modulator=modulator)

            result = TrialResult(
                trial_id    = trial_id,
                correct     = bool(scores.get('accuracy', 0.0) >= 0.5)
                              if trial_id >= getattr(task, '_n_conditioning', 0)
                              else None,
                response    = response,
                scores      = scores,
                state_trace = [],
                metadata    = {
                    'trial_type': 'learning' if is_learning else 'test',
                    'wall_time':  wall_time,
                    'compiled':   compiled,
                },
            )
            trial_results.append(result)
            wall_times.append(wall_time)

            if hasattr(model, 'weight_stats'):
                weight_stats_history.append(model.weight_stats)

            if self.verbose and (trial_id + 1) % self.report_every == 0:
                acc  = scores.get('accuracy', scores.get('performance', 0.0))
                cr   = scores.get('cr_rate', None)
                loop = 'ELIG' if use_elig_scan else ('JIT' if compiled else 'PY')
                msg  = (f"  Trial {trial_id+1:4d}/{task.n_trials} | "
                        f"acc={acc:.3f} | time={wall_time:.1f}s | {loop}")
                if cr is not None:
                    msg += f" | cr={cr:.1f}Hz"
                if weight_stats_history:
                    msg += f" | w={weight_stats_history[-1].get('mean',0):.4f}"
                print(msg)

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
