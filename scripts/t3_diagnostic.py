"""
T3 Mechanism Diagnostic — D→E Sequence Learning

Run each model in a separate invocation to avoid MPS thread exhaustion:

    python scripts/t3_diagnostic.py --model cadex
    python scripts/t3_diagnostic.py --model lif
    python scripts/t3_diagnostic.py --plot

--model cadex|lif  Run one model, save diagnostics to figures/t3_diag_{model}.npz
--plot             Load both .npz files and write figures/t3_mechanism.png (no simulation)

Diagnostics recorded per learning trial:
  1. Mean D→E feedforward weight
  2. Output spikes in the 50ms bridge window after D fires (D→E spillover)
  3. Output spikes in the E-window (200–220ms)
"""

import os
os.environ["JAX_PLATFORMS"] = "mps"

import argparse
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


FIGURES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'figures')


def npz_path(model_name: str) -> str:
    return os.path.join(FIGURES_DIR, f't3_diag_{model_name}.npz')


# ------------------------------------------------------------------
# Model construction + configuration — exact mirror of run.py T3
# ------------------------------------------------------------------

def build_t3_model(model_name: str, seed: int = 0):
    from oi_bench.models.cadex.network import CAdExNetwork
    from oi_bench.models.baselines.lif_network import LIFNetwork

    if model_name == 'cadex':
        return CAdExNetwork(
            n_input=100, n_output=50,
            alpha=0.85, conn_prob=0.5, dt=0.1,
            plasticity=True, homeostasis=True,
            seed=seed,
        )
    elif model_name == 'lif':
        return LIFNetwork(
            n_input=100, n_output=50,
            conn_prob=0.5, dt=0.1,
            I_background=150.0,
            plasticity=True, homeostasis=True,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


def configure_model_for_t3(model, model_name: str):
    if hasattr(model, 'configure_homeostasis'):
        model.configure_homeostasis(r_target=25.0, trial_dur_ms=250.0, enabled=False)

    stdp_cfg = {
        'cadex': {'A2_plus': 0.001, 'A3_plus': 0.000, 'A2_minus': 0.001},
        'lif':   {'A2_plus': 0.001, 'A3_plus': 0.000, 'A2_minus': 0.001},
    }[model_name]
    if hasattr(model, 'configure_stdp'):
        model.configure_stdp(**stdp_cfg)

    if hasattr(model, 'rec_synapse') and hasattr(model.rec_synapse, 'configure_stdp'):
        model.rec_synapse.configure_stdp(A2_plus=0.003, A3_plus=0.000, A2_minus=0.003)

    # tau_syn=50ms (NMDA-like): D retains exp(-30/50)=0.55 at E onset.
    # g_max=3.0 (vs default 1.5): doubles recurrent drive.
    if hasattr(model, 'rec_synapse'):
        if hasattr(model.rec_synapse, 'tau_syn'):
            model.rec_synapse.tau_syn = 50.0
        if hasattr(model.rec_synapse, 'g_max'):
            model.rec_synapse.g_max = 3.0

    # Per-trial state reset models the ITI: prevents CAdEx adaptation (w)
    # from trial N suppressing D/E in trial N+1 (period-2 alternation).
    model._reset_state_per_trial = True

    if hasattr(model, 'configure_eligibility_trace'):
        model.configure_eligibility_trace(enabled=False)

    if hasattr(model, 'disable_timer'):
        model.disable_timer()


# ------------------------------------------------------------------
# Simulation
# ------------------------------------------------------------------

def run_and_save(model_name: str, seed: int = 0):
    from oi_bench.tasks.temporal.sequence_prediction import SequencePredictionTask
    from oi_bench.harness.runner import BenchmarkRunner

    print(f"\n{'='*50}")
    print(f"  T3 diagnostic: {model_name}")
    print(f"{'='*50}")

    model = build_t3_model(model_name, seed=seed)
    configure_model_for_t3(model, model_name)

    task = SequencePredictionTask(n_learning_trials=200, n_test_trials=100)
    task.record_diagnostics = True

    runner = BenchmarkRunner(verbose=True, report_every=50)
    runner.run(model=model, task=task, model_name=model_name)

    out = npz_path(model_name)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    np.savez(
        out,
        DE_weight            = np.array(task._diag_DE_weight),
        DE_weight_init       = np.array(task._diag_DE_weight_init
                                        if task._diag_DE_weight_init is not None
                                        else float('nan')),
        D_spikes_in_E_window = np.array(task._diag_D_spikes_in_E_window),
        E_spikes             = np.array(task._diag_E_spikes),
        E_spikes_test        = np.array(task._diag_E_spikes_test),
    )
    print(f"\nDiagnostics saved → {out}")

    diag = dict(np.load(out))
    print("\n--- Summary ---")
    print(f"  DE_weight init → final:  {float(diag['DE_weight_init']):.4f} → {diag['DE_weight'][-10:].mean():.4f}")
    print(f"  D→E bridge (last 10):    {diag['D_spikes_in_E_window'][-10:].mean():.1f} spikes")
    print(f"  E-window learning:       {diag['E_spikes'][-10:].mean():.1f} spikes")
    print(f"  E-window test:           {diag['E_spikes_test'].mean():.1f} spikes")


# ------------------------------------------------------------------
# Plotting (no simulation — loads saved .npz files)
# ------------------------------------------------------------------

def plot():
    colors = {'cadex': '#1565C0', 'lif': '#C62828'}
    labels = {'cadex': 'CAdEx (α=0.85)', 'lif': 'LIF'}

    results = {}
    for model_name in ('cadex', 'lif'):
        path = npz_path(model_name)
        if not os.path.exists(path):
            print(f"Missing: {path}  —  run --model {model_name} first")
            sys.exit(1)
        results[model_name] = dict(np.load(path))

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle(
        "T3 Sequence Prediction — D→E Learning Mechanism\n"
        "CAdEx vs LIF: learning trials (top) and test trials (bottom)",
        fontsize=11, fontweight='bold',
    )

    # --- Row 1: learning trial diagnostics ---
    learning_metrics = [
        ('DE_weight',            'D→E synaptic weight\n(learning trials 1–200)',       'Weight (a.u.)'),
        ('D_spikes_in_E_window', 'Output spikes in D→E bridge\n(50ms after D fires)',  'Spike count'),
        ('E_spikes',             'Output spikes in E-window\n(200–220ms, full sequence)', 'Spike count'),
    ]
    for ax, (key, title, ylabel) in zip(axes[0], learning_metrics):
        for model_name, diag in results.items():
            y = diag[key]
            x = np.arange(1, len(y) + 1)
            ax.plot(x, y, color=colors[model_name], linewidth=1.4,
                    label=labels[model_name])
        ax.set_title(title, fontsize=9)
        ax.set_xlabel('Learning trial')
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.set_xlim(1, max(len(d[key]) for d in results.values()))

    # --- Row 2, left: D→E weight — initial vs end-of-learning bar chart ---
    ax_bar = axes[1, 0]
    n_models = len(results)
    x_pos    = np.arange(n_models)
    width    = 0.35
    model_names = list(results.keys())
    init_vals  = [float(results[m]['DE_weight_init']) for m in model_names]
    final_vals = [float(results[m]['DE_weight'][-1])  for m in model_names]
    bars_init  = ax_bar.bar(x_pos - width / 2, init_vals,  width,
                            color=[colors[m] for m in model_names],
                            alpha=0.35, label='Initial (no STDP)')
    bars_final = ax_bar.bar(x_pos + width / 2, final_vals, width,
                            color=[colors[m] for m in model_names],
                            alpha=1.0,  label='End of learning (trial 200)')
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels([labels[m] for m in model_names], fontsize=9)
    ax_bar.set_ylabel('Mean D→E weight (a.u.)')
    ax_bar.set_title('D→E weight: initial vs end of learning\n(STDP effect on D-input → output connections)', fontsize=9)
    ax_bar.legend(fontsize=8)

    # --- Row 2, middle: E-window spikes during test trials ---
    ax_test = axes[1, 1]
    for model_name, diag in results.items():
        y = diag['E_spikes_test']
        x = np.arange(1, len(y) + 1)
        ax_test.plot(x, y, color=colors[model_name], linewidth=1.4,
                     label=labels[model_name])
    ax_test.set_title('E-window output spikes — TEST trials\n(A→B→C only; E fires only from learned D→E chain)', fontsize=9)
    ax_test.set_xlabel('Test trial (201–300)')
    ax_test.set_ylabel('Spike count')
    ax_test.legend(fontsize=8)
    ax_test.set_xlim(1, max(len(d['E_spikes_test']) for d in results.values()))

    # --- Row 2, right: mean test E-spikes bar (summary) ---
    ax_sum = axes[1, 2]
    mean_test = [float(results[m]['E_spikes_test'].mean()) for m in model_names]
    ax_sum.bar(x_pos, mean_test, 0.5,
               color=[colors[m] for m in model_names], alpha=0.85)
    ax_sum.set_xticks(x_pos)
    ax_sum.set_xticklabels([labels[m] for m in model_names], fontsize=9)
    ax_sum.set_ylabel('Mean E-window spikes')
    ax_sum.set_title('Mean E-window activity (test)\n(higher = better sequence prediction)', fontsize=9)
    for i, v in enumerate(mean_test):
        ax_sum.text(i, v + 0.05, f'{v:.1f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, 't3_mechanism.png')
    os.makedirs(FIGURES_DIR, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Saved → {out}")


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='T3 mechanism diagnostic')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--model', choices=['cadex', 'lif'],
                       help='Run one model and save diagnostics to figures/t3_diag_{model}.npz')
    group.add_argument('--plot', action='store_true',
                       help='Load saved .npz files and write figures/t3_mechanism.png')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    if args.plot:
        plot()
    else:
        run_and_save(args.model, seed=args.seed)
