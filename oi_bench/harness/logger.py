"""
BenchmarkLogger — Structured logging and HDF5 output for OI-Bench runs.

Saves RunResult objects to HDF5 with a JSON manifest for reproducibility.
Every run gets a unique directory under results/ with:
  - run.h5       : all trial data, weights, metrics
  - manifest.json: run metadata, git hash, config snapshot
  - summary.csv  : one row per (model, task) with key metrics

HDF5 structure:
  /{model}__{task}/
    metadata/
      model_name, task_name, learning_index, n_trials  (attrs)
    trials/
      trial_{i}/
        scores/   (attrs — scalar floats)
        response  (dataset — n_output,)
        correct, trial_type, wall_time  (attrs)
    weight_stats/
      trial_{i}/  (attrs — mean, std, frac_silent, etc.)
    metrics/
      learning_index, convergence_trial, asymptotic_performance,
      sample_efficiency  (attrs)

Design: uses require_group / delete+recreate for datasets so reruns
into the same run_id are safe (e.g. after a crash mid-run).
"""

from __future__ import annotations
import json
import csv
import subprocess
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np

from oi_bench.harness.runner import RunResult
from oi_bench.metrics.learning_curve import learning_curve_stats
from oi_bench.metrics.plasticity import plasticity_stats


class BenchmarkLogger:
    """
    Logs benchmark run results to HDF5 + JSON manifest + CSV summary.

    Parameters
    ----------
    results_dir : str
        Root directory for results. Default 'results/'.
    run_id : str or None
        Unique run identifier. If None, generated from timestamp.
    """

    def __init__(
        self,
        results_dir: str = "results",
        run_id: str      = None,
    ):
        self.results_dir = Path(results_dir)
        self.run_id      = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir     = self.results_dir / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.h5_path       = self.run_dir / "run.h5"
        self.manifest_path = self.run_dir / "manifest.json"
        self.summary_path  = self.run_dir / "summary.csv"

        # Load existing manifest if resuming a crashed run
        if self.manifest_path.exists():
            with open(self.manifest_path) as f:
                self._manifest = json.load(f)
        else:
            self._manifest = {
                'run_id':    self.run_id,
                'timestamp': datetime.now().isoformat(),
                'git_hash':  self._get_git_hash(),
                'runs':      [],
            }

        # Load existing summary rows if resuming
        self._summary_rows = []
        if self.summary_path.exists():
            with open(self.summary_path, newline='') as f:
                reader = csv.DictReader(f)
                self._summary_rows = list(reader)

        print(f"  Logger: results → {self.run_dir}")

    def _get_git_hash(self) -> str:
        try:
            result = subprocess.run(
                ['git', 'rev-parse', '--short', 'HEAD'],
                capture_output=True, text=True, timeout=5
            )
            return result.stdout.strip()
        except Exception:
            return 'unknown'

    def _write_dataset(self, group: h5py.Group, name: str, data: np.ndarray) -> None:
        """
        Write a dataset, replacing it if it already exists.
        Needed when resuming into an existing HDF5 file after a crash.
        """
        if name in group:
            del group[name]
        group.create_dataset(name, data=data, compression='gzip')

    def log(self, result: RunResult) -> None:
        """
        Save one RunResult to HDF5 and update manifest + summary.

        Safe to call multiple times with the same run_id — existing
        datasets are overwritten rather than duplicated.
        """
        key = f"{result.model_name}__{result.task_name}"

        with h5py.File(self.h5_path, 'a') as f:
            # Remove existing group for this key if present (clean overwrite)
            if key in f:
                del f[key]
            grp = f.create_group(key)

            # Metadata
            meta = grp.create_group('metadata')
            meta.attrs['model_name']     = result.model_name
            meta.attrs['task_name']      = result.task_name
            meta.attrs['learning_index'] = result.learning_index
            meta.attrs['n_trials']       = len(result.trial_results)

            # Trial data
            trials_grp = grp.create_group('trials')
            for r in result.trial_results:
                t_grp = trials_grp.create_group(f"trial_{r.trial_id:04d}")

                # Scores as attrs
                scores_grp = t_grp.create_group('scores')
                for k, v in r.scores.items():
                    scores_grp.attrs[k] = float(v)

                # Response as dataset
                t_grp.create_dataset(
                    'response',
                    data=np.array(r.response, dtype=np.float32),
                    compression='gzip',
                )

                # Trial metadata as attrs
                t_grp.attrs['correct']    = int(r.correct) if r.correct is not None else -1
                t_grp.attrs['trial_type'] = r.metadata.get('trial_type', 'unknown')
                t_grp.attrs['wall_time']  = float(r.metadata.get('wall_time', 0.0))

            # Weight stats history
            if result.weight_stats_per_trial:
                ws_grp = grp.create_group('weight_stats')
                for i, ws in enumerate(result.weight_stats_per_trial):
                    w_grp = ws_grp.create_group(f"trial_{i:04d}")
                    for k, v in ws.items():
                        if v is not None:
                            w_grp.attrs[k] = float(v)

            # Computed metrics
            lc = learning_curve_stats(result.trial_results)
            metrics_grp = grp.create_group('metrics')
            metrics_grp.attrs['learning_index']         = result.learning_index
            metrics_grp.attrs['convergence_trial']      = lc['convergence_trial']
            metrics_grp.attrs['asymptotic_performance'] = lc['asymptotic_performance']
            metrics_grp.attrs['sample_efficiency']      = lc['sample_efficiency']

        # Update manifest — replace existing entry for this key if present
        self._manifest['runs'] = [
            r for r in self._manifest['runs']
            if r.get('key') != key
        ]
        self._manifest['runs'].append({
            'key':                 key,
            'model_name':          result.model_name,
            'task_name':           result.task_name,
            'learning_index':      result.learning_index,
            'n_trials':            len(result.trial_results),
            'total_wall_time_min': sum(result.wall_time_per_trial) / 60.0,
        })
        self._save_manifest()

        # Update summary — replace existing row for this key if present
        lc = learning_curve_stats(result.trial_results)
        self._summary_rows = [
            r for r in self._summary_rows
            if not (r.get('model') == result.model_name
                    and r.get('task') == result.task_name)
        ]
        self._summary_rows.append({
            'model':                  result.model_name,
            'task':                   result.task_name,
            'learning_index':         f"{result.learning_index:.4f}",
            'convergence_trial':      lc['convergence_trial'],
            'asymptotic_performance': f"{lc['asymptotic_performance']:.4f}",
            'sample_efficiency':      f"{lc['sample_efficiency']:.4f}",
            'wall_time_min':          f"{sum(result.wall_time_per_trial)/60:.1f}",
        })
        self._save_summary()

        print(f"  Logged: {key} | LI={result.learning_index:.4f}")

    def _save_manifest(self) -> None:
        with open(self.manifest_path, 'w') as f:
            json.dump(self._manifest, f, indent=2)

    def _save_summary(self) -> None:
        if not self._summary_rows:
            return
        fieldnames = list(self._summary_rows[0].keys())
        with open(self.summary_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self._summary_rows)

    def print_summary(self) -> None:
        if not self._summary_rows:
            print("No results logged yet.")
            return
        print(f"\n{'='*75}")
        print(f"  OI-Bench Results — Run {self.run_id}")
        print(f"{'='*75}")
        print(f"  {'Model':<20} {'Task':<30} {'LI':>6} {'Conv':>6} {'Asym':>6}")
        print(f"  {'-'*20} {'-'*30} {'-'*6} {'-'*6} {'-'*6}")
        for row in self._summary_rows:
            print(f"  {row['model']:<20} {row['task']:<30} "
                  f"{row['learning_index']:>6} "
                  f"{str(row['convergence_trial']):>6} "
                  f"{row['asymptotic_performance']:>6}")
        print(f"{'='*75}\n")
