# OI-Bench

**A substrate-agnostic benchmark suite for evaluating learning in organoid intelligence simulations.**

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-0.9.2-orange.svg)](https://github.com/google/jax)
[![BrainPy](https://img.shields.io/badge/BrainPy-2.7.8-green.svg)](https://brainpy.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

OI-Bench is the first standardized, reproducible benchmark suite for quantifying **learning** in organoid intelligence (OI) simulations. Unlike prior work that evaluates simulation fidelity against wet-lab recordings, OI-Bench measures plasticity, adaptation, and task performance — providing a community yardstick analogous to MLPerf for ML or the Allen Brain Observatory for in-vivo recording.

**Key features:**
- **Substrate-agnostic adapter interface** — any OI model (SNN, reservoir, RNN) implements `OIModel` and plugs in with zero modification to tasks or metrics
- **Reference model** — CAdEx neurons with fractional membrane capacitance (Lundstrom 2008), triplet STDP (Pfister & Gerstner 2006), and homeostatic plasticity (Turrigiano 1998)
- **6 benchmark tasks** across 3 learning axes: associative, temporal, working memory
- **Principled metrics** — Learning Index, weight entropy, transfer entropy, homeostatic efficacy

## Installation

```bash
git clone https://github.com/JustJay7/oi-bench.git
cd oi-bench
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install applejax          # Apple M1/M2 Metal backend
pip install -e .
```

**Requirements:** Python 3.13, Apple Silicon Mac (M1/M2/M3) or CUDA GPU

## Quick Start

```python
from oi_bench.models.cadex.network import CAdExNetwork
from oi_bench.tasks.associative.classical_conditioning import ClassicalConditioningTask
from oi_bench.harness.runner import BenchmarkRunner

model  = CAdExNetwork(n_input=100, n_output=50)
task   = ClassicalConditioningTask()
runner = BenchmarkRunner()
result = runner.run(model, task, model_name="cadex")

print(f"Learning Index: {result.learning_index:.4f}")
```

## Implementing Your Own Model

```python
from oi_bench.core.adapter import OIModel
from oi_bench.core.types import Stimulus, ModelState

class MyModel(OIModel):
    @property
    def n_input(self): return 100
    @property
    def n_output(self): return 50
    @property
    def dt(self): return 0.1

    def reset(self): ...
    def step(self, stimulus: Stimulus) -> ModelState: ...
```

Any model implementing `OIModel` can be evaluated against all benchmark tasks.

## Benchmark Tasks

| ID | Task | Axis | Protocol |
|---|---|---|---|
| T1 | Classical Conditioning | Associative | CS+US pairing → CR |
| T2 | Pattern Completion | Associative | Degraded input → full pattern |
| T3 | Sequence Prediction | Temporal | Truncated sequence → completion |
| T4 | Interval Timing | Temporal | Reproduce target ISI |
| T5 | Delay Match-to-Sample | Working Memory | Sample → delay → match |
| T6 | N-Back | Working Memory | Online memory updating |

## Models

| Model | Description | Learning |
|---|---|---|
| `CAdExNetwork` | Reference: CAdEx + fractional membrane + triplet STDP | STDP + Homeostasis |
| `LIFNetwork` | Ablation: standard LIF + triplet STDP | STDP + Homeostasis |
| `LiquidStateMachine` | Ablation: fixed spiking reservoir | Readout only |

## Architecture

```
oi_bench/
├── core/          # OIModel ABC, BenchmarkTask ABC, shared types
├── models/
│   ├── cadex/     # Reference CAdEx implementation
│   └── baselines/ # LIF and LSM baselines
├── tasks/         # Benchmark task implementations
├── harness/       # BenchmarkRunner
└── metrics/       # Learning curve, plasticity, information, stability
```

## Citation

```bibtex
@article{oibench2026,
  title   = {OI-Bench: A Substrate-Agnostic Benchmark Suite for
             Evaluating Learning in Organoid Intelligence Simulations},
  author  = {},
  journal = {},
  year    = {2026},
  url     = {https://github.com/JustJay7/oi-bench}
}
```

*Citation will be updated upon paper submission.*

## License

MIT License. See [LICENSE](LICENSE).
