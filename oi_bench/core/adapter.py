"""
OIModel — Abstract base class for all OI simulation models.

This is the central contract of OI-Bench. Any model that implements
OIModel can be evaluated by any task in the suite with zero modification
to either side. The adapter pattern makes the benchmark substrate-agnostic:
SNN, reservoir, RNN, or future wet-lab bridge implementations all plug in
by implementing this interface.

Design decisions:
  - step() must be JAX-jittable: no Python-side branching on array values
  - pre_trial() and post_trial() are optional hooks with no-op defaults
  - n_input and n_output are properties, not constructor args, so models
    can compute them dynamically from their architecture
  - ModelState.weights is Optional — reservoirs and RNNs may not expose
    internal weights, and the benchmark handles this gracefully
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from .types import Stimulus, ModelState


class OIModel(ABC):
    """
    Abstract base class for all OI simulation models in OI-Bench.

    Implementors must provide:
      n_input    : int property — number of input neurons/channels
      n_output   : int property — number of output neurons/channels
      dt         : float property — simulation timestep in ms
      reset()    : reinitialise to baseline state (called between episodes)
      step()     : advance one timestep given a Stimulus, return ModelState

    Optionally override:
      pre_trial()  : called once before each trial begins
      post_trial() : called once after each trial ends

    Usage example:
        class MyCAdExNetwork(OIModel):
            @property
            def n_input(self): return 100
            @property
            def n_output(self): return 50
            @property
            def dt(self): return 0.1

            def reset(self): ...
            def step(self, stimulus): ...

        model = MyCAdExNetwork()
        runner = BenchmarkRunner(config)
        runner.register_model("cadex", model)
        results = runner.run()
    """

    @property
    @abstractmethod
    def n_input(self) -> int:
        """Number of input neurons/channels."""
        ...

    @property
    @abstractmethod
    def n_output(self) -> int:
        """Number of output neurons/channels."""
        ...

    @property
    @abstractmethod
    def dt(self) -> float:
        """Simulation timestep in ms."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """
        Full state reset to baseline.

        Called at the start of each episode (before pre_trial of trial 0).
        Must reset membrane potentials, adaptation variables, calcium,
        synaptic conductances, and STDP traces to initial conditions.
        Synaptic weights are NOT reset — they persist across trials within
        an episode to allow learning to accumulate.
        """
        ...

    @abstractmethod
    def step(self, stimulus: Stimulus) -> ModelState:
        """
        Advance simulation by one timestep dt.

        Parameters
        ----------
        stimulus : Stimulus
            Input delivered this timestep. Use stimulus.current for
            current injection and stimulus.spike_train for spike input.

        Returns
        -------
        ModelState
            State snapshot after this timestep. spikes and membrane
            are required. weights and extras are optional.

        Implementation requirements:
          - Must be JAX-jittable: no Python-side branching on array values
          - Use jnp.where / lax.cond instead of if/else on arrays
          - All state must be stored in BrainPy Variable containers
        """
        ...

    def pre_trial(self, trial_id: int) -> None:
        """
        Optional hook called once before each trial.

        Use for trial-level resets that differ from full episode resets,
        e.g. resetting STDP traces while preserving weights, or setting
        up trial-specific network state.

        Default: no-op.
        """
        pass

    def post_trial(
        self,
        trial_id: int,
        state_trace: list[ModelState],
        modulator: float = 1.0,
    ) -> None:
        """
        Optional hook called once after each trial ends.

        Use for:
          - Applying homeostatic plasticity (ITI mechanism)
          - Weight consolidation
          - Three-factor neuromodulation (pass modulator signal here)

        Parameters
        ----------
        trial_id : int
            Zero-indexed trial number.
        state_trace : list[ModelState]
            Full per-step state trace from this trial.
            Empty if record=False.
        modulator : float
            Three-factor neuromodulatory signal. Default 1.0 (neutral).
            Task can pass reward, surprise, or other signals here.

        Default: no-op.
        """
        pass
