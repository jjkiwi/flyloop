"""A rate model with per-cell-type gain, for the parts a spiking model cannot reach.

Run 4 established, on real MaleCNS wiring, why a looming stimulus never reaches
LC4 in the leaky integrate-and-fire model: the columnar pathway is far below
single-spike threshold at every stage (129 synapses per pair into the medulla,
9-13 into the motion detectors, 5-7 into LC4), and the model has no mechanism to
bridge that. It has one global synaptic weight and no spontaneous activity, so a
pathway built on massive convergence of tiny inputs carries nothing.

This wraps ``connectome_interpreter.MultilayeredNetwork``, which supplies
exactly the three things that are missing:

* **biases per cell type** -- baseline activity, so inhibition has something to
  act on. Without it the fly's glutamatergic ON pathway is mute (Run 4).
* **slopes per cell type** -- excitability, in place of one global weight.
* **divisive normalisation** -- gain control.

and, critically, it is meant to run on **input-proportion** weights rather than
synapse counts. A connection of 7 synapses is negligible in absolute terms, but
as a fraction of a target that pools thousands of them it is not. That is how
LC4 is reached, and it is why ``load_dataset(..., matrix="inprop")`` exists.

**This model is not fitted.** ``MultilayeredNetwork`` can be trained against
measured firing rates, but no visual firing-rate targets ship with the package,
so biases and slopes here are set by hand. Everything it produces is therefore
a statement about the wiring under assumed excitability, not a prediction. Say
so in any result that uses it.

**The time axis is the synaptic-hop axis.** This is the trap in the API and it
is worth stating twice. Frame ``t`` of the input is injected at hop ``t``: the
first frame propagates through every hop, the last through one. So a pattern
that is large early gets far more amplification than the same pattern arriving
late, purely as a function of when it appears.

That makes this the right tool for "does activity from this pattern reach LC4
through N synapses" and the **wrong** tool for "is the response selective for an
approaching object over a receding one". Direction of motion needs temporal
dynamics this model does not have; comparing a looming sequence against a
receding one here measures the hop count, not the stimulus. Use
:func:`steady_state` to hold a pattern constant across hops whenever the
question is about the pathway rather than about time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..connectome.schema import Connectome

_INSTALL_HINT = (
    "The rate model needs connectome-interpreter:\n"
    "    pip install 'flyloop[data]'"
)


@dataclass
class RateResult:
    """Activation of every neuron across the time steps of a stimulus."""

    activations: np.ndarray  # (n_neurons, time_steps)
    populations: pd.DataFrame  # time step x population mean activation
    sensory: np.ndarray

    def peak(self) -> pd.Series:
        """Highest activation each population reaches."""
        return self.populations.max().sort_values(ascending=False)

    def report(self, threshold: float = 1e-3, top: int = 12) -> str:
        peak = self.peak()
        live = peak[peak > threshold]
        lines = [
            f"rate model: {self.activations.shape[0]:,} neurons, "
            f"{self.activations.shape[1]} time steps"
        ]
        if len(live):
            lines += [f"    {k:12s} peak {v:.4f}" for k, v in live.head(top).items()]
        else:
            lines.append("    nothing rose above threshold")
        silent = sorted(peak[peak <= threshold].index)
        if silent:
            lines.append(f"  silent: {', '.join(silent)}")
        return "\n".join(lines)


class RateBrain:
    """Per-cell-type rate dynamics over a connectome."""

    def __init__(
        self,
        connectome: Connectome,
        sensory: np.ndarray,
        *,
        num_layers: int = 6,
        default_bias: float = 0.0,
        tanh_steepness: float = 5.0,
        bias_by_type: dict[str, float] | None = None,
        slope_by_type: dict[str, float] | None = None,
        threshold: float = 0.01,
    ):
        try:
            from connectome_interpreter import MultilayeredNetwork
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(_INSTALL_HINT) from exc
        import torch

        if connectome.meta.get("matrix_kind") == "syncount":
            raise ValueError(
                "this model expects input-proportion weights, not synapse counts. "
                "Load with matrix='inprop' -- see the module docstring for why."
            )

        self.c = connectome
        self.sensory = np.asarray(sensory, dtype=np.int64)
        self.num_layers = num_layers
        self._torch = torch

        # MultilayeredNetwork puts *input* neurons in the columns, so it wants
        # (post, pre) -- the transpose of Connectome.W, which is (pre, post).
        weights = connectome.W.T.tocoo()
        idx = torch.from_numpy(np.vstack([weights.row, weights.col]).astype(np.int64))
        val = torch.from_numpy(weights.data.astype(np.float32))
        sparse = torch.sparse_coo_tensor(
            idx, val, (connectome.n, connectome.n)
        ).coalesce()

        types = connectome.neurons["type"].astype(str).to_numpy()
        idx_to_group = dict(enumerate(types))

        self.net = MultilayeredNetwork(
            all_weights=sparse,
            sensory_indices=self.sensory.tolist(),
            num_layers=num_layers,
            threshold=threshold,
            tanh_steepness=tanh_steepness,
            idx_to_group=idx_to_group,
            default_bias=default_bias,
            bias_dict=bias_by_type or None,
            slope_dict=slope_by_type or None,
        )

    @property
    def values(self):
        """The network's coalesced weight values, editable in place."""
        return self.net.all_weights.values()

    def plasticity(self, **kwargs):
        """Attach dopamine-gated KC->MBON plasticity to this network."""
        from .dopamine import mushroom_body_plasticity

        return mushroom_body_plasticity(
            self.c, self.values, self.net.all_weights.indices(), **kwargs
        )

    def run(
        self, inputs: np.ndarray, *, record: dict[str, np.ndarray] | None = None
    ) -> RateResult:
        """Run a ``(n_sensory, time_steps)`` stimulus through the network."""
        torch = self._torch
        x = torch.as_tensor(np.asarray(inputs, dtype=np.float32))
        if x.ndim != 2 or x.shape[0] != len(self.sensory):
            raise ValueError(
                f"inputs must be (n_sensory={len(self.sensory)}, time_steps), "
                f"got {tuple(x.shape)}"
            )
        with torch.no_grad():
            acts = self.net(x)
        acts = np.asarray(acts.detach().cpu().numpy(), dtype=np.float32)

        pops = {}
        for label, rows in (record or {}).items():
            if len(rows):
                pops[label] = acts[rows].mean(axis=0)
        return RateResult(
            activations=acts,
            populations=pd.DataFrame(pops),
            sensory=self.sensory,
        )


def steady_state(pattern: np.ndarray, hops: int) -> np.ndarray:
    """Hold one input pattern constant across every synaptic hop.

    The model injects input frame ``t`` at hop ``t``, so a varying sequence
    confounds the stimulus with how much propagation it has had. Repeating a
    single pattern removes that confound and asks the only question this model
    can answer cleanly: where does activity from *this* pattern end up.
    """
    pattern = np.asarray(pattern, dtype=np.float32).reshape(-1, 1)
    return np.repeat(pattern, hops, axis=1)


def population_index(
    c: Connectome, types: tuple[str, ...], *, side: str | None = None
) -> dict[str, np.ndarray]:
    """Neuron rows per cell type, optionally restricted to one side."""
    out: dict[str, np.ndarray] = {}
    kinds = c.neurons["type"].astype(str).to_numpy()
    sides = (
        c.neurons["side"].astype(str).to_numpy()
        if "side" in c.neurons.columns
        else None
    )
    for t in types:
        m = kinds == t
        if side and sides is not None:
            m &= sides == side
        rows = np.flatnonzero(m)
        if len(rows):
            out[f"{t}_{side}" if side else t] = rows
    return out
