"""Leaky integrate-and-fire dynamics over a connectome.

The equations and default constants follow the reference implementation of
Shiu et al., *Nature* (2024), which is the model the whole fly-connectome
simulation wave is built on:

    dv/dt = (v_rest - v + g) / tau_m      (frozen while refractory)
    dg/dt = -g / tau_syn                  (frozen while refractory)

A presynaptic spike adds ``sign * n_synapses * w_syn`` to the postsynaptic
``g`` after a fixed axonal delay.  On threshold crossing the neuron emits a
spike, ``v`` is reset, ``g`` is zeroed and the neuron is refractory.

Integration is exponential-Euler, which is exact for this linear system when
``g`` is held constant across a timestep, so a 0.1 ms step is accurate rather
than merely stable.

Delivery is event driven: the cost of a step scales with the number of neurons
that actually spiked times their out-degree, not with the size of the network.
That is what makes 165k neurons tractable without a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import scipy.sparse as sp

from ..connectome.schema import Connectome


@dataclass(frozen=True)
class LIFParams:
    """Model constants, in SI units (volts, seconds).

    Defaults are the values from the Shiu et al. reference implementation.
    ``w_syn`` is the one genuinely free parameter of that model: it sets the
    overall synaptic gain and is the first thing to sweep when a network is
    silent or saturated.
    """

    v_rest: float = -52e-3
    v_reset: float = -52e-3
    v_threshold: float = -45e-3
    tau_m: float = 20e-3
    tau_syn: float = 5e-3
    t_refractory: float = 2.2e-3
    t_delay: float = 1.8e-3
    w_syn: float = 0.275e-3
    dt: float = 1e-4
    #: Scaling of an external Poisson input spike relative to one real synapse.
    poisson_gain: float = 250.0

    def with_(self, **kw) -> LIFParams:
        """Return a copy with the given fields replaced."""
        return replace(self, **kw)


@dataclass
class RunResult:
    """Outcome of :meth:`LIFBrain.run`."""

    duration: float
    spike_counts: np.ndarray
    population_rates: dict[str, np.ndarray]
    bin_edges: np.ndarray

    def rate(self, neurons: np.ndarray) -> float:
        """Mean firing rate in Hz over the whole run for a set of neurons."""
        if len(neurons) == 0:
            return 0.0
        return float(self.spike_counts[neurons].sum() / (len(neurons) * self.duration))


class PoissonDrive:
    """External Poisson stimulation of chosen neurons, standing in for sensory
    input or for optogenetic activation.

    Rates are per-neuron in Hz and may be updated between steps, which is how
    the closed loop injects what the eye currently sees.
    """

    def __init__(self, n: int, rng: np.random.Generator):
        self._rates = np.zeros(n, dtype=np.float32)
        self._rng = rng

    def set(self, neurons: np.ndarray, rate_hz: float | np.ndarray) -> None:
        """Set the drive rate for ``neurons`` (Hz).  Set 0 to switch off."""
        self._rates[neurons] = rate_hz

    def clear(self) -> None:
        self._rates[:] = 0.0

    @property
    def rates(self) -> np.ndarray:
        return self._rates

    def sample(self, dt: float) -> np.ndarray:
        """Number of external input spikes arriving at each neuron this step."""
        active = self._rates > 0
        out = np.zeros(len(self._rates), dtype=np.float32)
        if active.any():
            out[active] = self._rng.poisson(self._rates[active] * dt).astype(np.float32)
        return out


class LIFBrain:
    """A connectome running as a spiking network."""

    def __init__(
        self,
        connectome: Connectome,
        params: LIFParams | None = None,
        *,
        seed: int = 0,
        backend: str = "numpy",
    ):
        self.c = connectome
        self.p = params or LIFParams()
        self.backend = backend
        self.rng = np.random.default_rng(seed)
        self.n = connectome.n

        # W is stored pre x post, so summing the rows of the neurons that spiked
        # gives the input each postsynaptic neuron receives.
        self.W = connectome.W.tocsr()

        self._delay_steps = max(1, int(round(self.p.t_delay / self.p.dt)))
        self._refrac_steps = int(round(self.p.t_refractory / self.p.dt))
        self._decay_v = float(np.exp(-self.p.dt / self.p.tau_m))
        self._decay_g = float(np.exp(-self.p.dt / self.p.tau_syn))

        self.drive = PoissonDrive(self.n, self.rng)
        self.reset()

        if backend == "torch":
            self._init_torch()

    # ------------------------------------------------------------------ state

    def reset(self) -> None:
        """Return every neuron to rest, empty the delay line and clear the drive.

        Clearing the external drive is deliberate: leaving stimulation in place
        across a reset silently contaminates the next trial, which is exactly
        how a "baseline" condition ends up looking identical to the stimulus
        condition.
        """
        p = self.p
        if hasattr(self, "drive"):
            self.drive.clear()
        self.v = np.full(self.n, p.v_rest, dtype=np.float32)
        self.g = np.zeros(self.n, dtype=np.float32)
        self.refrac = np.zeros(self.n, dtype=np.int32)
        self._delay_line = [
            np.empty(0, dtype=np.int64) for _ in range(self._delay_steps)
        ]
        self._delay_pos = 0
        self.t = 0.0
        self.step_count = 0
        self.last_spikes = np.empty(0, dtype=np.int64)

    def _init_torch(self) -> None:  # pragma: no cover - requires optional dep
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "backend='torch' needs PyTorch: pip install 'flyloop[gpu]'"
            ) from exc
        self._torch = torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(dev)
        coo = self.W.tocoo()
        idx = torch.from_numpy(np.vstack([coo.row, coo.col]).astype(np.int64))
        val = torch.from_numpy(coo.data.astype(np.float32))
        # Transposed so that (W^T @ spikes) gives per-postsynaptic input.
        self._Wt = torch.sparse_coo_tensor(
            idx.flip(0), val, (self.n, self.n), device=self._device
        ).coalesce()

    # ------------------------------------------------------------------- step

    def _deliver(self, spike_idx: np.ndarray) -> np.ndarray:
        """Summed signed synapse counts arriving at each neuron from ``spike_idx``."""
        if len(spike_idx) == 0:
            return np.zeros(self.n, dtype=np.float32)
        if self.backend == "torch":  # pragma: no cover - requires optional dep
            torch = self._torch
            s = torch.zeros(self.n, device=self._device)
            s[torch.from_numpy(spike_idx).to(self._device)] = 1.0
            return torch.sparse.mm(self._Wt, s.unsqueeze(1)).squeeze(1).cpu().numpy()
        # Sparse row-vector times sparse matrix: cost scales with the spikes.
        s = sp.csr_matrix(
            (np.ones(len(spike_idx), dtype=np.float32),
             (np.zeros(len(spike_idx), dtype=np.int64), spike_idx)),
            shape=(1, self.n),
        )
        return np.asarray((s @ self.W).todense(), dtype=np.float32).ravel()

    def step(self) -> np.ndarray:
        """Advance one timestep.  Returns the indices of neurons that spiked."""
        p = self.p

        # 1. Spikes emitted t_delay ago arrive now, plus any external drive.
        arriving = self._delay_line[self._delay_pos]
        inp = self._deliver(arriving) * p.w_syn
        ext = self.drive.sample(p.dt)
        if ext.any():
            inp += ext * p.w_syn * p.poisson_gain

        active = self.refrac <= 0

        # 2. Synaptic input adds to g whether or not the cell is refractory;
        #    only the *decay* of g and the integration of v are frozen.
        self.g += inp

        # 3. Exponential-Euler update of v with g held constant over the step.
        target = p.v_rest + self.g
        self.v = np.where(
            active, target + (self.v - target) * self._decay_v, p.v_reset
        ).astype(np.float32)
        self.g = np.where(active, self.g * self._decay_g, self.g).astype(np.float32)

        # 4. Threshold, reset, refractory.
        spiking = active & (self.v > p.v_threshold)
        spike_idx = np.flatnonzero(spiking).astype(np.int64)
        if len(spike_idx):
            self.v[spike_idx] = p.v_reset
            self.g[spike_idx] = 0.0
            self.refrac[spike_idx] = self._refrac_steps
        self.refrac[~spiking] -= 1
        np.maximum(self.refrac, 0, out=self.refrac)

        # 5. Push this step's spikes into the delay line.
        self._delay_line[self._delay_pos] = spike_idx
        self._delay_pos = (self._delay_pos + 1) % self._delay_steps

        self.t += p.dt
        self.step_count += 1
        self.last_spikes = spike_idx
        return spike_idx

    # -------------------------------------------------------------------- run

    def run(
        self,
        duration: float,
        *,
        record: dict[str, np.ndarray] | None = None,
        bin_size: float = 10e-3,
        progress: bool = False,
    ) -> RunResult:
        """Run for ``duration`` seconds, binning population rates as we go.

        ``record`` maps a label to an array of neuron indices; the returned
        :class:`RunResult` carries each label's firing rate in Hz per bin.
        """
        p = self.p
        n_steps = int(round(duration / p.dt))
        steps_per_bin = max(1, int(round(bin_size / p.dt)))
        n_bins = int(np.ceil(n_steps / steps_per_bin))
        record = record or {}

        counts = np.zeros(self.n, dtype=np.int64)
        binned = {k: np.zeros(n_bins, dtype=np.float64) for k in record}
        membership = {
            k: np.zeros(self.n, dtype=bool) for k in record
        }
        for k, idx in record.items():
            membership[k][idx] = True

        for i in range(n_steps):
            spk = self.step()
            if len(spk):
                counts[spk] += 1
                b = i // steps_per_bin
                for k in record:
                    binned[k][b] += membership[k][spk].sum()
            if progress and i % max(1, n_steps // 10) == 0:  # pragma: no cover
                print(f"  {100 * i / n_steps:3.0f}%  t={self.t * 1e3:7.1f} ms", flush=True)

        rates = {}
        for k, idx in record.items():
            denom = max(len(idx), 1) * (steps_per_bin * p.dt)
            rates[k] = binned[k] / denom

        return RunResult(
            duration=n_steps * p.dt,
            spike_counts=counts,
            population_rates=rates,
            bin_edges=np.arange(n_bins + 1) * steps_per_bin * p.dt,
        )


# --------------------------------------------------------------------- theory

def peak_psp_factor(p: LIFParams) -> float:
    """Peak membrane deflection produced by a unit step in ``g``.

    For the two-time-constant system above, a presynaptic event that adds ``G``
    to ``g`` does **not** move the membrane by ``G``.  The membrane chases a
    target that is already decaying, so the peak deflection is

        u_peak = G * k,    k = (tau_syn / (tau_syn - tau_m)) * (x^a - x^b)

    which for the default 5 ms / 20 ms constants is about 0.157.

    This is worth internalising before tuning anything: with the reference
    ``w_syn`` of 0.275 mV, a single presynaptic spike needs roughly 160
    synapses to take a resting neuron to threshold on its own.  Reasoning about
    connectome edges one synapse at a time will mislead you.
    """
    tm, ts = p.tau_m, p.tau_syn
    if abs(tm - ts) < 1e-12:  # degenerate case: alpha-function limit
        return float(np.exp(-1.0))
    t_peak = (tm * ts) / (tm - ts) * np.log(tm / ts)
    return float(ts / (ts - tm) * (np.exp(-t_peak / ts) - np.exp(-t_peak / tm)))


def synapses_to_threshold(p: LIFParams) -> float:
    """How many simultaneous synapses one spike needs to fire a resting neuron."""
    gap = p.v_threshold - p.v_rest
    return float(gap / (p.w_syn * peak_psp_factor(p)))
