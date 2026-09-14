# flyloop

An embodied *Drosophila* sensorimotor loop: a connectome-derived spiking brain
that sees through a compound eye, decides through real descending neurons, and
drives a body.

```
body.observe() -> compound eye -> temporal filter -> Poisson drive
  -> LIF brain -> descending readout -> locomotor command -> body.step()
```

It runs end to end on a laptop, with no GPU and no connectome download, against
a synthetic fixture -- so the plumbing can be tested before the science starts.

## Why another one of these

The June 2026 release of MaleCNS v1.0 produced a wave of projects wiring the fly
connectome into games, browsers and robots. Most of them show the system doing
something impressive and stop there.

This one is built around the opposite question: **how would we know if it isn't
working?** So it ships with controls, a non-connectome baseline, and a set of
numbers that make the model's limits explicit.

## Quick start

```bash
pip install -e .
flyloop selftest       # end-to-end check on the synthetic fixture
flyloop theory         # model constants and what they imply
flyloop looming        # escape experiment, with controls
flyloop baseline       # the ten-line controller you have to beat
```

`flyloop selftest` should end in `PASS`. It runs the whole loop and the
acceptance experiment in well under a minute.

## What the numbers say before you start

`flyloop theory` prints this, and it is worth internalising:

| quantity | value |
|---|---|
| membrane time constant | 20 ms |
| synaptic time constant | 5 ms |
| firing threshold above rest | 7 mV |
| per-synapse weight `w_syn` | 0.275 mV |
| **peak PSP per unit of `g`** | **0.157** |
| **synapses needed for one spike to fire a resting neuron** | **~162** |

A connectome edge of a few synapses cannot fire anything. Behaviour in this
model comes from populations firing together, which is why reasoning about
individual connectome edges is misleading, and why a readout fitted on top of
160k neurons will always find *something*.

## What this is not

The connectome is a wiring diagram, not a working brain. This model, like the
Shiu et al. model it follows, has:

- no real synaptic strengths -- only synapse counts times one free gain
- no gap junctions
- no neuromodulation (dopamine, octopamine, serotonin), so no hunger, arousal
  or internal state
- no plasticity, no learning
- zero basal firing: an unstimulated network is completely silent

In *Drosophila*, glutamate is predominantly **inhibitory** (GluCl-alpha).
Several community reimplementations treat it as excitatory; the network then
saturates and the authors compensate by tuning the readout until something
moves. `flyloop.connectome.signs` encodes the correct signs and there is a test
pinning it.

If you see a fly-connectome project reporting a "consciousness index" or
"integrated information" for its simulation, that is not a result.

## Layout

| module | what it owns |
|---|---|
| `connectome/` | wiring: `signs`, `schema`, `synthetic`, `flywire`, `malecns` |
| `brain/` | LIF dynamics, event-driven delivery, PSP theory helpers |
| `vision/` | hexagonal ommatidia, adaptation, ommatidium-to-neuron mapping |
| `motor/` | descending-neuron readout; tripod gait (explicitly not connectome-derived) |
| `body/` | kinematic stub, NeuroMechFly adapter, the `Body` protocol |
| `experiments/` | looming acceptance test with controls, reactive baseline |

The `Body` protocol is narrow on purpose: brain and body exchange a panorama and
a locomotor command, nothing else. Swapping the stub for MuJoCo, or later for a
hexapod over a socket, touches one file.

## Stages

1. **Brain on the desk** -- load a connectome, run LIF, check known circuits. *Done.*
2. **Eyes** -- camera to ommatidia to input currents. *Done.*
3. **Descending readout** -- DNa01, DNa02, MDN, DNp09, GF as the animal's API. *Done.*
4. **A body** -- kinematic stub done; NeuroMechFly adapter written, needs MuJoCo.
5. **A physical hexapod** -- brain on a workstation, body on a Pi, over a socket.

See `docs/PLAN.md` for the roadmap and `docs/DATA.md` for getting real data.

## Data and licences

No connectome data is bundled. `docs/DATA.md` covers both datasets.

- **MaleCNS v1.0** -- brain *and* nerve cord, 166,700 neurons, **CC-BY**.
  The one to use for embodied work: it has the leg motor neurons.
- **FlyWire v783** -- female brain, 139,255 neurons, **CC BY-NC**. Brain only:
  it stops at the neck, so it can issue descending commands but cannot reach
  the legs.

This code is MIT. The data is not; check the dataset licence for your use.

## Credits

The model follows Shiu et al., *Nature* (2024). Connectomes from FlyWire
(Dorkenwald et al., *Nature* 2024) and the Janelia FlyEM MaleCNS project. The
biomechanical body is NeuroMechFly v2 / FlyGym (NeLy-EPFL).
