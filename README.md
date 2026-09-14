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
working?** So it ships with control graphs, a non-connectome baseline, and a set
of numbers that make the model's limits explicit.

## Standing on other people's work

The parts of this problem that other people have solved properly are imported,
not rewritten:

| what | from | why |
|---|---|---|
| retinotopy | `connectome-interpreter` (MIT) | bundles the Nern 2024 and Matsliah 2024 columnar tables -- 892 hexagonal columns of MaleCNS, each naming its own L1, L2, Mi1 ... by body ID |
| biomechanical body | `flygym` 2.x | NeuroMechFly v2, plus the measured ommatidial lattice (721 per eye) and the raw-image-to-hex conversion |
| optic lobe | `flyvis` (MIT) | connectome-constrained visual model, *Nature* 2024, with pretrained weights |
| methodology | [`ommatid`](https://github.com/FutureJJ/ommatid) (MIT) | pre-registered hypotheses and shuffled-wiring control graphs, run on real data and real hardware |

**Python version window.** `flygym >= 2.1` needs Python >= 3.12 and `flyvis`
caps at < 3.13, so a full install with both extras must run on **Python 3.12**.
The core package and everything in CI works from 3.10 up.

## Control graphs

A closed loop built on a real connectome will produce *some* behaviour. So will
one built on a graph with the same degree sequence and the wiring shuffled.

```bash
flyloop controls
```

runs the looming experiment on the real graph and on three controls -- rewired
(degrees preserved, targets shuffled), relabelled (topology preserved, cell
identities permuted), and sign-scrambled (transmitters permuted). On the
synthetic fixture:

```
  graph                peak Hz  latency s  escape  carried by
  original               181.3       0.68   100%  DNp04
  rewired                  0.0          -     0%  -
  relabelled              19.3       2.06   100%  DNp02
  signs_scrambled         57.4       2.16   100%  DNp04
```

Note what the controls exposed: **binary escape rate cannot tell these graphs
apart** -- three of the four escape in 100% of trials. Only a continuous
statistic does. That is why the primary measure here is peak escape-population
firing rate rather than a hit count.

## It runs on the real connectome

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 \
    https://github.com/YijieYin/connectome_data_prep ~/connectome_data_prep
flyloop --data-root ~/connectome_data_prep activation --drive LC4 --side L --rate 50
```

MaleCNS v1.0: 161,429 neurons, 5,988,538 connections, 86,516,520 synapses,
loaded in about six seconds. Driving the left LC4 population, the looming
detectors, and reading the descending neurons against three control graphs:

| population | original | rewired | relabelled | signs scrambled |
|---|---:|---:|---:|---:|
| DNp04_L | **325.0** | 0.0 | 0.0 | 350.0 |
| DNp02_L | **225.0** | 0.0 | 0.0 | 305.0 |
| DNp01_L | **195.0** | 0.0 | 0.0 | 355.0 |
| DNp09_L | **0.0** | 0.0 | 0.0 | 285.0 |
| total spikes | 13,324 | 1,594 | 870,981 | 890,053 |

Three things to read off that table, in `docs/RESULTS.md` with the caveats:

- **The response needs the measured wiring.** A degree-preserving rewire kills
  it completely -- every readout at zero.
- **DNp04 and DNp02 carry it; DNp09 is silent.** That reproduces what ommatid
  measured on a robot across 210 trials, from an independent code path.
- **Two of the three controls are unusable at this scale.** Relabelling and
  sign-scrambling push the network into saturation -- ~880,000 spikes against
  13,324 -- so their rates mean nothing. Only the rewired control is
  interpretable. A control that preserves E/I balance per neuron is needed.

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

**And the textbook escape pathway may not be the one that fires.** The obvious
design reads escape off DNp01 (the giant fibre) and DNp09. The ommatid project
ran that against real MaleCNS wiring on a hexapod in September 2026 and measured
DNp01, DNp09 and MDN at 0 Hz in all 210 trials, with the looming signal carried
by DNp04 and DNp02 instead. So the escape channel here is a set of populations
and the readout reports *which* one fired, as a result rather than an
assumption. Their pre-registered optomotor and phototaxis hypotheses were not
met at all.

## Layout

| module | what it owns |
|---|---|
| `connectome/` | wiring: `signs`, `schema`, `synthetic`, `flywire`, `malecns` |
| `brain/` | LIF dynamics, event-driven delivery, PSP theory helpers |
| `vision/` | hexagonal ommatidia, adaptation, ommatidium-to-neuron mapping |
| `motor/` | descending-neuron readout; tripod gait (explicitly not connectome-derived) |
| `connectome/controls.py` | rewired, relabelled and sign-scrambled control graphs |
| `vision/columns.py` | real retinotopy from published columnar tables |
| `body/` | kinematic stub, NeuroMechFly adapter, the `Body` protocol |
| `experiments/` | looming acceptance test with controls, reactive baseline |

The `Body` protocol is narrow on purpose: brain and body exchange a panorama and
a locomotor command, nothing else. Swapping the stub for MuJoCo, or later for a
hexapod over a socket, touches one file.

## Stages

1. **Brain on the desk** -- load a connectome, run LIF, check known circuits. *Done.*
2. **Eyes** -- camera to ommatidia to input currents. *Done.*
3. **Descending readout** -- DNa01, DNa02, MDN, DNp09, GF as the animal's API. *Done.*
4. **A body** -- kinematic stub done; NeuroMechFly v2 adapter written against the
   FlyGym 2.x API but **not yet executed** (needs Python 3.12 and MuJoCo).
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
