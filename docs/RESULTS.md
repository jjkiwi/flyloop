# Run log

Every run against real data, with the parameters needed to repeat it and the
reasons not to over-read it. Modelled on `ommatid/docs/runs.md`, which is the
right standard for this kind of work.

---

## Run 1 — LC4 activation on MaleCNS, with control graphs

**Date** 2026-09-14
**Data** MaleCNS v1.0, prepared release from
[`YijieYin/connectome_data_prep`](https://github.com/YijieYin/connectome_data_prep)
(`mcns_all_neuron_meta.csv` + `mcns_syncount_all_neuron.npz`), CC-BY.
**Graph** 161,429 neurons; 5,988,538 connections and 86,516,520 synapses after
`min_synapses=5`; E/I synapse ratio 1.58; 1.9% of neurons unsigned.
**Model** LIF with the Shiu et al. constants, `sign_source="flyloop"`.
**Protocol** Poisson drive on LC4-left at 50 Hz for 200 ms, seed 0. Control
graphs at seed 0.

```bash
flyloop --data-root <clone> activation --drive LC4 --side L --rate 50
```

### Result (mean rate over the 200 ms window, Hz)

| population | original | rewired | relabelled | signs scrambled |
|---|---:|---:|---:|---:|
| DNp04_L | **325.0** | 0.0 | 0.0 | 350.0 |
| DNp02_L | **225.0** | 0.0 | 0.0 | 305.0 |
| DNp01_L | **195.0** | 0.0 | 0.0 | 355.0 |
| DNp01_R | 60.0 | 0.0 | 0.0 | 290.0 |
| MDN_L | 7.5 | 0.0 | 15.0 | 37.5 |
| DNa01_R | 5.0 | 0.0 | 0.0 | 320.0 |
| DNp09_L / DNp09_R | **0.0** | 0.0 | 0.0 | 285.0 |
| DNa02_L / DNa02_R | 0.0 | 0.0 | 0.0 | ~315 |
| LC4 (driven) | 46.5 | 44.8 | 70.5 | 158.5 |
| **total spikes** | **13,324** | 1,594 | 870,981 | 890,053 |

### What it supports

1. **The response is specific to the measured wiring.** On a degree-preserving
   rewire the entire descending response disappears — every readout at 0.0 Hz,
   and network activity falls to 1,594 spikes from 13,324. Same degrees, same
   transmitters, same drive; different targets, no response.

2. **The looming signal is carried by DNp04 and DNp02, and DNp09 is silent.**
   This reproduces, from a different code path and without a robot, camera or
   optic-lobe model, what the ommatid project measured on hardware across 210
   trials: DNp04 and DNp02 carry the looming-evoked rise while DNp09 stays at
   0 Hz. Two independent implementations agreeing on which neurons do and do
   not respond is worth more than either one alone.

3. **The response is lateralised.** Driving LC4 on the left recruits DNp02_L
   and DNp04_L and leaves their right-hand partners at 0 Hz. DNp01 is the
   exception, firing on both sides (195 Hz left, 60 Hz right), as a giant fibre
   with a known contralateral connection should.

### What it does not support, and the caveats that matter

- **Only the rewired control is interpretable.** The relabelled and
  sign-scrambled graphs both run away: ~880,000 spikes against the real graph's
  13,324, with every descending population firing at 200–350 Hz. That is not a
  response, it is a network with no functional excitation/inhibition balance
  saturating. Their high rates say nothing about the real graph, and reading
  them as "the control also responds" would be wrong. Permuting transmitters
  across 161,429 neurons destroys E/I balance too thoroughly to be a control at
  this scale.

- **The rates are ordinal, not cardinal.** 325 Hz is close to the model's
  refractory ceiling of 1/2.2 ms = 455 Hz. A 50 Hz Poisson drive straight onto
  LC4 is far stronger than anything a real visual stimulus delivers, so these
  numbers rank the populations but do not estimate firing rates.

- **DNp01 fires here and did not in ommatid.** They measured DNp01 at 0 Hz in
  every trial. The difference is almost certainly the drive: they arrive at LC4
  through a camera, an optic-lobe model and a calibrated gain, while this run
  injects current into LC4 directly and hard. It localises the disagreement to
  the front end rather than to the wiring, and it is the obvious next
  experiment — sweep the drive rate and find where DNp01 drops out.

- **This is an activation experiment, not behaviour.** No eye, no body, no
  closed loop. It says which descending neurons a connectome connects to LC4
  strongly enough to fire. It says nothing about what the animal would do.

### Next

- Sweep the LC4 drive rate from 5 to 200 Hz and find the threshold at which
  each descending population recruits — that is where the DNp01 disagreement
  with ommatid should resolve.
- Repeat with LPLC2 (185 neurons), the other looming-sensitive population.
- Ablate DNp04 and DNp02 and confirm the response disappears.
- Build a control graph that preserves E/I balance per neuron, since the
  current relabel and sign-scramble controls are unusable at whole-CNS scale.

---

## Run 2 — LC4 drive-strength sweep, and where DNp01 drops out

**Date** 2026-09-15
**Data, graph, model** as Run 1.
**Protocol** Poisson drive on LC4-left at ten rates from 5 to 200 Hz, 200 ms
per trial, **six seeds per rate**, median reported. Repeated on a
degree-preserving rewire (seed 1).

```bash
flyloop --data-root <clone> sweep --drive LC4 --side L --seeds 6 --control
```

This was the experiment Run 1 said was needed: Run 1 measured DNp01 firing at
195 Hz where ommatid measured it at 0 Hz, and blamed the drive path rather than
the wiring. That is testable by varying the drive.

### Dose-response, real wiring (Hz, median of 6 seeds)

| drive Hz | DNp04_L | DNp02_L | DNp01_L | DNp01_R | DNp09_L/R | DNp04_R |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 67.5 | 17.5 | 7.5 | 12.5 | **0.0** | 0.0 |
| 10 | 145.0 | 57.5 | 10.0 | 27.5 | **0.0** | 0.0 |
| 15 | 177.5 | 80.0 | 15.0 | 45.0 | **0.0** | 0.0 |
| 20 | 215.0 | 115.0 | 40.0 | 50.0 | **0.0** | 0.0 |
| 30 | 262.5 | 157.5 | 85.0 | 55.0 | **0.0** | 0.0 |
| 50 | 315.0 | 212.5 | 195.0 | 60.0 | **0.0** | 0.0 |
| 75 | 350.0 | 257.5 | 277.5 | 55.0 | **0.0** | 0.0 |
| 100 | 365.0 | 285.0 | 315.0 | 70.0 | **0.0** | 0.0 |
| 150 | 385.0 | 317.5 | 352.5 | 72.5 | **0.0** | 0.0 |
| 200 | 395.0 | 335.0 | 365.0 | 60.0 | **0.0** | 0.0 |

On the rewired graph every one of these populations is **0.0 Hz at every drive
rate**, while total network activity still scales smoothly with drive (69 spikes
at 5 Hz to 5,520 at 200 Hz). The graph conducts; it just never reaches these
neurons.

### What it settles

1. **The Run 1 disagreement with ommatid was drive strength, and the sweep
   shows exactly where.** DNp01 is not missing from the pathway — it is the
   *last* of the three to recruit. At the bottom of the sweep the order is
   DNp04 (67.5 Hz) ≫ DNp02 (17.5) > DNp01 (7.5), and DNp01 only overtakes
   DNp02 above about 75 Hz. ommatid measured, through a camera and a calibrated
   gain, DNp04 +12.8 Hz > DNp02 +5.6 Hz > DNp01 0 Hz. Same ordering, same
   regime. Run 1 drove LC4 at 50 Hz, by which point DNp01 is already at 195 Hz
   — far past anything a visual stimulus delivers.

2. **DNp09 is not weakly recruited. It is not recruited at all.** Zero on both
   sides at all ten drive rates across a fortyfold range. A single silent
   measurement can be a threshold effect; a flat zero across the whole sweep is
   a statement about the LC4→DNp09 pathway in this connectome under this model.

3. **The response is strictly ipsilateral except for DNp01.** DNp04_R and
   DNp02_R stay at 0.0 Hz throughout while their left partners saturate.
   DNp01_R does respond, rising to about 70 Hz and then flattening — the
   shape expected of a contralateral connection rather than a direct one.

4. **The rewired control is as clean as this control gets.** Flat zero for
   every descending population up to 100 Hz drive. It does start to leak above
   that — DNa01_L 7.5 Hz at 150, DNp02_R 10 Hz at 200 — so the control is
   trustworthy in the physiological range and should not be leaned on at the
   top of the sweep.

### Caveats

- **The bottom of the sweep is noisy and the tool now says so.** Seed-to-seed
  spread in total spikes is 8.2x at 5 Hz, 2.7x at 10 Hz, 3.6x at 30 Hz and
  3.7x at 100 Hz, against 1.1–1.2x elsewhere. With Poisson drive that sparse
  the whole-network response turns on whether a few input spikes coincide.
  Those four rates are excluded from the recruitment-threshold table by
  default, which is why it starts at 15 Hz. The dose-response *ordering* is
  stable across seeds; the absolute spike counts at 5 and 10 Hz are not.
- **Rates near 400 Hz are at the refractory ceiling** of 1/2.2 ms = 455 Hz.
  The top half of the table is saturated and ranks populations without
  estimating rates.
- **DNa01, DNa02 and MDN show nothing credible.** Their non-zero entries are
  at or below what the rewired control produces at the same drive.
- **Still an activation experiment.** No eye, no body, no behaviour.

### Next

- Repeat with LPLC2 (185 neurons), the other looming-sensitive population, and
  check whether it recruits DNp09 where LC4 does not.
- Ablate DNp04 and DNp02 and confirm the remaining response collapses.
- Calibrate the drive: find the LC4 rate that a real looming stimulus produces
  through flyvis, so the sweep can be read at the animal's own operating point
  instead of in arbitrary units.
- Build an E/I-preserving control graph, since relabelling and sign-scrambling
  remain unusable at whole-CNS scale (Run 1).

---

## Run 3 — LPLC2, and asking the graph instead of the simulation

**Date** 2026-09-15
**Data, graph, model** as Run 1.
**Protocol** as Run 2, driving LPLC2-left instead of LC4-left. Plus a
parameter-free structural analysis of the wiring itself.

```bash
flyloop --data-root <clone> sweep --drive LPLC2 --side L --seeds 6 --control
```

### The structural answer came first, and settled most of it

Before running anything, the connectome was asked directly how many synapses
each source makes onto each descending neuron
(`flyloop.connectome.pathways.direct_drive`):

| target | from LC4-L | from LPLC2-L |
|---|---:|---:|
| DNp04_L | 6,811 | 1,957 |
| DNp01_L | 3,782 | 2,636 |
| DNp02_L | 2,279 | **0** |
| DNp09_L | **0** | 68 |
| DNa01, DNa02, MDN | 0 | 0 |

**DNp09 receives exactly zero synapses from LC4.** Runs 1 and 2 spent a
fortyfold sweep of drive strength establishing that DNp09 never responds to
LC4; the graph says why in one query, with no model and no free parameters.
That is the order these questions should be asked in.

### Sweep result, LPLC2 (Hz, median of 6 seeds)

| drive Hz | DNp01_L | DNp04_L | DNp01_R | DNp02_L | DNp09_L/R |
|---:|---:|---:|---:|---:|---:|
| 5 | 57.5 | 15.0 | 15.0 | **0.0** | **0.0** |
| 20 | 75.0 | 45.0 | 45.0 | **0.0** | **0.0** |
| 50 | 112.5 | 135.0 | 95.0 | **0.0** | **0.0** |
| 100 | 192.5 | 235.0 | 110.0 | **0.0** | **0.0** |
| 200 | 325.0 | 310.0 | 162.5 | **0.0** | **0.0** |

On the rewired control, **every** descending population is never recruited at
any rate.

### What it settles

1. **No, LPLC2 does not recruit DNp09 either.** Its 68 direct synapses are 1.5%
   of DNp09's total excitation, and that is not enough at any drive strength
   from 5 to 200 Hz. The question Run 2 posed is answered negatively.

2. **Structure predicts function exactly where it is absolute.** LPLC2 makes
   zero synapses onto DNp02, and DNp02 reads 0.0 Hz at every LPLC2 drive rate.
   LC4 makes zero onto DNp09, and DNp09 reads 0.0 Hz at every LC4 drive rate.
   Where the graph says "no path", the model agrees without exception.

3. **The recruitment order reverses between the two sources.** Under LC4 it is
   DNp04 ≫ DNp02 > DNp01; under LPLC2 it is DNp01 ≫ DNp04, with DNp04 catching
   up only above about 50 Hz. Same three cells, same model, opposite ordering —
   which is a property of the wiring rather than of the readout.

### A predictor that worked once and then failed

The obvious explanation for the LC4 ordering is not raw synapse count but the
**share of the target's total excitation** the source supplies:

| target | LC4 synapses | total E | total I | E/I | LC4's share of E |
|---|---:|---:|---:|---:|---:|
| DNp04 | 6,811 | 9,329 | 2,107 | 4.43 | **0.730** |
| DNp02 | 2,279 | 4,993 | 2,562 | 1.95 | **0.456** |
| DNp01 | 3,782 | 13,027 | 7,028 | 1.85 | **0.290** |
| DNp09 | 0 | 4,393 | 2,730 | 1.61 | **0.000** |

That ranks all four correctly, and explains the otherwise odd fact that DNp01
recruits *later* than DNp02 despite 66% more direct LC4 synapses: DNp01 is a
much larger target (13,027 excitatory synapses) collecting 3.3× more inhibition
than DNp04, so the same absolute input moves it proportionally less.

**Then it fails on LPLC2.** The shares there are 0.210 for DNp04 and 0.202 for
DNp01 — essentially identical — yet at 5 Hz drive DNp01 fires at 57.5 Hz
against DNp04's 15.0. The metric predicts a tie and the simulation gives a
fourfold difference.

So: a useful first-order predictor for LC4, falsified as a general rule by the
very next source tested. Something beyond the monosynaptic share matters —
most likely the indirect paths, or which inhibitory populations each source
co-recruits. Recorded here as a failed hypothesis rather than quietly dropped.

### Caveats

- The bottom of the LPLC2 sweep is noisier than LC4's: seed spread 10.3× at
  5 Hz, 3.6× at 10, 2.9× at 15. Those three rates are excluded from the
  threshold table, so "DNp01_L recruited at 20 Hz" is the defensible statement
  and the 5 Hz row above is indicative only.
- LPLC2 drive is weaker overall: 1,674 spikes at 5 Hz against LC4's 7,042.
- Rates above ~300 Hz remain at the refractory ceiling.
- Still an activation experiment. No eye, no body, no behaviour.

---

## Note — flyvis assessment, and why the calibration plan changes

**Date** 2026-09-15

Run 2 proposed calibrating the drive through `flyvis`, so the sweep could be
read at the animal's operating point instead of in arbitrary Hz. flyvis 1.2.0
was installed and tested. What it can and cannot do here:

**Works offline.** It installs on Python 3.11, and `Network()` builds from the
connectome it bundles (`flyvis/connectome/fib25-fib19_v2.2.json`) with no
download — 734 free and 2,959 fixed parameters. Its stimulus generators
(`Flashes`, `MovingBar`, `MovingEdge`, `Dots`) are synthetic and need nothing.

**It has no LC4.** Its 65 cell types run R1–R8, L1–L5, Lawf, Am, C2/C3, CT1,
Mi1–Mi15, T1–T5, Tm and TmY. It models the retina, lamina and medulla up to the
T4/T5 motion detectors and stops *before* the lobula columnar projection
neurons. So it cannot produce an LC4 firing rate directly, which was the whole
point of the proposal.

**Its download hosts are blocked here anyway** — the Sintel training set
(`files.is.tue.mpg.de`) and moving-MNIST (`www.cs.toronto.edu`). An untrained
network's rates are meaningless, so without pretrained weights it cannot
calibrate anything from this sandbox.

### The better plan this reveals

The interface between the two models was never going to be LC4; it is the
**shared columnar cell types**. MaleCNS contains L1–L5, Mi1, Tm and T4/T5 *with
hex column assignments already* (23,720 neurons, 892 columns right and 875
left — see Run 1). So the arbitrary drive can be removed without flyvis at all:

> Present a looming pattern in hex coordinates to **L1**, using the real
> retinotopy from `ColumnMap.from_hex_metadata`, and let LC4 activity emerge
> from MaleCNS's own optic lobe rather than being injected.

`connectome-interpreter`, already a dependency, ships `looming_stimulus()` and
`make_sine_stim()` that generate exactly such patterns in hex coordinates — the
same stimulus family ommatid used (looming disc, drifting grating, bright
hemifield).

That closes the real gap in Runs 1–3, which is not the units of the drive but
that current is injected into LC4 rather than light being shown to an eye. It
also makes the optomotor and phototaxis hypotheses testable, which injection
into LC4 never could.

flyvis stays a dependency worth keeping for a machine with network access and
Python 3.12, where its pretrained optic-lobe model is a stronger front end than
anything derived from connectivity alone.

---

## Run 4 — showing the fly something, and why it does not work

**Date** 2026-09-15
**Data, graph, model** as Run 1.
**Protocol** A looming disc painted onto the fly's own hexagonal columns
(892 on the right eye, from `assignedOlHex`), delivered to the lamina monopolar
cells L1 and L2 in the columns it covers, 8 frames × 25 ms after a 100 ms
silent baseline. Plus direct activation of whole L1 and L2 populations, and a
structural audit of the pathway.

This was the step Runs 1–3 needed: they injected current into LC4, so nothing
between the eye and LC4 was ever exercised.

### The headline result is negative, and its cause is identifiable

**A looming stimulus presented to the eye does not reach LC4.** Not at any
background level tried, not at any drive rate. The LC4 responses in Runs 1–3
remain valid statements about LC4's *output* wiring, but LC4 can be reached by
injection and not by light.

Two independent reasons, both measured:

**1. Inhibitory pathways transmit nothing in a network with no basal firing.**
L1 — the first cell of the ON pathway — is glutamatergic, and glutamate is
inhibitory in *Drosophila*. It makes 74,170 synapses onto Mi1. Driving it:

| condition | L1 | Mi1 |
|---|---:|---:|
| no background, L1 off | 0.0 Hz | 0.0 Hz |
| no background, L1 driven at 200 Hz | 127.9 Hz | **0.0 Hz** |
| Mi1 held at 60 Hz background, L1 off | 0.0 Hz | 30.6 Hz |
| Mi1 held at 60 Hz background, L1 driven | 127.7 Hz | **16.6 Hz** |

Driving L1 as hard as the model allows leaves its target at exactly zero,
because there is nothing to inhibit. Give Mi1 a tonic rate and the same drive
suppresses it by 46%. The Shiu et al. model's zero basal firing is a documented
limitation; this is the consequence that does not seem to be written down
anywhere: **it does not merely make the network quiet, it makes every
sign-inverting pathway mute, and the fly's entire ON pathway with it.**

`PoissonDrive.set_background()` now exists for this, and it survives the
per-frame `clear()`.

**2. The columnar pathway to LC4 is far below threshold at every stage.**
A single spike needs about 162 synapses to fire a resting neuron in this model
(`flyloop theory`). Median synapses per connected pair:

| connection | pairs | median syn/pair | one spike enough? |
|---|---:|---:|---|
| L2 → Tm1 | 909 | 129 | borderline |
| L2 → Tm2 | 1,074 | 128 | borderline |
| L1 → Mi1 | 915 | 85 | no |
| Tm1 → T5a | 1,785 | 9 | no |
| Tm2 → T5a | 2,039 | 13 | no |
| Tm2 → LC4 | 1,076 | 7 | no |
| **T5a → LC4** | **3** | **5** | no |
| **T4a → LC4** | **0** | — | **no connection at all** |

The chain attenuates at every step: 129 synapses into the medulla, 9–13 into
the motion detectors, 5–7 into LC4. LC4 is reached only by massive convergence
of tiny inputs, and this model has no mechanism — no basal activity, no learned
gain, no graded transmission — to bridge that. Driving the *entire* L2
population on one side confirms it: at 100 Hz nothing downstream fires at all;
at 400 Hz, LC4 reaches 1.3 Hz. A stimulus covering at most 169 of 892 columns
delivers a small fraction of that.

### What this means for the project

- **Runs 1–3 stand, with their scope narrowed.** They measured LC4's downstream
  wiring, and that measurement is unaffected. They did not, and could not, say
  anything about vision.
- **The optomotor and phototaxis hypotheses remain untestable here**, for the
  same reason ommatid could not confirm them: the front end does not deliver.
  Their negative result now has a mechanism rather than only a measurement.
- **This is why flyvis is trained.** A connectome-constrained model *learns*
  its gains rather than deriving them from synapse counts. Synapse count times
  one global weight cannot carry a signal through a pathway built on
  convergence. That is an argument for the trained optic-lobe model as the
  front end, not a detail.

### Caveats

- Descending populations have 1–2 neurons per side, so in a 200 ms window one
  spike is 5 Hz. The ±10–20 Hz deltas seen with background on are one or two
  spikes and should not be read as responses.
- Background was applied uniformly, which is crude: real spontaneous rates are
  cell-type specific. A uniform 5 Hz already produced 255,538 spikes per 200 ms
  and pushed parts of the network into saturation.
- The drive sign is applied by hand: no photoreceptor stage exists, so "dark
  object drives L1 and L2" stands in for the histaminergic inversion.

### Next

- Front end from `flyvis` (trained, on a machine with network access and Python
  3.12), feeding its T4/T5 or Tm outputs into MaleCNS at the shared cell types.
- Or calibrate per-cell-type gain in this model against published firing rates,
  which is what the single global `w_syn` is standing in for.
- Either way, **do not build the closed loop on the current front end.** It
  would run, and it would mean nothing.

---

## Run 5 — a rate model gets the signal through

**Date** 2026-09-15
**Data** MaleCNS v1.0, **input-proportion** weights
(`mcns_inprop_all_neuron.npz`), signed by presynaptic transmitter.
**Model** `connectome_interpreter.MultilayeredNetwork` — tanh rate units with
per-cell-type bias and slope — wrapped as `flyloop.brain.RateBrain`.
**Protocol** A dark disc on the right eye's real hex columns, delivered to L1
and L2, held constant across 8 synaptic hops.

Run 4 showed the spiking model cannot carry a visual signal to LC4, because the
pathway is far below single-spike threshold at every stage and there is nothing
to amplify it. This run changes two things at once: **input-proportion weights**
instead of synapse counts, and **per-cell-type gain and bias** instead of one
global weight and silence.

### The signal arrives

Peak activation, real graph, largest disc (169 of 892 columns):

```
L1/L2 0.189  ->  Tm1 0.129, Tm2 0.146  ->  T5a 0.123  ->  LC4 0.118  ->  DNp04 0.328
```

The whole chain conducts. That is the thing Runs 1–4 could not do, and the
reason is the weighting: a 7-synapse connection is negligible in absolute terms
but a target pooling thousands of them receives a real fraction of its input.
Convergence is what reaches LC4, and only a proportional weighting represents it.

### Dose-response in disc size, with the control graph

Peak activation, no bias:

| disc columns | LC4 | DNp04 | DNp02 | DNp01 | DNp09 |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.004 | 0.012 | 0.000 | 0.000 | 0.000 |
| 19 | 0.056 | 0.179 | 0.085 | 0.100 | 0.012 |
| 61 | 0.147 | 0.453 | 0.230 | 0.262 | 0.036 |
| 127 | 0.270 | 0.731 | 0.404 | 0.451 | 0.059 |
| 169 | 0.340 | 0.828 | 0.478 | 0.539 | 0.066 |
| **rewired, 169** | 0.078 | **0.000** | 0.060 | *0.960* | *0.260* |

**DNp04 and DNp02 are wiring-specific.** DNp04 is exactly 0.000 on the
degree-preserving rewire at every disc size while reaching 0.828 on the real
graph. DNp09 stays weakest throughout, as the zero LC4→DNp09 synapse count from
Run 3 predicts.

### Baseline bias wakes the ON pathway — in a second model class

Run 4 found that with no basal firing the glutamatergic ON pathway is mute.
Setting a uniform bias in this completely different model reproduces it:

| bias | Mi1 | T4a | T5a | LC4 | DNp04 |
|---:|---:|---:|---:|---:|---:|
| 0.0 | **0.000** | **0.000** | 0.123 | 0.118 | 0.328 |
| 0.1 | 0.100 | 0.131 | 0.239 | 0.306 | 0.758 |
| 0.3 | 0.291 | 0.372 | 0.612 | 0.730 | 0.984 |

A spiking model and a rate model, sharing nothing but the connectome, agree:
without baseline activity the fly's ON pathway carries no signal at all.

### Two things this run does not support

**The control graph fails for DNp01, so no claim is made about it.** On the
rewired graph DNp01 reaches 0.960 — *higher* than the real graph's 0.539 — and
DNp09 likewise rises. Both are high in-degree targets (DNp01 collects 13,027
excitatory synapses), and a random rewiring hands high in-degree cells a large
draw by construction. Degree-preserving rewiring is therefore informative for
low in-degree targets and misleading for high ones. That sharpens the
outstanding control-graph debt from Run 1 rather than settling it.

**A first attempt at this comparison was invalid and is recorded as such.**
Presenting looming as a sequence gave receding a *larger* response than looming,
which looked like a finding and was an API misuse: `MultilayeredNetwork` injects
input frame *t* at synaptic hop *t*, so its time axis is the hop axis. A pattern
that is large early gets eight hops of propagation and one that is large late
gets one. The comparison measured hop count. `steady_state()` now holds a
pattern constant across hops, `RateBrain`'s docstring states the trap, and a
test pins it.

The corollary matters: **this model cannot test direction of motion at all.**
Optomotor responses need temporal dynamics it does not have, so the optomotor
hypothesis stays out of reach here — now for a stated structural reason rather
than an unexplained failure.

### Caveats

- **Nothing is fitted.** Biases and slopes are hand-set; no visual firing-rate
  targets ship with the package. Every number is a statement about wiring under
  assumed excitability, not a prediction.
- Activations are arbitrary units in [0, 1], not firing rates.
- The bias is uniform across all 161,429 neurons, which is crude.
- Only the right eye and one disc position were tested.

### Next

- Fit slopes and biases against published cell-type firing rates, which is what
  `train_model` is for and what would turn these units into predictions.
- A control that preserves in-degree *and* input composition, since the current
  one breaks on high in-degree targets.
- For motion and the optomotor hypothesis, a model with real temporal dynamics
  is required — the LIF has that but cannot conduct, and this one conducts but
  has no time. Neither alone is enough.

---

## Run 6 — flying at a target, with dopamine that grows as it closes

**Date** 2026-09-15
**Data, model** MaleCNS v1.0, input-proportion weights, rate model as Run 5.
**Protocol** A dark target at (1.4, 0.6) with the fly starting at the origin
facing +x, so it begins 23 degrees off-axis. Each control step renders the
target onto the fly's own hex columns, runs the network to steady state over 5
synaptic hops, takes turn as ``DNa02_R - DNa02_L``, and moves the body. 45
steps. Reward is delivered to the **PAM cluster** -- the fly's own 316
reward-signalling dopaminergic neurons -- scaled by closeness, and depresses its
own KC->MBON synapses (61,210 of them) wherever dopamine and Kenyon-cell
activity coincide.

```bash
flyloop --data-root <clone> approach --steps 45
```

### It flies at the target, and the fixation is wiring-specific

| condition | final distance | closed | mean \|bearing\| | KC→MBON depression |
|---|---:|---:|---:|---:|
| original, dopamine lr 0.5 | 0.379 | 75.1% | **10.78°** | 0.0119 |
| original, dopamine lr 0.05 | 0.379 | 75.1% | **10.78°** | 0.0013 |
| original, no dopamine | 0.379 | 75.1% | **10.78°** | 0.0000 |
| rewired, dopamine lr 0.5 | 0.655 | 57.0% | **39.48°** | 0.1378 |
| rewired, dopamine lr 0.05 | 0.655 | 57.0% | 39.48° | 0.0175 |
| rewired, no dopamine | 0.655 | 57.0% | 39.48° | 0.0000 |

Bearing falls from 23 degrees toward 6 and stays there: the animal turns to face
the target and holds it there. That is object fixation, and it is the real
behaviour this circuit is known for.

**Mean |bearing| is the number that discriminates, not distance closed.** The
rewired control still closes 57% of the distance because the loop has a baseline
forward drive — it simply walks. What it cannot do is aim: 39.5 degrees of
mean bearing error against the real graph's 10.8.

### Dopamine works, and changes nothing here

The plasticity is real and measurable. Depression scales with the learning rate
exactly as it should (0.0119 at lr 0.5, 0.0013 at lr 0.05, exactly 0.0000 with
dopamine off), so the fly's own KC->MBON synapses are being depressed by its own
PAM neurons in proportion to how close the target is.

**And the trajectory is identical to five decimal places whether dopamine is on
or off.** Not similar — identical.

The graph said why before the simulation confirmed it, which is the order these
questions should be asked in:

- **MBONs supply about 1% of DNa02's input.** Summed over all 97 mushroom body
  output neurons onto both DNa02 cells: 0.0104 of input proportion. The
  connection exists and is monosynaptic; it is just negligible.
- **Kenyon cells barely respond to vision.** During the visual task their mean
  peak activation is 0.0023, with 235 of 4,064 above 0.01. The learning rule is
  a coincidence detector between dopamine and KC activity, and there is almost
  no KC activity to coincide with.

**The mushroom body is an olfactory learning centre, and it is not in the visual
steering loop.** Reward-modulated plasticity at the fly's real plasticity site
cannot change visual fixation, because visual fixation does not route through
that site. That is a statement about the animal, not a limitation of the code.

A curiosity worth noting: the rewired control shows *more* depression (0.138 vs
0.012), because random rewiring feeds the Kenyon cells input they do not
normally get, so there is more coincidence to detect. Plasticity is more active
on the scrambled brain than the real one.

### Caveats

- **Reward shaping is not biology.** Scaling reward continuously with distance
  is a reinforcement-learning convenience; real PAM neurons signal reward
  delivery and prediction error, not a distance gradient.
- Nothing is fitted: learning rate, gains and the forward-speed floor are all
  set by hand.
- One target position, one starting pose, no repeats. The bearing difference is
  large and consistent across six runs, but this is a demonstration rather than
  a measurement with error bars.
- The loop is quasi-static, as Run 5 requires: each step assumes the network
  settles. Fine for approaching an object, useless for motion direction.

### Two bugs found on the way, both recorded because both were silent

**A sign convention that turned the fly away from the target.** Bearing was
computed counter-clockwise-positive while the eye projection put the right eye
at positive azimuth and the body turned right on a positive command. The animal
steered smoothly and confidently in the wrong direction. Fixed by defining
bearing positive-to-the-right once, and pinned by three tests.

**Plasticity that edited the wrong synapses.** Entry positions were computed
from the COO matrix *before* `coalesce()`, which sorts entries — so the learning
rule rewrote unrelated parts of the connectome and reported a depression of
-9.21. It looked like a number. A test now builds the tensor the way the model
does, coalescing included, and checks the located weights are the right ones.

### Next

- Pair reward with an **olfactory** stimulus, where the mushroom body actually
  is the substrate. That is where dopamine-gated KC->MBON plasticity should
  change behaviour, and the same machinery is already in place.
- Repeat the approach from several starting bearings with seeds, to turn the
  fixation result into a measurement.

---

## Run 7 — fixation measured, and dopamine moved to where it works

**Date** 2026-09-15
**Data, model** MaleCNS v1.0, input-proportion weights, rate model.

Two follow-ups from Run 6: turn the one-anecdote fixation result into a
measurement, and move the reward to the modality the mushroom body is actually
for.

### Fixation, across five starting bearings

Target 1.6 away, fly started 40, 20, 0, -20 and -40 degrees off-axis, 30 control
steps, no dopamine.

| start | original: final bearing | rewired: final bearing |
|---:|---:|---:|
| −40° | **−12.2°** | −56.3° |
| −20° | **−7.5°** | −6.7° |
| 0° | **−4.7°** | +5.9° |
| +20° | **−1.8°** | +28.8° |
| +40° | **+2.5°** | +29.4° |

| graph | mean \|bearing\| | s.d. |
|---|---:|---:|
| original | **13.5°** | 9.5 |
| rewired | **25.1°** | 16.8 |

The real graph converges toward the target from every starting bearing, and the
further off-axis it begins the more it turns. The rewired graph **diverges** at
large bearings: from −40° it ends at −56°, from +20° at +29°, from +40° at +29°.
It is not merely worse at aiming, it turns the wrong way. Fixation is a property
of the measured wiring, now across five conditions rather than one.

Distance closed barely differs (0.449 vs 0.413) and remains the wrong statistic,
for the reason Run 6 gave: the loop walks forward regardless.

### Calibrating the Kenyon cell code, because conditioning is impossible without it

Run 6 ended by proposing an olfactory version of the reward task. The olfactory
pathway is fully present -- 2,635 ORNs with proper glomerular names, 197
glomerular projection neurons -- and it is what drives the mushroom body:
Kenyon cell input is 3,547 units of input proportion from central-brain
intrinsic neurons (projection neurons and the APL) against **19.3 from the
entire visual pathway**. Run 6's finding, confirmed from the input side.

But at the model's default gain the odour code is not sparse, and a dense code
makes odour-specific learning impossible by construction:

| KC slope | KCs active | sparseness | overlap of two odours |
|---:|---:|---:|---:|
| 5.0 | 3,579 | 88.1% | 99.8% |
| 1.0 | 3,318 | 81.6% | 87.4% |
| 0.6 | 1,795 | 44.2% | 43.9% |
| 0.45 | 732 | 18.0% | 19.8% |
| **0.35** | **181** | **4.5%** | **6.1%** |
| 0.30 | 66 | 1.6% | 4.5% |
| 0.10 | 0 | 0.0% | — |

In the animal roughly 5% of Kenyon cells respond to a given odour. **0.35
reproduces that**, and is now `SPARSE_KC_SLOPE`. The window is narrow -- a
factor of three either way gives a dense code or a dead one -- which is worth
knowing: the mushroom body's sparse code is not an emergent property of the
connectome in this model, it has to be imposed by per-cell-type gain.

### Differential conditioning: dopamine changes something at last

The fly approaches an odour source over 16 trials; odour and dopamine both grow
as it nears (reward 0.135 → 0.654). A second odour is experienced throughout
without reward.

| condition | Δ paired | Δ unpaired | depression |
|---|---:|---:|---:|
| A rewarded, B not | −0.007565 | −0.004180 | 0.219 |
| B rewarded, A not (swap) | −0.004156 | −0.005829 | 0.184 |
| A rewarded, **dopamine off** | **0.000000** | **0.000000** | 0.000 |
| **rewired graph**, A rewarded | −0.000000 | 0.000000 | 0.0001 |

**Dopamine now changes the mushroom body's output.** With dopamine off the
change is exactly zero, so every effect below is reward-driven.

**And the naive statistic is contaminated.** The learning index (unpaired minus
paired change) is +0.0034 in one direction and −0.0017 in the other: it does
*not* flip when the reward is swapped. Odour A falls more than odour B whichever
one is rewarded, because the two odours drive different numbers of Kenyon cells
and start from different response magnitudes. Reporting that index alone would
have claimed learning that the swap does not support.

The valid comparison is the same odour, rewarded versus not:

- **Odour A: −0.00757 when rewarded, −0.00583 when not.** It falls 30% further
  when it is the rewarded odour. A real, reward-specific effect.
- **Odour B: −0.00416 when rewarded, −0.00418 when not.** Identical to three
  decimal places. **No learning at all.**

So the effect is present, modest, and asymmetric between odours. Odour B
activates 1.5% of Kenyon cells against A's 4.5%, and 18% of B's cells lie inside
A's set while only 6% of A's lie inside B's — so B has both a weaker handle on
the output and more of its cells depressed by A's pairing.

### What this run does not support

- **The rewired control is uninformative here, not passed.** Its depression is
  0.0001 and its odour overlap exactly 0.000: the scrambled mushroom body does
  not respond to odours at all, so it cannot show whether the effect needs the
  real wiring. A control that stays silent has not controlled anything.
- **One odour pair, one seed, no repeats.** The A/B asymmetry means the choice
  of glomeruli matters, and two odours is not a sample.
- Learning rate, KC slope and reward scale are all set by hand. The KC slope is
  at least calibrated against a published constraint; the others are not.

### Next

- More odour pairs matched for sparseness, so the asymmetry does not dominate.
- A control graph that preserves the mushroom body's structure while scrambling
  elsewhere — the current rewire destroys the thing under test.
- Read the descending neurons the MBONs actually reach (DNp52, DNg104, DNa03 —
  not DNa02) and ask whether the depression changes their output, which would
  connect conditioning to behaviour rather than stopping at the MBON.

---

## Run 8 — does the learning reach behaviour? No, and the reason is structural

**Date** 2026-09-15
**Data, model** MaleCNS v1.0, input-proportion weights, rate model, KC slope 0.35.
**Question** Run 7 showed dopamine depresses the KC→MBON synapse in an
odour-specific way. Does that change reach the descending neurons — the only
cells that can move the animal?

The readout was extended from the mushroom body to the eight descending neuron
types MBONs actually project to, and conditioning was run in both directions so
the same odour could be compared rewarded against unrewarded.

### The answer

Change in peak activation for odour A, rewarded versus not:

| population | A rewarded | A not rewarded | within-odour effect | dopamine off |
|---|---:|---:|---:|---:|
| **MBON** | −0.007565 | −0.005829 | **−0.001736** | 0.0 |
| DNg104 | −0.000227 | −0.000180 | −0.000047 | 0.0 |
| DNp42 | −0.000016 | −0.000025 | +0.000009 | 0.0 |
| DNge138 | +0.000045 | +0.000044 | +0.000001 | 0.0 |
| DNp52, DNge151, DNg34, DNge150, DNa03 | 0.000000 | 0.000000 | **0.000000** | 0.0 |

**The learning stops at the mushroom body.** The largest descending effect is
3% of the MBON effect, on a neuron whose absolute response is 0.007, and five of
the eight do not move at all. With dopamine off every number is exactly zero, so
the MBON effect is real and reward-driven — it simply goes nowhere.

### Why: learning and output are handled by different MBONs

Per MBON type, the Kenyon cell input it receives (where plasticity acts) against
its output onto descending neurons (what could change behaviour):

| MBON | KC input | descending output |
|---|---:|---:|
| MBON14 | **3.54** | **0.0000** |
| MBON09 | 3.53 | 0.0044 |
| MBON07 | 3.39 | 0.0001 |
| MBON12 | 3.24 | 0.0010 |
| … | | |
| MBON33 | 0.32 | **0.3620** |
| MBON20 | 0.77 | 0.1040 |
| MBON35 | 0.19 | 0.1018 |

The two quantities are **anticorrelated, r = −0.42**. Only 12 of 97 MBONs drive
descending neurons at all, and they are not the ones the plasticity lands on.
The cell with the most Kenyon cell input has exactly zero descending output.

**And it is not a path-length problem.** The learning-site MBONs sit one or two
hops from a descending neuron; the connections exist and are negligible. Adding
hops would not help.

### What this means

In the animal, the learned valence of an odour is read out by the **MBON
ensemble as a whole** — the balance across compartments, shaped by MBON-to-MBON
interactions — rather than by reading the cells the plasticity lands on. A
feedforward model that weights every edge by input proportion propagates the
strong direct paths and loses exactly that kind of distributed, balance-based
readout.

So this is a statement about the model class as much as about the connectome:
**dopamine-gated plasticity at the right synapse, with the right sparse code,
still does not produce a behavioural change, because the readout the animal uses
is not the one a feedforward proportional model computes.**

That is the third time in this project the same lesson has appeared in a
different form: Run 4 (inhibition is mute without baseline activity), Run 5 (a
convergent pathway needs proportional weighting), and now this. Each is a place
where the connectome is necessary and not sufficient.

### Caveats

- One odour pair, one seed. The A/B asymmetry from Run 7 persists: odour B shows
  no learning at the MBON either, so its descending numbers say nothing.
- The rewired control remains uninformative for conditioning, as Run 7 stated.
- DNge151 changes by −0.0101 for odour B, but identically whether B was rewarded
  or not — odour-driven, not reward-driven. The within-odour comparison is what
  catches that; the naive before/after difference would have reported it as a
  large learning effect.

### Next

- Model the MBON ensemble readout explicitly: the difference between
  approach-promoting and avoidance-promoting compartments, rather than a mean
  over 97 cells.
- Or accept the boundary and report it: this is where a connectome-derived
  feedforward model stops, and it stops for a reason that can be stated.

---

## Run 9 — reading the mushroom body as a balance

**Date** 2026-09-15
**Data, model** MaleCNS v1.0, input-proportion weights, rate model, KC slope 0.35.
**Question** Run 8 concluded that olfactory learning does not reach the
descending neurons, and blamed the readout: averaging 97 MBONs of opposite sign
and very different output weight throws the signal away. Does reading the
*balance* instead recover it?

### The readout

Every MBON is given a valence: its net signed influence on the descending
neurons that drive approach (`DNa01`, forward walking) minus those that drive
withdrawal (`MDN` backward, `DNp09` stop), propagated four synapses with a decay
of 0.5 per hop. The readout is then the valence-weighted sum of MBON activity
rather than its mean.

Nothing is hand-labelled from the literature. The weights come out of the signed
connectome; the only prior knowledge is the behavioural role of those four
descending neurons, which this project has used since its first run. On MaleCNS
the result splits the population sensibly: 20 MBONs positive, 17 negative, the
strongest approach-weighted being MBON32, MBON30 and MBON35 -- the same
output-stage cells Run 8 identified as the ones that actually drive descending
neurons.

### First attempt: over-trained, and misleading

Run at the same settings as Run 8 (learning rate 1.0, 16 trials) the balance
showed a within-odour effect of −0.000002 for **both** odours: a reward-driven
change, since dopamine off gave exactly zero, but no odour specificity at all.

That was an artefact of over-training. Those settings depress 22% of all KC→MBON
weight, which is a near-global change that swamps any odour-specific component.
**The learning rate has to be low enough that the depression stays selective**,
and nothing in the earlier runs made that constraint visible.

### With matched, weaker training

Relative change in each readout, in percent; negative means the readout fell
further when that odour was the rewarded one.

| learning rate | depression | readout | odour A effect | odour B effect |
|---|---:|---|---:|---:|
| 0.05 | 1.3% | MBON mean | −0.413 pp | −0.111 pp |
| 0.05 | 1.3% | **balance** | **−0.446 pp** | +0.103 pp |
| 0.20 | 4.9% | MBON mean | −1.581 pp | −0.092 pp |
| 0.20 | 4.9% | **balance** | **−1.700 pp** | **−0.409 pp** |

**The balance is the better readout, and the gain is where it was predicted.**
For odour B — which showed essentially nothing in the mean (−0.092 pp) — the
balance recovers a reward-specific effect 4.4× larger (−0.409 pp). For odour A
it is modestly larger in both conditions. The direction is the one
dopamine-gated depression predicts in every case at lr 0.2.

Because the valence weights are each MBON's influence on the motor output, a
change in the balance *is* a change in the net mushroom body drive onto
descending neurons. So Run 8's conclusion needs narrowing rather than reversing:
**the learned signal does reach the motor output, at about 1.7% of the balance,
and Run 8 could not see it because it was averaging.**

### What this still does not support

- **The effect is small and most of the change is not reward-driven.** Of an
  8.3% total drop in odour A's balance at lr 0.2, only 1.7 points are specific
  to having been rewarded. The rest is odour-driven depression that happens to
  both odours.
- **It does not restore behaviour.** A 1.7% shift in the net drive onto DNa01,
  MDN and DNp09 is far below what moved the animal in the visual fixation runs,
  where the wiring-specific difference in bearing was 13.5° against 25.1°.
- Run 8's structural explanation stands and explains the size: the MBONs where
  plasticity lands and the MBONs with motor output are anticorrelated at
  r = −0.42, so weighting by output necessarily up-weights the cells that learn
  least.
- One odour pair, one seed, two learning rates. Odour B's sign flips between
  them, so at lr 0.05 the training is too weak to resolve.
- The decay of 0.5 per hop is a modelling choice, not a measurement.

### Next

- Sweep the learning rate properly to find where reward specificity peaks
  against total depression — the two trade off and nothing here has located the
  optimum.
- Several odour pairs matched for sparseness, so a sign flip in one odour is
  not the whole B result.

## Run 10 — the same readout, in a body with legs

**Date** 2026-09-15
**Data, model** MaleCNS v1.0, input-proportion weights, rate model, 5 hops.
**Body** NeuroMechFly v2 in MuJoCo via FlyGym 1.x (`flygym-gymnasium`), 42
actuated joints, tripod gait with stumbling and retraction correction, 0.1 ms
physics step, 50 ms control step. `MUJOCO_GL=disable`: full physics, no pixels.
**Question** Every result so far moved a point on a plane. Does the same
unfitted descending readout — the left-right difference of DNa02 — steer a body
that has legs, mass and contact physics?

### The measurement had to be designed before the experiment

A walking NeuroMechFly does not go straight. Commanded to walk straight for 2 s
it drifts **+14.6°, −0.4° and −40.6°** on three body seeds — a spread of 55°,
several times larger than any steering signal expected here. A single episode
therefore measures gait noise, not behaviour.

So each measurement is a **mirror pair**: the same body seed walks once at a
target 35° to its right and once at the mirror image 35° to its left. The gait
noise is common to both; only the stimulus differs. The statistic is

    fixation = (Δθ_left − Δθ_right) / 2

positive when the fly turned toward the target in both mirror images.

### Result

Target 25 mm away at ±35°, 30 control steps (1.5 s), 3 body seeds per graph.

| graph | fixation | commanded turn | mean \|bearing\| |
|---|---:|---:|---:|
| MaleCNS | **+37.72° ± 1.17** | +0.0504 ± 0.0013 | 9.7° |
| rewired control | **+3.00° ± 1.38** | +0.0021 ± 0.0042 | 43.9° |

**The fly aims, and aiming needs the measured wiring.** On MaleCNS the angle to
the target falls from 35° to about zero in 0.8 s and is then held: −35° → −3.2°
with the target on the left, +35° → −5.8° with it on the right. The rewired
control does not aim at all (+3.00°, twelve times smaller, and its commanded
turn is statistically indistinguishable from zero).

Body drift is genuinely cancelled rather than merely averaged: the three seeds
span 55° of intrinsic drift but only 2.3° of fixation.

### The control hides a trap, and it is the reason for the mirror design

Read naively, the control looks like it *does* something: mean |bearing| 43.9°
against 9.7°, a 4.5× separation that would have made a publishable-looking
number. But split by side, the rewired fly's bearing goes −35° → −32.9° on the
left and +35° → **+58.8°** on the right. It did not respond to the stimulus in
either case; both episodes simply drifted left, which happens to hold a
left-hand target roughly in place and push a right-hand target away.

**A one-sided experiment here would have reported a real-looking steering effect
produced entirely by the gait.** The mirror-pair statistic reports +3.00°,
which is the honest answer.

### What the gait contributes

The commanded turn is small and nearly constant — +0.050, about 5% of the
controller's range — yet it produces 37.7° of heading change. The tripod gait
integrates a small steady difference in descending drive over many steps. So the
connectome supplies a *direction*, and the body supplies the *gain*. Neither
number alone describes the behaviour.

### Following the indirect routes: 4.1x more coupling, same conclusion

Run 12 as first written coupled MBONs to steering through the direct connection
only. The literature says the main route is indirect, so this traces the input
backwards instead. Input proportions make that well defined -- a neuron's input
column sums to 1, so it is a distribution over where its input came from, and
applying the matrix again pushes that distribution one step further back.

| steps back | share of DNa02's input originating in MBONs |
|---:|---:|
| 1 (direct) | 0.5214% |
| 2 | **0.6307%** |
| 3 | 0.5461% |
| 4 | 0.4441% |
| **1-4 together** | **2.1423%, or 4.11x the direct connection** |

The two-step share is larger than the direct one, so the indirect routes really
are the bigger ones. Re-running the sweep with the fuller coupling moves the
threshold by the same factor it moved the coupling:

| gain | strength | mean modulation | closed |
|---:|---:|---:|---:|
| 1 | 0.021 | 1.000000 | 11.10 mm |
| **50,000** | 1071 | **0.851348** | 7.83 mm |
| 100,000 | 2142 | 0.951360 | 10.38 mm |

**About 50,000x, against 200,000x for the direct connection alone** -- a factor
of four, which is the factor the coupling grew by. Everything else stands: at
gain 1 the modulation is still 1.000000 to six decimals and the fly closes the
same 11.10 mm. Four orders of magnitude instead of five is still four orders of
magnitude.

(The 100,000 row closing *more* than the 50,000 row is not a reversal. Once the
modulation reaches its zero floor the trajectories diverge, so the two rows are
different paths rather than the same path pushed harder.)

### The intermediates are not the ones we were told to expect

The relayed literature summary named DNa03 as the main indirect route. Ranking
the two-step paths by how much they actually carry does not agree:

| route MBON -> X -> DNa02 | contribution |
|---|---:|
| via LAL051 | 0.0855% |
| via LAL171 | 0.0839% |
| via LAL172 | 0.0760% |
| via LAL173 | 0.0560% |
| via LAL170 | 0.0295% |
| via LAL018 | 0.0265% |
| via CB0356 | 0.0252% |
| **via DNa03** | **0.0226% (eighth)** |

The carriers are lateral accessory lobe neurons, not DNa03, which comes eighth.
That is not obviously in conflict with the steering literature -- the LAL is the
premotor hub DNa02 sits downstream of, and LAL013 and LAL010 both appear in the
descriptions of this circuit -- but the specific claim that DNa03 is the main
mushroom-body route to DNa02 is not what this connectome says. The claim was
relayed to us rather than read at source (see `docs/LITERATURE.md`), so the
disagreement may be in the relay rather than in the paper.

### What this does not show

- **The fly aims but does not arrive.** In 1.5 s it closes 8.1–8.4 mm of 25 mm
  and ends 16.8 mm out. Turning costs forward speed; a straight-walking
  NeuroMechFly covers 27 mm in 2 s. Nothing here tests approach to contact.
- **The gait is not connectome-derived.** The CPG, stumbling correction and
  retraction rules are FlyGym's hand-written controller. The connectome supplies
  exactly two numbers per control step. A claim that "the connectome walks the
  fly" would be false.
- **Vision is analytic.** Bearing and angular size are computed from the pose and
  painted onto the hex columns, not rendered through the fly's optics. This is
  what the other connectome-in-a-body projects do, and it does not change what
  the brain receives, but it is not a test of the optics.
- **The brain is quasi-static.** The rate model's time axis is the synaptic-hop
  axis (Run 5), so it has no memory between control steps. Fixation is a
  defensible use of that approximation; anything about motion direction is not.
- **Dopamine is present and irrelevant**, as Runs 6 and 7 established: the
  mushroom body is not in the visual steering loop.
- Three body seeds, one bearing, one distance, one episode length.

### Next

- Sweep the bearing. A single 35° offset cannot distinguish a proportional
  controller from a bang-bang one, and the near-constant commanded turn
  (+0.050 ± 0.0013 across seeds) hints at saturation rather than proportionality.
- Longer episodes, to ask whether it reaches the target or orbits it.
- The `relabel_neurons` and `scramble_signs` controls, which this run skipped for
  time — each mirror pair costs 11 minutes of wall clock.

## Run 11 — where in the animal the decision actually lives

**Date** 2026-09-15
**Data, model** MaleCNS v1.0, input-proportion weights, rate model, 5 hops.
**Question** Run 10 showed the readout steers a physical fly. Which neurons, and
where in the animal, carry the part of the signal that decides *which way*?

### Method, and why it is a difference

The prepared files carry one spatial number per neuron — `somaLocation`, the
cell body's position — for 138,496 of 161,429 cells (85.8%). There are no
skeletons in them and neuPrint and Codex are both unreachable from here, so
what can be drawn is a cloud of cell bodies, not morphology.

The highlight is not an activation but a **difference**: activation with the
object 35° to the right minus activation with its mirror image 35° to the left.
This is the per-neuron form of Run 10's mirror pair, and it is the same argument.
A cell that responds equally to both is reporting *that an object exists*; only
a cell whose response flips carries *which way to turn*.

### The decision narrows by an order of magnitude

Mean |lateralisation| among cells that respond at all:

| stage | mean \|lateralisation\| | responding cells |
|---|---:|---:|
| optic lobe | 0.248 | 22,792 |
| visual projection | 0.146 | 5,461 |
| central brain | 0.078 | 9,754 |
| **descending** | **0.045** | **631** |
| nerve cord | 0.021 | 2,298 |

41,377 neurons respond to the stimulus; side information is strong at the retina
and has fallen 5.5× by the time it reaches the descending neurons. That is not a
loss — it is the funnel the animal has to build, since 22,792 optic lobe cells
cannot each issue a motor command. But it does set the scale of everything this
project measures downstream: the steering command is a small difference riding
on a much larger common signal, which is exactly what Run 10 saw when a
commanded turn of 0.050 produced 37.7° of heading.

### The cell types it picks out were not chosen by us

The most lateralised types, ranked by mean |lateralisation|:

| type | mean \|lat\| |
|---|---:|
| PVLP025, CB1099 | 0.644 |
| PVLP097 | 0.610 |
| PVLP111 | 0.602 |
| PVLP078 | 0.596 |
| **DNp04** | **0.554** |

PVLP is the posterior ventrolateral protocerebrum — where LC4 and LPLC2
terminate, the object and looming pathway. Nothing in the analysis knows that;
the ranking comes from the signed connectome and the stimulus alone.

**DNp04 is the interesting one.** It is the highest-ranked descending neuron,
and it is the same cell the ommatid project measured as carrying the looming
signal on real MaleCNS wiring when the textbook choice, DNp01, sat at 0 Hz in
all 210 trials. Two independent routes to the same unexpected neuron.

Along the pathway the gradient is orderly: L1 (lamina) ±0.10, T5a ±0.134,
LC4 ±0.17, LPLC2 ±0.24, DNp01 ±0.23, **DNa02 −0.065 / +0.083** — the steering
neuron this project reads, right-preferring on the right side. MDN, which drives
backward walking, sits at ≈0 on both sides: not lateralised, which is correct,
because backing up is not a steering decision.

### What this does not show

- **These are cell bodies, not neurites.** Somata sit on the rind around the
  neuropil, so the picture is a shell. It is not the morphology render it
  resembles, and it says nothing about where a neuron's processes go.
- **14% of neurons have no located soma** and are simply absent from the figure,
  including some that respond.
- The rate model is quasi-static and the activation is its peak over 5 hops, so
  "responds" means "receives signal within 5 synapses", not a measured firing
  rate.
- Superclass is the dataset's own label; the stage table inherits whatever is
  wrong with it.
- One bearing, one object size, no control graph. A rewired control would show
  whether the *spatial* organisation of the highlight is wiring-specific; the
  funnel's shape probably is not, since it partly reflects how many cells each
  stage has.

### Next

- The same map on a rewired graph, to separate "this is what the wiring does"
  from "this is what any funnel-shaped network does".
- Highlight the reward pathway the same way, using the PAM cluster rather than a
  visual stimulus.

## Run 12 — a reflex with a learned base, and the size of the gap

**Date** 2026-09-16
**Data, model** MaleCNS v1.0, input-proportion weights, rate model, 5 hops,
Kenyon cell slope 0.35, learning rate 0.2, 8 conditioning trials.
**Question** The visual reflex works (Run 10) and the mushroom body learns
(Run 9). Can the learned valence bias the reflex, the way it does in the animal,
and how much of the behaviour does it move?

### The connectome chose the architecture

Two measurements decided the design before any code was written:

| path | input share |
|---|---:|
| LC4 + LPLC2 → Kenyon cells | **0.0000%** |
| MBON → DNa02 | **0.5256%** |
| MBON → DNa03 | 1.9613% |
| DNa03 → DNa02 | 1.1616% |
| MBON → DNa01, DNp09 | 0.0000% |

A seen object cannot be learned through the mushroom body, so whatever is
learned has to be an odour. And MBONs reach steering through one narrow door.
That is the arrangement the animal has: **the odour says whether to approach,
the sight says which way**, and the mushroom body biases a command it does not
generate.

An independent check fell out of this. Li et al. (*eLife* 2020;9:e62576) name
MBON32 and MBON31 as DNa02's direct mushroom body input; ranking MBON types here
by input share, with no reference to that paper, puts those two on top at 0.279%
and 0.219% — 0.498% of the 0.526% total. See `docs/LITERATURE.md`.

### The measurement had to be rebuilt twice

**First attempt: trained odour against control odour.** −0.00025 against
+0.00434, which looks like a large learned effect. It is not one. With no
training at all the same two odours sit at −0.00025 and +0.00434. The difference
is odour identity; eight pairing trials move each by about 2% of itself.

**Second attempt: subtract each odour's untrained baseline**, measured on the
odour alone. Also wrong. The behaving fly has an object in view as well, and
that moves the readout by 0.001 — twenty times the effect being isolated.

**What works is a paired readout inside the step.** The same input is run twice,
once on the trained weights and once on the naive ones, and the learned
component is their difference. Before any training it is zero by construction,
which is now a test. The second pass costs 0.16 s against the physics step's
1.8 s, so exactness is nearly free.

### Result: the learned signal is real, and four orders of magnitude too small

Trained and behaving in the same odour, 8 trials, 40 control steps, target 25 mm
away at 35°.

| gain | coupling strength | mean modulation | closed | mean \|bearing\| |
|---:|---:|---:|---:|---:|
| **1** (measured anatomy) | 0.005 | **1.000000** | 11.28 mm | 8.86° |
| 1,000 | 5.3 | 0.999568 | 11.28 mm | 8.87° |
| 10,000 | 52.6 | 0.995682 | 11.23 mm | 8.99° |
| 40,000 | 210 | 0.970284 | 10.92 mm | 9.59° |
| **70,000** | 368 | **0.860790** | 9.60 mm | 12.46° |
| 100,000 | 526 | 0.073395 | 0.73 mm | 28.70° |

At the measured coupling the learned component of the descending drive is
**−8.2 × 10⁻⁵** and the approach drive moves by less than a millionth. Behaviour
is identical to Run 10 to two decimal places. **About 70,000× the measured
MBON→DNa02 coupling is where learning first bends the path**, and by 100,000×
the learned aversion cancels approach almost completely — the fly closes 0.73 mm
instead of 11.28.

This is the sharpest form of what Runs 8 and 9 found. Not "1.7% of the balance"
but: the learned component of the steering command is about eight parts in a
hundred thousand.

### In a body with legs the gap is wider still, and the reason is Run 10's

The table above is the kinematic stub. Repeating it on NeuroMechFly, same
odour, same seed, moves the threshold by another factor of three:

| gain | mean modulation | min modulation | closed | mean \|bearing\| |
|---:|---:|---:|---:|---:|
| 1 | 1.000000 | 1.000000 | 11.10 mm | 7.05° |
| 70,000 | 0.991924 | 0.984712 | 10.99 mm | 7.15° |
| **200,000** | 0.901843 | 0.000000 | 9.09 mm | 9.79° |
| 500,000 | 0.892176 | 0.000000 | 9.28 mm | 9.05° |

At 70,000x -- enough to bend the stub's path by 1.7 mm -- the physical fly gives
up 0.11 mm. It takes about **200,000x** to move it, and past that the effect
saturates, because the modulation floors at zero and cannot push further.

**This is the same fact as Run 10, seen from the other side.** There, a
commanded turn of 0.050 became 37.7 degrees of heading, and the conclusion was
that the gait supplies the gain. A tripod gait integrates the descending command
over many steps, so a small *steady* difference accumulates -- and a *transient*
dip washes out. The learned modulation is transient: it tracks an odour readout
that moves with the fly's own view. The body amplifies one and attenuates the
other, and which it does to a given signal depends on the signal's time course,
not its size.

So the honest headline is the wider number. **A learned association in this
connectome needs about five orders of magnitude more coupling than it has
before it changes what a physical fly does.**

### The controls that make it readable

- **No odour, any gain.** At gain 100,000 with nothing in the air the modulation
  is exactly 1.000000000 and the fly closes 11.39 mm. The learned machinery
  cannot leak into behaviour through some other route.
- **No training, any odour.** The learned component is exactly 0, by
  construction of the paired readout.
- **Direction is never overridden.** The modulation floors at zero, so a learned
  aversion can cancel approach but not reverse the turn. Which way to go is the
  visual pathway's answer, and 0.5% of DNa02's input does not get to overrule it.

### What this does not show

- **Two bodies give two thresholds**, 70,000x and 200,000x, and neither is "the"
  answer: they differ because the gait filters the command, not because one
  measurement is better. Quoting either without the body is meaningless.
- **Past saturation the sweep stops measuring gain.** At 200,000x and above the
  modulation hits its zero floor within the episode, so higher gains change the
  fraction of time spent floored rather than the depth of the effect. The 500,000
  row closing *more* than the 200,000 row is that, not a reversal.
- **The gain figures are upper bounds.** The coupling models only the direct
  MBON→DNa02 connection, and the literature calls the indirect route through
  DNa03 the main one — which this connectome supports, MBON→DNa03 being nearly
  four times stronger. Routing through DNa03 would cut the required gain by a
  factor of a few, not by orders of magnitude.
- **The learned component is not gain-independent.** It drifts from −8.2e−5 at
  gain 1 to −2.2e−3 at gain 100,000, because modulation changes behaviour, which
  changes what the eye sees, which changes the readout. The high-gain rows are
  measuring a different trajectory, not the same one harder.
- **Kenyon cells are driven directly** with a synthetic sparse code, not through
  projection neurons. In this connectome KC input is dominated by KC↔KC
  recurrence and APL feedback (10.6%), and that approximation carries over from
  Run 9 unchanged.
- One odour pair, one bearing, one distance, one seed, one learning rate.

### Next

- Route the coupling through DNa03 and re-measure, now that the literature says
  that is the main path.
- Sweep the learning rate against the number of trials; 8 trials at lr 0.2
  depresses 1.05% of KC→MBON weight, and nothing here locates the optimum.

## Run 13 — the controller's open-loop shape, and a lean nobody asked for

**Date** 2026-09-16
**Data, model** MaleCNS v1.0, input-proportion weights, rate model, 5 hops. No
body: this is the readout alone, probed with an object at 19 bearings.
**Question** Run 10 noted the commanded turn barely varied and flagged possible
saturation. Is the steering readout proportional to where the object is, or is
it bang-bang? And since the literature reports the relationship as *linear*
(Rayshubskiy, Holtz & Wilson, eLife 102230), does this model reproduce it?

### It is proportional, and roughly linear where it matters

| bearing | −30° | −10° | 0° | +10° | +30° | +60° | +90° |
|---|---:|---:|---:|---:|---:|---:|---:|
| turn | −0.0655 | −0.0156 | **+0.0117** | +0.0363 | +0.0824 | +0.0624 | +0.0320 |

Not saturated: the command spans −0.066 to +0.082 and tracks bearing. Over
|bearing| ≤ 60° a straight line fits with **R² = 0.845**, slope 1.25 × 10⁻³ per
degree. So the linear proportionality the recordings describe is approximately
what this wiring produces, which is a real agreement and not one the model was
fitted to. Past 30° the curve turns over, as a tuning curve should.

There is also a hard rectification the "see-saw" description does not have:
DNa02_L is *exactly* zero for every positive bearing and DNa02_R exactly zero
below −30°. In the animal one copy is inhibited as the other is excited; here
the contralateral copy simply falls below threshold and stops.

### The lean: a centred object still says "go right"

An object dead ahead commands **+0.0117**, which is **14% of the strongest turn
anywhere in the sweep**, to the fly's right. At the closed loop's turn gain of
3.0 that is a standing command of +0.035 held for the whole episode.

Three checks locate it, and the first two rule out the obvious suspects.

- **It is not the stimulus.** At bearing 0 the projection lights exactly 156
  columns in each eye. Identical input, asymmetric output.
- **It is not a lopsided reconstruction.** MaleCNS is symmetric in aggregate:
  73,467 neurons on the left against 73,638 on the right, and across 373 cell
  types with 20 or more cells the median left-right asymmetry is exactly zero,
  with only 4 types differing by more than 20%.
- **It is not a resting offset.** With no stimulus at all both DNa02 sit at
  exactly 0.00000.

What is uneven is this particular pathway. **LC4 has 71 cells on the left and 55
on the right**, a 13% imbalance, and LC4 is what carries object position to the
steering neuron. The bias is already present at the LC4 stage (LC4_R − LC4_L =
+0.022 for a centred object) and survives every population statistic we tried —
mean, maximum, and fraction active all show it, the last with the opposite sign.
So it is in the wiring of the visual-to-steering path, not in how that path is
summarised. The controls below show it is not in the *measured* wiring
specifically: a rewired graph leans harder.

### The controls: the proportionality is wiring-specific, the lean is not

| graph | peak turn | centre bias | % of peak | linear R² |
|---|---:|---:|---:|---:|
| **MaleCNS** | 0.0824 | +0.0117 | 14% | **0.845** |
| rewired, degree-preserving | 0.7246 | −0.1587 | 22% | **0.024** |
| signs scrambled | 1.0000 | 0.0000 | 0% | — |

Two separate conclusions, and they point opposite ways.

**The linear tuning needs the measured wiring.** A degree-preserving rewiring
still produces a turn command -- a large one, nine times the peak -- but it has
almost no relationship to where the object is: R² falls from 0.845 to 0.024.
So the agreement with the recorded linear relationship is a property of the
connectome and not of any network with this degree sequence, which is the
strongest form the claim can take here.

**The lean is not.** The rewired control leans harder than the real graph, 22%
against 14%, and in the opposite direction. So "MaleCNS pushes right" overstates
it: networks of this shape lean, and this one happens to lean right by 14%. The
LC4 count imbalance is a plausible contributor, not a demonstrated cause.

The sign-scrambled control is the runaway this project's sign table exists to
prevent: both DNa02 pinned at exactly 1.0, so their difference is exactly zero
at every bearing. A steering readout that reports 0.000 everywhere is not
balanced, it is saturated, and the two look identical in the summary statistic.
That is why `peak_turn` is reported next to the bias.

### Why this matters more than its size

**Every single-sided measurement in this model inherits it.** A 14%-of-full-scale
standing rightward push, sustained across an episode, is the same order as the
effects this project measures. Run 10's mirror-pair design cancels it exactly,
and Run 10 justified that design by the gait's drift alone — the brain had a
second, independent reason for it that we had not yet found.

`flyloop.experiments.bias` measures this, so a future change to the visual
pathway can be checked against it rather than assumed symmetric.

### What this does not show

- **One object size, one hop count.** The tuning curve is for a 5.7° half-width
  probe at 5 synaptic hops. Both change the shape.
- **The lit-column counts are only nearly mirrored** away from centre — 187/24 at
  −30° against 20/188 at +30°, for instance. Those few columns are a small extra
  asymmetry on top of the wiring's, and the report records them so the two can
  be told apart.
- **It says nothing about the real animal.** A 13% left-right difference in an
  LC4 count is as likely to be reconstruction completeness as biology, and
  nothing here can distinguish those. The rewired control leaning harder makes
  the LC4 imbalance a candidate rather than an explanation.
- The linear fit is over one connectome and one probe; R² = 0.845 is a decent
  line through a curve that is visibly not straight, not a demonstration that
  the model is linear.

### Next

- Check whether the FlyWire female brain shows the same sign of lean, which
  would separate specimen from method.
- Sweep the probe's angular size. If the linear range scales with it, the
  controller is reading angular position; if it does not, it is reading which
  columns are lit, which is a weaker claim.

## Run 14 — the same structure in a second animal, after a bug that hid it

**Date** 2026-09-16
**Data** MaleCNS v1.0 (male, brain and nerve cord, CC-BY) against FlyWire v783
(female, brain only, CC BY-NC). Different animals, different sex, different
reconstruction teams, same input-proportion preparation.
**Question** Runs 12 and 13 rest on structural claims about one connectome. Do
they hold in another?

### The bug first, because without it this run said the opposite

The first pass reported **MBON to DNa02 of 0.0000% in FlyWire** against 0.526%
in MaleCNS, which would have meant the junction the whole hybrid is built on is
specific to one animal. It was not a biological difference. It was the loader.

These prepared files do not promise that the metadata CSV is in matrix row
order, and they are not consistent about it. MaleCNS ships sorted, its `idx`
column running 0 to 161,428. **FlyWire does not: its first metadata row is
matrix row 90,908.** A loader that trusts CSV order therefore reads one dataset
correctly and mislabels every neuron in the other, with nothing raising.

What caught it was not a test but a sanity check on the answer. The cell
labelled DNa02 in FlyWire appeared to draw **17.8% of its input from Tm3 and
11.5% from Mi1** -- medulla cells, which cannot plausibly feed a descending
steering neuron -- while the identical query on MaleCNS returned PS049, PFL3 and
the LAL types the literature names. Implausible neighbours, not a crash.

`_in_matrix_order` now reorders by `idx`, refuses a duplicated or non-permuted
one rather than guessing, and six tests pin it, including one that loads the
same miniature dataset in two row orders and requires an identical graph.

### With that fixed, the structure replicates

| quantity | MaleCNS | FlyWire | |
|---|---:|---:|---|
| MBON → DNa02 | 0.5256% | **0.5151%** | replicates |
| MBON → DNa03 | 1.9613% | 2.5338% | replicates |
| MBON → DNp09 | 0.0000% | 0.0000% | replicates exactly |
| **LC4 + LPLC2 → Kenyon cells** | **0.0000%** | **0.0000%** | replicates exactly |
| coupling, direct | 0.5214% | 0.5131% | replicates |
| coupling, within 4 hops | 2.1423% | 2.8876% | replicates |
| indirect / direct ratio | 4.11× | 5.63× | replicates |
| MBON → MDN | 0.4345% | 0.9726% | same order, 2× apart |

The strongest result is the one nothing in the pipeline was told to look for:

| | MaleCNS | FlyWire |
|---|---|---|
| strongest MBON into DNa02 | MBON31, 0.279% | MBON32, 0.276% |
| second | MBON32, 0.219% | MBON31, 0.210% |
| those two, as a share of all MBON input | **98%** | **99%** |

**The same two cell types, at the same strengths, in two independently
reconstructed animals**, carrying essentially all of the mushroom body's direct
access to the steering neuron. Li et al. (*eLife* 2020;9:e62576) name MBON32 and
MBON31 from the hemibrain; this is that result reproduced twice more, from
matrices, with no reference to the paper. The rank order swaps between the two
and the magnitudes differ by under 5%, which is about what two reconstructions
of one circuit should look like.

Run 12's central structural facts therefore are not artefacts of one specimen.
Vision cannot be learned through the mushroom body in either animal -- exactly
zero, twice -- and the learned valence reaches steering through one narrow,
reproducible door.

### What does not replicate: Run 13's lean

| | MaleCNS | FlyWire |
|---|---:|---:|
| LC4, left/right | 71 / 55, **−12.7%** | 54 / 50, **−3.8%** |
| LPLC2, left/right | 94 / 91, −1.6% | 108 / 102, −2.9% |

The LC4 imbalance that Run 13 offered as a candidate for the 14% centre bias is
three times smaller in FlyWire. Together with the rewired control leaning harder
than the real graph, that settles it: **the lean is not a property of fly wiring,
it is a property of this reconstruction and of networks of this shape.** Run 13's
caution was warranted and can now be stated as a conclusion.

### What this does not show

- **No behaviour was run on FlyWire.** It has no hex retinotopy in these files,
  so the tuning curve of Run 13 and every closed-loop result needs MaleCNS. This
  run compares structure only.
- **FlyWire is brain-only.** It stops at the neck, so anything about the nerve
  cord is MaleCNS alone.
- **Two reconstructions are not two independent measurements of nature.** They
  share conventions, cell-type nomenclature and in places the same annotators,
  so agreement bounds reconstruction noise better than it bounds biology.
- The sign table was applied identically to both; a transmitter prediction wrong
  in the same way in both files would replicate happily.

### Next

- The loader now reads FlyWire, so `flyloop info --dataset fafb` and the atlas
  work on it. The obvious follow-up is the sign-source comparison across both.
- FlyWire's own `sign` column against ours, on 139,102 neurons, as a second
  check on the glutamate rule.

## Run 15 — the sign table against 300,000 neurons, and a hole in one dataset

**Date** 2026-09-17
**Data** MaleCNS v1.0 (161,429 neurons) and FlyWire v783 (139,102), each of
which ships its own `sign` column alongside its transmitter prediction.
**Question** `flyloop.connectome.signs` is the most consequential table in the
project. Two datasets assigned signs independently. Do they agree with it?

### The claim that matters is unanimous

| | MaleCNS | FlyWire |
|---|---:|---:|
| glutamatergic neurons | 29,058 | 24,917 |
| their sign, per the dataset | **−1, all of them** | **−1, all of them** |

**53,975 neurons across two reconstructions, no exceptions.** Glutamate is
inhibitory in the adult central brain, which is what Liu & Wilson measured
(*PNAS* 2013;110:10294–10299) and what this project has encoded since its first
commit. The community reimplementations that treat it as excitatory disagree
with both datasets as well as with the literature.

Acetylcholine (+1) and GABA (−1) agree everywhere too.

### Overall agreement, and what the gap is made of

| dataset | agree | disagree | rate |
|---|---:|---:|---:|
| MaleCNS | 153,487 | 7,942 | 95.08% |
| FlyWire | 137,523 | 1,579 | 98.86% |

Every disagreement is one of two things, and neither is a fast-transmission
error.

**Neuromodulators, by our choice.** Both datasets give dopamine, octopamine and
serotonin +1; we give them 0. That is deliberate and documented: the LIF model
has no mechanism for neuromodulation, and pretending it is fast excitation would
be worse than leaving it out. 392 + 101 + 48 neurons in MaleCNS, 559 + 65 + 854
in FlyWire.

**Histamine, where we differ from MaleCNS on substance.** MaleCNS signs its
4,937 histaminergic neurons +1. Histamine in *Drosophila* gates a chloride
channel (HisCl1/ort), so the photoreceptor synapse is sign-inverting, and we
sign it −1. This is the same class of error as the glutamate one, in a dataset's
own column rather than in a reimplementation.

### FlyWire has no histamine at all, and that inverts its visual system

FlyWire's transmitter prediction returns no histamine for any neuron. Its
photoreceptors are labelled as something else:

| | MaleCNS | FlyWire |
|---|---|---|
| R1-6 | histamine (435) | **acetylcholine (8,325)** |
| R7 | histamine (482) | **glutamate (1,340)** |
| R8 | histamine (481) | **acetylcholine (1,324)** |
| all photoreceptors | 3,145, all histamine | 10,989, none histamine |

This is not a curiosity in a corner of the data. **R1-6 supplies 48.1% of L1's
input in FlyWire** — L1 being the lamina cell every visual model starts from. A
sign taken from the transmitter label therefore inverts the first synapse of the
visual system, and inverts the largest one.

`flyloop.connectome.signs.photoreceptor_check` now tests this, and the loader
warns when a dataset fails it. MaleCNS passes; FlyWire warns on load.

**So this project's visual work cannot move to FlyWire without correcting
photoreceptor signs by cell type rather than by transmitter.** Run 14's
structural comparison is unaffected: it touches mushroom body and descending
connectivity, where transmitter labels agree.

### What this does not show

- **Agreeing with a dataset's column is not agreeing with nature.** These
  columns are largely derived from the same kind of transmitter prediction, so
  two datasets can be wrong together — as they are, together, about histamine
  and modulators relative to our table.
- **It does not validate the magnitudes**, only the signs. Every weight in this
  project is still an input proportion times one free gain.
- MaleCNS's optic lobe shows a different picture from FlyWire's at L1: R1-6 is
  not among L1's top inputs there at all, where in FlyWire it is 48%. That is a
  reconstruction coverage difference this run does not attempt to resolve.

### Fixed the same day

`sign_vector` now takes an optional cell-type column and lets identity override
the transmitter label -- **for photoreceptors only**, because they are the one
population whose transmitter is beyond question from the name alone. A general
licence to sign cells by their names would be a far worse bug than the one it
fixes.

After it, FlyWire's 9,073 photoreceptor-to-L1 edges are all inhibitory, where
before they were all excitatory. MaleCNS is untouched, which is the point: its
labels were already right, and its headline numbers are unchanged to six
decimals.

The pattern was checked for false positives rather than assumed safe. It matches
10 type names in MaleCNS and 3 in FlyWire, and all 13 are genuine photoreceptor
subtypes: `R1-R6`, `R7d/p/y`, `R8d/p/y` and the `_unclear` variants. Types like
`Rostrum` and `RIM` do not match.

### Next

- Check whether the `unclear` transmitter class in MaleCNS (2,464 neurons, sign
  0 here) concentrates anywhere that matters.
