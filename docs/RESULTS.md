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
