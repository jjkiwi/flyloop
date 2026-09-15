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
