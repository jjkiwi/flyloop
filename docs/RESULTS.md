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
