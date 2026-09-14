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
