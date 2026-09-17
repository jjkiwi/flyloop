# What the literature says about the claims this project makes

Four load-bearing claims were checked against published work in September 2026.
Two came back with the science confirmed and the **citation wrong**, which is
the reason this file records who verified what rather than just listing papers.

## Verified here, against the sources

### DNa02 steers by its left-right difference — confirmed, exactly

Rayshubskiy, Holtz & Wilson, *eLife* 102230 (first posted as bioRxiv
2020.04.04.024703). Dual recordings from both copies of DNa02 show the fly's
**rotational velocity is linearly proportional to the right-left difference in
DNa02 activity** — a "see-saw", where excitation of one copy is accompanied by
inhibition of the contralateral one.

This project's readout is `turn = DNa02_R - DNa02_L`, used unchanged since its
first run. It is the published relationship, with the same functional form, so
it should not be described as an assumption of ours. Related: DNa02 receives
direct input from PFL3 cells, which compare head direction against a goal.

Two limits on leaning on it:
- DNa02 sits **below DNa03 and LAL013** in the steering hierarchy (Westeinde
  et al.), so reading DNa02 alone reads one level of a stack.
- Yang et al., *Cell* 2024 (`Fine-grained descending control of steering in
  walking Drosophila`) report structure a scalar turn cannot express: DNa02
  shortens strides on the *inside* of a turn.

**Commonly miscited as Current Biology 2020.** It is eLife.

### Glutamate is inhibitory in the adult central brain — confirmed

Liu & Wilson, *PNAS* 2013;110:10294–10299, "Glutamate is an inhibitory
neurotransmitter in the Drosophila olfactory system". Iontophoresed glutamate
hyperpolarises every major antennal-lobe cell type; the effect is abolished by
RNAi knockdown of GluCl-alpha. A third of antennal-lobe local neurons are
glutamatergic.

**Commonly miscited as Nature Neuroscience 16:966–973.** It is PNAS.

**Both datasets agree, unanimously.** MaleCNS and FlyWire each ship a `sign`
column, and across **53,975 glutamatergic neurons in the two of them together**
every single one is −1. See Run 15.

**Why reimplementations get the sign backwards.** At the larval neuromuscular
junction glutamate *is* excitatory, via GluRII. A project that flips the sign is
usually importing the neuromuscular fact into the adult central brain.
`flyloop/connectome/signs.py` encodes the central-brain fact and a test pins it.

### The mushroom body connectome paper exists and covers MBON output

Li F. et al., *eLife* 2020;9:e62576, "The connectome of the adult Drosophila
mushroom body provides insights into function" (Janelia, Columbia and others).

## Relayed, not independently verified here

These came from a literature search run outside this project. The science is
plausible and consistent with what we measure, but the sources were not read
directly, so treat the specifics as second-hand.

- **DNp04 is a looming-responsive descending neuron.** Namiki et al., *eLife*
  2018, is cited as showing DNp04 innervating the whole LC4 glomerulus and
  projecting to the lower tectulum; DNp02 + DNp04 coactivation is reported to
  produce a slower backward escape. No published *ranking* of looming DNs by
  lateralisation exists, so our "DNp04 ranks highest" is a new observation
  rather than a replication.
- **DNa02 receives direct input from MBON32 and MBON31** (Li et al. 2020,
  Fig. 25), described as weaker than the DNa03 route, with no percentage
  published.

## Where our numbers meet theirs

| claim | literature | measured here |
|---|---|---|
| DNa02 turn signal | linear in right − left | `turn = DNa02_R - DNa02_L` |
| MBON → DNa02 direct | "weaker", no number | **0.526%** of input |
| which MBONs | MBON31, MBON32 | those two are **0.498% of the 0.526%** |
| main MB → steering route | indirect, via DNa03 | MBON → DNa03 **1.96%**, DNa03 → DNa02 **1.16%** |
| glutamate sign | inhibitory (GluCl-alpha) | −1, pinned by a test |

Two of these are worth stating plainly.

**The MBON31/32 result is an independent confirmation, now twice over.** Nothing
in our pipeline knows that paper. Ranking MBON types by their input share to
DNa02 puts MBON31 and MBON32 on top in **both** connectomes:

| | MaleCNS (male) | FlyWire (female) |
|---|---|---|
| strongest | MBON31, 0.279% | MBON32, 0.276% |
| second | MBON32, 0.219% | MBON31, 0.210% |
| share of all MBON input | 98% | 99% |

Li et al. name the pair from the hemibrain; two further reconstructions agree,
with the rank order swapped and the magnitudes within 5%. See Run 14.

**The indirect routes are the larger ones, but DNa03 is not their carrier.**
`flyloop/brain/coupling.py` now traces the input backwards as well
(`coupling_hops=4`), and the collected indirect share is 2.14% against 0.52%
direct -- 4.1x, which is the direction the relayed claim pointed. Ranking the
actual two-step paths, though, puts **LAL051, LAL171, LAL172 and LAL173** on top
and **DNa03 eighth**, at 0.0226%. The lateral accessory lobe is the premotor hub
DNa02 sits downstream of, so this is not at odds with the steering literature;
it is at odds with the specific relayed sentence naming DNa03. Since that
sentence was summarised for us rather than read at source, the disagreement may
be in the relay. Recorded in Run 12 with the numbers.

## Provenance

Sources for the verified section were searched and read during the session of
2026-09-16. Anything in "relayed" was supplied as a summary and is marked as
such deliberately: two of the four citations in that summary had the wrong
journal, and the same check has not been done on the rest.
