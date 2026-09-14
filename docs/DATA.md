# Getting real connectome data

Nothing here is bundled: the datasets are large and carry their own licences.

## Which dataset

|  | MaleCNS v1.0 | FlyWire v783 |
|---|---|---|
| animal | adult male | adult female |
| coverage | brain **+ ventral nerve cord** | brain only |
| neurons | 166,700 | 139,255 |
| leg motor neurons | yes | **no** |
| licence | CC-BY | CC BY-NC |
| released | June 2026 | 2024 |

**For an embodied project, use MaleCNS.** FlyWire stops at the neck. A FlyWire
model can produce a descending command but there is nothing downstream of it to
move a leg, so the last stage has to be invented -- which defeats the point.

## MaleCNS

Site: <https://male-cns.janelia.org/> (explore, download, cite).

Whole-CNS work wants the CSV export that backs the neuPrint database, not live
queries:

```python
from flyloop.connectome.malecns import from_dump
c = from_dump("data/malecns-v1.0", min_synapses=5)
c.save("data/cx-malecns")
print(c.report())
```

For a named subcircuit, query neuPrint directly. This needs an auth token from
your neuPrint account:

```python
from neuprint import NeuronCriteria
from flyloop.connectome.malecns import from_neuprint
c = from_neuprint(criteria=NeuronCriteria(type="LC4"), token="...")
```

`from_neuprint` refuses a criteria-less call on purpose: pulling the whole CNS
over HTTP will not work.

## FlyWire

Export the v783 data products from Codex: <https://codex.flywire.ai/api/download>.
Put `classification.csv.gz` and `connections.csv.gz` in one directory:

```python
from flyloop.connectome.flywire import load_flywire
c = load_flywire("data/flywire-783", min_synapses=5)
```

`min_synapses=5` is the conventional FlyWire threshold and removes most
reconstruction noise. Lower it only deliberately.

## Check the data before trusting it

Always, on every fresh load:

```bash
flyloop --connectome data/cx-malecns info
```

Three things to look at in the report:

1. **Unsigned neurons.** These are neurons whose transmitter this project does
   not recognise, or which are neuromodulatory. Their *outgoing synapses do
   nothing*. A large fraction means much of the graph is silently disconnected.
2. **E/I synapse ratio.** Wildly lopsided means the sign mapping is wrong for
   this dataset's transmitter labels.
3. **Unrecognised transmitter labels.** Listed explicitly. Map them in
   `flyloop/connectome/signs.py` rather than letting them default to zero.

## Retinotopy, the part that bites

`ColumnMap.from_connectome` needs to know which neuron each ommatidium drives.
The synthetic fixture makes this trivial. Real data does not: you need the
dataset's own optic-lobe column annotations, and without them the mapping is
arbitrary.

An arbitrary retinotopy still spikes, still produces commands and still looks
like it works. Nothing crashes. Every motion computation downstream is simply
wrong. `ColumnMap` raises rather than guessing when the counts do not line up;
do not work around that by truncating until the shapes match.

## Stage 4: the biomechanical body

```bash
pip install 'flyloop[body]'     # FlyGym + MuJoCo
```

FlyGym returns per-ommatidium intensities directly, already in the fly's
retinotopic coordinates. Use `FlyGymBody.ommatidia()`, not `observe()` through
`CompoundEye` -- resampling them a second time scrambles the retinotopy you just
got for free.

## Retinotopy, solved

The section above said retinotopy is the part that bites. It is, but the data
exists and someone has already packaged it.

`connectome-interpreter` (MIT) bundles two published columnar tables:

| dataset | source | columns | gives you |
|---|---|---|---|
| `mcns_right` | Nern et al. 2024 | **892** | body ID of each column's L1, L2, L3, L5, Mi1, Mi4, Mi9, C2, C3, Tm1, Tm2, Tm4, Tm9, Tm20, T1 in MaleCNS |
| `fafb_right` | Matsliah et al. 2024 | 796 | the same for the right optic lobe of FlyWire |

```bash
pip install 'flyloop[data]'
flyloop columns --dataset mcns_right
```

```python
from flyloop.vision import ColumnMap, CompoundEye
m = ColumnMap.from_columnar_table(connectome, CompoundEye(892), cell_type="L1")
```

892 is also the number of retinotopic columns the ommatid robot samples its
camera onto, which is a good sign this is the table everyone doing this ends up
using.

Three things to know before relying on it:

1. **Right optic lobe only.** There is no left-eye table, and the left eye's
   column identities cannot be derived from the right one. `ColumnMap` therefore
   returns a *monocular* map by default. `mirror=True` fills the left eye by
   rank order on the assumption of developmental symmetry; it is recorded in
   `map.assumptions` so a result can declare it. Do not use it for anything that
   depends on binocular geometry.
2. **The file has 920 rows for 892 columns.** Twenty-eight columns are listed
   twice as exact duplicates. They are collapsed on load; left in, two ommatidia
   would share an input neuron.
3. **Pick the input cell type deliberately.** L1 and L2 are the large monopolar
   cells carrying the ON and OFF pathways out of the lamina. L1 is the closest a
   connectome-only model gets to "light lands here".

The extra is heavy for what it delivers -- `connectome-interpreter` pulls in
torch and CUDA wheels for the sake of two small CSVs. If that becomes a problem,
the tables are MIT-licensed and can be vendored with attribution.

## The route that actually works from a restricted network

`neuprint.janelia.org`, `male-cns.janelia.org`, `codex.flywire.ai`, Zenodo and
Hugging Face are all unreachable from some sandboxes. GitHub usually is not.

[`YijieYin/connectome_data_prep`](https://github.com/YijieYin/connectome_data_prep)
publishes the major fly connectomes already reduced to a `scipy.sparse` matrix
plus a metadata CSV — MaleCNS, FAFB/FlyWire, BANC, hemibrain, MANC and the
larva — and it is a plain git clone:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 \
    https://github.com/YijieYin/connectome_data_prep ~/connectome_data_prep   # 3.7 GB
flyloop --data-root ~/connectome_data_prep info
```

```python
from flyloop.connectome.data_prep import load_dataset
c = load_dataset("~/connectome_data_prep", "malecns", min_synapses=5)
```

MaleCNS loads in about six seconds: 161,429 neurons, 5,988,538 connections and
86,516,520 synapses at `min_synapses=5`.

### Three things to check on this data

**Matrix orientation is `(pre, post)`, and it is verifiable.** The columns of
`inprop` sum to 1.0 for 99.8% of neurons, because that file normalises by the
*recipient's* total input. It matches `Connectome.W` with no transpose. Check it
rather than assuming it, for any dataset.

**The signs disagree with ours on 5% of neurons.** The datasets apply one rule:
glutamate and GABA inhibitory, everything else excitatory. This project treats
histamine as inhibitory — it gates a chloride channel in the fly visual system
exactly as glutamate does — and gives neuromodulators no fast sign at all.

| transmitter | flyloop | dataset | neurons |
|---|---:|---:|---:|
| histamine | −1 | +1 | 4,937 |
| unclear | 0 | +1 | 2,464 |
| dopamine | 0 | +1 | 392 |
| octopamine | 0 | +1 | 101 |
| serotonin | 0 | +1 | 48 |

The histamine row is all the photoreceptors, so this choice lands directly on
vision. `sign_report()` prints the table and `--sign-source dataset` switches
to theirs. Decide it; do not inherit it.

**`min_synapses` is a real filter.** At 5, a quarter of the connections survive
but 72% of the synapses do. Most edges are one or two synapses.

### Retinotopy comes free with this metadata

The prepared MaleCNS metadata carries `assignedOlHex1`/`assignedOlHex2` for
23,720 optic-lobe neurons on **both** sides — 892 distinct columns for the right
eye, 875 for the left. That is better than the published columnar tables, which
are right-eye only:

```python
from flyloop.vision import ColumnMap, CompoundEye
m = ColumnMap.from_hex_metadata(c, CompoundEye(721), cell_type="L1")
assert not m.monocular
```

Use `from_hex_metadata` when the connectome has the hex columns, and fall back
to `from_columnar_table` when it does not.
