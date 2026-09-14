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
