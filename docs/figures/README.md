# Figures

`run-log.html` is the visual summary of the run log — six panels, one per
finding, each carrying its control. Published as an artifact; the file here is
the source.

## Regenerating the data

`structure.json` is recomputed from the connectome and must not be edited by
hand:

```bash
flyloop --data-root ~/connectome_data_prep info   # confirm the graph first
python -c "
from flyloop.connectome.data_prep import load_dataset
from flyloop.experiments.olfactory import mbon_learning_vs_output
c = load_dataset('~/connectome_data_prep', 'malecns', matrix='inprop')
print(mbon_learning_vs_output(c).head())
"
```

`figures.json` merges that with the measured run results. Those come from
`docs/RESULTS.md` and are **transcribed, not recomputed** — each run takes
minutes on a CPU, and the log is the record of what was measured. If a run is
repeated and its numbers move, update both files together or the page will
disagree with the log.

## Palette

The categorical colours are the validated default from the `dataviz` skill,
checked with its own validator against both surfaces used here
(`#f7f8f7` light, `#141715` dark). Two light-mode slots sit below 3:1 contrast,
so every chart ships direct labels and a table view — that relief is required,
not optional. Do not substitute colours without re-running the validator.

## Run 10 — the embodied figure

`embodied.png` is regenerated from measured data, not transcribed:

```bash
MUJOCO_GL=disable flyloop --data-root ~/connectome_data_prep \
    embodied --control --out docs/figures/embodied/summary.csv
python docs/figures/plot_embodied.py
```

`embodied/summary.csv` and `embodied/trajectories.csv` are the raw run output.
Each mirror pair costs about 11 minutes of CPU (30 control steps x 2 episodes,
~5.4 s per step: 3.5 s of rate model and 1.8 s of physics), so the full figure
with its control is roughly an hour.

The bearing panel folds left-target episodes onto the right before averaging.
That is only legitimate because the design is a mirror pair by construction —
the two episodes share a body seed and differ solely in the sign of the
stimulus. Do not reuse that fold on data that is not paired.
