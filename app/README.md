# Episode player

A browser view of one recorded run: the fly walking on the left, its brain on
the right, both scrubbing on one timeline.

```bash
python -m http.server 8000 --directory app
# then open http://127.0.0.1:8000/
```

It needs a static server. Opening `index.html` from the filesystem fails --
ES modules and `fetch` both refuse `file://`.

## Regenerating the data

`app/data/` is not in git: it is 8.9 MB of binaries that the pipeline rebuilds.

```bash
flyloop --data-root ~/connectome_data_prep record --physics --out app/data
python -c "
from flyloop.viz.fly_geometry import export_rig
export_rig('app/data')"
```

The first writes `episode.{json,bin}` and `atlas.{json,bin}`; the second writes
`rig.json` and `meshes.bin`. A 40-step physics episode takes about 90 seconds,
almost all of it MuJoCo.

`--gain N` records the same run with the learned coupling scaled. The player
reads the gain out of the manifest and says so on screen, so an amplified run
cannot be mistaken for the measured one.

## What the panels are, and are not

**The fly is posed, not animated.** Each frame sets 42 joint angles from the
log and solves the exported rig's forward kinematics, which agrees with MuJoCo
to 8.9e-16 mm. Nothing is interpolated or eased.

**The brain panel is cell bodies.** The prepared MaleCNS files carry one spatial
number per neuron -- the soma position -- and no skeletons, so this is a cloud of
somata on the rind around the neuropil, not the rainbow neurite render it
resembles. Every soma is coloured by its *cell type's* activation, because
per-neuron activity would be 194 MB an episode against 2.6 MB per type.

**Playback is not simulation.** One control step costs about 1.99 s of wall
clock to simulate 0.05 s. There is no live mode and there is not going to be one.

## Things the format made us handle

- **Training frames have no pose.** `joints`, `x`, `y` and `theta` are NaN for
  them: the fly is on a rig, not walking. The player hides the body and shows
  the mushroom body instead. Posing a fly from NaN draws a corpse sliding across
  the floor and reads as a physics bug.
- **The two type vocabularies differ.** The atlas names every type with a
  located soma (11,384); the episode names only types that were live in the run
  (7,653 of 11,751). They are matched by name, and unmatched somata stay dark.
- **float16 needs decoding by hand.** `type_activity` is half-precision and
  there is no `Float16Array` to rely on, so `js/data.js` builds a 65,536-entry
  lookup once and has a self-test that runs at startup.
- **Three.js is vendored.** `cdn.jsdelivr.net` is unreachable from the machine
  this was built on (the egress proxy answers 000), so a CDN importmap would
  have shipped a viewer nobody had ever seen render. `vendor/` holds the
  upstream r169 files unmodified.
