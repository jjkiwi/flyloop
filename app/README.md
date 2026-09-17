# Episode player

A browser view of one recorded run: the fly walking on the left, its brain on
the right, both scrubbing on one timeline.

```bash
flyloop --data-root ~/connectome_data_prep serve
# then open http://127.0.0.1:8000/
```

That serves the viewer **and** a run endpoint, so the panel at the top can hand
the fly a stimulus you chose and show you what it did. A plain static server
works too (`python -m http.server 8000 --directory app`) but only replays
episodes recorded earlier; the run button will say so rather than hang.

Either way it needs a server. Opening `index.html` from the filesystem fails --
ES modules and `fetch` both refuse `file://`.

## Handing it a stimulus

Set the object's bearing, distance and size, choose whether an odour is in the
air, how many conditioning trials come first, and press run. About **10 seconds**
later the page reloads on the result.

The first run costs about 30 s because it loads the connectome and builds the
rate model. Every run after that is ~10 s: the model is kept alive and only the
learning is reset, since neither the wiring nor the MBON ensemble depends on
what the fly has been taught.

`body: physics` swaps in NeuroMechFly and costs about 2 s per control step. The
form estimates the wait before you commit to it.

**Nothing about this is live.** A control step costs 0.29 s of wall clock to
simulate 0.05 s even on the fast body. What the endpoint buys is not real time,
it is the ability to change something and look again.

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

`--gain N` records the same run with the learned coupling scaled, and
`--coupling-hops N` changes how far back the MBON share of DNa02's input is
traced -- 1 is the direct connection (0.53%), 4 collects the indirect routes
(2.14%, and see Run 12 for why those are the larger ones). Episodes are named
`gain<N>h<hops>` and the picker labels both, because two runs at the same gain
with different hop counts are different experiments and "gain 50,000x" twice in
a list would hide that.

The four episodes worth having:

```bash
for spec in "1 1" "1 4" "70000 1" "50000 4"; do
  set -- $spec
  flyloop --data-root ~/connectome_data_prep record --physics --out app/data       --gain $1 --coupling-hops $2
done
```

The two at gain 1 are the anatomy as measured, on either coupling, and both do
nothing visible. The other two are where each coupling first bends the path.

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


## One file you can email

```bash
python app/build_single.py app/data flyloop-demo.html
```

Folds the viewer, three.js and two recorded episodes into a single ~4 MB HTML
file that opens by double-clicking. No server, no network, no install.

It replays; it cannot run a new stimulus, because there is no Python behind it.
The panel is removed rather than left there greyed out.

Two things about `file://` shape how it is built, and both are worth knowing
before editing `build_single.py`:

- **A page opened from disk cannot fetch its neighbours.** Every file is its own
  origin. So the assets are gzipped, base64'd and inlined, and `js/data.js`
  checks `globalThis.__FLYLOOP_ASSETS` before reaching for the network. Served
  normally that map is absent and nothing changes.
- **A module script cannot import from `file://` either**, but an *inline*
  module with no imports is allowed. So everything is concatenated into one:
  three.js and OrbitControls each inside their own function, because three
  exports a class called `Controls` and OrbitControls imports one, and flat
  concatenation makes that a redeclaration that fails to parse.

The meshes are decimated to 59,045 triangles from 502,781 for this build --
`export_rig(decimate_to=60000)` -- which takes `meshes.bin` from 6.7 MB to 1.0.
The fly is visibly faceted up close and fine at the distance the camera sits.
