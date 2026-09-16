// Reading the four files flyloop writes, exactly as their writers lay them out.
//
//   rig.json      flyloop/viz/fly_geometry.py: export_rig()
//   meshes.bin    per mesh, vertexCount*3 float32 positions then indexCount
//                 uint32 indices, both at byte offsets recorded in rig.json
//   episode.json  flyloop/app/episode.py: Episode.save()
//   episode.bin   one chunk per array, 4-byte aligned, dtype and offset in
//                 manifest.arrays
//   atlas.json /  write_atlas(): int16 quantised soma xyz, then a uint16 index
//   atlas.bin     into type_names
//
// Nothing here guesses a field name: every key used below appears in one of
// those two writers.

export const DATA = "./data/";

// ------------------------------------------------------------- float16
//
// type_activity is float16 and there is no Float16Array in engines older than
// 2025, so the half words are decoded by hand. A 64k-entry lookup table costs
// 256 kB once and turns the per-frame decode into a table read, which matters:
// a frame is 7,653 values and the colour update touches 138,496 points.

function halfToFloat(h) {
  const sign = (h & 0x8000) ? -1 : 1;
  const exp = (h >> 10) & 0x1f;
  const frac = h & 0x03ff;
  if (exp === 0) return sign * Math.pow(2, -24) * frac;          // subnormal
  if (exp === 0x1f) return frac ? NaN : sign * Infinity;         // Inf / NaN
  return sign * Math.pow(2, exp - 25) * (0x400 + frac);          // normal
}

const HALF = new Float32Array(65536);
for (let i = 0; i < 65536; i++) HALF[i] = halfToFloat(i);

export function decodeHalf(u16, out) {
  const dst = out && out.length === u16.length ? out : new Float32Array(u16.length);
  for (let i = 0; i < u16.length; i++) dst[i] = HALF[u16[i]];
  return dst;
}

/** Cases with a known answer, checked at startup so a broken decode is loud. */
export function halfSelfTest() {
  const cases = [
    [0x0000, 0], [0x8000, -0], [0x3c00, 1], [0xbc00, -1], [0xc000, -2],
    [0x4000, 2], [0x3800, 0.5], [0x7bff, 65504], [0x0400, 6.103515625e-5],
    [0x0001, 5.960464477539063e-8], [0x3555, 0.333251953125],
    [0x2e66, 0.09997558593750], [0xb266, -0.1998291015625],
  ];
  const fails = [];
  for (const [bits, want] of cases) {
    const got = HALF[bits];
    if (!(Math.abs(got - want) <= Math.abs(want) * 1e-7)) fails.push(`0x${bits.toString(16)}: ${got} != ${want}`);
  }
  if (!Number.isNaN(HALF[0x7e00])) fails.push("0x7e00 should decode to NaN");
  if (HALF[0x7c00] !== Infinity) fails.push("0x7c00 should decode to +Infinity");
  if (HALF[0xfc00] !== -Infinity) fails.push("0xfc00 should decode to -Infinity");
  return { pass: fails.length === 0, fails };
}

// --------------------------------------------------------------- fetch

async function getJSON(name) {
  const r = await fetch(DATA + name);
  if (!r.ok) throw new Error(`${DATA}${name}: HTTP ${r.status}`);
  return r.json();
}

async function getBuffer(name) {
  const r = await fetch(DATA + name);
  if (!r.ok) throw new Error(`${DATA}${name}: HTTP ${r.status}`);
  return r.arrayBuffer();
}

const VIEWS = {
  float32: Float32Array, float16: Uint16Array, uint32: Uint32Array,
  int16: Int16Array, uint16: Uint16Array, int32: Int32Array, float64: Float64Array,
};

/** One entry of a manifest `arrays` index, as a typed array over the buffer.
 *  float16 comes back as the raw Uint16Array: decode it with decodeHalf. */
function arrayView(index, buf, name) {
  const e = index[name];
  if (!e) throw new Error(`array ${name} is not in the manifest`);
  const Ctor = VIEWS[e.dtype];
  if (!Ctor) throw new Error(`array ${name} has dtype ${e.dtype}, which this reader does not know`);
  const count = e.shape.reduce((a, b) => a * b, 1);
  if (e.offset % Ctor.BYTES_PER_ELEMENT) {
    throw new Error(`array ${name} starts at byte ${e.offset}, not aligned for ${e.dtype}`);
  }
  return new Ctor(buf, e.offset, count);
}

// ---------------------------------------------------------------- load

/** The recorded episodes, newest format first. Rig and atlas are shared. */
export async function listEpisodes() {
  const idx = await getJSON("episodes.json");
  return idx.episodes;
}

export async function load(onStatus = () => {}, epPath = "") {
  onStatus("rig.json + meshes.bin");
  const [rig, meshBuf] = await Promise.all([getJSON("rig.json"), getBuffer("meshes.bin")]);

  onStatus("episode.json + episode.bin");
  const [episode, epBuf] = await Promise.all([
    getJSON(epPath + "episode.json"), getBuffer(epPath + "episode.bin"),
  ]);

  onStatus("atlas.json + atlas.bin");
  const [atlas, atlasBuf] = await Promise.all([getJSON("atlas.json"), getBuffer("atlas.bin")]);

  const frames = episode.frames;

  // Scalars. Every column the writer emitted is here; which ones exist depends
  // on the body, so read what the manifest lists rather than a fixed list.
  const scalars = {};
  for (const key of Object.keys(episode.arrays)) {
    if (key.startsWith("scalar/")) scalars[key.slice(7)] = arrayView(episode.arrays, epBuf, key);
  }

  // type_activity: (frames, live types) float16, decoded per frame on demand.
  const taEntry = episode.arrays["type_activity"];
  const taRaw = arrayView(episode.arrays, epBuf, "type_activity");
  const nTypes = taEntry.shape[1];

  // joints: (frames, 42) float32, or absent on the kinematic stub. Rows for the
  // training phase are all-NaN by construction -- the fly is on a rig.
  const joints = episode.has_joints ? arrayView(episode.arrays, epBuf, "joints") : null;
  const nJoints = joints ? episode.arrays["joints"].shape[1] : 0;

  // Atlas: quantised int16 positions, back to the dataset's own voxel space.
  const nSoma = atlas.neurons;
  const qpos = arrayView(atlas.arrays, atlasBuf, "position");
  const typeIndex = arrayView(atlas.arrays, atlasBuf, "type_index");
  const off = atlas.quantised_offset;   // -32767
  const position = new Float32Array(nSoma * 3);
  for (let i = 0; i < nSoma; i++) {
    for (let k = 0; k < 3; k++) {
      position[i * 3 + k] = (qpos[i * 3 + k] - off) * atlas.scale[k] + atlas.origin[k];
    }
  }

  // The two type vocabularies are different sets: the atlas names every type
  // with a located soma (11,384), the episode names only the types that were
  // live during the run (7,653 of 11,751). They are matched BY NAME; a soma
  // whose type has no episode column keeps no value and is drawn grey.
  const epIndexByName = new Map(episode.type_names.map((n, i) => [n, i]));
  const atlasToEpisode = new Int32Array(atlas.type_names.length);
  let matchedTypes = 0;
  for (let i = 0; i < atlas.type_names.length; i++) {
    const j = epIndexByName.has(atlas.type_names[i]) ? epIndexByName.get(atlas.type_names[i]) : -1;
    atlasToEpisode[i] = j;
    if (j >= 0) matchedTypes++;
  }
  let matchedSomata = 0;
  for (let i = 0; i < nSoma; i++) if (atlasToEpisode[typeIndex[i]] >= 0) matchedSomata++;

  const frameActivity = new Float32Array(nTypes);
  function activityAt(frame) {
    const base = frame * nTypes;
    for (let i = 0; i < nTypes; i++) frameActivity[i] = HALF[taRaw[base + i]];
    return frameActivity;
  }

  const phaseOf = (frame) => episode.phases.find((p) => frame >= p.start && frame < p.stop) || null;

  return {
    rig, meshBuf, episode, atlas,
    frames, scalars, joints, nJoints, nTypes,
    soma: { n: nSoma, position, typeIndex, atlasToEpisode, matchedTypes, matchedSomata },
    activityAt, phaseOf, raw: { taRaw },
  };
}
