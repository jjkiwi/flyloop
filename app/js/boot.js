/**
 * Entry point: load the recording, build the two scenes, wire the timeline.
 *
 * Everything here is playback. The episode was computed offline because a
 * control step costs about 1.99 s of wall clock to simulate 0.05 s, so there is
 * no version of this that runs live, and pretending otherwise would be the one
 * dishonest thing a viewer of this project could do.
 */
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { load, listEpisodes, halfSelfTest } from "./data.js";
import { Rig } from "./fk.js";

const $ = (id) => document.getElementById(id);
const boot = $("boot");
const bootStatus = $("boot-status");

/** Sequential ramp for activation. Dark background, so it starts near black. */
function ramp(v, out, i) {
  const t = Math.max(0, Math.min(1, v));
  // black -> indigo -> cyan -> warm white, chosen to stay legible where most
  // of the population sits (low values) instead of spending the range on peaks.
  const r = t < 0.5 ? 0.25 * t : 0.5 * t * t + 0.5 * (t - 0.5);
  const g = t < 0.5 ? 0.9 * t * t : 0.45 + 1.1 * (t - 0.5);
  const b = t < 0.5 ? 0.35 + 1.1 * t : 0.9 - 0.25 * (t - 0.5);
  out[i] = Math.min(1, r * 1.6);
  out[i + 1] = Math.min(1, g);
  out[i + 2] = Math.min(1, b);
}

main().catch((err) => {
  bootStatus.textContent = "failed";
  const pre = $("boot-error");
  pre.hidden = false;
  pre.textContent = String(err && err.stack ? err.stack : err);
  console.error(err);
});

async function main() {
  halfSelfTest();
  const episodes = await listEpisodes();
  const want = new URLSearchParams(location.search).get("ep");
  const chosen = episodes.find((e) => e.name === want) || episodes[0];
  // Switching episodes reloads the page rather than rebuilding the scenes.
  // meshes.bin and atlas.bin are served with the same URL either way, so the
  // browser cache makes the swap cheap and the code stays honest about what is
  // being reloaded.
  const pick = $("episode-pick");
  // Label with the coupling as well as the gain: two episodes at the same gain
  // but different hop counts are different experiments, and "gain 50,000x"
  // twice in a list would hide that.
  pick.innerHTML = episodes
    .map((e) => {
      const hops = e.coupling_hops || 1;
      const route = hops === 1 ? "direct" : `${hops} hops`;
      const pct = e.coupling_share ? ` ${(e.coupling_share * 100).toFixed(2)}%` : "";
      return `<option value="${e.name}"${e === chosen ? " selected" : ""}>` +
             `gain ${e.gain.toLocaleString()}\u00d7 \u00b7 ${route}${pct}</option>`;
    })
    .join("");
  pick.addEventListener("change", () => {
    location.search = `?ep=${encodeURIComponent(pick.value)}`;
  });

  const D = await load((s) => (bootStatus.textContent = s), chosen.path);
  window.__flyloop = D; // for poking at from the console

  const man = D.episode;
  $("provenance").textContent =
    `${man.connectome}, ${man.neurons.toLocaleString()} neurons — ` +
    `${man.body}, ${man.frames} frames at ${man.control_dt}s, ` +
    `train odour "${man.train_odour}", behaving in "${man.behave_odour}"`;

  $("gain-value").textContent = man.gain.toFixed(man.gain < 10 ? 1 : 0) + "×";
  $("gain-note").textContent =
    man.gain === 1
      ? `the anatomy as measured (MBON→DNa02 is ${(man.coupling_share * 100).toFixed(2)}% of input). ` +
        `The learned bias moves the approach drive by a fraction of a percent, which is real and invisible. ` +
        `That is the result, not a bug.`
      : `${man.gain}× the measured MBON→DNa02 coupling. Anything visible here is amplified.`;
  if (man.gain !== 1) $("badge-gain").classList.add("warn");

  const world = buildWorld(D);
  const brain = buildBrain(D);
  const traces = buildTraces(D);

  wireTimeline(D, (frame) => {
    world.show(frame);
    brain.show(frame);
    traces.show(frame);
    updateHud(D, frame);
  });

  wireStimulus(D);

  boot.classList.add("gone");
  setTimeout(() => boot.remove(), 400);
}

/* ------------------------------------------------------------------ world */

function buildWorld(D) {
  const canvas = $("world-canvas");
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0a0b10);
  scene.fog = new THREE.Fog(0x0a0b10, 40, 160);

  // MuJoCo is Z-up and so is the exported rig; tell three rather than rotating
  // every vector into a Y-up world.
  const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 500);
  camera.up.set(0, 0, 1);
  camera.position.set(-16, -30, 20);
  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;

  scene.add(new THREE.AmbientLight(0xffffff, 0.55));
  const key = new THREE.DirectionalLight(0xfff0dd, 1.5);
  key.position.set(-20, -30, 40);
  scene.add(key);
  const rim = new THREE.DirectionalLight(0x88aaff, 0.7);
  rim.position.set(30, 20, 10);
  scene.add(rim);

  const grid = new THREE.GridHelper(120, 60, 0x4a5570, 0x232a3a);
  grid.rotation.x = Math.PI / 2;
  scene.add(grid);

  // The target, at the position and radius the run actually used.
  const t = D.episode.target;
  const target = new THREE.Mesh(
    new THREE.CylinderGeometry(t.radius, t.radius, 8, 32),
    new THREE.MeshStandardMaterial({ color: 0x9a5cff, roughness: 0.5, emissive: 0x2a1050 })
  );
  target.rotation.x = Math.PI / 2;
  target.position.set(t.x, t.y, 4);
  scene.add(target);

  // The fly: one mesh per geom, parented to a group per body so the solved
  // forward kinematics can be written straight onto the groups.
  const rig = new Rig(D.rig);
  const bodyGroups = D.rig.bodies.map(() => new THREE.Group());
  const flyRoot = new THREE.Group();
  const material = new THREE.MeshStandardMaterial({
    color: 0xd9c08a, roughness: 0.62, metalness: 0.08, flatShading: false,
  });
  const geoms = meshGeometries(D);
  D.rig.bodies.forEach((b, i) => {
    for (const g of b.geoms) {
      const geo = geoms.get(g.mesh);
      if (!geo) continue;
      const m = new THREE.Mesh(geo, material);
      m.position.fromArray(g.pos);
      m.quaternion.fromArray(g.quat);
      bodyGroups[i].add(m);
    }
    flyRoot.add(bodyGroups[i]);
  });
  scene.add(flyRoot);

  // Where the fly has been, drawn up to the current frame only.
  const path = new THREE.Line(
    new THREE.BufferGeometry().setAttribute(
      "position", new THREE.BufferAttribute(new Float32Array(D.frames * 3), 3)
    ),
    new THREE.LineBasicMaterial({ color: 0x5ce0b0 })
  );
  scene.add(path);

  let follow = true;
  $("follow").addEventListener("change", (e) => (follow = e.target.checked));
  $("view-arena").addEventListener("click", () => {
    follow = false; $("follow").checked = false;
    camera.position.set(-10, -34, 26); controls.target.set(t.x / 2, t.y / 2, 0);
  });
  $("view-fly").addEventListener("click", () => {
    follow = true; $("follow").checked = true;
  });

  function resize() {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
  new ResizeObserver(resize).observe(canvas);
  resize();

  const overlay = $("train-overlay");
  function show(frame) {
    const phase = D.phaseOf(frame);
    const training = !phase || phase.name === "train";
    // The stylesheet keys the overlay off <body>, not off the element, so the
    // side panels can restyle for the training phase too.
    document.body.classList.toggle("phase-train", training);
    flyRoot.visible = !training;
    path.visible = !training;
    if (training) {
      $("train-trial").textContent = `trial ${frame + 1} of ${phase ? phase.stop : "?"}`;
      showTrainCard(D, frame);
      return;
    }
    const x = D.scalars.x[frame], y = D.scalars.y[frame], th = D.scalars.theta[frame];

    rig.setLoggedAngles(D.joints
      ? D.joints.subarray(frame * D.nJoints, (frame + 1) * D.nJoints)
      : new Float32Array(D.nJoints));
    rig.solve();
    for (let i = 0; i < bodyGroups.length; i++) {
      bodyGroups[i].position.set(rig.pos[i * 3], rig.pos[i * 3 + 1], rig.pos[i * 3 + 2]);
      bodyGroups[i].quaternion.set(
        rig.quat[i * 4], rig.quat[i * 4 + 1], rig.quat[i * 4 + 2], rig.quat[i * 4 + 3]
      );
    }
    flyRoot.position.set(x || 0, y || 0, 0);
    flyRoot.rotation.set(0, 0, th || 0);

    const start = D.episode.phases.find((p) => p.name === "behave").start;
    const pos = path.geometry.attributes.position;
    let n = 0;
    for (let f = start; f <= frame; f++, n++) {
      pos.array[n * 3] = D.scalars.x[f];
      pos.array[n * 3 + 1] = D.scalars.y[f];
      pos.array[n * 3 + 2] = 0.25;
    }
    path.geometry.setDrawRange(0, n);
    pos.needsUpdate = true;

    if (follow) {
      // Aim between the fly and the target rather than at the fly. Following
      // the animal alone fills the frame with a fly and no reason for what it
      // is doing.
      controls.target.lerp(
        new THREE.Vector3(((x || 0) + t.x) / 2, ((y || 0) + t.y) / 2, 1.5), 0.2
      );
    }
  }

  (function tick() {
    requestAnimationFrame(tick);
    controls.update();
    renderer.render(scene, camera);
  })();

  return { show };
}

/** Slice the one flat mesh buffer into indexed BufferGeometries. */
function meshGeometries(D) {
  const out = new Map();
  for (const [name, m] of Object.entries(D.rig.meshes)) {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(
      new Float32Array(D.meshBuf, m.positionOffset, m.vertexCount * 3), 3
    ));
    g.setIndex(new THREE.BufferAttribute(
      new Uint32Array(D.meshBuf, m.indexOffset, m.indexCount), 1
    ));
    g.computeVertexNormals();   // normals are not exported; it saves a third of the file
    out.set(name, g);
  }
  return out;
}

/* ------------------------------------------------------------------ brain */

function buildBrain(D) {
  const canvas = $("brain-canvas");
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x07070b);

  const n = D.soma.n;
  const pos = new Float32Array(n * 3);
  // Centre on the brain's own bounding box and flip x so the fly's right eye
  // lands on the viewer's right -- you are facing the animal.
  let lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < n * 3; i++) {
    const k = i % 3;
    lo[k] = Math.min(lo[k], D.soma.position[i]);
    hi[k] = Math.max(hi[k], D.soma.position[i]);
  }
  const mid = lo.map((v, k) => (v + hi[k]) / 2);
  const span = Math.max(...hi.map((v, k) => v - lo[k]));
  for (let i = 0; i < n; i++) {
    pos[i * 3] = -(D.soma.position[i * 3] - mid[0]) / span;
    pos[i * 3 + 1] = -(D.soma.position[i * 3 + 1] - mid[1]) / span;
    pos[i * 3 + 2] = -(D.soma.position[i * 3 + 2] - mid[2]) / span;
  }
  const colors = new Float32Array(n * 3);

  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  const points = new THREE.Points(geo, new THREE.PointsMaterial({
    size: 0.0042, vertexColors: true, sizeAttenuation: true,
    transparent: true, opacity: 0.95, depthWrite: false,
  }));
  scene.add(points);

  const camera = new THREE.PerspectiveCamera(38, 1, 0.01, 40);
  camera.up.set(0, 0, 1);
  const views = {
    front: [0, -1.5, 0], side: [1.5, 0, 0], top: [0, 0, 1.5],
  };
  camera.position.set(...views.front);
  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.target.set(0, 0, 0);
  for (const b of document.querySelectorAll("[data-brainview]")) {
    b.addEventListener("click", () => camera.position.set(...views[b.dataset.brainview]));
  }

  $("brain-counts").textContent =
    n.toLocaleString() + " somata, " + D.soma.matchedSomata.toLocaleString() + " live";
  $("brain-unmatched").textContent =
    (n - D.soma.matchedSomata).toLocaleString() + " dark: type never above threshold.";
  paintColorbar();

  function resize() {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
  new ResizeObserver(resize).observe(canvas);
  resize();

  const attr = geo.attributes.color;
  function show(frame) {
    const act = D.activityAt(frame);
    const { typeIndex, atlasToEpisode } = D.soma;
    for (let i = 0; i < n; i++) {
      const col = atlasToEpisode[typeIndex[i]];
      if (col < 0) {
        colors[i * 3] = 0.09; colors[i * 3 + 1] = 0.095; colors[i * 3 + 2] = 0.12;
      } else {
        ramp(act[col], colors, i * 3);
      }
    }
    attr.needsUpdate = true;
  }

  (function tick() {
    requestAnimationFrame(tick);
    controls.update();
    renderer.render(scene, camera);
  })();

  return { show };
}

function paintColorbar() {
  const c = $("colorbar-canvas");
  const ctx = c.getContext("2d");
  const img = ctx.createImageData(c.width, c.height);
  const rgb = new Float32Array(3);
  for (let x = 0; x < c.width; x++) {
    ramp(x / (c.width - 1), rgb, 0);
    for (let y = 0; y < c.height; y++) {
      const o = (y * c.width + x) * 4;
      img.data[o] = rgb[0] * 255; img.data[o + 1] = rgb[1] * 255;
      img.data[o + 2] = rgb[2] * 255; img.data[o + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
}

/* ----------------------------------------------------------------- traces */

const TRACES = [
  { key: "turn", label: "turn command (DNa02 R−L)", color: "#5ce0b0", signed: true },
  { key: "bearing", label: "angle to target (deg)", color: "#ffb347", signed: true },
  { key: "distance", label: "distance (mm)", color: "#7aa2ff", signed: false },
  { key: "modulation", label: "learned modulation ×", color: "#ff5ca8", signed: false },
  { key: "depression", label: "KC→MBON depression", color: "#c08cff", signed: false },
  { key: "reward", label: "reward (PAM)", color: "#ffd166", signed: false },
];

function buildTraces(D) {
  const canvas = $("trace-canvas");
  const ctx = canvas.getContext("2d");
  const rows = TRACES.filter((t) => D.scalars[t.key]);

  function resize() {
    const r = Math.min(devicePixelRatio, 2);
    canvas.width = canvas.clientWidth * r;
    canvas.height = canvas.clientHeight * r;
    ctx.setTransform(r, 0, 0, r, 0, 0);
  }
  new ResizeObserver(() => { resize(); draw(current); }).observe(canvas);
  resize();

  let current = 0;
  function draw(frame) {
    current = frame;
    const W = canvas.clientWidth, H = canvas.clientHeight;
    ctx.clearRect(0, 0, W, H);
    const pad = 8, rowH = (H - pad * 2) / rows.length;
    // Labels live in a left gutter. Without one the polyline is drawn straight
    // through its own caption and neither can be read.
    const gutter = 152, x0 = pad + gutter, plotW = W - pad - x0;
    const behave = D.episode.phases.find((p) => p.name === "behave");

    rows.forEach((t, r) => {
      const y0 = pad + r * rowH, h = rowH - 14;
      const v = D.scalars[t.key];
      let lo = Infinity, hi = -Infinity;
      for (let f = 0; f < D.frames; f++) {
        const x = v[f];
        if (Number.isFinite(x)) { lo = Math.min(lo, x); hi = Math.max(hi, x); }
      }
      if (!Number.isFinite(lo)) { lo = 0; hi = 1; }
      if (hi - lo < 1e-9) { hi = lo + 1e-9; }

      ctx.fillStyle = "#8a8a9a";
      ctx.font = "10px ui-monospace, monospace";
      ctx.fillText(t.label, pad, y0 + 9);
      const now = v[frame];
      ctx.fillStyle = t.color;
      ctx.textAlign = "right";
      ctx.fillText(Number.isFinite(now) ? now.toFixed(4) : "n/a", W - pad, y0 + 9);
      ctx.textAlign = "left";

      // The shaded span marks the behaviour phase; before it the fly is on a rig.
      if (behave) {
        ctx.fillStyle = "rgba(120,140,200,0.07)";
        const bx = x0 + (behave.start / (D.frames - 1)) * plotW;
        ctx.fillRect(bx, y0 + 12, W - pad - bx, h);
      }

      ctx.beginPath();
      let started = false;
      for (let f = 0; f < D.frames; f++) {
        const x = x0 + (f / (D.frames - 1)) * plotW;
        const val = v[f];
        if (!Number.isFinite(val)) { started = false; continue; }
        const y = y0 + 12 + h - ((val - lo) / (hi - lo)) * h;
        if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
      }
      ctx.strokeStyle = t.color;
      ctx.lineWidth = 1.4;
      ctx.stroke();

      const cx = x0 + (frame / (D.frames - 1)) * plotW;
      ctx.strokeStyle = "rgba(255,255,255,0.35)";
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(cx, y0 + 12); ctx.lineTo(cx, y0 + 12 + h); ctx.stroke();
    });
  }

  return { show: draw };
}

/* --------------------------------------------------------------- the rest */

function showTrainCard(D, frame) {
  for (const row of document.querySelectorAll("#train-overlay .mb-row")) {
    const k = row.dataset.k;
    const v = D.scalars[k] ? D.scalars[k][frame] : NaN;
    const span = k === "depression" ? 0.02 : k === "reward" ? 1 : 0.005;
    const frac = Math.max(0, Math.min(1, Math.abs(v) / span));
    row.querySelector(".mb-bar i").style.width = `${frac * 100}%`;
    row.querySelector(".mb-num").textContent =
      Number.isFinite(v) ? (Math.abs(v) < 0.001 ? v.toExponential(2) : v.toFixed(5)) : "--";
  }
  $("train-caveat").textContent =
    "Dopamine and Kenyon cell activity coincide, so KC→MBON synapses depress. " +
    "That much is the measured rule. Whether it reaches the legs is what the " +
    "behaviour phase shows, and at gain 1 the answer is no.";
}

function updateHud(D, frame) {
  const phase = D.phaseOf(frame);
  $("hud-phase").textContent = phase ? phase.name.toUpperCase() : "--";
  $("hud-frame").textContent = phase ? phase.detail : "";
  const keys = ["bearing", "distance", "turn", "forward", "modulation"];
  $("hud-state").innerHTML = keys
    .filter((k) => D.scalars[k])
    .map((k) => {
      const v = D.scalars[k][frame];
      return `<span class="k">${k}</span><span class="v">${
        Number.isFinite(v) ? v.toFixed(3) : "n/a"
      }</span>`;
    })
    .join("");
}

function wireTimeline(D, onFrame) {
  const slider = $("frame");
  slider.max = String(D.frames - 1);
  const playBtn = $("play");
  const speed = $("speed");

  const bar = $("phasebar");
  bar.innerHTML = D.episode.phases
    .map((p) => {
      const w = ((p.stop - p.start) / D.frames) * 100;
      return `<span class="seg ${p.name}" style="width:${w}%" title="${p.detail}">${p.name}</span>`;
    })
    .join("");

  let frame = 0, playing = false, last = 0, acc = 0;
  function set(f) {
    frame = Math.max(0, Math.min(D.frames - 1, f));
    slider.value = String(frame);
    $("tl-frame").textContent = `frame ${frame}`;
    $("tl-time").textContent = `t = ${D.scalars.t[frame].toFixed(2)} s`;
    onFrame(frame);
  }
  slider.addEventListener("input", () => set(Number(slider.value)));
  playBtn.addEventListener("click", () => {
    playing = !playing;
    playBtn.textContent = playing ? "pause" : "play";
    playBtn.classList.toggle("on", playing);
    last = performance.now();
  });
  addEventListener("keydown", (e) => {
    if (e.key === " ") { e.preventDefault(); playBtn.click(); }
    if (e.key === "ArrowRight") set(frame + 1);
    if (e.key === "ArrowLeft") set(frame - 1);
  });

  (function tick(now) {
    requestAnimationFrame(tick);
    if (!playing) { last = now; return; }
    acc += (now - last) * Number(speed.value);
    last = now;
    const step = D.episode.control_dt * 1000;
    while (acc >= step) {
      acc -= step;
      set(frame >= D.frames - 1 ? 0 : frame + 1);
    }
  })(performance.now());

  set(0);
}


/* ------------------------------------------------------------- stimulus */

/**
 * The form that hands a stimulus to the server and reloads on the result.
 *
 * A run is a page reload rather than an in-place swap. The rig and the atlas
 * come from the browser cache either way, and the alternative -- tearing down
 * two WebGL scenes and rebuilding them from a new episode -- is a lot of state
 * to get wrong for no visible gain.
 */
function wireStimulus(D) {
  const panel = $("stim-panel");
  const toggle = $("stim-toggle");
  const form = $("stim-form");
  const status = $("stim-status");
  const runBtn = $("stim-run");

  toggle.addEventListener("click", () => {
    panel.hidden = !panel.hidden;
    toggle.textContent = panel.hidden ? "set a stimulus" : "hide";
  });

  const bearing = $("f-bearing");
  const showBearing = () => {
    const v = Number(bearing.value);
    $("o-bearing").textContent =
      `${v > 0 ? "+" : ""}${v}\u00b0 ${v === 0 ? "(dead ahead)" : v > 0 ? "right" : "left"}`;
  };
  bearing.addEventListener("input", showBearing);
  showBearing();

  // Seed the form from the episode on screen, so "run" starts from what you
  // are looking at rather than from defaults you did not choose.
  const spec = D.episode.spec;
  if (spec) {
    for (const [id, key] of [
      ["f-bearing", "bearing"], ["f-distance", "distance"], ["f-radius", "radius"],
      ["f-train", "train_trials"], ["f-gain", "gain"], ["f-steps", "steps"],
    ]) {
      if (spec[key] !== undefined) $(id).value = spec[key];
    }
    if (spec.odour !== undefined) $("f-odour").value = spec.odour || "none";
    if (spec.coupling_hops) $("f-hops").value = String(spec.coupling_hops);
    if (spec.body) $("f-body").value = spec.body;
    showBearing();
  }

  const estimate = () => {
    const steps = Number($("f-steps").value) + Number($("f-train").value);
    const per = $("f-body").value === "physics" ? 2.0 : 0.29;
    return Math.round(steps * per);
  };
  const showEstimate = () => {
    $("body-hint").textContent =
      $("f-body").value === "physics"
        ? `legs and contact physics, about ${estimate()} s for this run`
        : "calibrated to the physics body it stands in for";
  };
  for (const id of ["f-body", "f-steps", "f-train"]) {
    $(id).addEventListener("input", showEstimate);
  }
  showEstimate();

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    runBtn.disabled = true;
    const began = performance.now();
    const tick = setInterval(() => {
      const s = ((performance.now() - began) / 1000).toFixed(0);
      status.textContent = `running\u2026 ${s}s of roughly ${estimate()}s`;
    }, 500);

    try {
      const res = await fetch("./api/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          bearing: Number(bearing.value),
          distance: Number($("f-distance").value),
          radius: Number($("f-radius").value),
          odour: $("f-odour").value,
          train_trials: Number($("f-train").value),
          steps: Number($("f-steps").value),
          gain: Number($("f-gain").value),
          coupling_hops: Number($("f-hops").value),
          body: $("f-body").value,
        }),
      });
      const out = await res.json();
      if (!res.ok) throw new Error(out.error || res.statusText);
      clearInterval(tick);
      status.textContent = `done in ${out.seconds}s, loading\u2026`;
      location.search = `?ep=${encodeURIComponent(out.name)}`;
    } catch (err) {
      clearInterval(tick);
      // A failed run is usually a server that is not the run endpoint -- the
      // plain static server has no /api/run -- so say which one is in front.
      status.textContent = `failed: ${err.message}. Started with \`flyloop serve\`?`;
      runBtn.disabled = false;
    }
  });
}
