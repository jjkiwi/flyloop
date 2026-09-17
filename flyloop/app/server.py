"""A local server that runs the fly on a stimulus you hand it.

Precomputed episodes were the only option while a control step cost 1.99 s --
1.8 s of that being MuJoCo. On the kinematic body, calibrated to the physics one
in Run 10 (13.7 mm/s, 4.2 rad/s), a step costs 0.29 s, and the expensive part
turns out not to be the run at all: building a :class:`~flyloop.hybrid.HybridLoop`
takes 15.4 s, nearly all of it the rate model and the MBON ensemble, and neither
depends on what the fly has learned or what it is looking at.

So this holds one built loop and calls
:meth:`~flyloop.hybrid.HybridLoop.reset_learning` between requests. A stimulus
comes back in about 13 seconds instead of 30. That is not live -- nothing here
will be live -- but it is short enough to change something and look again, which
is the loop that matters for asking questions of a model.

**The physics body is still available and still slow.** ``body: "physics"``
costs about 2 s per control step, so a 40-step episode is 90 seconds. The server
will do it; it just says so first.

Nothing about the brain, the readout or the signs changes between the two. The
only thing that differs is what the descending command moves.
"""

from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import dataclass
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from ..experiments.embodied import target_at
from ..hybrid import CONTROL_ODOUR, TRAINED_ODOUR

#: Default arena geometry, matching the recorded episodes.
DEFAULT_DISTANCE = 25.0
DEFAULT_RADIUS = 2.5

#: More glomeruli than this in one odour is almost certainly a mistake -- the
#: whole antennal lobe at once is not an odour, it is every odour.
MAX_GLOMERULI = 12

#: A run this long on the physics body takes a minute and a half.
SLOW_BODY_WARNING = 20


def _glomeruli_field(value, which: str) -> tuple[str, ...]:
    """Coerce whatever JSON supplied into a tuple of glomerulus names.

    A browser sends a list, a saved manifest round-trips a list, and a hand
    written request might send one string. All three are accepted; anything
    else, and anything empty, is refused here rather than deep inside the
    network where the failure reads as a flat odour response.

    Names are *not* checked against a hard coded list -- :meth:`FlyService.run`
    checks them against the connectome that is actually loaded, which is the
    only list that is true.
    """
    if isinstance(value, str):
        value = [value]
    try:
        names = [str(g).strip() for g in value]
    except TypeError as exc:
        raise ValueError(f"{which} glomeruli must be a list of names") from exc
    names = [g for g in names if g]
    if not names:
        raise ValueError(f"the {which} odour needs at least one glomerulus")
    if len(names) > MAX_GLOMERULI:
        raise ValueError(
            f"the {which} odour asks for {len(names)} glomeruli; "
            f"at most {MAX_GLOMERULI}"
        )
    # Order carries no meaning -- the odour vector is a set of ORNs -- so a
    # repeat is silently one glomerulus rather than an error.
    return tuple(dict.fromkeys(names))


@dataclass
class Spec:
    """What the browser asked for. Everything has a default that works."""

    bearing: float = 35.0
    distance: float = DEFAULT_DISTANCE
    radius: float = DEFAULT_RADIUS
    odour: str | None = "trained"
    train_trials: int = 8
    train_odour: str = "trained"
    steps: int = 40
    gain: float = 1.0
    coupling_hops: int = 1
    body: str = "kinematic"
    #: Which glomeruli each of the two odours is made of. "trained" is the one
    #: dopamine is paired with; "control" is the one that is not, and exists so
    #: the first can be measured against something. Names are validated against
    #: the loaded connectome, not against a list kept here -- MaleCNS has 53
    #: and a different dataset will have different ones.
    trained_glomeruli: tuple[str, ...] = TRAINED_ODOUR
    control_glomeruli: tuple[str, ...] = CONTROL_ODOUR

    @classmethod
    def from_json(cls, raw: dict) -> "Spec":
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"unknown fields: {sorted(unknown)}")
        spec = cls(**{k: raw[k] for k in raw})
        if spec.body not in ("kinematic", "physics"):
            raise ValueError(f"body must be kinematic or physics, not {spec.body!r}")
        if spec.odour not in (None, "none", "trained", "control"):
            raise ValueError(f"unknown odour {spec.odour!r}")
        if spec.train_odour not in ("trained", "control"):
            raise ValueError(f"unknown train_odour {spec.train_odour!r}")
        if not 1 <= spec.steps <= 400:
            raise ValueError("steps must be between 1 and 400")
        if not 0 <= spec.train_trials <= 100:
            raise ValueError("train_trials must be between 0 and 100")
        spec.trained_glomeruli = _glomeruli_field(spec.trained_glomeruli, "trained")
        spec.control_glomeruli = _glomeruli_field(spec.control_glomeruli, "control")
        if set(spec.trained_glomeruli) == set(spec.control_glomeruli):
            raise ValueError(
                "the trained and control odours are the same glomeruli, so "
                "training would have nothing to be measured against"
            )
        if spec.odour == "none":
            spec.odour = None
        return spec

    def key(self) -> tuple:
        """What a cached loop must match. Everything else is per-run."""
        return (self.gain, self.coupling_hops, self.body)


class FlyService:
    """One loaded connectome, one built loop per configuration."""

    def __init__(self, data_root: str, dataset: str = "malecns"):
        self.data_root = data_root
        self.dataset = dataset
        self._connectome = None
        self._loops: dict[tuple, object] = {}
        self._lock = threading.Lock()
        self.status = "idle"

    def connectome(self):
        if self._connectome is None:
            from ..connectome.data_prep import load_dataset

            self.status = "loading connectome"
            self._connectome = load_dataset(self.data_root, self.dataset, matrix="inprop")
        return self._connectome

    def glomeruli(self) -> dict:
        """Every odour the loaded connectome can actually present, with counts.

        Loads the connectome if nothing has yet, which costs about 25 s the
        first time -- so the browser asks for this while the page is coming up
        rather than when the picker is opened.
        """
        from ..experiments.olfactory import glomeruli

        counts = glomeruli(self.connectome())
        return {
            "connectome": self.connectome().name,
            "glomeruli": [
                {"name": str(name), "orns": int(n)} for name, n in counts.items()
            ],
            "defaults": {"trained": list(TRAINED_ODOUR), "control": list(CONTROL_ODOUR)},
        }

    def loop(self, spec: Spec):
        key = spec.key()
        if key not in self._loops:
            from ..hybrid import HybridLoop

            self.status = "building the rate model"
            body = None
            if spec.body == "physics":
                from ..body.nmf_body import NeuroMechFlyBody

                body = NeuroMechFlyBody(control_dt=0.05, seed=0)
            self._loops[key] = HybridLoop(
                self.connectome(),
                target=target_at(spec.bearing, distance=spec.distance),
                gain=spec.gain,
                coupling_hops=spec.coupling_hops,
                body=body,
            )
        return self._loops[key]

    def run(self, spec: Spec, out_dir: Path) -> dict:
        """Run one stimulus and write the episode where the viewer can fetch it."""
        from .episode import record_episode, write_index

        # One request at a time: a HybridLoop carries mutable weights and a body,
        # and two concurrent runs would interleave into each other's learning.
        with self._lock:
            started = time.time()
            loop = self.loop(spec)
            loop.target = target_at(spec.bearing, distance=spec.distance)
            loop.target.radius = spec.radius
            # Cheap -- an odour is a vector over the ORNs -- so the glomeruli
            # are per-request and do not force the 15 s rebuild that gain and
            # coupling_hops do. An unknown name raises KeyError here, which the
            # handler turns into a 400 with the list of names that do exist.
            loop.set_odours(spec.trained_glomeruli, spec.control_glomeruli)
            loop.reset_learning()
            # Before training, not after: this is a property of the glomeruli,
            # and measuring it on the trained weights would invite reading it
            # as something the pairing did. NaN -- no Kenyon cell fires at all
            # -- becomes null, because NaN is not JSON a browser will parse.
            code = {
                k: None if isinstance(v, float) and math.isnan(v) else v
                for k, v in loop.kc_code().items()
            }
            # Carried into the manifest so the form can reopen showing the
            # stimulus that produced whatever is on screen.
            loop._spec = spec.__dict__

            self.status = "running"
            episode = record_episode(
                loop,
                train_trials=spec.train_trials,
                behave_steps=spec.steps,
                train_odour=spec.train_odour,
                behave_odour=spec.odour,
            )
            episode.manifest["kc_code"] = code
            name = f"run{int(started)}"
            episode.save(out_dir / "episodes" / name)
            write_index(out_dir)
            self.status = "idle"
            return {
                "name": name,
                "path": f"episodes/{name}/",
                "seconds": round(time.time() - started, 1),
                "spec": spec.__dict__,
                # How the two odours land on the Kenyon cells. An odour no
                # Kenyon cell responds to cannot be learned, and two odours
                # that overlap completely cannot be told apart -- both are
                # properties of the glomeruli chosen and not of the training,
                # so they are reported next to the run that used them rather
                # than left to be discovered.
                "kc_code": code,
                **{
                    k: episode.manifest[k]
                    for k in ("gain", "coupling_share", "coupling_hops", "live_types")
                },
            }


def make_handler(service: FlyService, root: Path):
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(root), **kw)

        def log_message(self, fmt, *args):  # pragma: no cover - quieter console
            pass

        def _json(self, code: int, payload: dict) -> None:
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802 - required name
            if self.path == "/api/status":
                return self._json(200, {"status": service.status})
            if self.path == "/api/glomeruli":
                try:
                    return self._json(200, service.glomeruli())
                except Exception as exc:  # pragma: no cover - surfaced to browser
                    return self._json(500, {"error": f"{type(exc).__name__}: {exc}"})
            return super().do_GET()

        def do_POST(self):  # noqa: N802 - required name
            if self.path != "/api/run":
                return self._json(404, {"error": "no such endpoint"})
            length = int(self.headers.get("Content-Length", 0))
            try:
                spec = Spec.from_json(json.loads(self.rfile.read(length) or b"{}"))
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                return self._json(400, {"error": str(exc)})
            try:
                return self._json(200, service.run(spec, root / "data"))
            except (KeyError, ValueError) as exc:
                # A glomerulus this connectome does not have, or a pair with
                # nothing to compare. Both are the request's fault, and the
                # message already names what would have worked.
                service.status = "idle"
                # str() of a KeyError is the repr of its argument, so the
                # message arrives at the browser wrapped in quotes it did not
                # have. Take the argument itself.
                message = exc.args[0] if exc.args else str(exc)
                return self._json(400, {"error": str(message)})
            except Exception as exc:  # pragma: no cover - surfaced to the browser
                service.status = "idle"
                return self._json(500, {"error": f"{type(exc).__name__}: {exc}"})

    return Handler


def serve(
    data_root: str, *, root: Path | None = None, port: int = 8000, dataset: str = "malecns"
) -> None:
    """Serve the viewer and the run endpoint from ``root`` (default: ``app/``)."""
    root = Path(root or Path(__file__).resolve().parents[2] / "app")
    service = FlyService(data_root, dataset)
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(service, root))
    print(f"flyloop: http://127.0.0.1:{port}/  (serving {root})")
    print("  the connectome loads on the first run, which costs about 25 s;")
    print("  every run after that is about 13 s on the kinematic body.")
    server.serve_forever()
