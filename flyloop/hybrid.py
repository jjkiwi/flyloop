"""A reflex with a learned base: odour says whether, sight says which way.

This is the only arrangement of the two systems that this connectome permits,
and the wiring is what decides it rather than a modelling preference:

* **LC4 and LPLC2 supply 0.0000% of Kenyon cell input.** Whatever the fly
  learns here cannot be something it saw. It has to be an odour.
* **MBONs supply 0.53% of DNa02's input** and nothing at all to DNa01 or DNp09,
  so the learned valence has exactly one door into steering, and it is a narrow
  one.

So the loop runs in two phases. In **training** the fly meets an odour paired
with dopamine addressed to its own PAM cluster -- see
:mod:`flyloop.brain.dopamine` on what that does and does not mean -- and the
coincidence
depresses KC->MBON synapses -- the measured rule, not an abstract reward
signal. In **behaviour** it looks at an object and walks, with the visual
reflex of Run 10 unchanged, while the learned odour valence scales how hard it
approaches.

**What the two phases share is the weight matrix and nothing else.** Training
does not move the body and behaviour does not deliver dopamine; the only thing
that crosses between them is the set of depressed synapses. That separation is
deliberate, because it is the one that makes the result interpretable: if
behaviour changes, it changed through the synapses.

**The honest part is the gain.** At the measured 0.53% a fully trained valence
moves the approach drive by a fraction of a percent, which is real and
invisible. :class:`~flyloop.brain.coupling.LearnedBias` scales that, and every
episode records the gain it ran at, so no result from here can be read without
it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .body.kinematic import Arena, KinematicBody
from .brain.coupling import LearnedBias, learned_bias
from .brain.dopamine import dopaminergic, proximity_reward
from .brain.rate import population_index, rate_brain, steady_state
from .connectome.schema import Connectome
from .experiments.olfactory import (
    SPARSE_KC_SLOPE,
    kc_slopes,
    mushroom_body,
    odour,
    olfactory_receptors,
)
from .loop_rate import Target
from .motor.descending import LocomotorCommand
from .vision.hexproject import HexWorldView

#: Glomeruli of the odour the fly is trained on, and of the one it is not.
TRAINED_ODOUR = ("DM1", "DM4")
CONTROL_ODOUR = ("DA1", "VA2")

#: Populations worth a trace in the side panel of an application.
WATCH = ("DNa02", "DNa01", "LC4", "LPLC2", "MDN", "DNp09")

#: Kinematic stand-ins for the physics body, measured from it rather than
#: guessed: NeuroMechFly covered 27.4 mm in 2.0 s walking straight, and swung
#: 96 deg in 0.4 s under a full turn command. Matching them means an episode can
#: be roughed out on the stub in milliseconds and re-run in physics unchanged.
#: Everything here is millimetres, because Target is.
WALK_MM_PER_S = 13.7
TURN_RAD_PER_S = 4.2


@dataclass
class Phase:
    """One recorded segment of an episode, so a player can label the timeline."""

    name: str
    start: int
    stop: int
    detail: str = ""


class HybridLoop:
    """Visual fixation, with an odour valence the fly was taught beforehand."""

    def __init__(
        self,
        connectome: Connectome,
        *,
        target: Target | None = None,
        hops: int = 5,
        dt: float = 0.05,
        bias: float = 0.0,
        turn_gain: float = 3.0,
        gain: float = 1.0,
        coupling_hops: int = 1,
        learning_rate: float = 0.2,
        reward_scale: float = 8.0,
        kc_slope: float = SPARSE_KC_SLOPE,
        trained_odour: tuple[str, ...] = TRAINED_ODOUR,
        body=None,
        view: HexWorldView | None = None,
    ):
        self.c = connectome
        self.target = target or Target()
        self.hops = hops
        self.dt = dt
        self.turn_gain = turn_gain
        self.reward_scale = reward_scale

        self.view = view or HexWorldView(connectome)
        self.pam = dopaminergic(connectome, cluster="PAM")
        self.orn = olfactory_receptors(connectome)
        self.mb = mushroom_body(connectome)

        self.sensory = np.unique(np.concatenate([self.view.sensory, self.pam, self.orn]))
        pos = {int(v): i for i, v in enumerate(self.sensory)}
        self._visual = np.array([pos[int(v)] for v in self.view.sensory], dtype=np.int64)
        self._pam = np.array([pos[int(v)] for v in self.pam], dtype=np.int64)
        self._orn = np.array([pos[int(v)] for v in self.orn], dtype=np.int64)

        # Kenyon cells get the shallow slope that reproduces the ~5% sparse
        # odour code; without it the "learned" representation is dense and the
        # plasticity stops being selective (Run 9).
        self.brain = rate_brain(
            connectome,
            self.sensory,
            num_layers=hops,
            default_bias=bias,
            slope_by_type=kc_slopes(connectome, kc_slope),
        )
        self.plastic = self.brain.plasticity(learning_rate=learning_rate)
        self.learned: LearnedBias = learned_bias(
            connectome, gain=gain, coupling_hops=coupling_hops
        )

        self.odours = {
            name: odour(connectome, glom, self.orn)
            for name, glom in (("trained", trained_odour), ("control", CONTROL_ODOUR))
        }
        self.trained_odour = trained_odour
        #: The KC->MBON weights as they were before any training, so every
        #: step can be measured against itself rather than against a separate
        #: baseline run. A baseline taken on the odour alone is not comparable:
        #: the behaving fly also has an object in view, and that moved the
        #: readout by 0.001 -- twenty times the learned effect it was meant to
        #: isolate. Paired within the step is the only honest comparison.
        self._naive = np.asarray(self.plastic.initial).copy()

        self.record = {}
        for side in ("L", "R"):
            self.record.update(population_index(connectome, WATCH, side=side))

        self.body = body or KinematicBody(
            Arena([self.target.as_pillar()]),
            dt=dt,
            speed=WALK_MM_PER_S,
            turn_rate=TURN_RAD_PER_S,
        )
        self.phases: list[Phase] = []
        self.rows: list[dict] = []
        #: Called with the full (neurons, hops) activation after every frame of
        #: either phase. A recorder uses it to keep per-cell-type activity
        #: without the loop having to know what a recorder is.
        self.on_frame = None
        self.reset()

    @property
    def body_has_joints(self) -> bool:
        """Whether the body can report leg posture. The kinematic stub cannot."""
        return callable(getattr(self.body, "joint_angles", None))

    def _paired_readout(self, inp: np.ndarray):
        """Run one input twice: on the trained weights and on the naive ones.

        Returns ``(result, trained, naive)``. The learned component is their
        difference, which cancels everything the odour and the object do
        innately and leaves only what the pairing changed. It costs a second
        pass through the network -- 0.16 s against the physics step's 1.8 s, so
        the exactness is close to free.
        """
        res = self.brain.run(inp, record=self.record)
        trained = self.learned.readout(res.activations)
        held = np.asarray(self.brain.values[self.plastic.entry_index]).copy()
        self.brain.values[self.plastic.entry_index] = self._naive
        naive = self.learned.readout(self.brain.run(inp).activations)
        self.brain.values[self.plastic.entry_index] = held
        return res, trained, naive

    def reset_learning(self) -> None:
        """Put the KC->MBON weights back as they were before any training.

        Building a loop costs about 15 s, almost all of it the rate model and
        the MBON ensemble, and neither depends on what the fly has learned. So
        an interactive server keeps one loop alive and calls this between runs
        instead of rebuilding, which turns a 30 s request into a 13 s one.
        """
        self.brain.values[self.plastic.entry_index] = self._naive
        self.plastic.history.clear()
        self.learned.baseline = 0.0

    def reset(self) -> None:
        self.body.reset()
        self.t = 0.0
        self.rows = []
        self.phases = []

    # ------------------------------------------------------------- stimulus

    def _input(
        self, *, visual=None, reward: float = 0.0, smell: np.ndarray | None = None
    ) -> np.ndarray:
        v = np.zeros(len(self.sensory), dtype=np.float32)
        if visual is not None:
            v[self._visual] = visual
        if reward:
            v[self._pam] = reward
        if smell is not None:
            v[self._orn] = smell
        return v

    # ------------------------------------------------------------- training

    def train(self, n_trials: int = 8, *, odour_name: str = "trained") -> pd.DataFrame:
        """Pair an odour with dopamine, and let the synapses record it.

        The body does not move: this is the fly on a rig, learning a smell.
        Dopamine grows across trials the way it does when a source is
        approached, which is the shaping rule Run 6 introduced and the only one
        that makes a graded association learnable here.
        """
        start = len(self.rows)
        smell = self.odours[odour_name]
        out = []
        for trial in range(n_trials):
            # Closer each trial, so both the odour and the reward grow.
            distance = self.reward_scale * (1.0 - trial / max(n_trials, 1))
            reward = proximity_reward(distance, scale=self.reward_scale)
            # Paired here too, not just in the behaviour phase. Without it the
            # training rows report the raw readout as if it were the learned
            # change, which looks like a large effect on trial one and is really
            # just the odour's innate response.
            inp = steady_state(self._input(smell=smell * 1.0, reward=reward), self.hops)
            res, _, naive = self._paired_readout(inp)
            self.learned.baseline = naive
            acts = res.activations
            depression = self.plastic.step(acts[self.plastic.kc].max(axis=1), reward)
            if self.on_frame is not None:
                self.on_frame(acts, phase="train")
            row = self._row(
                phase="train",
                acts=acts,
                res=res,
                reward=reward,
                depression=depression,
                modulation=float("nan"),
                extra={"trial": trial, "odour": odour_name},
            )
            self.rows.append(row)
            out.append(row)
            # Training has no physical duration, but a player needs both phases
            # on one axis, so each trial advances the clock like a control step.
            self.t += self.dt
        self.phases.append(
            Phase("train", start, len(self.rows), f"{n_trials} trials, odour {odour_name}")
        )
        return pd.DataFrame(out)

    # ------------------------------------------------------------ behaviour

    def run(
        self,
        n_steps: int = 30,
        *,
        odour_name: str | None = "trained",
        progress: bool = False,
    ) -> pd.DataFrame:
        """Walk at the target, with the learned valence scaling the approach."""
        start = len(self.rows)
        smell = self.odours[odour_name] if odour_name else None
        out = []
        for i in range(n_steps):
            row = self.step(smell=smell, odour_name=odour_name)
            out.append(row)
            if progress:  # pragma: no cover
                print(
                    f"    {i:3d}  d={row['distance']:.1f}  bearing={row['bearing']:+6.1f}"
                    f"  turn={row['turn']:+.3f}  mod={row['modulation']:.4f}",
                    flush=True,
                )
            if row["distance"] <= self.target.radius:
                break
        self.phases.append(
            Phase("behave", start, len(self.rows), f"odour {odour_name or 'none'}")
        )
        return pd.DataFrame(out)

    def step(self, *, smell=None, odour_name: str | None = None) -> dict:
        st = self.body.state()
        bearing, half_width, dist = self.target.seen_from(st["x"], st["y"], st["theta"])

        inp = steady_state(
            self._input(visual=self.view.pattern([(bearing, half_width)]), smell=smell),
            self.hops,
        )
        res, trained, naive = self._paired_readout(inp)
        self.learned.baseline = naive
        peak = res.peak()
        turn = float(peak.get("DNa02_R", 0.0) - peak.get("DNa02_L", 0.0))
        forward = 0.5 * (peak.get("DNa01_L", 0.0) + peak.get("DNa01_R", 0.0))

        # The learned valence scales the approach; it does not steer. With no
        # odour in the air the factor is exactly 1 and this is Run 10 unchanged.
        mod = self.learned.modulation(res.activations) if smell is not None else 1.0
        cmd = LocomotorCommand(
            forward=float(np.clip((0.4 + forward) * mod, 0.0, 1.0)),
            turn=float(np.clip(self.turn_gain * turn * mod, -1.0, 1.0)),
        )
        if self.on_frame is not None:
            self.on_frame(res.activations, phase="behave")
        row = self._row(
            phase="behave",
            acts=res.activations,
            res=res,
            reward=0.0,
            depression=self.plastic.depression(),
            modulation=mod,
            extra={
                "x": st["x"],
                "y": st["y"],
                "theta": st["theta"],
                "distance": dist,
                "bearing": bearing,
                "half_width": half_width,
                "turn": cmd.turn,
                "forward": cmd.forward,
                "odour": odour_name or "none",
            },
        )
        self.body.step(cmd)
        self.t += self.dt
        self.rows.append(row)
        return row

    # --------------------------------------------------------------- record

    def _row(self, *, phase, acts, res, reward, depression, modulation, extra) -> dict:
        peak = res.peak()
        row = {
            "t": self.t,
            "phase": phase,
            "reward": reward,
            "depression": depression,
            "modulation": modulation,
            "mbon_readout": self.learned.readout(acts),
            "learned_component": self.learned.learned_component(acts),
            "baseline": self.learned.baseline,
            "gain": self.learned.gain,
        }
        row.update({k: float(peak.get(k, 0.0)) for k in self.record})
        row.update(extra)
        return row

    def log(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)
