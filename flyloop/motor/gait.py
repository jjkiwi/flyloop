"""A tripod gait generator: locomotor command -> joint angles.

This is *not* connectome-derived and is not pretending to be.  It is the
minimal piece of engineering needed to turn a two-number command into
eighteen servo targets, and it exists so the hexapod stays upright while the
interesting question -- what the brain sends down -- is being studied.

Keeping it explicitly separate matters: when the robot walks, this is the part
that deserves the credit for walking.  The fly's contribution is the command.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Front, middle, hind leg on each side.
LEGS = ("LF", "LM", "LH", "RF", "RM", "RH")
#: Tripod groups: these three legs swing together.
TRIPOD_A = ("LF", "RM", "LH")
TRIPOD_B = ("RF", "LM", "RH")


@dataclass
class GaitParams:
    base_frequency: float = 6.0  # Hz at full forward drive
    stride_amplitude: float = 0.5  # rad of coxa swing
    lift_amplitude: float = 0.35  # rad of femur lift
    turn_asymmetry: float = 0.8  # how strongly turning shortens the inner stride


class TripodGait:
    """Central pattern generator producing 3 joint angles per leg."""

    def __init__(self, params: GaitParams | None = None):
        self.p = params or GaitParams()
        self.phase = 0.0

    def reset(self) -> None:
        self.phase = 0.0

    def step(self, forward: float, turn: float, dt: float) -> dict[str, np.ndarray]:
        """Advance the CPG and return ``{leg: [coxa, femur, tibia]}`` in radians."""
        p = self.p
        speed = abs(float(forward))
        direction = 1.0 if forward >= 0 else -1.0
        self.phase = (self.phase + 2 * np.pi * p.base_frequency * speed * dt) % (2 * np.pi)

        out: dict[str, np.ndarray] = {}
        for leg in LEGS:
            offset = 0.0 if leg in TRIPOD_A else np.pi
            ph = self.phase + offset
            side_sign = -1.0 if leg[0] == "L" else 1.0
            # Turning shortens the stride on the inside of the turn.
            inner = turn * side_sign
            stride = p.stride_amplitude * (1.0 - p.turn_asymmetry * max(inner, 0.0))
            coxa = direction * stride * np.sin(ph) * speed
            # Legs lift only during the swing half-cycle.
            femur = p.lift_amplitude * max(np.cos(ph), 0.0) * speed
            tibia = -0.5 * femur
            out[leg] = np.array([coxa, femur, tibia], dtype=np.float32)
        return out

    @staticmethod
    def flatten(joints: dict[str, np.ndarray]) -> np.ndarray:
        """Flatten to the 18-element vector a hexapod servo bus expects."""
        return np.concatenate([joints[leg] for leg in LEGS])
