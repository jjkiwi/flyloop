from .base import Body
from .kinematic import Arena, KinematicBody, Pillar, cluttered_arena, looming_arena

__all__ = [
    "Arena",
    "Body",
    "KinematicBody",
    "Pillar",
    "cluttered_arena",
    "looming_arena",
]


def flygym_body(**kwargs):
    """Construct a :class:`FlyGymBody`, importing MuJoCo only when asked to."""
    from .flygym_body import FlyGymBody

    return FlyGymBody(**kwargs)
