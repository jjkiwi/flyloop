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


def neuromechfly_body(**kwargs):
    """Construct a :class:`NeuroMechFlyBody`, importing MuJoCo only when asked."""
    from .nmf_body import NeuroMechFlyBody

    return NeuroMechFlyBody(**kwargs)


def flygym_body(**kwargs):
    """Construct a :class:`FlyGymBody`, importing MuJoCo only when asked to."""
    from .flygym_body import FlyGymBody

    return FlyGymBody(**kwargs)
