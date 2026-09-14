"""flyloop -- an embodied sensorimotor loop driven by a connectome-derived brain.

The package is deliberately split so that each stage of the project can be run,
tested and falsified on its own:

    connectome/  the wiring diagram (real MaleCNS / FlyWire, or synthetic)
    brain/       leaky integrate-and-fire dynamics over that wiring
    vision/      camera frames -> ommatidial sampling -> input currents
    motor/       descending-neuron spike rates -> locomotor command
    body/        the thing that moves (kinematic stub, or NeuroMechFly)
    experiments/ falsifiable tests and non-connectome baselines
"""

__version__ = "0.1.0"
