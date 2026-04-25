"""Configuration parameters for the cart+double-pendulum environment."""

from dataclasses import dataclass


@dataclass
class HoldConfig:
    """Configuration for the physics environment and controller.
    
    Attributes:
        u_limit: Maximum control force magnitude.
        x_limit: Maximum cart position before failure.
        max_steps: Maximum steps per episode.
        dt: Time step for simulation.
        fall_threshold: Angle threshold for pendulum fall detection.
        kx, vx: PD gains for cart position/velocity.
        k1, kw1: PD gains for first pendulum.
        k2, kw2: PD gains for second pendulum.
    """
    u_limit: float = 200.0
    x_limit: float = 8.0
    max_steps: int = 5000
    dt: float = 1.0 / 240.0

    # failure threshold (wide enough to let the system evolve)
    fall_threshold: float = 1.0

    # PD gains (stabilization baseline)
    kx: float = 0.0
    vx: float = 0.1
    k1: float = 15.0
    kw1: float = 3.0
    k2: float = 10.0
    kw2: float = 2.5
